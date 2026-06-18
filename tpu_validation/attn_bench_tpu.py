#!/usr/bin/env python3
"""Prefill / decode GQA attention microbench for TPU v6e (real-TPU side; JAX).

GOAL: measure real v6e latency for the SAME GQA attention shapes we run in
PyTorchSim (flash-SDPA template), so we can validate the simulator. We sweep the
sequence / KV-cache length and compare the *scaling* (slope = marginal cost per
token) between sim and real -- the absolute offset differs because the real TPU
has a fixed per-launch host/dispatch overhead (tens of us) that dominates at the
small B=1 shapes, which the cycle simulator does not model the same way.

Layout note: jax.nn.dot_product_attention expects [batch, seq, heads, head_dim].
GQA is native: q has Hq heads, k/v have Hkv heads, Hq % Hkv == 0.

  prefill: q_seq = kv_seq = S,  is_causal=True   (standard causal prefill)
  decode : q_seq = 1, kv_seq = S, is_causal=False (one new token attends all KV)

Run on the TPU VM:
    python3 attn_bench_tpu.py
Writes attn_bench_results.json + per-config XProf traces under ./tb_logs_attn/.
"""
import os, time, json, argparse
import jax, jax.numpy as jnp

# LLAMA4_TP8 per-device GQA shape (matches the sim runs)
B, HQ, HKV, D = 1, 5, 1, 128
SEQS = [256, 512, 1024, 2048, 4096]
DTYPES = {"fp16": jnp.float16, "bf16": jnp.bfloat16}
WARMUP, MEASURE, PROFILE = 20, 100, 5
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_attn")
SCALE = 1.0 / (D ** 0.5)


def make_attn(is_causal):
    @jax.jit
    def fn(q, k, v):
        return jax.nn.dot_product_attention(
            q, k, v, scale=SCALE, is_causal=is_causal, implementation="xla")
    return fn


def bench(mode, S, dtype_name, dtype, batch):
    Lq = S if mode == "prefill" else 1
    is_causal = (mode == "prefill")
    key = jax.random.PRNGKey(0)
    # [batch, seq, heads, head_dim]
    q = (jax.random.normal(key, (batch, Lq, HQ, D), jnp.float32) * 0.1).astype(dtype)
    k = (jax.random.normal(key, (batch, S,  HKV, D), jnp.float32) * 0.1).astype(dtype)
    v = (jax.random.normal(key, (batch, S,  HKV, D), jnp.float32) * 0.1).astype(dtype)
    fn = make_attn(is_causal)

    out = fn(q, k, v); out.block_until_ready()           # compile

    logdir = os.path.join(LOGROOT, f"{mode}_{dtype_name}_S{S}_B{batch}")
    os.makedirs(logdir, exist_ok=True)
    # Dump optimized HLO so we can tell whether attention ran as ONE fused
    # kernel/custom-call vs decomposed into separate dot/softmax ops.
    try:
        hlo = fn.lower(q, k, v).compile().as_text()
        open(os.path.join(logdir, "optimized_hlo.txt"), "w").write(hlo)
        n_fusion = hlo.count("fusion(")
        n_custom = hlo.count("custom-call")
        n_dot = hlo.count(" dot(")
        flash = ("attention" in hlo.lower()) or (n_custom > 0)
        print(f"    [HLO] fusions={n_fusion} custom-call={n_custom} dot={n_dot} "
              f"-> {'FUSED/flash custom-call' if flash else 'DECOMPOSED (separate ops)'}")
    except Exception as e:
        print(f"    [HLO] dump failed: {e}")
        n_fusion = n_custom = n_dot = -1

    for _ in range(WARMUP):
        out = fn(q, k, v)
    out.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        out = fn(q, k, v)
    out.block_until_ready()
    avg_s = (time.perf_counter() - t0) / MEASURE

    with jax.profiler.trace(logdir):
        for _ in range(PROFILE):
            out = fn(q, k, v)
        out.block_until_ready()

    # FLOPs: QK^T (2*Lq*S*D) + softmax*V (2*Lq*S*D), over HQ heads, batch.
    # causal prefill ~ halves the work; report both raw and effective.
    causal_factor = 0.5 if is_causal else 1.0
    flop = 2.0 * (2.0 * Lq * S * D) * HQ * batch * causal_factor
    tflops = flop / avg_s / 1e12
    print(f"{mode:>7} S={S:>5} {dtype_name} B={batch}: {avg_s*1e6:9.2f} us/iter  {tflops:8.3f} TFLOP/s")
    return dict(mode=mode, S=S, dtype=dtype_name, batch=batch, Lq=Lq,
                Hq=HQ, Hkv=HKV, D=D, is_causal=is_causal,
                avg_us=avg_s * 1e6, tflops=tflops,
                hlo_fusions=n_fusion, hlo_custom_call=n_custom, hlo_dot=n_dot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=B)
    ap.add_argument("--modes", default="prefill,decode")
    ap.add_argument("--dtypes", default="fp16,bf16")
    a = ap.parse_args()
    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    print(f"shape: Hq={HQ} Hkv={HKV} D={D}  warmup={WARMUP} measure={MEASURE}\n")
    results = []
    for mode in a.modes.split(","):
        for dtype_name in a.dtypes.split(","):
            for S in SEQS:
                try:
                    results.append(bench(mode, S, dtype_name, DTYPES[dtype_name], a.batch))
                except Exception as e:
                    print(f"{mode} S={S} {dtype_name}: FAILED {e}")
            print()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_bench_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"saved {out}")
    print(f"XProf traces in {LOGROOT}/  (scp back; read device-side op duration per config)")


if __name__ == "__main__":
    main()
