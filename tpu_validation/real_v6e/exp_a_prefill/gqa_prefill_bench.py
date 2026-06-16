#!/usr/bin/env python3
"""EXP-A: LLAMA4_TP8 GQA attention prefill latency on real TPU v6e.

Attention-only, causal, bf16 (Mosaic flash is bf16-only).
Shapes match PyTorchSim sim kernel for apples-to-apples comparison:
  q  [B, S, Hq=5, D=128]
  kv [B, S, Hkv=1, D=128]   (GQA group G = Hq/Hkv = 5)

Timing harness mirrors llama_layer_bench.py: 30 warmup → 50 timed → 5-iter profile.
Outputs:
  - JSONL line per (B, S) into gqa_prefill_results.jsonl
  - TensorBoard profile under tb_logs_gqa_prefill/<dtype>_b{B}_s{S}/

  python gqa_prefill_bench.py --seq 2048 --batch 1 --dtype bf16
"""
import os, time, json, argparse, math
import jax, jax.numpy as jnp

# LLAMA4_TP8 per-partition
HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS = 128, 5, 1
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_gqa_prefill")
DTYPES = {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--profile_iters", type=int, default=5)
    p.add_argument("--no_profile", action="store_true")
    a = p.parse_args()

    dt, B, S = DTYPES[a.dtype], a.batch, a.seq
    D, Hq, Hkv = HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS
    scale = 1.0 / math.sqrt(D)
    print(f"JAX {jax.__version__} devs={jax.devices()}  prefill B={B} S={S} Hq={Hq} Hkv={Hkv} D={D} {a.dtype}")

    # Pre-generate `iters` distinct random input sets (so each measured iter sees fresh data
    # → defeats constant-folding and L1/HBM caching artifacts).
    key = jax.random.PRNGKey(0)
    def rnd(shape, k):
        return jax.random.normal(k, shape, jnp.float32).astype(dt)

    # Layout: jax.nn.dot_product_attention default expects (B, T, N, H) = BSNH
    def make_inputs(seed):
        ks = jax.random.split(jax.random.PRNGKey(seed), 3)
        q  = rnd((B, S, Hq,  D), ks[0])
        k_ = rnd((B, S, Hkv, D), ks[1])
        v  = rnd((B, S, Hkv, D), ks[2])
        return q, k_, v

    # Compile once
    @jax.jit
    def fwd(q, k, v):
        # GQA: jax.nn.dot_product_attention broadcasts KV heads when Hq % Hkv == 0
        return jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=scale)

    # Warmup (force compile)
    q0, k0, v0 = make_inputs(0)
    for _ in range(a.warmup):
        out = fwd(q0, k0, v0)
    out.block_until_ready()

    # Build distinct random inputs ahead of timing
    inputs = [make_inputs(seed + 1) for seed in range(a.iters)]
    # Pre-pin to device by running fwd once on each (not timed) — actually skip; jit handles dispatch.

    # Timed loop: distinct inputs per iter
    t0 = time.perf_counter()
    for q, k_, v in inputs:
        out = fwd(q, k_, v)
    out.block_until_ready()
    dt_total = time.perf_counter() - t0
    avg_us = dt_total / a.iters * 1e6

    # Profile capture
    logdir = None
    if not a.no_profile:
        logdir = os.path.join(LOGROOT, f"{a.dtype}_b{B}_s{S}")
        os.makedirs(logdir, exist_ok=True)
        with jax.profiler.trace(logdir):
            for _ in range(a.profile_iters):
                out = fwd(q0, k0, v0)
            out.block_until_ready()

    # FLOPs sanity: 2 * 2 * B * Hq * S * S * D (QK^T + PV), causal => /2
    flops = 2 * 2 * B * Hq * S * S * D
    flops_causal = flops / 2
    tflops = flops_causal / (avg_us * 1e-6) / 1e12

    print(f"RESULT prefill B={B} S={S} {a.dtype}: {avg_us:.2f} us/iter  "
          f"({tflops:.1f} TFLOPS w/ causal)  profile={logdir}")
    res = dict(exp="A", op="prefill_gqa_flash", batch=B, seq=S, Hq=Hq, Hkv=Hkv, D=D,
               dtype=a.dtype, iters=a.iters, avg_us=avg_us, causal=True,
               flops_causal=flops_causal, achieved_tflops=tflops,
               profile_dir=logdir)
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gqa_prefill_results.jsonl")
    with open(out_json, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
