#!/usr/bin/env python3
"""Can TPU run prefill attention as ONE fused kernel (like PyTorchSim)?

Compares, on the SAME GQA prefill shape, two TPU paths:
  (A) XLA default  : jax.nn.dot_product_attention(implementation="xla")
                     -> DECOMPOSED (separate QK / softmax / PV fusions)
  (B) Pallas flash : jax.experimental.pallas.ops.tpu.flash_attention
                     -> single fused Mosaic custom-call (online softmax, no
                        score materialization), i.e. the TPU analogue of our
                        PyTorchSim mlir_kernel_0 flash template.

For each, dump optimized HLO and count custom-call / fusion / dot so we can tell
whether it is ONE fused kernel. bf16 only (TPU has no fp16 matmul).
"""
import os, time, json, functools
import jax, jax.numpy as jnp
from jax.nn import dot_product_attention as xla_dpa

# LLAMA4_TP8 GQA prefill shape
B, HQ, HKV, D = 1, 5, 1, 128
SEQS = [256, 512, 1024, 2048]
DT = jnp.bfloat16
SCALE = 1.0 / (D ** 0.5)
WARMUP, MEASURE, PROFILE = 20, 50, 5
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_fused")

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention, BlockSizes
    HAVE_PALLAS = True
except Exception as e:
    HAVE_PALLAS = False
    print("PALLAS import failed:", e)


def hlo_summary(fn, *args):
    h = fn.lower(*args).compile().as_text()
    return dict(custom_call=h.count("custom-call"), fusion=h.count("fusion("),
                dot=h.count("dot("), conv=h.count("convolution("), text=h)


def bench(label, fn, args, S):
    out = fn(*args); jax.block_until_ready(out)          # compile
    logdir = os.path.join(LOGROOT, f"{label}_S{S}")
    os.makedirs(logdir, exist_ok=True)
    try:
        s = hlo_summary(fn, *args)
        open(os.path.join(logdir, "optimized_hlo.txt"), "w").write(s["text"])
        kind = "ONE FUSED custom-call" if s["custom_call"] >= 1 else "DECOMPOSED"
        print(f"    [HLO {label} S{S}] custom-call={s['custom_call']} fusion={s['fusion']} "
              f"dot={s['dot']} conv={s['conv']} -> {kind}")
    except Exception as e:
        print(f"    [HLO {label} S{S}] dump failed: {e}"); s = {}
    for _ in range(WARMUP):
        out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        out = fn(*args)
    jax.block_until_ready(out)
    us = (time.perf_counter() - t0) / MEASURE * 1e6
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE):
            out = fn(*args)
        jax.block_until_ready(out)
    print(f"  {label:14s} S={S:>5}: {us:9.2f} us/iter")
    return dict(label=label, S=S, avg_us=us, **{k: s.get(k) for k in ("custom_call","fusion","dot","conv")})


def main():
    print(f"JAX {jax.__version__}  devices={jax.devices()}  pallas={HAVE_PALLAS}")
    results = []
    key = jax.random.PRNGKey(0)
    for S in SEQS:
        # XLA path expects [B, seq, heads, D]; GQA native (Hkv<Hq).
        q1 = (jax.random.normal(key, (B, S, HQ, D), jnp.float32) * 0.1).astype(DT)
        k1 = (jax.random.normal(key, (B, S, HKV, D), jnp.float32) * 0.1).astype(DT)
        v1 = (jax.random.normal(key, (B, S, HKV, D), jnp.float32) * 0.1).astype(DT)
        xla_fn = jax.jit(functools.partial(xla_dpa, scale=SCALE, is_causal=True,
                                           implementation="xla"))
        try:
            results.append(bench("xla_decomp", xla_fn, (q1, k1, v1), S))
        except Exception as e:
            print(f"  xla_decomp S={S} FAILED {e}")

        if HAVE_PALLAS:
            # Pallas flash expects [B, heads, seq, D], MHA -> repeat KV to HQ heads.
            qP = jnp.transpose(q1, (0, 2, 1, 3))                       # [B,HQ,S,D]
            kP = jnp.broadcast_to(jnp.transpose(k1, (0, 2, 1, 3)), (B, HQ, S, D))
            vP = jnp.broadcast_to(jnp.transpose(v1, (0, 2, 1, 3)), (B, HQ, S, D))
            b = min(S, 512)
            bs = BlockSizes(block_q=b, block_k_major=b, block_k=b, block_b=1,
                            block_q_major_dkv=b, block_k_major_dkv=b,
                            block_k_dkv=b, block_q_dkv=b,
                            block_k_major_dq=b, block_k_dq=b, block_q_dq=b)
            flash_fn = jax.jit(functools.partial(flash_attention, causal=True,
                                                 sm_scale=SCALE, block_sizes=bs))
            try:
                results.append(bench("pallas_fused", flash_fn, (qP, kP, vP), S))
            except Exception as e:
                print(f"  pallas_fused S={S} FAILED {e}")
        print()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_fused_results.json")
    json.dump([{k: v for k, v in r.items() if k != "text"} for r in results],
              open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
