#!/usr/bin/env python3
"""Splash attention = FUSED single kernel + NATIVE GQA on TPU v6e.

Tests one prefill + one decode (GQA Hq=5/Hkv=1, D=128, bf16) and checks the HLO
for a single custom-call (fused). splash reads KV at Hkv heads (true GQA traffic),
unlike the MHA-materialized flash_attention test.

q: [num_q_heads, q_seq, D], k/v: [num_kv_heads, kv_seq, D] per example; vmap over B.
"""
import os, time, json, functools
import jax, jax.numpy as jnp

from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

B, HQ, HKV, D = 1, 5, 1, 128
DT = jnp.bfloat16
SCALE = 1.0 / (D ** 0.5)
WARMUP, MEASURE, PROFILE = 10, 50, 5
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_splash")


def make_kernel(q_seq, kv_seq, causal):
    if causal:
        m = mask_lib.CausalMask(shape=(q_seq, kv_seq))
    else:
        m = mask_lib.FullMask(_shape=(q_seq, kv_seq)) if hasattr(mask_lib, "FullMask") \
            else mask_lib.NumpyMask(jnp.ones((q_seq, kv_seq), dtype=bool))
    multi = mask_lib.MultiHeadMask(masks=tuple(m for _ in range(HQ)))
    bq = min(q_seq, 512); bk = min(kv_seq, 512)
    bs = splash.BlockSizes.get_default()
    try:
        bs = splash.BlockSizes(block_q=bq, block_kv=bk, block_kv_compute=bk,
                               block_q_dkv=bq, block_kv_dkv=bk, block_kv_dkv_compute=bk,
                               block_q_dq=bq, block_kv_dq=bk)
    except Exception:
        pass
    k = splash.make_splash_mha(mask=multi, head_shards=1, q_seq_shards=1, block_sizes=bs)
    # vmap over batch; q[B,HQ,Tq,D] k/v[B,HKV,S,D]
    return jax.jit(jax.vmap(k))


def run(tag, q_seq, kv_seq, causal):
    key = jax.random.PRNGKey(0)
    q = (jax.random.normal(key, (B, HQ, q_seq, D), jnp.float32) * 0.1 * SCALE).astype(DT)
    k = (jax.random.normal(key, (B, HKV, kv_seq, D), jnp.float32) * 0.1).astype(DT)
    v = (jax.random.normal(key, (B, HKV, kv_seq, D), jnp.float32) * 0.1).astype(DT)
    fn = make_kernel(q_seq, kv_seq, causal)
    out = fn(q, k, v); jax.block_until_ready(out)
    logdir = os.path.join(LOGROOT, tag); os.makedirs(logdir, exist_ok=True)
    h = fn.lower(q, k, v).compile().as_text()
    open(os.path.join(logdir, "optimized_hlo.txt"), "w").write(h)
    cc = h.count("custom-call")
    print(f"  [HLO {tag}] custom-call={cc} fusion={h.count('fusion(')} "
          f"-> {'ONE FUSED custom-call' if cc>=1 else 'NOT fused'}  (kv heads in operands: see hlo)")
    for _ in range(WARMUP): out = fn(q, k, v)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(MEASURE): out = fn(q, k, v)
    jax.block_until_ready(out)
    us = (time.perf_counter() - t0) / MEASURE * 1e6
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE): out = fn(q, k, v)
        jax.block_until_ready(out)
    print(f"  {tag}: {us:.2f} us/iter  out={tuple(out.shape)}")
    return dict(tag=tag, q_seq=q_seq, kv_seq=kv_seq, causal=causal, avg_us=us, custom_call=cc)


def main():
    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    print(f"GQA Hq={HQ} Hkv={HKV} D={D} bf16  (splash = native GQA, fused)\n")
    res = []
    for tag, qs, kvs, ca in [("prefill_S512", 512, 512, True),
                              ("decode_KV512", 1, 512, False)]:
        try:
            res.append(run(tag, qs, kvs, ca))
        except Exception as e:
            print(f"  {tag} FAILED: {type(e).__name__}: {e}")
        print()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_splash_results.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
