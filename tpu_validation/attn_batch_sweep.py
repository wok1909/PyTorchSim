#!/usr/bin/env python3
"""Batch-size sweep of fused GQA attention (splash) on TPU v6e.

Same GQA shape (Hq=5,Hkv=1,D=128,S/KV=512,bf16) as the sim comparison; sweeps
batch B in {1,4,8,16,32,64,128} for prefill (causal) and decode (q=1, non-causal).
Saves an XProf trace per (mode,B) under tb_logs_batch/ so the host can extract the
device-side kernel duration (jit_unnamed module wall duration), plus prints a
warmup'd wall-clock per-iter (converges to device time at large B).
"""
import os, time, json, functools
import jax, jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

HQ, HKV, D = 5, 1, 128
S = 512
BATCHES = [1, 4, 8, 16, 32, 64, 128]
DT = jnp.bfloat16
SCALE = 1.0 / (D ** 0.5)
WARMUP, MEASURE, PROFILE = 10, 30, 5
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_batch")


def make_kernel(q_seq, kv_seq, causal):
    if causal:
        m = mask_lib.CausalMask(shape=(q_seq, kv_seq))
    else:
        m = mask_lib.FullMask(_shape=(q_seq, kv_seq)) if hasattr(mask_lib, "FullMask") \
            else mask_lib.NumpyMask(jnp.ones((q_seq, kv_seq), dtype=bool))
    multi = mask_lib.MultiHeadMask(masks=tuple(m for _ in range(HQ)))
    bq = min(q_seq, 512); bk = min(kv_seq, 512)
    try:
        bs = splash.BlockSizes(block_q=bq, block_kv=bk, block_kv_compute=bk,
                               block_q_dkv=bq, block_kv_dkv=bk, block_kv_dkv_compute=bk,
                               block_q_dq=bq, block_kv_dq=bk)
    except Exception:
        bs = splash.BlockSizes.get_default()
    k = splash.make_splash_mha(mask=multi, head_shards=1, q_seq_shards=1, block_sizes=bs)
    return jax.jit(jax.vmap(k))


def run(mode, B, q_seq, kv_seq, causal):
    key = jax.random.PRNGKey(0)
    q = (jax.random.normal(key, (B, HQ, q_seq, D), jnp.float32) * 0.1 * SCALE).astype(DT)
    k = (jax.random.normal(key, (B, HKV, kv_seq, D), jnp.float32) * 0.1).astype(DT)
    v = (jax.random.normal(key, (B, HKV, kv_seq, D), jnp.float32) * 0.1).astype(DT)
    fn = make_kernel(q_seq, kv_seq, causal)
    out = fn(q, k, v); jax.block_until_ready(out)
    for _ in range(WARMUP):
        out = fn(q, k, v)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        out = fn(q, k, v)
    jax.block_until_ready(out)
    wall_us = (time.perf_counter() - t0) / MEASURE * 1e6
    logdir = os.path.join(LOGROOT, f"{mode}_B{B}")
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE):
            out = fn(q, k, v)
        jax.block_until_ready(out)
    print(f"{mode:>7} B={B:>3}: wall {wall_us:8.2f} us/iter  out={tuple(out.shape)}")
    return dict(mode=mode, B=B, q_seq=q_seq, kv_seq=kv_seq, wall_us=wall_us)


def main():
    print(f"JAX {jax.__version__}  {jax.devices()}  GQA Hq={HQ} Hkv={HKV} D={D} S={S} bf16")
    res = []
    for mode, qs, kv, ca in [("prefill", S, S, True), ("decode", 1, S, False)]:
        for B in BATCHES:
            try:
                res.append(run(mode, B, qs, kv, ca))
            except Exception as e:
                print(f"{mode} B={B} FAILED: {type(e).__name__}: {e}")
        print()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_batch_results.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved {out}; traces in {LOGROOT}/")


if __name__ == "__main__":
    main()
