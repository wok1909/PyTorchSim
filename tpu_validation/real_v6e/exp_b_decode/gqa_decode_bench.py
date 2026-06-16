#!/usr/bin/env python3
"""EXP-B: LLAMA4_TP8 GQA attention decode latency on real TPU v6e.

Single-token decode against KV cache (q_len=1).
Shapes:
  q       [B, 1, Hq=5,  D=128]
  kcache  [B, S, Hkv=1, D=128]
  vcache  [B, S, Hkv=1, D=128]
Non-causal (decode sees full cache).

Two implementations:
  --impl flash : jax.nn.dot_product_attention(q_len=1)  (Mosaic flash, bf16-only)
  --impl naive : explicit einsum + softmax + einsum    (fp16 OK = matches sim exactly)

Timing harness mirrors llama_layer_bench.py.

  python gqa_decode_bench.py --seq 2048 --dtype fp16 --impl naive
  python gqa_decode_bench.py --seq 2048 --dtype bf16 --impl flash
"""
import os, time, json, argparse, math
import jax, jax.numpy as jnp

HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS = 128, 5, 1
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_gqa_decode")
DTYPES = {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, required=True, help="KV cache length S")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--impl", default="flash", choices=["flash", "naive"])
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--profile_iters", type=int, default=5)
    p.add_argument("--no_profile", action="store_true")
    a = p.parse_args()

    dt, B, S = DTYPES[a.dtype], a.batch, a.seq
    D, Hq, Hkv = HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS
    scale = 1.0 / math.sqrt(D)
    print(f"JAX {jax.__version__} devs={jax.devices()}  decode B={B} S={S} Hq={Hq} Hkv={Hkv} D={D} "
          f"{a.dtype} impl={a.impl}")

    def rnd(shape, k):
        return jax.random.normal(k, shape, jnp.float32).astype(dt)

    def make_inputs(seed):
        ks = jax.random.split(jax.random.PRNGKey(seed), 3)
        q = rnd((B, 1, Hq,  D), ks[0])
        kc = rnd((B, S, Hkv, D), ks[1])
        vc = rnd((B, S, Hkv, D), ks[2])
        return q, kc, vc

    if a.impl == "flash":
        @jax.jit
        def fwd(q, kc, vc):
            return jax.nn.dot_product_attention(q, kc, vc, is_causal=False, scale=scale)
    else:  # naive: matches sim's explicit-ops decode
        G = Hq // Hkv
        @jax.jit
        def fwd(q, kc, vc):
            # repeat KV heads to match Hq
            kr = jnp.repeat(kc, G, axis=2)  # [B, S, Hq, D]
            vr = jnp.repeat(vc, G, axis=2)
            # transpose to BHSD for batched matmul
            qh = q.transpose(0, 2, 1, 3)   # [B, Hq, 1, D]
            kh = kr.transpose(0, 2, 1, 3)  # [B, Hq, S, D]
            vh = vr.transpose(0, 2, 1, 3)
            sc = (qh @ kh.transpose(0, 1, 3, 2)) * scale         # [B, Hq, 1, S]
            pr = jax.nn.softmax(sc.astype(jnp.float32), -1).astype(dt)
            o = (pr @ vh).transpose(0, 2, 1, 3)                   # [B, 1, Hq, D]
            return o

    q0, kc0, vc0 = make_inputs(0)
    for _ in range(a.warmup):
        out = fwd(q0, kc0, vc0)
    out.block_until_ready()

    inputs = [make_inputs(s + 1) for s in range(a.iters)]
    t0 = time.perf_counter()
    for q, kc, vc in inputs:
        out = fwd(q, kc, vc)
    out.block_until_ready()
    avg_us = (time.perf_counter() - t0) / a.iters * 1e6

    logdir = None
    if not a.no_profile:
        logdir = os.path.join(LOGROOT, f"{a.impl}_{a.dtype}_b{B}_s{S}")
        os.makedirs(logdir, exist_ok=True)
        with jax.profiler.trace(logdir):
            for _ in range(a.profile_iters):
                out = fwd(q0, kc0, vc0)
            out.block_until_ready()

    # Decode memory traffic dominates: ~ (Hkv*S*D + Hq*D) * 2 bytes per chip per token
    bytes_kv = B * Hkv * S * D * 2 * 2  # K + V, bf16/fp16=2B
    flops = 2 * 2 * B * Hq * 1 * S * D   # QK^T + PV
    bw_gbs = bytes_kv / (avg_us * 1e-6) / 1e9

    print(f"RESULT decode B={B} S={S} {a.dtype} impl={a.impl}: {avg_us:.2f} us/iter  "
          f"(KV BW≈{bw_gbs:.0f} GB/s, flops={flops/1e9:.2f} GFLOPs)  profile={logdir}")
    res = dict(exp="B", op=f"decode_gqa_{a.impl}", batch=B, seq=S, Hq=Hq, Hkv=Hkv, D=D,
               dtype=a.dtype, iters=a.iters, avg_us=avg_us,
               kv_bytes=bytes_kv, kv_bw_gbs=bw_gbs, flops=flops,
               profile_dir=logdir)
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gqa_decode_results.jsonl")
    with open(out_json, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
