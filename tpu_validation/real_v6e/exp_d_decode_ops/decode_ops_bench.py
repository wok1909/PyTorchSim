#!/usr/bin/env python3
"""Per-OP decode benchmark — one Llama 3.1 8B decoder layer's ops, isolated.

decode = 1 new token, KV-cache length S. Every op written EXPLICITLY (no flash /
no opaque dispatch) so we know exactly what runs; matches PyTorchSim's op-level
execution. fp16 (decode attention is naive -> no Mosaic, fp16 OK = sim match).

  python decode_ops_bench.py --op qkv  --batch 8 --seq 2048
  python decode_ops_bench.py --op attn --batch 8 --seq 2048
ops: rmsnorm rope qkv attn oproj ffn
"""
import os, time, json, argparse, math
import jax, jax.numpy as jnp

D, N_H, N_KV, HD, FFN, THETA = 4096, 32, 8, 128, 14336, 500000.0
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_decode")
DTYPES = {"fp16": jnp.float16, "bf16": jnp.bfloat16, "fp32": jnp.float32}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True,
                   choices=["rmsnorm", "rope", "qkv", "attn", "attn_flash", "oproj", "ffn"])
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--seq", type=int, required=True, help="KV-cache length S")
    p.add_argument("--tile_size", type=int, default=512, help="Flash-Decode KV tile size T")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    a = p.parse_args()
    dt, B, S = DTYPES[a.dtype], a.batch, a.seq
    print(f"JAX {jax.__version__} {jax.devices()}  op={a.op} B={B} S={S} {a.dtype}")
    k = jax.random.split(jax.random.PRNGKey(0), 12)
    rn = lambda shp, kk: (jax.random.normal(kk, shp, jnp.float32) * 0.02).astype(dt)
    scale = 1.0 / math.sqrt(HD)

    if a.op == "rmsnorm":
        x = rn((B, D), k[0]); w = jnp.ones((D,), dt)
        @jax.jit
        def f(x):
            s = jax.lax.rsqrt(jnp.mean(x.astype(jnp.float32) ** 2, -1, keepdims=True) + 1e-5)
            return (x.astype(jnp.float32) * s).astype(dt) * w
        args = (x,)
    elif a.op == "rope":
        q = rn((B, N_H, HD), k[0]); cs = rn((1, HD), k[1]); sn = rn((1, HD), k[2])
        @jax.jit
        def f(q):
            q1, q2 = jnp.split(q, 2, -1)
            return q * cs[:, None, :] + jnp.concatenate([-q2, q1], -1) * sn[:, None, :]
        args = (q,)
    elif a.op == "qkv":   # x[B,D] -> Q[B,D], K/V[B,N_KV*HD]  (GEMM M=B)
        x = rn((B, D), k[0]); Wq = rn((D, D), k[1]); Wk = rn((D, N_KV * HD), k[2]); Wv = rn((D, N_KV * HD), k[3])
        @jax.jit
        def f(x):
            return x @ Wq, x @ Wk, x @ Wv
        args = (x,)
    elif a.op == "attn":  # GQA-native: K/V stay N_KV(8) heads, read once, broadcast over G=4 query heads
        G = N_H // N_KV
        q = rn((B, N_KV, G, HD), k[0])      # query heads grouped per KV head
        Kc = rn((B, N_KV, S, HD), k[1]); Vc = rn((B, N_KV, S, HD), k[2])  # 8-head KV cache (GQA-sized)
        @jax.jit
        def f(q, Kc, Vc):
            sc = jnp.einsum('bgkh,bgsh->bgks', q, Kc) * scale   # [B,N_KV,G,S]  (Kc read 1x per kv head)
            pr = jax.nn.softmax(sc.astype(jnp.float32), -1).astype(dt)
            return jnp.einsum('bgks,bgsh->bgkh', pr, Vc)        # [B,N_KV,G,HD]
        args = (q, Kc, Vc)
    elif a.op == "attn_flash":  # Flash-Decode GQA (matches tests/test_gqa_decode.py: tile KV + online softmax)
        G = N_H // N_KV
        T = min(a.tile_size, S)
        nt = (S + T - 1) // T
        q = rn((B, N_KV, G, HD), k[0]); Kc = rn((B, N_KV, S, HD), k[1]); Vc = rn((B, N_KV, S, HD), k[2])
        @jax.jit
        def f(q, Kc, Vc):
            pad = nt * T - S
            K = jnp.pad(Kc, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else Kc
            V = jnp.pad(Vc, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else Vc
            Kt = K.reshape(B, N_KV, nt, T, HD); Vt = V.reshape(B, N_KV, nt, T, HD)
            sc = jnp.einsum('bgkh,bgnth->bgnkt', q, Kt).astype(jnp.float32) * scale   # [B,N_KV,nt,G,T]
            lmax = jnp.max(sc, -1, keepdims=True)                  # tile-local max
            lexp = jnp.exp(sc - lmax)
            lsum = jnp.sum(lexp, -1, keepdims=True)
            sv = jnp.einsum('bgnkt,bgnth->bgnkh', lexp.astype(dt), Vt)   # [B,N_KV,nt,G,HD]
            gmax = jnp.max(lmax, 2, keepdims=True)                 # online reduction over tiles
            resc = jnp.exp(lmax - gmax)
            csv = jnp.sum(sv * resc.astype(dt), 2)
            csum = jnp.sum(lsum * resc, 2)
            return (csv / jnp.maximum(csum, 1e-12).astype(dt))     # [B,N_KV,G,HD]
        args = (q, Kc, Vc)
    elif a.op == "oproj":
        o = rn((B, D), k[0]); Wo = rn((D, D), k[1])
        @jax.jit
        def f(o):
            return o @ Wo
        args = (o,)
    elif a.op == "ffn":   # SwiGLU
        x = rn((B, D), k[0]); Wg = rn((D, FFN), k[1]); Wu = rn((D, FFN), k[2]); Wd = rn((FFN, D), k[3])
        @jax.jit
        def f(x):
            return (jax.nn.silu(x @ Wg) * (x @ Wu)) @ Wd
        args = (x,)

    for _ in range(30):
        out = f(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(50):
        out = f(*args)
    jax.block_until_ready(out)
    avg_us = (time.perf_counter() - t0) / 50 * 1e6

    logdir = os.path.join(LOGROOT, f"{a.op}_{a.dtype}_b{B}_s{S}")
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for _ in range(5):
            out = f(*args)
        jax.block_until_ready(out)

    print(f"RESULT op={a.op} B={B} S={S} {a.dtype}: {avg_us:.2f} us/iter -> {logdir}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode_ops_results.jsonl"), "a") as fo:
        fo.write(json.dumps(dict(op=a.op, batch=B, seq=S, dtype=a.dtype, avg_us=avg_us)) + "\n")


if __name__ == "__main__":
    main()
