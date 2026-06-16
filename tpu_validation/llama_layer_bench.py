#!/usr/bin/env python3
"""One Llama 3.1 8B decoder layer on TPU v6e — prefill / decode latency + profile.

Random weights (latency is shape/dtype-determined, not value-dependent). fp16.
Prefill uses TPU Pallas flash attention (causal); decode uses KV-cache attention
(seq_q=1 over context length S). RMSNorm + RoPE + GQA + SwiGLU, same arch both modes.

  python llama_layer_bench.py --mode prefill --seq 2048 --batch 8 --dtype fp16
  python llama_layer_bench.py --mode decode  --seq 2048 --batch 8 --dtype fp16
"""
import os, time, json, argparse, math
import jax, jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

# Llama 3.1 8B per-layer architecture
D, N_H, N_KV, HD, FFN, THETA = 4096, 32, 8, 128, 14336, 500000.0
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_llama")
DTYPES = {"fp16": jnp.float16, "bf16": jnp.bfloat16, "fp32": jnp.float32}


def rmsnorm(x, w, eps=1e-5):
    s = jax.lax.rsqrt(jnp.mean(x.astype(jnp.float32) ** 2, -1, keepdims=True) + eps)
    return (x.astype(jnp.float32) * s).astype(x.dtype) * w


def make_rope(S, offset, dt):
    inv = 1.0 / (THETA ** (jnp.arange(0, HD, 2, dtype=jnp.float32) / HD))
    t = jnp.arange(offset, offset + S, dtype=jnp.float32)
    f = jnp.outer(t, inv)
    emb = jnp.concatenate([f, f], -1)
    return jnp.cos(emb)[None, :, None, :].astype(dt), jnp.sin(emb)[None, :, None, :].astype(dt)


def rope(x, cos, sin):  # x: [B, S, H, HD]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return x * cos + jnp.concatenate([-x2, x1], axis=-1) * sin


def gqa_repeat(x):  # [B, S, N_KV, HD] -> [B, S, N_H, HD]
    return jnp.repeat(x, N_H // N_KV, axis=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["prefill", "decode"])
    p.add_argument("--seq", type=int, required=True, help="prefill prompt len / decode KV-cache len")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    a = p.parse_args()
    dt, B, S = DTYPES[a.dtype], a.batch, a.seq
    print(f"JAX {jax.__version__} dev={jax.devices()}  {a.mode} B={B} S={S} {a.dtype}")
    k = jax.random.PRNGKey(0)

    def rnd(shape, kk):
        return jax.random.normal(kk, shape, jnp.float32).astype(dt) * 0.02
    ks = jax.random.split(k, 9)
    Wq = rnd((D, N_H * HD), ks[0]); Wk = rnd((D, N_KV * HD), ks[1]); Wv = rnd((D, N_KV * HD), ks[2])
    Wo = rnd((N_H * HD, D), ks[3]); Wg = rnd((D, FFN), ks[4]); Wu = rnd((D, FFN), ks[5]); Wd = rnd((FFN, D), ks[6])
    n1 = jnp.ones((D,), dt); n2 = jnp.ones((D,), dt)
    scale = 1.0 / math.sqrt(HD)

    if a.mode == "prefill":
        x = rnd((B, S, D), ks[7])
        cos, sin = make_rope(S, 0, dt)

        @jax.jit
        def fwd(x):
            with jax.named_scope("attn"):
                h = rmsnorm(x, n1)
                q = (h @ Wq).reshape(B, S, N_H, HD); kk = (h @ Wk).reshape(B, S, N_KV, HD); vv = (h @ Wv).reshape(B, S, N_KV, HD)
                q = rope(q, cos, sin); kk = rope(kk, cos, sin)
                # GQA-native, TPU-optimized attention (q:[B,S,N_H,HD], kk/vv:[B,S,N_KV,HD])
                o = jax.nn.dot_product_attention(q, kk, vv, is_causal=True, scale=scale)
                o = o.reshape(B, S, D)
                x = x + o @ Wo
            with jax.named_scope("ffn"):
                h = rmsnorm(x, n2)
                x = x + (jax.nn.silu(h @ Wg) * (h @ Wu)) @ Wd
            return x
        inputs = (x,)
    else:  # decode
        x = rnd((B, 1, D), ks[7])
        kc = rnd((B, S, N_KV, HD), ks[8]); vc = rnd((B, S, N_KV, HD), ks[0])
        cos, sin = make_rope(1, S, dt)

        @jax.jit
        def fwd(x, kc, vc):
            with jax.named_scope("attn"):
                h = rmsnorm(x, n1)
                q = (h @ Wq).reshape(B, 1, N_H, HD); kn = (h @ Wk).reshape(B, 1, N_KV, HD); vn = (h @ Wv).reshape(B, 1, N_KV, HD)
                q = rope(q, cos, sin); kn = rope(kn, cos, sin)
                K = gqa_repeat(jnp.concatenate([kc, kn], 1)); V = gqa_repeat(jnp.concatenate([vc, vn], 1))  # [B,S+1,N_H,HD]
                qh = q.transpose(0, 2, 1, 3); Kh = K.transpose(0, 2, 1, 3); Vh = V.transpose(0, 2, 1, 3)
                sc = (qh @ Kh.transpose(0, 1, 3, 2)) * scale  # [B,N_H,1,S+1]
                pr = jax.nn.softmax(sc.astype(jnp.float32), -1).astype(dt)
                o = (pr @ Vh).transpose(0, 2, 1, 3).reshape(B, 1, D)
                x = x + o @ Wo
            with jax.named_scope("ffn"):
                h = rmsnorm(x, n2)
                x = x + (jax.nn.silu(h @ Wg) * (h @ Wu)) @ Wd
            return x
        inputs = (x, kc, vc)

    for _ in range(30):
        out = fwd(*inputs)
    out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(50):
        out = fwd(*inputs)
    out.block_until_ready()
    avg_us = (time.perf_counter() - t0) / 50 * 1e6

    logdir = os.path.join(LOGROOT, f"{a.mode}_{a.dtype}_b{B}_s{S}")
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for _ in range(5):
            out = fwd(*inputs)
        out.block_until_ready()

    print(f"RESULT {a.mode} B={B} S={S} {a.dtype}: {avg_us:.2f} us/iter  -> {logdir}")
    res = dict(mode=a.mode, batch=B, seq=S, dtype=a.dtype, avg_us=avg_us)
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_results.jsonl")
    with open(out_json, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
