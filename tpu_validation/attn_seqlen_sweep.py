"""Seq-length sweep of fused GQA attention (splash) on TPU v6e, B=1.
Prefill (causal, q_seq=kv_seq=S) and decode (q_seq=1, kv_seq=S), bf16,
GQA Hq=5/Hkv=1/D=128. Profiles each point so we can extract DEVICE-time
(not just wall) vs S from the traces. Writes tb_logs_seq/<tag>/.
"""
import os, time, json, functools
import jax, jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

HQ, HKV, D = 5, 1, 128
DT = jnp.bfloat16
SCALE = 1.0 / (D ** 0.5)
SEQS = [256, 512, 1024, 2048, 4096]
WARMUP, MEASURE, PROFILE = 10, 50, 5
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_seq")


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


def run(mode, S, causal):
    q_seq = S if mode == "prefill" else 1
    key = jax.random.PRNGKey(0)
    q = (jax.random.normal(key, (1, HQ, q_seq, D), jnp.float32) * 0.1 * SCALE).astype(DT)
    k = (jax.random.normal(key, (1, HKV, S, D), jnp.float32) * 0.1).astype(DT)
    v = (jax.random.normal(key, (1, HKV, S, D), jnp.float32) * 0.1).astype(DT)
    fn = make_kernel(q_seq, S, causal)
    out = fn(q, k, v); jax.block_until_ready(out)
    tag = f"{mode}_S{S}"
    logdir = os.path.join(LOGROOT, tag); os.makedirs(logdir, exist_ok=True)
    for _ in range(WARMUP): out = fn(q, k, v)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(MEASURE): out = fn(q, k, v)
    jax.block_until_ready(out)
    wall = (time.perf_counter() - t0) / MEASURE * 1e6
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE): out = fn(q, k, v)
        jax.block_until_ready(out)
    print(f"  {tag}: wall={wall:.2f}us")
    return dict(mode=mode, S=S, q_seq=q_seq, wall_us=wall)


def main():
    print(f"JAX {jax.__version__} {jax.devices()}  GQA Hq={HQ} Hkv={HKV} D={D} bf16  seq-sweep")
    res = []
    for mode, causal in [("prefill", True), ("decode", False)]:
        for S in SEQS:
            try:
                res.append(run(mode, S, causal))
            except Exception as e:
                print(f"  {mode}_S{S} FAILED: {type(e).__name__}: {e}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_seqlen_results.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
