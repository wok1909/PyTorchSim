"""2D (batch x seq) sweep of fused GQA splash attention on TPU v6e.
decode (q=1, non-causal) + prefill (causal, q_seq=kv_seq=S), bf16,
GQA Hq=5/Hkv=1/D=128. Profiles each (mode,B,S) so we can extract DEVICE-time.
Writes tb_logs_2d/<mode>_B<B>_S<S>/.
"""
import os, time, json
import jax, jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

HQ, HKV, D = 5, 1, 128
DT = jnp.bfloat16
SCALE = 1.0 / (D ** 0.5)
BATCHES = [1, 8, 32, 128]
SEQS = [512, 1024, 2048, 4096]
WARMUP, MEASURE, PROFILE = 8, 40, 5
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_2d")


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


def run(mode, B, S, causal):
    q_seq = S if mode == "prefill" else 1
    key = jax.random.PRNGKey(0)
    q = (jax.random.normal(key, (B, HQ, q_seq, D), jnp.float32) * 0.1 * SCALE).astype(DT)
    k = (jax.random.normal(key, (B, HKV, S, D), jnp.float32) * 0.1).astype(DT)
    v = (jax.random.normal(key, (B, HKV, S, D), jnp.float32) * 0.1).astype(DT)
    fn = make_kernel(q_seq, S, causal)
    out = fn(q, k, v); jax.block_until_ready(out)
    tag = f"{mode}_B{B}_S{S}"
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
    dev = _devtime(logdir)   # extract on-VM so preemption during scp can't lose it
    print(f"  RES {tag}: wall={wall:.2f}us dev={dev:.3f}us" if dev else
          f"  RES {tag}: wall={wall:.2f}us dev=NA", flush=True)
    return dict(mode=mode, B=B, S=S, wall_us=wall, dev_us=dev)


def _devtime(logdir):
    import gzip, glob, statistics, time as _t
    try:
        _t.sleep(0.3)                           # let trace flush
        g = glob.glob(os.path.join(logdir, "plugins/profile/*/*.trace.json.gz"))
        if not g:
            return None
        d = json.load(gzip.open(g[0])); ev = d.get("traceEvents", [])
        pn = {e.get("pid"): e.get("args", {}).get("name", "") for e in ev if e.get("name") == "process_name"}
        dp = [p for p, n in pn.items() if "/device:TPU" in str(n)]
        jit = [e["dur"] for e in ev if e.get("ph") == "X" and "dur" in e and e.get("pid") in dp
               and e.get("name", "").startswith("jit")]
        return statistics.median(jit) if jit else None
    except Exception as e:
        print(f"    (devtime err: {type(e).__name__}: {e})", flush=True)
        return None


def main():
    print(f"JAX {jax.__version__} {jax.devices()}  2D grid B={BATCHES} S={SEQS}", flush=True)
    res = []
    for mode, causal in [("decode", False), ("prefill", True)]:
        for B in BATCHES:
            for S in SEQS:
                try:
                    res.append(run(mode, B, S, causal))
                except Exception as e:
                    print(f"  {mode}_B{B}_S{S} FAILED: {type(e).__name__}: {e}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_2d_results.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
