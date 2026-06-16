#!/usr/bin/env python3
"""Vector add (C = A + B) benchmark on TPU v6e — for PyTorchSim validation.

Pure memory-bound (OI ~ 0). Sweeps size x dtype, warmup=100, measure=100 (avg).
Inputs are DISTINCT per iteration (pool of random vectors, capped by an HBM
budget so huge sizes still fit — consecutive iterations always differ; small
sizes are fully distinct over the whole run). Captures a TensorBoard/XLA
profile per (size,dtype).

Run on the TPU VM (after `pip install -U "jax[tpu]"`):
    python3 vadd_bench.py
Profiles + results land under ./ (this script's dir) = tpu_validation/.
"""
import os, time, json
import jax, jax.numpy as jnp

SIZES  = [1_048_576, 4_194_304, 16_777_216, 67_108_864]   # 1M, 4M, 16M, 64M elements
DTYPES = {"fp16": jnp.float16, "bf16": jnp.bfloat16}
WARMUP = 100
MEASURE = 100
PROFILE = 5                                  # iterations captured in the trace
POOL_BUDGET_BYTES = 8 * (2**30)              # cap distinct-input pool at ~8 GB HBM

HBM_GBPS = 1638 * (2**30) / 1e9   # ~1759 GB/s (v6e)
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs")


def make_pool(key, gen, pair_bytes, n_needed):
    """Pre-generate distinct inputs, capping the pool so it fits POOL_BUDGET_BYTES.
    Returns (pool, pool_n). Consecutive iterations always use a different entry;
    if all n_needed entries fit, every iteration is fully distinct."""
    pool_n = max(2, min(n_needed, int(POOL_BUDGET_BYTES // max(1, pair_bytes))))
    keys = jax.random.split(key, pool_n)
    return [gen(k) for k in keys], pool_n


def bench(N, dtype_name, dtype):
    elem = 2 if dtype_name != "fp32" else 4

    def gen(k):
        k1, k2 = jax.random.split(k)
        a = jax.random.normal(k1, (N,), jnp.float32).astype(dtype)
        b = jax.random.normal(k2, (N,), jnp.float32).astype(dtype)
        return a, b

    pair_bytes = 2 * N * elem
    pool, P = make_pool(jax.random.PRNGKey(0), gen, pair_bytes, WARMUP + MEASURE + PROFILE)

    @jax.jit
    def vadd(x, y):
        return x + y

    idx = 0
    for _ in range(WARMUP):
        a, b = pool[idx % P]; out = vadd(a, b); idx += 1
    out.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        a, b = pool[idx % P]; out = vadd(a, b); idx += 1
    out.block_until_ready()
    t1 = time.perf_counter()
    avg_s = (t1 - t0) / MEASURE

    logdir = os.path.join(LOGROOT, f"vadd_{dtype_name}_{N}")
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE):
            a, b = pool[idx % P]; out = vadd(a, b); idx += 1
        out.block_until_ready()

    bytes_ = 3.0 * N * elem            # read A + read B + write C
    bw = bytes_ / avg_s / 1e9
    print(f"VADD N={N:>10} {dtype_name:>4}: {avg_s*1e6:9.2f} us/iter  "
          f"{bw:8.1f} GB/s ({100*bw/HBM_GBPS:5.1f}%)  pool={P}")
    return dict(op="vadd", N=N, dtype=dtype_name, avg_us=avg_s*1e6,
                bw_gbps=bw, bw_pct=100*bw/HBM_GBPS, pool=P)


def main():
    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    print(f"warmup={WARMUP} measure={MEASURE}\n")
    results = []
    for dtype_name, dtype in DTYPES.items():
        for N in SIZES:
            try:
                results.append(bench(N, dtype_name, dtype))
            except Exception as e:
                print(f"VADD N={N} {dtype_name}: FAILED {e}")
        print()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vadd_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"saved {out}")
    print(f"profiles in {LOGROOT}/  (scp back for TensorBoard)")


if __name__ == "__main__":
    main()
