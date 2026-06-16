#!/usr/bin/env python3
"""GEMV (y = A @ x, A square N x N) benchmark on TPU v6e — memory-bound matrix op.

Complements gemm_bench (compute-bound) and vadd_bench (pure BW): GEMV reads the
full N x N matrix once per call (arithmetic intensity ~ 1/elem), so it is
HBM-bound. Sweeps size x dtype, warmup=100, measure=100 (avg). Inputs are
DISTINCT per iteration (pool of random A,x capped by an HBM budget so huge sizes
still fit — consecutive iterations always differ). Captures a TensorBoard/XLA
profile per (size,dtype).

Run on the TPU VM (after `pip install -U "jax[tpu]"`):
    python3 gemv_bench.py
Profiles land in ./tb_logs/gemv_<dtype>_<N>/  — scp these back for TensorBoard.
"""
import os, time, json
import jax, jax.numpy as jnp

SIZES  = [1024, 2048, 4096, 8192, 16384]    # square A: N x N, vector x: N
DTYPES = {"fp16": jnp.float16, "bf16": jnp.bfloat16}
WARMUP = 100
MEASURE = 100
PROFILE = 5                                  # iterations captured in the trace
POOL_BUDGET_BYTES = 8 * (2**30)              # cap distinct-input pool at ~8 GB HBM

PEAK_TFLOPS = {"bf16": 918.0, "fp16": 918.0}
HBM_GBPS = 1638 * (2**30) / 1e9            # ~1759 GB/s
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
        A = jax.random.normal(k1, (N, N), jnp.float32).astype(dtype)
        x = jax.random.normal(k2, (N,), jnp.float32).astype(dtype)
        return A, x

    pair_bytes = N * N * elem + N * elem
    pool, P = make_pool(jax.random.PRNGKey(0), gen, pair_bytes, WARMUP + MEASURE + PROFILE)

    @jax.jit
    def mv(A, x):
        return A @ x

    idx = 0
    for _ in range(WARMUP):
        A, x = pool[idx % P]; out = mv(A, x); idx += 1
    out.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        A, x = pool[idx % P]; out = mv(A, x); idx += 1
    out.block_until_ready()
    t1 = time.perf_counter()
    avg_s = (t1 - t0) / MEASURE

    logdir = os.path.join(LOGROOT, f"gemv_{dtype_name}_{N}")
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE):
            A, x = pool[idx % P]; out = mv(A, x); idx += 1
        out.block_until_ready()

    flops = 2.0 * N * N
    tflops = flops / avg_s / 1e12
    bytes_ = (N * N + 2 * N) * elem      # read A + read x + write y
    bw = bytes_ / avg_s / 1e9
    pk = PEAK_TFLOPS[dtype_name]
    print(f"GEMV N={N:>6} {dtype_name:>4}: {avg_s*1e6:9.2f} us/iter  "
          f"{tflops:7.2f} TFLOPS ({100*tflops/pk:5.2f}%)  "
          f"{bw:8.1f} GB/s ({100*bw/HBM_GBPS:5.1f}%)  pool={P}")
    return dict(op="gemv", N=N, dtype=dtype_name, avg_us=avg_s*1e6,
                tflops=tflops, mfu_pct=100*tflops/pk, bw_gbps=bw,
                bw_pct=100*bw/HBM_GBPS, pool=P)


def main():
    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    print(f"warmup={WARMUP} measure={MEASURE}\n")
    results = []
    for dtype_name, dtype in DTYPES.items():
        for N in SIZES:
            try:
                results.append(bench(N, dtype_name, dtype))
            except Exception as e:
                print(f"GEMV N={N} {dtype_name}: FAILED {e}")
        print()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"saved {out}")
    print(f"profiles in {LOGROOT}/  (scp back for TensorBoard)")


if __name__ == "__main__":
    main()
