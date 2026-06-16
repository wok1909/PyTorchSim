#!/usr/bin/env python3
"""GEMM (square M=K=N) benchmark on TPU v6e — for PyTorchSim validation.

Sweeps square matmul over sizes x dtypes, warmup=100, measure=100 (avg).
Inputs are DISTINCT per iteration (pool of random tensors, capped by an HBM
budget so huge sizes still fit — consecutive iterations always differ; small
sizes are fully distinct over the whole run). Captures a TensorBoard/XLA
profile per (size,dtype) and prints latency/TFLOPS/BW.

Run on the TPU VM (after `pip install -U "jax[tpu]"`):
    python3 gemm_bench.py
Profiles land in ./tb_logs/gemm_<dtype>_<N>/  — scp these back for TensorBoard.
"""
import os, time, json
import jax, jax.numpy as jnp

SIZES  = [512, 1024, 2048, 4096, 8192]      # square N x N x N
DTYPES = {"fp16": jnp.float16, "bf16": jnp.bfloat16}
WARMUP = 100
MEASURE = 100
PROFILE = 5                                  # iterations captured in the trace
POOL_BUDGET_BYTES = 8 * (2**30)              # cap distinct-input pool at ~8 GB HBM

# v6e peak (for util %): BF16 918 TFLOPS, HBM 1.6 TB/s (1638 GiBps)
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
        a = jax.random.normal(k1, (N, N), jnp.float32).astype(dtype)
        b = jax.random.normal(k2, (N, N), jnp.float32).astype(dtype)
        return a, b

    pair_bytes = 2 * N * N * elem
    pool, P = make_pool(jax.random.PRNGKey(0), gen, pair_bytes, WARMUP + MEASURE + PROFILE)

    @jax.jit
    def mm(x, y):
        return x @ y

    idx = 0
    # warmup (JIT compile + stabilize)
    for _ in range(WARMUP):
        a, b = pool[idx % P]; out = mm(a, b); idx += 1
    out.block_until_ready()

    # timed measure
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        a, b = pool[idx % P]; out = mm(a, b); idx += 1
    out.block_until_ready()
    t1 = time.perf_counter()
    avg_s = (t1 - t0) / MEASURE

    # profile capture (separate short run)
    logdir = os.path.join(LOGROOT, f"gemm_{dtype_name}_{N}")
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE):
            a, b = pool[idx % P]; out = mm(a, b); idx += 1
        out.block_until_ready()

    flops = 2.0 * N**3
    tflops = flops / avg_s / 1e12
    bytes_ = 3.0 * N * N * elem          # read 2 mats + write 1
    bw = bytes_ / avg_s / 1e9
    pk = PEAK_TFLOPS[dtype_name]
    print(f"GEMM N={N:>5} {dtype_name:>4}: {avg_s*1e6:9.1f} us/iter  "
          f"{tflops:7.1f} TFLOPS ({100*tflops/pk:5.1f}%)  "
          f"{bw:8.1f} GB/s ({100*bw/HBM_GBPS:5.1f}%)  pool={P}")
    return dict(op="gemm", N=N, dtype=dtype_name, avg_us=avg_s*1e6,
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
                print(f"GEMM N={N} {dtype_name}: FAILED {e}")
        print()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"saved {out}")
    print(f"profiles in {LOGROOT}/  (scp back for TensorBoard)")


if __name__ == "__main__":
    main()
