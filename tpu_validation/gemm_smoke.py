#!/usr/bin/env python3
"""GEMM 256x256 smoke test on TPU v6e — pipeline check before full sweep.

Single size (256), bf16. Pre-generates a DISTINCT random (a,b) pair for every
iteration (no reuse, no cycling) so repeated calls never hit a warm cache
(PyTorchSim assumes full HBM load every time, so we avoid artificial cache
reuse on TPU). Generation happens up front, so the timed/profiled loops stay
pure GEMM with no RNG cost mixed in.

Run on TPU VM (after pip install -U "jax[tpu]"):
    python3 gemm_smoke.py
"""
import os, time, json
import jax, jax.numpy as jnp

N = 256
DTYPE = jnp.bfloat16
WARMUP = 100
MEASURE = 100
PROFILE = 5          # iterations captured in the TensorBoard trace

PEAK_TFLOPS = 918.0
HBM_GBPS = 1638 * (2**30) / 1e9
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs")


def main():
    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    key = jax.random.PRNGKey(0)

    # One distinct random (a, b) pair per iteration -> no reuse, no cache reuse.
    total = WARMUP + MEASURE + PROFILE
    keys = jax.random.split(key, total)
    As, Bs = [], []
    for k in keys:
        k1, k2 = jax.random.split(k)
        As.append(jax.random.normal(k1, (N, N), jnp.float32).astype(DTYPE))
        Bs.append(jax.random.normal(k2, (N, N), jnp.float32).astype(DTYPE))

    @jax.jit
    def mm(x, y):
        return x @ y

    w0, m0, p0 = 0, WARMUP, WARMUP + MEASURE

    for i in range(WARMUP):
        out = mm(As[w0 + i], Bs[w0 + i])
    out.block_until_ready()

    t0 = time.perf_counter()
    for i in range(MEASURE):
        out = mm(As[m0 + i], Bs[m0 + i])
    out.block_until_ready()
    t1 = time.perf_counter()
    avg_s = (t1 - t0) / MEASURE

    logdir = os.path.join(LOGROOT, f"gemm_smoke_bf16_{N}")
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for i in range(PROFILE):
            out = mm(As[p0 + i], Bs[p0 + i])
        out.block_until_ready()

    flops = 2.0 * N**3
    tflops = flops / avg_s / 1e12
    bytes_ = 3.0 * N * N * 2
    bw = bytes_ / avg_s / 1e9
    print(f"\nGEMM N={N} bf16 (distinct/iter): {avg_s*1e6:.2f} us/iter  "
          f"{tflops:.2f} TFLOPS ({100*tflops/PEAK_TFLOPS:.1f}%)  "
          f"{bw:.1f} GB/s ({100*bw/HBM_GBPS:.1f}%)")
    res = dict(op="gemm", N=N, dtype="bf16", distinct_per_iter=True,
               avg_us=avg_s*1e6, tflops=tflops, mfu_pct=100*tflops/PEAK_TFLOPS)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "gemm_smoke_result.json"), "w"), indent=2)
    print(f"profile -> {logdir}")


if __name__ == "__main__":
    main()
