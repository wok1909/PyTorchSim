#!/usr/bin/env python3
"""VPU-PEAK microbench for TPU v6e  (real-TPU side; JAX).

GOAL: measure the v6e VPU's *compute* throughput in isolation, to decide whether
PyTorchSim's VPU should be modeled at vlen-256 or vlen-128.

WHY this kernel: a small VMEM-resident array gets a long chain of K elementwise
fused-multiply-add steps, each followed by a relu. The array is read once / written
once (HBM ~0 after that); the K compute steps run on-chip -> the VPU is the
bottleneck (NOT the MXU, NOT HBM). The relu is a non-linearity that stops XLA from
algebraically folding the linear recurrence y=y*c1+c2 into a closed form, so all K
steps actually execute. No transcendentals (so the identical kernel also compiles in
PyTorchSim, which has an fp16-exp/tanh codegen bug).

Run on the TPU VM:
    python3 vpu_peak_bench.py
Reads achieved TFLOP/s per (N,K). ALSO captures an XProf trace per config under
./tb_logs_vpu/ -- in XProf check: HBM BW util should be LOW (confirms VPU-bound),
and read the per-core FLOP rate.

Compare the achieved VPU TFLOP/s to:  128 lanes x 8 sublanes x 2 flop x clock(~0.94GHz)
~= 1.9 TFLOP/s.  If measured ~1.9 -> confirms the 8-sublane structure (=> sim vlen-256
is the right throughput match; vlen-128 would be ~2x too slow).
"""
import os, time, json
import jax, jax.numpy as jnp

# (N, K): N = square side (VMEM-resident), K = number of FMA+relu steps (compute knob)
CONFIGS = [(512, 200), (1024, 200), (2048, 100), (1024, 400)]
DTYPES  = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}
WARMUP, MEASURE, PROFILE = 30, 100, 5
LOGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs_vpu")
# v6e VPU peak estimate for reference (128 lanes x 8 sublanes x 2 flop x ~0.94 GHz)
VPU_PEAK_REF_TFLOPS = 128 * 8 * 2 * 0.94e9 / 1e12


def make_kernel(K, c1, c2, lo):
    @jax.jit
    def fn(y):
        for _ in range(K):
            y = jnp.maximum(y * c1 + c2, lo)   # FMA then relu-ish clamp (non-foldable, VPU-only)
        return y
    return fn


def bench(N, K, dtype_name, dtype):
    elem = 2 if dtype_name != "fp32" else 4
    key = jax.random.PRNGKey(0)
    x = (jax.random.normal(key, (N, N), jnp.float32) * 0.1).astype(dtype)
    # coefficients chosen so values stay bounded (|c1|<1) -> no overflow over K steps
    c1 = jnp.asarray(0.99, dtype); c2 = jnp.asarray(0.001, dtype); lo = jnp.asarray(-1.0, dtype)
    fn = make_kernel(K, c1, c2, lo)

    out = fn(x); out.block_until_ready()                 # compile
    for _ in range(WARMUP):
        out = fn(x)
    out.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        out = fn(x)
    out.block_until_ready()
    t1 = time.perf_counter()
    avg_s = (t1 - t0) / MEASURE

    logdir = os.path.join(LOGROOT, f"vpu_{dtype_name}_N{N}_K{K}")
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for _ in range(PROFILE):
            out = fn(x)
        out.block_until_ready()

    flop  = 2.0 * N * N * K            # mul + add per element per step (relu not counted)
    bytes_ = 2.0 * N * N * elem        # read input + write output once (K steps are on-chip)
    tflops = flop / avg_s / 1e12
    bw_gbps = bytes_ / avg_s / 1e9
    print(f"VPU N={N:>5} K={K:>4} {dtype_name:>4}: {avg_s*1e6:9.2f} us/iter  "
          f"{tflops:7.2f} TFLOP/s  (HBM~{bw_gbps:6.1f} GB/s -> want LOW)  "
          f"[ref VPU peak ~{VPU_PEAK_REF_TFLOPS:.2f}]")
    return dict(N=N, K=K, dtype=dtype_name, avg_us=avg_s*1e6, tflops=tflops, bw_gbps=bw_gbps)


def main():
    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    print(f"VPU peak reference (128x8x2x0.94GHz) ~= {VPU_PEAK_REF_TFLOPS:.2f} TFLOP/s")
    print(f"warmup={WARMUP} measure={MEASURE}\n")
    results = []
    for dtype_name in ("fp16", "bf16", "fp32"):  # fp16 = matches sim/our pipeline; bf16 = native v6e; fp32 ref
        for N, K in CONFIGS:
            try:
                results.append(bench(N, K, dtype_name, DTYPES[dtype_name]))
            except Exception as e:
                print(f"VPU N={N} K={K} {dtype_name}: FAILED {e}")
        print()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vpu_peak_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"saved {out}")
    print(f"XProf traces in {LOGROOT}/  (scp back; check HBM util LOW + FLOP rate)")
    print("\nDECISION: take the MAX achieved TFLOP/s (VPU peak). Compare to sim vlen-256 vs vlen-128.")


if __name__ == "__main__":
    main()
