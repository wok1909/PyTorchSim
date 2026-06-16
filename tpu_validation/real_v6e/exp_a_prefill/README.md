# EXP-A — GQA Flash Prefill (real TPU v6e)

## Purpose

Ground-truth latency for the **flash-prefill** path. Validates the sim kernel
`tests/test_gqa_prefill.py`. Target: sim within ±10% of these numbers in the
flash-plateau regime.

## What was measured

A single call to the JAX SDPA flash kernel — attention math only, no projections,
norms, RoPE, or FFN:

```python
out = jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=1/sqrt(128))
```

## Shape (LLAMA4_TP8 per-partition, matches sim)

| Param | Value |
|---|---|
| `HEAD_DIM (D)` | 128 |
| `NUM_Q_HEADS (Hq)` | 5 (= 40 total / TP8) |
| `NUM_KV_HEADS (Hkv)` | 1 (= 8 total / TP8) |
| GQA group `G` | Hq/Hkv = 5 |
| Layout | BSNH: `q [B,S,5,128]`, `k/v [B,S,1,128]` |
| dtype | **bf16** (Mosaic flash is bf16-only) |
| `scale` | 1 / √128 |
| causal | True |

## Sweep — 20 points

| Var | Values |
|---|---|
| `S` | 512, 1024, 2048, 4096, 8192 |
| `B` | 1, 2, 4, 8 |

## Methodology

1. 30 warmup iterations (forces JIT compile, populates caches)
2. **100 timed iterations** with **distinct random inputs per iter** (different PRNGKey)
   — defeats XLA constant-folding and unrealistic HBM-cache hits
3. Single `block_until_ready()` after the loop
4. `mean = total / 100`
5. After timing, capture a 5-iter JAX profile trace

## Results

Causal FLOPs/iter = `2 × 2 × B × Hq × S² × D / 2`. v6e peak: 918 TFLOPS bf16.

| B | S | avg us | TFLOPS | %peak | causal FLOPs/iter |
|---|---|---:|---:|---:|---:|
| 1 | 512 | 42.14 | 8.0 | 0.9% | 0.34G |
| 2 | 512 | 31.02 | 21.6 | 2.4% | 0.67G |
| 4 | 512 | 30.90 | 43.4 | 4.7% | 1.34G |
| 8 | 512 | 42.45 | 63.2 | 6.9% | 2.68G |
| 1 | 1024 | 30.52 | 44.0 | 4.8% | 1.34G |
| 2 | 1024 | 43.29 | 62.0 | 6.8% | 2.68G |
| 4 | 1024 | **76.60** | **70.1** | **7.6%** | 5.37G |
| 8 | 1024 | 436.13 | 24.6 | 2.7% | 10.74G |
| 1 | 2048 | 82.17 | 65.3 | 7.1% | 5.37G |
| 2 | 2048 | 422.34 | 25.4 | 2.8% | 10.74G |
| 4 | 2048 | 889.36 | 24.1 | 2.6% | 21.47G |
| 8 | 2048 | 1802.28 | 23.8 | 2.6% | 42.95G |
| 1 | 4096 | 860.45 | 25.0 | 2.7% | 21.47G |
| 2 | 4096 | 1758.46 | 24.4 | 2.7% | 42.95G |
| 4 | 4096 | 3561.96 | 24.1 | 2.6% | 85.90G |
| 8 | 4096 | 7109.72 | 24.2 | 2.6% | 171.80G |
| 1 | 8192 | 3522.22 | 24.4 | 2.7% | 85.90G |
| 2 | 8192 | 7016.09 | 24.5 | 2.7% | 171.80G |
| 4 | 8192 | 14498.15 | 23.7 | 2.6% | 343.60G |
| 8 | 8192 | 29351.71 | 23.4 | 2.6% | 687.19G |

## Observed regimes

Three distinct regimes by `B × S²`:

1. **Kernel-overhead** (`B·S² < ~1M`): 28–43 us latency floor regardless of size.
   Dispatch + setup overhead dominates the actual compute.
2. **Pre-flash compute** (~1M ≤ `B·S²` ≤ ~10M): peak observed **70 TFLOPS at B=4 S=1024**.
   Mosaic appears to use a materialized (non-tiled) kernel for this region.
3. **Flash plateau** (`B·S² > ~10M`): stable **~24 TFLOPS** sustained. Linear scaling
   in `B·S²`. This is the actual flash-attention kernel.

## Sim validation usage

- For `B·S² > 10M`: sim should match within **±10%**, with fixed `eff ≈ 0.026` of 918 TFLOPS
  (≈ 23.9 TFLOPS sustained).
- For smaller sizes: kernel choice differs; treat as a separate baseline if needed.

## Files

```
exp_a_prefill/
├── README.md                      ← this file
├── gqa_prefill_bench.py           ← the bench script (run on TPU VM)
├── gqa_prefill_results.jsonl      ← 20 final results
├── gqa_prefill_partial.jsonl      ← 5 results from a preempted first attempt
└── profile_prefill_b1_s2048/      ← representative TB profile
```

### JSONL fields

```
exp, op, batch, seq, Hq=5, Hkv=1, D=128, dtype, iters=100, avg_us,
causal=true, flops_causal, achieved_tflops, profile_dir (on-VM)
```

## How to reproduce

```bash
# On TPU v6e VM with JAX[tpu] installed
python3 gqa_prefill_bench.py --seq 2048 --batch 1 --dtype bf16
# Output: RESULT prefill B=1 S=2048 bf16: <us> us/iter ... + profile dir
```
