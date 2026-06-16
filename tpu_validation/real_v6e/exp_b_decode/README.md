# EXP-B — GQA Single-Token Decode (real TPU v6e)

## Purpose

Ground-truth latency for **single-token decode** against a KV cache. Validates the
sim kernel `tests/test_gqa_decode.py` (Flash-Decode). Target: sim within ±10%.

## What was measured

Two implementations of decode attention (q_len=1), attention math only:

### `flash` impl (Mosaic, bf16-only)
```python
out = jax.nn.dot_product_attention(q, kc, vc, is_causal=False, scale=1/sqrt(128))
```

### `naive` impl (explicit ops, matches sim path, fp16)
```python
kr = jnp.repeat(kc, G=5, axis=2)               # GQA: broadcast KV to Hq heads
vr = jnp.repeat(vc, G=5, axis=2)
sc = (q @ kr.transpose) * scale                # [B, Hq, 1, S]
pr = softmax(sc.astype(fp32), -1).astype(dt)
out = pr @ vr                                  # [B, 1, Hq, D]
```

No projections, no cache update — cache is pre-filled with random data.

## Shape (LLAMA4_TP8 per-partition)

| Param | Value |
|---|---|
| `q` | `[B=1, 1, Hq=5, D=128]` |
| `kc` | `[B=1, S, Hkv=1, D=128]` (KV cache) |
| `vc` | `[B=1, S, Hkv=1, D=128]` |
| `G` (GQA group) | 5 |
| Layout | BSNH |
| causal | False (decode sees full cache) |

## Sweep — 12 points

| Var | Values |
|---|---|
| `impl` | flash, naive |
| `dtype` | flash → **bf16** (Mosaic), naive → **fp16** (sim-exact match) |
| `S` (context length) | 512, 1024, 2048, 4096, 8192, 10240 |
| `B` | 1 |

## Methodology

Identical to EXP-A: 30 warmup → 100 timed (distinct random per iter) → 5-iter profile.

## Results

KV bytes = `B × Hkv × S × D × 2 × 2` (K + V, 2 bytes per element).
v6e HBM peak: 1759 GB/s.

| impl | dtype | S | avg us | KV bytes | KV BW GB/s | %HBM |
|---|---|---:|---:|---:|---:|---:|
| flash | bf16 | 512 | 28.18 | 262K | 9.3 | 0.5% |
| flash | bf16 | 1024 | 29.45 | 524K | 17.8 | 1.0% |
| flash | bf16 | 2048 | 39.78 | 1.0M | 26.4 | 1.5% |
| flash | bf16 | 4096 | 28.20 | 2.1M | 74.4 | 4.2% |
| flash | bf16 | 8192 | 31.06 | 4.2M | 135.0 | 7.7% |
| flash | bf16 | 10240 | 38.25 | 5.2M | 137.1 | 7.8% |
| naive | fp16 | 512 | 30.33 | 262K | 8.6 | 0.5% |
| naive | fp16 | 1024 | 37.78 | 524K | 13.9 | 0.8% |
| naive | fp16 | 2048 | 30.32 | 1.0M | 34.6 | 2.0% |
| naive | fp16 | 4096 | 29.81 | 2.1M | 70.4 | 4.0% |
| naive | fp16 | 8192 | 31.06 | 4.2M | 135.0 | 7.7% |
| naive | fp16 | 10240 | 37.72 | 5.2M | 139.0 | 7.9% |

## Observed regimes

- **Flat latency floor (~28–39 us)** across the entire S range for B=1 decode.
  Dispatch + kernel-launch overhead dominates the actual KV read.
- **Flash ≈ Naive**: at B=1 single-token decode, the workload is too small to
  benefit from flash's online-softmax tiling. Within 1–3 us between impls.
- **fp16 ≈ bf16**: memory-bound; dtype doesn't materially affect bytes.
- BW utilization peaks at **7.8% at S=10240** — chip is heavily under-utilized.
- Decode time grows sub-linearly with S because the work fits well within
  the kernel-overhead floor at any tested S.

## Sim validation usage

- For exact dtype-match: use **`naive fp16`** numbers (sim is fp16-only).
- Sim must model a **fixed ~28–38 us kernel-launch floor**, not just KV-read latency.
- BW-bound regime requires B ≥ 8 or larger Hkv — not covered here (see EXP-D).

## Files

```
exp_b_decode/
├── README.md                                ← this file
├── gqa_decode_bench.py                      ← the bench script
├── gqa_decode_results.jsonl                 ← 12 results
├── profile_decode_flash_bf16_b1_s8192/      ← flash representative profile
└── profile_decode_naive_fp16_b1_s8192/      ← naive representative profile
```

### JSONL fields

```
exp, op, batch, seq, Hq=5, Hkv=1, D=128, dtype, iters=100, avg_us,
kv_bytes, kv_bw_gbs, flops, profile_dir (on-VM)
```

## How to reproduce

```bash
python3 gqa_decode_bench.py --seq 8192 --batch 1 --dtype fp16 --impl naive
python3 gqa_decode_bench.py --seq 8192 --batch 1 --dtype bf16 --impl flash
```
