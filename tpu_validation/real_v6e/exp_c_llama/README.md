# EXP-C — Llama 3.1 8B (1 decoder layer) prefill + decode (real TPU v6e)

## Purpose

Broader reference: a full Llama 3.1 8B decoder layer's prefill and decode latency
on real v6e, to validate the **end-to-end** sim layer including RMSNorm + QKV +
RoPE + attention + Wo + SwiGLU FFN. Not a sim-kernel gate (that's A/B), but
provides context for "how much do non-attention ops cost in the whole layer".

## What was measured

One Llama 3.1 8B decoder layer:

```
x → RMSNorm → Wq/Wk/Wv → RoPE → attention (flash for prefill, naive for decode)
  → Wo → residual → RMSNorm → SwiGLU FFN (Wgate, Wup, Wdown) → residual
```

Random init weights. Same architecture for prefill and decode (only attention
path differs).

## Architecture (Llama 3.1 8B per-layer)

| Param | Value |
|---|---|
| `D` (model) | 4096 |
| `N_H` (query heads) | 32 |
| `N_KV` (kv heads) | 8 |
| `HD` (head dim) | 128 |
| `FFN` (intermediate) | 14336 |
| `θ` (RoPE base) | 500000 |
| dtype | bf16 |

Note: this is **different from EXP-A/B's LLAMA4_TP8 shape** (D=128, Hq=5, Hkv=1).
EXP-C is a "broader reference" with the full standard Llama 3.1 8B config.

## Sweep — 74 planned, 59 completed (80%)

Memory caps per S (FFN intermediate `B*S*FFN*2*2` must fit ~16GB):

| S | B values | Mode | Captured / Planned |
|---|---|---|---:|
| 512 | 1, 2, 4, 8, 16, 32, 64, 128 | both | 16/16 ✓ |
| 1024 | 1, 2, 4, 8, 16, 32, 64, 128 | both | 16/16 ✓ |
| 2048 | 1, 2, 4, 8, 16, 32, 64, 128 | both | **14/16** (prefill B=64,128 missing) |
| 4096 | 1, 2, 4, 8, 16, 32, 64 | both | **9/14** |
| 8192 | 1, 2, 4, 8, 16, 32 | both | **4/12** |
| **Total** | | | **59/74** |

Missing 15 points (mostly large-B prefill at S≥4096) due to two preempt events
mid-sweep. These would likely have hit memory pressure anyway (see "Caveats").

## Methodology

1. JIT compile + 30 warmup iters
2. **50 timed iterations** (same input — script uses fixed inputs after init)
3. `block_until_ready()` after loop
4. 5-iter profile capture
5. JSONL record per (mode, S, B)

Note: this script uses **same inputs every iter** (unlike EXP-A/B's distinct-per-iter).
Real-world numbers may differ slightly under non-cached-input conditions.

## Results — PREFILL

| B | S=512 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|---:|
| 1 | 441 us | 941 us | 2361 us | 7833 us | 27362 us |
| 2 | 768 us | 1643 us | 4245 us | **448128 us** ⚠ | — |
| 4 | 1726 us | 3561 us | 8880 us | — | — |
| 8 | 3705 us | 6951 us | 21005 us | **1844069 us** ⚠ | — |
| 16 | 7461 us | 16246 us | 43634 us | — | — |
| 32 | 17248 us | 33768 us | **594347 us** ⚠ | — | — |
| 64 | 35062 us | 64967 us | — | — | — |
| 128 | 67397 us | **214984 us** ⚠ | — | — | — |

⚠ = severely super-linear (likely memory pressure / HBM spilling). See Caveats.

## Results — DECODE (per-step / TPOT)

| B | S=512 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|---:|
| 1 | 416 us | 418 us | 420 us | 426 us | 587 us |
| 2 | 418 us | 418 us | 426 us | 567 us | 1286 us |
| 4 | 421 us | 424 us | 505 us | 1203 us | — |
| 8 | 428 us | 543 us | 1089 us | 2188 us | — |
| 16 | 490 us | 1056 us | 2002 us | 4106 us | — |
| 32 | 1053 us | 1918 us | 3693 us | 7939 us | — |
| 64 | 1933 us | 3656 us | 7396 us | 15666 us | — |
| 128 | 3669 us | 6970 us | 14297 us | — | — |

Decode is much cleaner — flat ~420 us floor for small B, near-linear scaling in
B*S above the floor. Behaves as expected for memory-bound decode.

## Key observations

### Prefill
- **Normal regime** (S ≤ 2048, B ≤ 16): clean scaling.
  Example: prefill 1 layer (1024, 8) ≈ 7 ms, full 32-layer model ≈ 224 ms.
- **OOM/spill regime** at S=4096 B=2 and beyond: 50–250× slowdowns indicate
  the FFN intermediate (`B·S·FFN·2 = 469MB` at B=2 S=4096) plus attention
  activation budget exceeds practical HBM working set.
- The script's CAP table was too optimistic at S ≥ 4096.

### Decode
- **Decode is FFN-dominated.** Compare to EXP-D: FFN alone is ~325 us, and full
  decode is ~420 us at small B. → FFN ≈ 75–80% of decode time.
- Per-layer decode at B=1 grows from 416 us (S=512) to 587 us (S=8192)
  — only 41% slower despite 16× more KV — confirms attention is small fraction.

### Layer total perspective
- For a 32-layer Llama 3.1 8B: 32 × per-layer numbers ≈ wall time.
  E.g., decode TPOT at B=1, S=4096 ≈ 32 × 426 us = **13.6 ms/token**.

## Caveats

- **2 preemption events** during this sweep — completed in two segments
  (initial + resume). See `source` field in JSONL.
- **OOM-suspect points** (marked ⚠ above) likely hit HBM working-set pressure;
  treat them as **upper bound** not steady-state. Do not use for sim calibration
  in the OOM regime.
- Script uses **same random inputs every iter** (unlike EXP-A/B distinct inputs)
  → potential minor underestimation due to cache reuse. For order-of-magnitude
  this doesn't matter.
- v6e-1 single-chip — no tensor parallelism. Real Llama deployment would shard
  across 8 chips (TP8) at minimum.

## Files

```
exp_c_llama/
├── README.md                ← this file
├── llama_layer_bench.py     ← the bench script
└── llama_results.jsonl      ← 59 results (50 from initial + 9 from resume)
```

### JSONL fields

```
exp="C", mode (prefill|decode), batch, seq, dtype, avg_us, source (which sweep run)
```

## How to reproduce

```bash
python3 llama_layer_bench.py --mode prefill --seq 2048 --batch 8 --dtype bf16
python3 llama_layer_bench.py --mode decode  --seq 2048 --batch 8 --dtype bf16
```

Or full sweep:
```bash
for S in 512 1024 2048 4096 8192; do
  for B in 1 2 4 8 16 32 64 128; do
    for MODE in prefill decode; do
      python3 llama_layer_bench.py --mode $MODE --seq $S --batch $B --dtype bf16
    done
  done
done
```
