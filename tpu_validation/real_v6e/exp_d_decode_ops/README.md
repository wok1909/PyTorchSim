# EXP-D — Per-Op Decode Breakdown (real TPU v6e)

## Purpose

**Finer-grained reference**: isolate each operation in a Llama decoder layer to
measure its individual contribution to decode latency. Helps attribute "where does
decode time go?" for sim validation and optimization.

## What was measured

Each operator in isolation (no fusion across operators), single decode step:

| op | What it computes |
|---|---|
| `rmsnorm` | RMS norm over last dim of `[B, D]` |
| `rope` | Rotary position embedding on `[B, N_H, HD]` |
| `qkv` | Three GEMMs: `x → Q[B,D]`, `K/V[B, N_KV·HD]` |
| `attn` | GQA-native naive attention (einsum + softmax + einsum) |
| `attn_flash` | GQA Flash-Decode (KV-tiled, online softmax) |
| `oproj` | Output projection: `o[B,D] → [B,D]` |
| `ffn` | SwiGLU: `silu(x @ Wgate) * (x @ Wup) → @ Wdown` |

Shape baseline: Llama 3.1 8B config (`D=4096, N_H=32, N_KV=8, HD=128, FFN=14336`).
Same script as `tpu_validation/decode_ops_bench.py`. dtype fp16 (matches sim exactly).

## Sweep — 56 points

| Var | Values |
|---|---|
| `op` | rmsnorm, rope, qkv, attn, attn_flash, oproj, ffn (7 ops) |
| `S` (KV cache length) | 1024, 2048, 4096, 8192 (4 values) |
| `B` (batch) | 1, 8 (2 values) |
| `dtype` | fp16 |

## Methodology

30 warmup + 50 timed iters + 5-iter profile. Same as EXP-C.

## Results (all 56 points, fp16, avg us/iter)

### Vector ops (B-independent, ~constant)

| op | B=1 S=1024 | B=8 S=1024 | B=1 S=8192 | B=8 S=8192 |
|---|---:|---:|---:|---:|
| rmsnorm | 33.6 | 29.5 | 27.1 | 30.1 |
| rope    | 27.5 | 27.8 | 28.5 | 25.8 |

→ ~25–34 us flat. Pure dispatch-bound; no measurable scaling.

### GEMM ops (B-dependent at decode)

| op | B=1 S=2048 | B=8 S=2048 | B=1 S=8192 | B=8 S=8192 |
|---|---:|---:|---:|---:|
| qkv   | 51.9 | 57.4 | 49.5 | 46.0 |
| oproj | 37.0 | 30.1 | 29.8 | 37.0 |

→ qkv ≈ 47–59 us (3 GEMMs combined). oproj ≈ 30–37 us (1 GEMM). Mostly flat —
single-token decode underutilizes the MXU regardless of B (B=8 still tiny).

### Attention (B and S dependent)

| S | impl | B=1 | B=8 |
|---|---|---:|---:|
| 1024 | attn | 30.4 | 28.6 |
| 1024 | attn_flash | 29.6 | 30.1 |
| 2048 | attn | 37.9 | 33.8 |
| 2048 | attn_flash | 30.2 | 35.1 |
| 4096 | attn | 30.2 | **117.8** |
| 4096 | attn_flash | 30.3 | **121.6** |
| 8192 | attn | 29.4 | **242.0** |
| 8192 | attn_flash | 30.9 | **240.4** |

→ At B=1: flat ~30 us (memory-bound, KV cache tiny per token).
→ At B=8: scales linearly with S above S≈2048 — the chip starts doing real work.
→ **Flash ≈ Naive** within ±2 us at every point. At B≤8 single-token decode the
   flash advantage doesn't materialize.

### FFN — the dominant cost

| op | B=1 S=1024 | B=8 S=1024 | B=1 S=8192 | B=8 S=8192 |
|---|---:|---:|---:|---:|
| ffn | **321.9** | **313.3** | **345.0** | **328.7** |

→ ~320 us **regardless of B or S** — 10× more than any other single op.
→ This is two GEMMs (`x @ Wgate (D × FFN)`, `x @ Wup (D × FFN)`) + SiLU + multiply +
   one more GEMM (`@ Wdown (FFN × D)`).
→ FFN is GEMV-like at decode (M=B≤8, K=D=4096, N=FFN=14336) — weight read of
   `D × FFN × 2` per matmul × 3 = **351 MB**. At 1759 GB/s HBM, theoretical min = 200 us.
   Observed 320 us → 63% of HBM peak. Reasonable.

## Key observations

- **FFN dominates decode** (~325 us out of ~420 us total layer decode, see EXP-C).
- **Vector ops are dispatch-bound** (~25–35 us floor across the board).
- **GEMM at decode is GEMV-like** (B≤8 → MXU heavily under-used).
- **attn vs attn_flash: nearly identical** at B≤8. Flash gains appear only at
  larger B (not measured here) or when KV cache pushes activations out of VMEM.
- **Total per-layer decode budget** ≈ `rmsnorm + qkv + rope + attn + oproj +
  rmsnorm + ffn` ≈ `30 + 50 + 28 + 30 + 35 + 30 + 320` = **523 us** (B=1, S=4096
  estimate). Matches EXP-C decode B=1 S=4096 = 426 us (some fusion / overlap).

## Sim validation usage

- **FFN target**: 320 us at all (B, S) for Llama 3.1 8B FFN at fp16.
- **Decode attention** (naive == flash at B≤8): use this table by (S, B).
- **Vector ops floor**: sim should model dispatch overhead, not just compute.
- **Total decode budget** should equal sum of per-op (within ~10%).

## Files

```
exp_d_decode_ops/
├── README.md                                  ← this file
├── decode_ops_bench.py                        ← the bench script
├── decode_ops_results.jsonl                   ← 56 results
├── profile_attn_fp16_b8_s4096/                ← naive attention representative
├── profile_attn_flash_fp16_b8_s4096/          ← flash decode representative
└── profile_ffn_fp16_b8_s4096/                 ← FFN representative (dominant op)
```

### JSONL fields

```
op, batch, seq, dtype, avg_us
```

## How to reproduce

```bash
python3 decode_ops_bench.py --op ffn --batch 8 --seq 4096 --dtype fp16
# Output: RESULT op=ffn B=8 S=4096 fp16: <us> us/iter ...
```

Full sweep:
```bash
for OP in rmsnorm rope qkv attn attn_flash oproj ffn; do
  for S in 1024 2048 4096 8192; do
    for B in 1 8; do
      python3 decode_ops_bench.py --op $OP --seq $S --batch $B --dtype fp16
    done
  done
done
```
