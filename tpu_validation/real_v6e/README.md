# Real TPU v6e — Sim Validation Reference Measurements

Ground-truth latency measurements on real TPU v6e for validating the PyTorchSim
TPUv6e config (a.k.a. "N-UPU"). Target: sim within ±10% of these numbers.

## Experiment summary

| Exp | What | Sweep | Points | Status |
|---|---|---|---:|---|
| **[exp_a_prefill/](exp_a_prefill/)** | GQA flash prefill (LLAMA4_TP8, attention-only) | 5 S × 4 B = 20 | 20 | ✅ complete |
| **[exp_b_decode/](exp_b_decode/)** | GQA decode (LLAMA4_TP8, flash vs naive) | 2 impl × 6 S × 1 B = 12 | 12 | ✅ complete |
| **[exp_c_llama/](exp_c_llama/)** | Llama 3.1 8B 1-layer (prefill + decode) | up to 74 | **59 / 74 (80%)** | ⚠ partial (preempt × 2) |
| **[exp_d_decode_ops/](exp_d_decode_ops/)** | Per-op decode breakdown | 7 ops × 4 S × 2 B = 56 | 56 | ✅ complete |
| **Total** | | | **147** | |

**Priority order per original spec**: A → B → C → D. A and B (the sim-kernel gates)
both fully completed. C/D add broader reference / per-op attribution.

## Environment

| Item | Value |
|---|---|
| Hardware | TPU **v6e-1** (1 chip, 1 TensorCore) |
| TPU mode | Preemptible (on-demand had 0 capacity all run) |
| Zone | `europe-west4-a` |
| GCP Project | `tpu-board` |
| Runtime image | `v2-alpha-tpuv6e` |
| JAX | 0.6.2 |
| libtpu | shipped with runtime |
| v6e peak refs | **918 TFLOPS bf16** · **1759 GB/s HBM** · ridge AI **522 FLOP/byte** |
| Date | 2026-06-08 |

## Common methodology

Every experiment uses:
1. Random init weights / inputs.
2. **JIT-compile + 30 warmup iterations**.
3. **50-100 timed iterations** (EXP-A/B use distinct random inputs per iter;
   EXP-C/D use a fixed input — script convention).
4. `block_until_ready()` once after the loop.
5. `mean = total / iters`.
6. **5-iter JAX profile trace** captured per point → TensorBoard `.xplane.pb` files.

The per-experiment README documents shape, sweep, and any deviations.

## How to read this directory

```
real_v6e/
├── README.md                          ← this file (orientation)
│
├── exp_a_prefill/                     ⭐ HIGHEST PRIORITY — sim-kernel gate
│   ├── README.md                      ← exp-specific doc
│   ├── gqa_prefill_bench.py
│   ├── gqa_prefill_results.jsonl
│   ├── gqa_prefill_partial.jsonl      ← 5 pts from a preempted attempt
│   └── profile_prefill_b1_s2048/      ← rep. TB profile
│
├── exp_b_decode/                      ⭐ HIGHEST PRIORITY — sim-kernel gate
│   ├── README.md
│   ├── gqa_decode_bench.py
│   ├── gqa_decode_results.jsonl
│   ├── profile_decode_flash_bf16_b1_s8192/
│   └── profile_decode_naive_fp16_b1_s8192/
│
├── exp_c_llama/                       broader reference
│   ├── README.md
│   ├── llama_layer_bench.py
│   └── llama_results.jsonl
│
└── exp_d_decode_ops/                  finer per-op attribution
    ├── README.md
    ├── decode_ops_bench.py
    ├── decode_ops_results.jsonl
    ├── profile_attn_fp16_b8_s4096/
    ├── profile_attn_flash_fp16_b8_s4096/
    └── profile_ffn_fp16_b8_s4096/
```

Each subdirectory is self-contained (bench script + raw JSONL + key profiles +
README). You can hand any one of them to a downstream user.

## Top-line findings

### EXP-A (prefill flash)
Three regimes by `B·S²`:
- Kernel-overhead floor (`B·S² < 1M`): 28–43 us
- Pre-flash compute peak: **70 TFLOPS at B=4 S=1024**
- Flash plateau: **~24 TFLOPS sustained** for `B·S² > 10M`

→ Sim should match flash-plateau numbers within ±10% using `eff ≈ 0.026` of peak.

### EXP-B (decode flash vs naive)
- **Flat 28–39 us** across all S (B=1) — kernel-launch overhead dominates.
- **Flash ≈ naive** at B=1 (within 1–3 us). Flash advantage doesn't manifest at
  single-token + tiny KV.
- BW utilization peaks at 7.8% (S=10240). Chip is heavily under-used.

→ For exact dtype-match: use **`naive fp16`** numbers.

### EXP-C (full Llama 3.1 8B 1-layer)
- **Decode is FFN-dominated** (~75–80% of layer time).
- Per-layer decode at B=1: 416–587 us across S=512..8192.
- Prefill at S ≥ 4096 with B ≥ 2 shows **OOM/spill behavior** (50–250× slowdowns)
  — treat as upper bounds, not steady-state.

### EXP-D (per-op decode)
- **FFN: ~320 us** (the single dominant op). 63% of HBM peak achieved.
- Vector ops (rmsnorm, rope): 25–35 us flat floor (dispatch-bound).
- GEMM ops (qkv, oproj): 30–60 us, GEMV-like at B≤8.
- Attention (naive == flash at B≤8): scales linearly with S above S≈2048 when B=8.

## Caveats

- **Preempt-driven gaps**: TPU v6e-1 preempt was reclaimed twice during this work.
  EXP-A first attempt died at 5 points (preserved as `gqa_prefill_partial.jsonl`).
  EXP-C lost 15 of 74 planned points.
- **OOM regime in EXP-C** at S ≥ 4096 with B ≥ 2 prefill: marked with ⚠ in
  exp_c_llama/README.md. Don't use those for sim calibration.
- **Distinct inputs per iter** only in EXP-A/B. EXP-C/D use script defaults
  (same input each iter) — minor potential underestimation from cache reuse.
- **Single chip (v6e-1)** — no ICI, no multi-host, no TP. Real deployments would
  shard across 8 chips minimum.
- Mosaic flash path was confirmed empirically (flash plateau at 24 TFLOPS,
  no-OOM at S=8192, sub-linear scaling in S²) — not by reading HLO. To verify
  exactly which kernel ran, dump HLO via `jax.jit(f).lower(*args).compile()`.

## Reproducing on a fresh TPU

```bash
# 1) Create a v6e-1 (or v6e-4/8 for larger sweeps)
NAME=okkyun-v6e-test ZONE=europe-west4-a
gcloud compute tpus tpu-vm create $NAME --project=tpu-board --zone=$ZONE \
  --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e --preemptible

# 2) Upload + install
gcloud compute tpus tpu-vm scp \
  exp_a_prefill/gqa_prefill_bench.py \
  exp_b_decode/gqa_decode_bench.py \
  exp_c_llama/llama_layer_bench.py \
  exp_d_decode_ops/decode_ops_bench.py \
  $NAME:~/ --zone=$ZONE
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE \
  --command='pip install -q -U "jax[tpu]"'

# 3) Run any subset (see each exp's README for the full sweep loop)
# 4) Pull results: gcloud compute tpus tpu-vm scp $NAME:~/*.jsonl . --zone=$ZONE
# 5) Delete: gcloud compute tpus tpu-vm delete $NAME --zone=$ZONE --quiet
```

## Sim-side artifacts (not in this directory — FYI)

The corresponding sim tests live in PyTorchSim:
- `tests/test_gqa_prefill.py` (sim EXP-A counterpart)
- `tests/test_gqa_decode.py` (sim EXP-B counterpart)

Both must run with `TOGSIM_CONFIG=configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml`
(otherwise they silently use the default tpuv3 128×128 config).
