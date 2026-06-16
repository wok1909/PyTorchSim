# Real TPU v6e Reference Experiments — Handoff Spec

**Purpose**: Measure **real TPU v6e** latencies to serve as ground truth for validating
the PyTorchSim TPUv6e config (= "N-UPU"). We are building/validating a tiled flash
**prefill** kernel and a Flash-**Decode** kernel in the simulator; we need matching
real-device numbers to confirm the sim is within ±10%.

This doc is self-contained so it can be handed to another agent. Everything runs in
**JAX on a real v6e-1 (1 chip, 1 TensorCore)**.

---

## 0. Getting the TPU (already in progress)

- An on-demand poller is running on the host: `bash /tmp/od_retry.sh` (creates
  `okkyun-v6e-od`, `--accelerator-type=v6e-1`, `--version=v2-alpha-tpuv6e`, across 7 zones).
- When grabbed, `/tmp/od_zone.txt` holds the zone and `/tmp/od_retry.log` shows `GOT_OD`.
- **Use on-demand, NOT preemptible** (preempt gets reclaimed mid-run).
- SSH in, install JAX TPU: `pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`
- Verify: `python3 -c "import jax; print(jax.devices())"` → should show 1 TpuDevice (v6e).
- **DELETE the TPU when done** (`gcloud compute tpus tpu-vm delete okkyun-v6e-od --zone=<z>`).

---

## 1. CRITICAL constraints (read first)

- **dtype**:
  - **Prefill (Mosaic flash)** is **bf16-only** on real TPU (fp16 → "Unsupported type 'f16'").
    Sim is fp16-only → run real prefill in bf16, sim in fp16, compare *timing* not values.
  - **Decode (naive, no Mosaic)** runs **fp16 fine** on real TPU → exact dtype match with sim
    (better, removes dtype as a confound for EXP-B).
- **Measure steady-state**: JIT-compile once (warmup), then time ≥100 iterations with
  **distinct random inputs per iter** (avoid constant-folding / cache effects), use
  `block_until_ready()`, report mean (and divide total by the iteration count, not by warmups).
- **v6e-1 spec** (for sanity): 1 TensorCore, 2 MXU @ 256×256, ~918 TFLOPS bf16,
  HBM 32GB @ 1600 GB/s, clock ~3502 MHz.

---

## 2. Attention shape under test (LLAMA4_TP8 per-partition)

This is the shape the simulator kernels use — match it exactly for apples-to-apples:

- `HEAD_DIM (D) = 128`
- `NUM_Q_HEADS (Hq) = 5`   (= 40 total / TP8)
- `NUM_KV_HEADS (Hkv) = 1` (=  8 total / TP8)   → GQA group size G = 5
- `scale = 1/sqrt(128)`

---

## 3. Experiments to run

### EXP-A — Attention **prefill** latency (HIGHEST PRIORITY)
Validates the new tiled flash-prefill sim kernel (`tests/test_gqa_prefill.py`).

- Op: causal flash attention, full self-attention (query len = key len = S).
- Shapes: `q [B, Hq=5, S, 128]`, `k/v [B, Hkv=1, S, 128]`, **causal=True**, **bf16**.
- Sweep: `S ∈ {512, 1024, 2048, 4096, 8192}`, `B = 1` (single request first).
- Implementation: use the JAX flash path — `jax.nn.dot_product_attention(..., is_causal=True)`
  with the Mosaic/Pallas flash backend, OR `flash_attention` from
  `jax.experimental.pallas.ops.tpu.flash_attention`. Confirm it lowers to a flash kernel
  (not materialized S×S) — check it doesn't OOM at S=8192.
- **Measure**: per-call latency (us). Report a table: S × latency.

### EXP-B — Attention **decode** latency
Validates the Flash-Decode sim kernel (`tests/test_gqa_decode.py`).

- Op: single-token decode against a KV cache.
- Shapes: `q [B, Hq=5, 1, 128]`, `k/v cache [B, Hkv=1, S, 128]`, **non-causal**, **bf16**.
- Sweep: `S (context len) ∈ {512, 1024, 2048, 4096, 8192, 10240}`, `B = 1`.
- Implementation: flash-decode / `jax.nn.dot_product_attention` with q length 1.
- **Measure**: per-call latency (us). Table: S × latency.

### EXP-C — Full Llama 3.1 8B **1 decoder layer** (broader reference)
Use existing script `tpu_validation/llama_layer_bench.py` (already written for this).
- Config baked in: `D=4096, N_H=32, N_KV=8, HD=128, FFN=14336`.
- Prefill: flash via `jax.nn.dot_product_attention`; decode: naive.
- Sweep: prefill `S ∈ {512,1024,2048,4096,8192}`, batch `B ∈ {1,2,4,8,16,32,64,128}`, bf16.
- **Measure**: prefill latency and per-step decode latency (TPOT), per (S,B).

### EXP-D — Per-op decode breakdown (optional, finer granularity)
Use existing `tpu_validation/decode_ops_bench.py` (ops: rmsnorm/rope/qkv/attn(GQA-native einsum)/
attn_flash(Flash-Decode)/oproj/ffn).
- Gives per-op latency to attribute where decode time goes.
- bf16 for flash ops; note the attn vs attn_flash variants.

---

## 4. Output format expected (so it slots into validation)

For each experiment, produce a table with these columns and a short note on the env:
- HW: v6e-1, zone, JAX/libtpu version
- dtype: bf16
- shape params (Hq, Hkv, D)
- sweep var (S and/or B)
- **latency (us)**, mean over ≥100 iters, distinct random inputs/iter

Save raw results as CSV/JSON under `tpu_validation/real_v6e/` and also extract a
TensorBoard trace for at least one point per experiment (to inspect kernel breakdown).

Priority order: **A → B → C → D**. A and B directly gate the current sim-kernel validation.

---

## 5. Existing assets to reuse
- `tpu_validation/llama_layer_bench.py` — EXP-C (Llama 8B 1-layer, JAX).
- `tpu_validation/decode_ops_bench.py` — EXP-D (per-op decode, JAX).
- For EXP-A/B you may need a small new JAX script (attention-only, the LLAMA4_TP8 shape) —
  ~40 lines, mirror the timing harness in the above scripts.

## 6. Sim side (for the comparison, FYI — not run on TPU)
- Sim prefill: `tests/test_gqa_prefill.py` (in PyTorchSim container, fp16).
- Sim decode:  `tests/test_gqa_decode.py`.
- Both must run with `TOGSIM_CONFIG=configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml`
  (else they silently use the default tpuv3 128×128 config!).
