# PyTorchSim TPUv6e Validation — Handoff

## 0. 한 줄 목표
PyTorchSim(=TOGSim) 시뮬레이터를 **실제 Google TPU v6e**와 대조 검증해서, 검증된 latency를 멀티-HW RAG 시뮬레이터의 LUT로 쓰는 게 최종 목표. 현재 단계는 **primitive op (GEMM / GEMV / VADD) latency가 실제 v6e와 얼마나 맞는지**, 그리고 **VPU vector-length 설정(256 vs 128)**이 정확도에 미치는 영향을 보는 것.

---

## 1. Ground truth: 실제 TPU v6e 스펙
(JAX `tpu_info.py` + Google 공식 + scaling book 기준)
- chip당 **TensorCore 1개**, TensorCore당 **MXU 2개**, MXU = **256×256** systolic array
- **bf16 918 TFLOPS** (일부 자료 920)
- **VPU: num_lanes = 128, num_sublanes = 8** ← 중요 (아래 §3)
- **VMEM ≈ 128 MiB/core**, CMEM = 0, SMEM = 1 MiB/core
- **HBM 32GB @ 1600 GB/s** (≈1.64 TB/s), HBM2-class

실제 측정 데이터는 Google TRC 임대 v6e-1에서 직접 측정 (bf16). 위치: `tpu_validation/real_v6e/` (exp_a_prefill, exp_b_decode, exp_c_llama, exp_d_decode_ops). 본 문서의 "real" 값은 primitive op latency (µs).

---

## 2. PyTorchSim config 와 정당성

파일: `configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml` (vlen-128)
와 `configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml` (vlen-256, base).
**두 config는 `vpu_vector_length_bits` (128 vs 256) 하나만 다르고 나머지는 전부 동일.**

| 파라미터 | 값 | 정당성 |
|---|---|---|
| `num_cores` | 1 | v6e는 chip당 TensorCore 1개 |
| `num_systolic_array_per_core` | 2 | TensorCore당 MXU 2개 |
| systolic array | 256×256 | MXU 크기 |
| `core_freq_mhz` | **3502** | 스펙에서 **역산**: 918e12 / (2 flop/MAC × 256×256 × 2 MXU) = 3502 MHz. (datasheet에 clock 미공개라 TFLOPS로부터 derive) |
| `vpu_num_lanes` | 256 | ⚠️ PyTorchSim 구조상 `systolic_size = vpu_num_lanes`로 **강하게 커플링**되어 있어 256으로 둘 수밖에 없음 (실제 v6e VPU는 128). §3 참고 |
| `vpu_vector_length_bits` | **128 (opt1) / 256 (base)** | 본 실험의 비교 변수. §3 |
| `vpu_spad_size_kb_per_lane` | 512 | 512KB × 256 lane = 128MB ≈ v6e VMEM(~128MiB). 단 이건 **calibration knob** — 우선 넉넉히 주고 validation 후 줄여나갈 예정 |
| `dram_freq_mhz` | 3125 | 3125 × 16ch × 32B × 2burst / 2 = 1600 GB/s 가 되도록 맞춤 |
| `dram_channels` | 16 | 1600 GB/s BW 타겟 (의사채널 포함) |
| `ramulator_config_path` | HBM2.yaml | v6e는 HBM2-class |
| `l2d_config` | 8MB (v4-identical) | calibration knob (v6e L2 미공개라 v4 값 차용) |
| `pytorchsim_functional_mode` | 0 | **functional(정확도 검증) OFF, timing-only**. latency만 측정 |
| `codegen_mapping_strategy` | autotune | tile mapping 자동 탐색 (§5 crash와 관련) |

**Calibration knob 주의**: clock(3502), VMEM(512KB/lane), L2(8MB), DRAM split은 datasheet에 없어서 derive/차용한 값. clock을 3502로 높게 잡은 게 일부 op를 우연히 맞췄을 가능성 있음 — 추후 검증 필요.

---

## 3. 왜 vlen 256 vs 128 을 비교하나 (핵심 동기)

문제: **실제 v6e VPU는 lane이 128개**인데, PyTorchSim은 애초에 `1 vector lane = 1 systolic array row`로 매핑하도록 구현돼 있어서 `vpu_num_lanes`를 systolic array 크기(256)와 분리할 수 없음.

→ 교수님 결정(**option 1**): config는 v6e 그대로(`vpu_num_lanes=256`) 두되, **`vpu_vector_length_bits`를 절반(256→128)으로 줄여 VPU compute throughput을 /2** 해서 128-lane VPU를 *throughput 관점*에서 모델링.

그래서 검증 질문은:
1. vlen-128(option1)이 vlen-256(raw)보다 실제 v6e에 더 가까운가?
2. 애초에 vlen 변경이 이 op들의 latency에 영향이 있긴 한가?

(다른 방안 — 실제 TPU latency 직접 사용 / TPUv4로 회귀 — 은 논문에 N-UPU spec 명시 어려움 / HW 너무 구형이라 기각됨. v6e config 유지가 결정사항.)

---

## 4. 실험 환경 / 재현 방법

- **실행 위치**: docker container `ok_torchsim_n01` (node n01), 이미지 `ok_torchsim_2.8:n01move`.
  PyTorchSim 소스는 host NFS `/home/shared/RAG/PyTorchSim` 를 컨테이너 `/workspace/PyTorchSim`에 **volume mount** (소스 수정이 즉시 반영됨).
- **파이프라인**: Python(torch) → torch.compile/inductor → MLIR template → mlir-opt → LLVM IR → llc → RISC-V obj → gem5 timing sim. LLVM 툴: `/riscv-llvm/bin/{mlir-opt,mlir-translate,llc}`.
- **중요**: template/codegen 수정 후엔 반드시 `rm -rf /tmp/torchinductor*` (inductor FX 해시가 안 바뀌어 stale cache 됨).
- **runner**: `tpu_validation/sim_bench.py --op {gemm,gemv,vadd} --size N --dtype fp16`
  (`TOGSIM_CONFIG` env로 config 지정). dtype은 **fp16** (PyTorchSim은 fp16/fp32만; bf16은 별개의 LLVM 이슈로 crash).
- **sweep 스크립트**:
  - `tpu_validation/revalidate_opt1.sh` → vlen-128, 결과 `revalidate_opt1_results.txt`
  - `tpu_validation/revalidate_base256.sh` → vlen-256, 결과 `revalidate_base256_results.txt`
  - 둘 다 **CMEM=0** (`unset SRAM_BUFFER_PLAN_PATH`, 모든 데이터 HBM streaming), fp16, functional-off, **동일 사이즈**. → vlen만 변수.
- **cycle → µs 변환**: `µs = cycles / 3502` (core_freq 3502 MHz). 두 config 동일 clock이라 공정 비교.
- sweep 사이즈: GEMM/GEMV N ∈ {512..16384}, VADD length ∈ {1M..67M}.

---

## 5. ⚠️ 실험을 돌리려고 먼저 고쳐야 했던 버그 (GEMM autotune crash)

vlen 실험을 돌리려 하니 **GEMM 4096/8192에서 autotune이 큰 tile을 고를 때 `llc`가 crash** (`LLVM ERROR: SmallVector unable to grow`).
- 원인: 출력 버퍼 zero-init이 **per-lane 65536-element 단일 `vector_store`** (= TILE_M×⌈TILE_N/vlane⌉, 예: 4096×16). **65536 = 2^16**이 LLVM RISC-V vector legalizer의 16-bit 카운터를 overflow.
- 수정: MLIR codegen에서 zero-init store가 threshold(**32768**)보다 크면 threshold-크기 chunk로 분할 (`emit_chunked_zero_init` helper, 11개 template 사이트). autotune tile 크기 제약은 안 검.
- **이 fix는 별도 브랜치에 있음**: `fix/autotune-large-tile-crash` (base=origin/develop, fork=github.com/wok1909/PyTorchSim, 1 commit). 검증 완료(GEMM 4096/8192 정상, 2048 regression 일치).
- **즉, 위 sweep 결과는 이 fix가 적용된 상태에서 측정됨.**

---

## 6. 현재 결과 (sim/real, 1.0이 완벽)

cycle은 `revalidate_*_results.txt` 참고. µs 변환·normalize·plot은 `tpu_validation/plot_vlen_compare.py` → `vlen_compare_bar.png`.

| op | N | vlen-256 | vlen-128 |
|---|---|---|---|
| GEMM | 512 / 1024 / 2048 / 4096 / 8192 | 0.67 / 0.97 / 1.05 / 0.99 / 0.79 | (동일) 0.67 / 0.97 / 1.05 / 0.99 / 0.79 |
| GEMV | 1024 / 2048 / 4096 / 8192 / 16384 | 0.63 / 0.89 / 1.10 / 1.25 / 1.10 | 0.79 / 1.06 / 1.32 / 1.25 / 1.10 |
| VADD | 1M / 4M / 16M / 67M | 0.80 / 1.02 / 1.07 / 1.08 | (동일) 0.80 / 1.02 / 1.07 / 1.08 |

**핵심 발견**:
1. **GEMM: vlen 256/128 사이클 완전 동일** — GEMM은 MXU-bound라 VPU vector length 무관. 중간 사이즈(1k~4k) ±5%, 양끝(512=0.67, 8192=0.79)이 outlier.
2. **VADD: vlen 256/128 완전 동일** — VADD는 DRAM-BW-bound(거대 배열 streaming)라 vector compute width 절반이 BW 병목을 안 바꿈.
3. **GEMV: 작은 N에서만 차이** (1024: 0.63→0.79 등). compute에 민감한 구간. 큰 N(8192·16384)은 BW-bound로 수렴해 256/128 동일.
4. **결론**: option-1(vlen 절반)은 이 3개 primitive op에서 거의 드러나지 않음 — 전부 MXU 또는 HBM-BW 병목이라. 의미 있는 차이는 GEMV 소형뿐.

---

## 7. Open questions / 다음 단계 후보
- VADD가 vlen 256/128 **비트 단위 동일** — `vpu_vector_length_bits`가 BW-bound 구간에서 정말 무효인지, 혹은 VADD codegen에 반영이 안 되는지 확인 필요. compute-bound microbenchmark로 vlen 효과를 분리해서 봐야 함.
- GEMV 4096 gap(1.10~1.32) 원인 (tiling / effective BW).
- GEMM 512(0.67)·8192(0.79) outlier — 작은 건 launch overhead 미모델링, 큰 건 effective BW 추정 차이로 추정.
- Phase-2 (attention prefill/decode)는 별도 config(vlen-256, `tpuv6e.yml`)로 측정된 옛 값 — vlen-128로 일관되게 재측정할지.
- calibration knob (clock 3502, VMEM 512KB/lane) 타당성 재검토.

## 파일 인덱스
- config: `configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml` (vlen128) / `..._tpuv6e.yml` (vlen256)
- runner: `tpu_validation/sim_bench.py`
- sweep: `tpu_validation/revalidate_opt1.sh` / `revalidate_base256.sh`
- raw 결과: `tpu_validation/revalidate_opt1_results.txt` / `revalidate_base256_results.txt`
- plot: `tpu_validation/plot_vlen_compare.py` → `vlen_compare_bar.png` (vlen 비교) / `plot_v6e_validation.py` → `v6e_validation_plots.png` (sim vs real)
- real 데이터: `tpu_validation/real_v6e/`
- GEMM fix: 브랜치 `fix/autotune-large-tile-crash`
