#!/bin/bash
# Post-fix (Fix1+Fix2) re-validation: GEMM/GEMV/VADD, vlen256 vs vlen128.
# Full cache clear (inductor + outputs) before EVERY run -> no stale/collision
# (Fix3 cache-key not done, so identical-MLIR kernels would otherwise reuse).
# fp16, functional OFF, CMEM=0, ABSOLUTE TOGSIM_CONFIG (relative -> silent deadlock).
set -u
cd /workspace/PyTorchSim
export TORCHSIM_DIR=/workspace/PyTorchSim
unset SRAM_BUFFER_PLAN_PATH
C256=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
C128=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml
OUT=/workspace/PyTorchSim/tpu_validation/_postfix_sweep_results.txt
: > "$OUT"

run() {  # op size
  local OP=$1 N=$2 c256 c128
  for tag in 256 128; do
    rm -rf /tmp/torchinductor* /workspace/PyTorchSim/outputs
    if [ "$tag" = 256 ]; then export TOGSIM_CONFIG=$C256; else export TOGSIM_CONFIG=$C128; fi
    L=/tmp/_pf_${OP}_${N}_${tag}.log
    python3 tpu_validation/sim_bench.py --op "$OP" --size "$N" --dtype fp16 > "$L" 2>&1
    local C=$(grep -oE "Total execution cycles: [0-9]+" "$L" | tail -1 | grep -oE "[0-9]+")
    if [ "$tag" = 256 ]; then c256=${C:-FAIL}; else c128=${C:-FAIL}; fi
  done
  local ratio="NA"
  if [[ "$c256" =~ ^[0-9]+$ && "$c128" =~ ^[0-9]+$ ]]; then
    ratio=$(awk "BEGIN{printf \"%.3f\", $c128/$c256}")
  fi
  printf "%-5s N=%-9s vlen256=%-10s vlen128=%-10s ratio(128/256)=%s\n" "$OP" "$N" "$c256" "$c128" "$ratio" | tee -a "$OUT"
}

echo "=== GEMM ===" | tee -a "$OUT"
for N in 512 1024 2048; do run gemm $N; done
echo "=== GEMV ===" | tee -a "$OUT"
for N in 1024 2048 4096; do run gemv $N; done
echo "=== VADD ===" | tee -a "$OUT"
for N in 1048576 4194304; do run vadd $N; done
echo "SWEEP_DONE" | tee -a "$OUT"
