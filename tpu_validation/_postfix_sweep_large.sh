#!/bin/bash
# Post-fix large-size sweep, FULLY ISOLATED (own dump path + inductor cache) so it
# cannot collide with other jobs sharing /workspace/PyTorchSim/outputs.
set -u
cd /workspace/PyTorchSim
export TORCHSIM_DIR=/workspace/PyTorchSim
unset SRAM_BUFFER_PLAN_PATH
export TORCHSIM_DUMP_PATH=/tmp/pf_large_out      # isolate outputs/
C256=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
C128=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml
OUT=/workspace/PyTorchSim/tpu_validation/_postfix_sweep_large_results.txt
: > "$OUT"

run() {  # op size
  local OP=$1 N=$2 c256 c128
  for tag in 256 128; do
    export TORCHINDUCTOR_CACHE_DIR=/tmp/pf_ti_$tag
    rm -rf /tmp/pf_ti_$tag /tmp/pf_large_out
    if [ "$tag" = 256 ]; then export TOGSIM_CONFIG=$C256; else export TOGSIM_CONFIG=$C128; fi
    L=/tmp/_pfL_${OP}_${N}_${tag}.log
    python3 tpu_validation/sim_bench.py --op "$OP" --size "$N" --dtype fp16 > "$L" 2>&1
    local C=$(grep -oE "Total execution cycles: [0-9]+" "$L" | tail -1 | grep -oE "[0-9]+")
    if [ "$tag" = 256 ]; then c256=${C:-FAIL}; else c128=${C:-FAIL}; fi
  done
  local ratio="NA"
  [[ "$c256" =~ ^[0-9]+$ && "$c128" =~ ^[0-9]+$ ]] && ratio=$(awk "BEGIN{printf \"%.3f\", $c128/$c256}")
  printf "%-5s N=%-9s vlen256=%-10s vlen128=%-10s ratio(128/256)=%s\n" "$OP" "$N" "$c256" "$c128" "$ratio" | tee -a "$OUT"
}

echo "=== GEMM large ===" | tee -a "$OUT"
for N in 4096 8192; do run gemm $N; done
echo "=== GEMV large ===" | tee -a "$OUT"
for N in 8192 16384; do run gemv $N; done
echo "=== VADD large ===" | tee -a "$OUT"
for N in 16777216 67108864; do run vadd $N; done
echo "LARGE_DONE" | tee -a "$OUT"
