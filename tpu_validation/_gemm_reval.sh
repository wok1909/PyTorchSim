#!/bin/bash
# Fresh GEMM cycle re-measure vs real v6e. vlen256 (v6e base), fp16, functional OFF.
# Isolated dump/cache. 8192 run TWICE to check autotune reproducibility.
set -u
cd /workspace/PyTorchSim
export TORCHSIM_DIR=/workspace/PyTorchSim
unset SRAM_BUFFER_PLAN_PATH
CFG=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
OUT=/workspace/PyTorchSim/tpu_validation/_gemm_reval_results.txt
: > "$OUT"
run() {  # N tag
  export TORCHSIM_DUMP_PATH=/tmp/gm_${1}_${2} TORCHINDUCTOR_CACHE_DIR=/tmp/gmti_${1}_${2}
  rm -rf /tmp/gm_${1}_${2} /tmp/gmti_${1}_${2}
  export TOGSIM_CONFIG=$CFG
  python3 tpu_validation/sim_bench.py --op gemm --size $1 --dtype fp16 > /tmp/gmr_${1}_${2}.log 2>&1
  local C=$(grep -oE "Total execution cycles: [0-9]+" /tmp/gmr_${1}_${2}.log | tail -1 | grep -oE "[0-9]+")
  echo "gemm N=$1 ($2) cycles=${C:-FAIL}" | tee -a "$OUT"
}
for N in 512 1024 2048 4096 8192; do run $N a; done
run 8192 b   # reproducibility check
echo GEMM_REVAL_DONE | tee -a "$OUT"
