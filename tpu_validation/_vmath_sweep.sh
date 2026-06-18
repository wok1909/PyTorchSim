#!/bin/bash
# VPU-compute-bound sweep: fixed small SRAM-resident size, vary K (elementwise
# op count), both vlen configs. If vlen is modeled on the VPU path, vlen256
# should be ~2x faster than vlen128 and the gap should grow with K.
set -u
cd /workspace/PyTorchSim
export TORCHSIM_DIR=/workspace/PyTorchSim
unset SRAM_BUFFER_PLAN_PATH

SIZE=65536
KS="25 50 100 200"
DT=fp32

run_config() {  # tag config
  local TAG=$1 CFG=$2
  export TOGSIM_CONFIG=$CFG
  local OUT=tpu_validation/_vmath_sweep_${TAG}.txt
  : > "$OUT"
  for K in $KS; do
    rm -rf /tmp/torchinductor*
    local L=tpu_validation/_vmath_${TAG}_k${K}.log
    python3 tpu_validation/vmath_bench.py --size $SIZE --iters $K --dtype $DT > "$L" 2>&1
    local CYC
    CYC=$(grep -oE "Total execution cycles: [0-9]+" "$L" | tail -1 | grep -oE "[0-9]+")
    echo "vmath size=$SIZE K=$K cycles=${CYC:-FAIL}" >> "$OUT"
    echo "[$TAG] K=$K -> cycles=${CYC:-FAIL}"
  done
}

run_config vlen256 /workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
run_config vlen128 /workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml
echo "ALL_DONE"
