#!/bin/bash
# VADD-only reproduction of HANDOFF §6 (vlen 256 vs 128).
# CMEM=0 (no SRAM_BUFFER_PLAN), fp16, functional OFF (config has functional_mode:0).
# Clears inductor cache before EVERY run so vlen256->vlen128 cannot share a stale
# FX artifact -> any equality observed is real, not cache contamination.
set -u
cd /workspace/PyTorchSim
export TORCHSIM_DIR=/workspace/PyTorchSim
unset SRAM_BUFFER_PLAN_PATH

SIZES="1048576 4194304 16777216 67108864"

run_config() {  # tag config
  local TAG=$1 CFG=$2
  export TOGSIM_CONFIG=$CFG
  local OUT=tpu_validation/_vadd_repro_${TAG}_results.txt
  : > "$OUT"
  echo "### config=$CFG (vpu_vector_length_bits=$(grep -oE 'vpu_vector_length_bits: [0-9]+' "$CFG" | grep -oE '[0-9]+'))" >> "$OUT"
  for N in $SIZES; do
    rm -rf /tmp/torchinductor*
    local L=tpu_validation/_vadd_repro_${TAG}_${N}.log
    python3 tpu_validation/sim_bench.py --op vadd --size "$N" --dtype fp16 > "$L" 2>&1
    local CYC
    CYC=$(grep -oE "Total execution cycles: [0-9]+" "$L" | tail -1 | grep -oE "[0-9]+")
    echo "vadd N=$N cycles=${CYC:-FAIL}" >> "$OUT"
    echo "[$TAG] vadd N=$N -> cycles=${CYC:-FAIL}"
  done
  echo "DONE_$TAG"
}

run_config vlen256 /workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
run_config vlen128 /workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml
echo "ALL_DONE"
