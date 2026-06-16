#!/bin/bash
# Re-validate GEMM/GEMV/VADD with BASE config (vlen 256, full vector throughput)
# and CMEM=0 (no SRAM_BUFFER_PLAN). fp16 sim, functional OFF. Reports cycles.
# Mirrors revalidate_opt1.sh (vlen 128) for a vlen-256 vs vlen-128 comparison.
cd /workspace/PyTorchSim
export TORCHSIM_DIR=/workspace/PyTorchSim
export TOGSIM_CONFIG=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
unset SRAM_BUFFER_PLAN_PATH   # CMEM = 0 (stream everything)
OUT=tpu_validation/revalidate_base256_results.txt
: > $OUT
run() {  # op size
  L=tpu_validation/_base256_${1}_${2}.log
  python3 tpu_validation/sim_bench.py --op $1 --size $2 --dtype fp16 > $L 2>&1
  CYC=$(grep -oE "Total execution cycles: [0-9]+" $L | tail -1 | grep -oE "[0-9]+")
  echo "$1 N=$2 cycles=${CYC:-FAIL}" >> $OUT
}
echo "=== GEMM ===" >> $OUT
for N in 512 1024 2048 4096 8192; do run gemm $N; done
echo "=== GEMV ===" >> $OUT
for N in 1024 2048 4096 8192 16384; do run gemv $N; done
echo "=== VADD ===" >> $OUT
for N in 1048576 4194304 16777216 67108864; do run vadd $N; done
echo "BASE256_DONE" >> $OUT
