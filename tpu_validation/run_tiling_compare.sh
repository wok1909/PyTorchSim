#!/bin/bash
# Compare codegen mapping strategies on GEMM: heuristic vs autotune(topk4) vs autotune(topk16).
# Isolates the tiling variable (same spad 128MB, v4 plan, HBM2). Logs per (strategy,N).
set -u
export SRAM_BUFFER_PLAN_PATH=/workspace/PyTorchSim/tpu_validation/gemm_plan_v6e.py
OUT=/workspace/PyTorchSim/tpu_validation/sim_logs_tiling
mkdir -p "$OUT"
cd /workspace/PyTorchSim
CFG=/workspace/PyTorchSim/configs

declare -A CONFIGS=(
  [heuristic]=$CFG/v6e_heuristic.yml
  [auto4]=$CFG/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
  [auto16]=$CFG/v6e_auto16.yml
)

for strat in heuristic auto4 auto16; do
  export TOGSIM_CONFIG=${CONFIGS[$strat]}
  for N in 1024 2048 4096; do
    echo "=== RUN $strat gemm $N ==="
    python tpu_validation/sim_bench.py --op gemm --size $N --dtype fp16 \
      > "$OUT/${strat}_gemm_${N}.log" 2>&1
    grep -E "Total execution cycles|Systolic array \[0\] utilization\(%\) [0-9].*active_cycles [1-9]|DRAM BW [0-9].* GB/s \([0-9]|SIMBENCH_DONE|LLVM ERROR|Error" \
      "$OUT/${strat}_gemm_${N}.log" | tail -4
  done
done
echo "ALL_TILING_DONE"
