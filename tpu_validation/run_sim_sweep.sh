#!/bin/bash
# Run the full sim sweep (gemm/gemv/vadd) at the real-device sizes, fp16.
# One isolated process per (op,size) -> clean per-run stats. Logs to sim_logs/.
set -u
CONFIG=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
export TOGSIM_CONFIG=$CONFIG
export SRAM_BUFFER_PLAN_PATH=/workspace/PyTorchSim/tpu_validation/gemm_plan_v6e.py  # v4 CMEM policy
OUT=/workspace/PyTorchSim/tpu_validation/sim_logs
mkdir -p "$OUT"
cd /workspace/PyTorchSim

run() {
  local op=$1 n=$2
  echo "=== RUN $op $n ==="
  python tpu_validation/sim_bench.py --op "$op" --size "$n" --dtype fp16 \
    > "$OUT/${op}_fp16_${n}.log" 2>&1
  grep -E "Total execution cycles|Systolic array \[0\]|Systolic array \[1\]|Vector unit util|DRAM BW|SIMBENCH_DONE|LLVM ERROR|InductorError" \
    "$OUT/${op}_fp16_${n}.log" | tail -6
}

for n in 512 1024 2048 4096 8192;            do run gemm $n; done
for n in 1024 2048 4096 8192 16384;          do run gemv $n; done
for n in 1048576 4194304 16777216 67108864;  do run vadd $n; done
echo "ALL_SWEEP_DONE"
