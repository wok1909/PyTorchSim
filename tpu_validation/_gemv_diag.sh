#!/bin/bash
# GEMV tile/micro-op diagnostic: 8192 vs 16384, both configs. Isolated dump/cache.
set -u
cd /workspace/PyTorchSim
export TORCHSIM_DIR=/workspace/PyTorchSim
unset SRAM_BUFFER_PLAN_PATH
C256=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
C128=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml
OUT=/workspace/PyTorchSim/tpu_validation/_gemv_diag_results.txt
: > "$OUT"

probe() {  # N tag cfg
  local N=$1 T=$2 CFG=$3
  export TORCHSIM_DUMP_PATH=/tmp/gv_${N}_${T} TORCHINDUCTOR_CACHE_DIR=/tmp/gvti_${N}_${T}
  rm -rf /tmp/gv_${N}_${T} /tmp/gvti_${N}_${T}
  export TOGSIM_CONFIG=$CFG
  python3 tpu_validation/sim_bench.py --op gemv --size $N --dtype fp16 > /tmp/gvp_${N}_${T}.log 2>&1
  local CYC W NI NO
  CYC=$(grep -oE "Total execution cycles: [0-9]+" /tmp/gvp_${N}_${T}.log | tail -1 | grep -oE "[0-9]+")
  W=$(grep -ohE "vector<[0-9]+x" /tmp/gv_${N}_${T}/outputs/*/*.mlir 2>/dev/null | sort | uniq -c | tr -s " " | tr "\n" "|")
  NI=$(grep -h "commitStats0.numInsts " /tmp/gv_${N}_${T}/outputs/*/m5out/stats.txt 2>/dev/null | awk "{s+=\$2} END{print s}")
  NO=$(grep -h "commitStats0.numOps " /tmp/gv_${N}_${T}/outputs/*/m5out/stats.txt 2>/dev/null | awk "{s+=\$2} END{print s}")
  echo "N=$N vlen$T: cyc=${CYC:-FAIL} numInsts=${NI:-NA} numOps=${NO:-NA} widths={$W}" | tee -a "$OUT"
}
probe 8192  256 $C256
probe 8192  128 $C128
probe 16384 256 $C256
probe 16384 128 $C128
echo DIAG_DONE | tee -a "$OUT"
