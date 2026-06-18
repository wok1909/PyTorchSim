#!/bin/bash
# Re-run on develop (+ our 4 fixes): vadd / gemm / gemv (one size each) + a high-reuse
# VPU-flops-only probe (poly). Each at vlen256 (tpuv6e.yml) and vlen128 (opt1.yml),
# with a FRESH cache dir per run (clears torchinductor + outputs, no hash collisions).
set -u
BASE=/workspace/PyTorchSim
FREQ=3502000000
PLAN=$BASE/tpu_validation/gemm_plan_v6e.py
declare -A CFG=( [v256]=systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
                 [v128]=systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml )

# op:size  (one representative size each); poly handled separately
GEMM_N=2048; GEMV_N=4096; VADD_N=4194304; POLY_N=65536; POLY_K=100

emit() {  # $1=label $2=logfile $3=flop(optional)
  local lbl=$1 log=$2 flop=${3:-}
  local cyc=$(grep -oE "Total execution cycles: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
  local vut=$(grep -oE "Vector unit utilization\(%\) [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+")
  local sys=$(grep -oE "Systolic array \[0\] utilization\(%\) [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
  local bw=$(grep -oE "DRAM BW [0-9.]+ GB/s" "$log" | tail -1 | grep -oE "[0-9.]+")
  local done=$(grep -cE "SIMBENCH_DONE|VMATH_DONE" "$log")
  local extra=""
  if [ -n "$flop" ] && [ -n "${cyc:-}" ]; then
    extra=$(python3 -c "print(f'  TFLOP/s={$flop/($cyc/$FREQ)/1e12:.3f}')")
  fi
  printf "%-22s done=%s cyc=%-9s vutil=%-6s sysutil=%-5s dramBW(last)=%-7s%s\n" \
    "$lbl" "${done:-0}" "${cyc:-NA}" "${vut:-NA}" "${sys:-NA}" "${bw:-NA}" "$extra"
}

for vl in v256 v128; do
  export TOGSIM_CONFIG=$BASE/configs/${CFG[$vl]}
  echo "############ vlen=$vl (${CFG[$vl]}) ############"

  # --- gemm / gemv / vadd via sim_bench.py (vadd last: largest) ---
  for spec in "gemm $GEMM_N" "gemv $GEMV_N" "vadd $VADD_N"; do
    set -- $spec; op=$1; n=$2
    run=/tmp/dev_${op}_${vl}; rm -rf "$run"; mkdir -p "$run/inductor"
    export TORCHSIM_DUMP_PATH="$run" TORCHINDUCTOR_CACHE_DIR="$run/inductor" TORCHSIM_DIR=$BASE
    [ "$op" = "gemm" ] && export SRAM_BUFFER_PLAN_PATH=$PLAN || unset SRAM_BUFFER_PLAN_PATH
    python3 $BASE/tpu_validation/sim_bench.py --op $op --size $n --dtype fp16 > "$run/out.log" 2>&1
    emit "$op N=$n [$vl]" "$run/out.log"
  done

  # --- high-reuse VPU-flops probe (poly: y=y*0.99+0.01 chain, low HBM) ---
  run=/tmp/dev_poly_${vl}; rm -rf "$run"; mkdir -p "$run/inductor"
  export TORCHSIM_DUMP_PATH="$run" TORCHINDUCTOR_CACHE_DIR="$run/inductor" TORCHSIM_DIR=$BASE
  unset SRAM_BUFFER_PLAN_PATH
  python3 $BASE/tpu_validation/vmath_bench.py --size $POLY_N --iters $POLY_K --dtype fp16 --kind poly > "$run/out.log" 2>&1
  emit "poly N=$POLY_N K=$POLY_K [$vl]" "$run/out.log" $((2*POLY_N*POLY_K))
  echo
done
echo "ALL_DEV_DONE"
