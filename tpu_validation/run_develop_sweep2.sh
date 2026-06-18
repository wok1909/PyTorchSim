#!/bin/bash
# develop env sweep: clear ALL caches, then vadd / gemv / gemm / vector-throughput(poly)
# at vlen256 (tpuv6e.yml) and vlen128 (opt1.yml). Fresh isolated cache dir per run too.
set -u
BASE=/workspace/PyTorchSim
FREQ=3502000000
PLAN=$BASE/tpu_validation/gemm_plan_v6e.py
declare -A CFG=( [v256]=systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
                 [v128]=systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml )
GEMM_N=2048; GEMV_N=4096; VADD_N=4194304; POLY_N=65536; POLY_K=100

echo "############ CLEARING CACHES ############"
rm -rf "$BASE/outputs" "$BASE/togsim_results" /tmp/torchinductor* /tmp/dev_* /tmp/smoke_* 2>/dev/null
echo "cleared: outputs/, togsim_results/, /tmp/torchinductor*, /tmp/dev_*, /tmp/smoke_*"
echo

emit() {  # $1=label $2=logfile $3=flop(optional)
  local lbl=$1 log=$2 flop=${3:-}
  local cyc=$(grep -oE "Total execution cycles: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
  local vut=$(grep -oE "Vector unit utilization\(%\) [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+")
  local sys=$(grep -oE "Systolic array \[0\] utilization\(%\) [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
  local bw=$(grep -oE "Total bandwidth [0-9.]+ GB/s" "$log" | tail -1 | grep -oE "[0-9.]+")
  local done=$(grep -cE "SIMBENCH_DONE|VMATH_DONE" "$log")
  local extra=""
  if [ -n "$flop" ] && [ -n "${cyc:-}" ]; then
    extra=$(python3 -c "print(f'  TFLOP/s={$flop/($cyc/$FREQ)/1e12:.3f}')")
  fi
  printf "%-26s done=%s cyc=%-9s vutil=%-6s sysutil=%-5s peakBW=%-6s%s\n" \
    "$lbl" "${done:-0}" "${cyc:-NA}" "${vut:-NA}" "${sys:-NA}" "${bw:-NA}" "$extra"
}

for vl in v256 v128; do
  export TOGSIM_CONFIG=$BASE/configs/${CFG[$vl]}
  echo "############ vlen=$vl (${CFG[$vl]}) ############"
  for spec in "gemm $GEMM_N" "gemv $GEMV_N" "vadd $VADD_N"; do
    set -- $spec; op=$1; n=$2
    run=/tmp/dev2_${op}_${vl}; rm -rf "$run"; mkdir -p "$run/inductor"
    export TORCHSIM_DUMP_PATH="$run" TORCHINDUCTOR_CACHE_DIR="$run/inductor" TORCHSIM_DIR=$BASE
    [ "$op" = "gemm" ] && export SRAM_BUFFER_PLAN_PATH=$PLAN || unset SRAM_BUFFER_PLAN_PATH
    timeout 1800 python3 $BASE/tpu_validation/sim_bench.py --op $op --size $n --dtype fp16 > "$run/out.log" 2>&1
    emit "$op N=$n [$vl]" "$run/out.log"
  done
  # vector throughput: SPAD-resident long FMA chain (poly), low HBM -> pure VPU flops
  run=/tmp/dev2_poly_${vl}; rm -rf "$run"; mkdir -p "$run/inductor"
  export TORCHSIM_DUMP_PATH="$run" TORCHINDUCTOR_CACHE_DIR="$run/inductor" TORCHSIM_DIR=$BASE
  unset SRAM_BUFFER_PLAN_PATH
  timeout 1800 python3 $BASE/tpu_validation/vmath_bench.py --size $POLY_N --iters $POLY_K --dtype fp16 --kind poly > "$run/out.log" 2>&1
  emit "vthroughput N=$POLY_N K=$POLY_K [$vl]" "$run/out.log" $((2*POLY_N*POLY_K))
  echo
done
echo "ALL_DEV2_DONE"
