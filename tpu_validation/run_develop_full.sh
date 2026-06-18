#!/bin/bash
# develop full sweep @ HBM3 1638 GB/s (vlen256), real-matching sizes, for real-vs-sim table.
set -u
BASE=/workspace/PyTorchSim
FREQ=3502000000
PLAN=$BASE/tpu_validation/gemm_plan_v6e.py
CFG=$BASE/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml   # vlen256, HBM3 32ch = 1638 GB/s
export TOGSIM_CONFIG=$CFG

echo "############ CLEARING CACHES ############"
rm -rf "$BASE/outputs" "$BASE/togsim_results" /tmp/torchinductor* /tmp/dev2_* /tmp/devf_* 2>/dev/null
echo "cleared. config=$(basename $CFG)  BW check:"
( cd $BASE/TOGSim/build/bin && timeout 15 ./Simulator --config $CFG --models_list /dev/null 2>&1 | grep -oE "Total bandwidth [0-9.]+ GB/s" | head -1 )
echo

emit() {  # $1=label $2=logfile $3=flop(optional)
  local lbl=$1 log=$2 flop=${3:-}
  local cyc=$(grep -oE "Total execution cycles: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
  local done=$(grep -cE "SIMBENCH_DONE|VMATH_DONE" "$log")
  local us="NA" extra=""
  if [ -n "${cyc:-}" ]; then us=$(python3 -c "print(f'{$cyc/$FREQ*1e6:.2f}')"); fi
  if [ -n "$flop" ] && [ -n "${cyc:-}" ]; then extra=$(python3 -c "print(f'  TFLOP/s={$flop/($cyc/$FREQ)/1e12:.3f}')"); fi
  printf "RESULT %-22s done=%s cyc=%-10s sim_us=%-9s%s\n" "$lbl" "${done:-0}" "${cyc:-NA}" "$us" "$extra"
}

run_op() { # op size
  local op=$1 n=$2
  local run=/tmp/devf_${op}_${n}; rm -rf "$run"; mkdir -p "$run/inductor"
  export TORCHSIM_DUMP_PATH="$run" TORCHINDUCTOR_CACHE_DIR="$run/inductor" TORCHSIM_DIR=$BASE
  [ "$op" = "gemm" ] && export SRAM_BUFFER_PLAN_PATH=$PLAN || unset SRAM_BUFFER_PLAN_PATH
  timeout 7000 python3 $BASE/tpu_validation/sim_bench.py --op $op --size $n --dtype fp16 > "$run/out.log" 2>&1
  emit "$op N=$n" "$run/out.log"
}

for n in 512 1024 2048 4096 8192;            do run_op gemm $n; done
echo
for n in 1024 2048 4096 8192 16384;          do run_op gemv $n; done
echo
for n in 1048576 4194304 16777216 67108864;  do run_op vadd $n; done
echo
# vector throughput (SPAD-resident FMA chain)
run=/tmp/devf_poly; rm -rf "$run"; mkdir -p "$run/inductor"
export TORCHSIM_DUMP_PATH="$run" TORCHINDUCTOR_CACHE_DIR="$run/inductor" TORCHSIM_DIR=$BASE
unset SRAM_BUFFER_PLAN_PATH
timeout 3000 python3 $BASE/tpu_validation/vmath_bench.py --size 65536 --iters 100 --dtype fp16 --kind poly > "$run/out.log" 2>&1
emit "vthroughput N=65536 K=100" "$run/out.log" $((2*65536*100))
echo "ALL_DEVF_DONE"
