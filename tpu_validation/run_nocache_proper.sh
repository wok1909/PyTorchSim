#!/bin/bash
# PROPER no-L2 sweep: l2d_type=nocache + AUTOTUNE mapping (consistent), HBM3 1638 GB/s.
# Clears ALL PyTorchSim caches first. vadd/gemm via topk=4; gemv via topk=1 (16384 tractable).
set -u
BASE=/workspace/PyTorchSim
FREQ=3502000000
PLAN=$BASE/tpu_validation/gemm_plan_v6e.py
CFG_T4=$BASE/configs/_cmp_nocache.yml        # autotune topk4 + nocache + HBM3
CFG_GV=$BASE/configs/_v6e_nocache_at.yml     # autotune topk1 + nocache (gemv, tractable)

echo "############ CLEAR ALL PYTORCHSIM CACHES ############"
rm -rf "$BASE/outputs" "$BASE/togsim_results" /tmp/torchinductor* /tmp/np_* \
       /tmp/cmp_* /tmp/cmp2_* /tmp/nc_* /tmp/fix_* /tmp/fix2_* /tmp/devf_* /tmp/dev2_* \
       /tmp/heur_* /tmp/at_* /tmp/min_* /tmp/final_* /tmp/tiny_* /tmp/smoke_* 2>/dev/null
echo "cleared. config l2d_type:"; grep -h l2d_type $CFG_T4 $CFG_GV
( cd $BASE/TOGSim/build/bin && timeout 15 ./Simulator --config $CFG_T4 --models_list /dev/null 2>&1 | grep -oE "Total bandwidth [0-9.]+ GB/s" | head -1 )
echo

emit() { local lbl=$1 log=$2
  local cyc=$(grep -oE "Total execution cycles: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
  local done=$(grep -cE "SIMBENCH_DONE|VMATH_DONE" "$log")
  local us="NA"; [ -n "${cyc:-}" ] && us=$(python3 -c "print(f'{$cyc/$FREQ*1e6:.1f}')")
  printf "RESULT %-16s done=%s cyc=%-10s sim_us=%s\n" "$lbl" "${done:-0}" "${cyc:-NA}" "$us"
}
run() { local op=$1 n=$2 cfg=$3 plan=${4:-}
  local r=/tmp/np_${op}_${n}; rm -rf "$r"; mkdir -p "$r/inductor"
  export TORCHSIM_DUMP_PATH="$r" TORCHINDUCTOR_CACHE_DIR="$r/inductor" TORCHSIM_DIR=$BASE TOGSIM_CONFIG=$cfg
  if [ -n "$plan" ]; then export SRAM_BUFFER_PLAN_PATH=$plan; else unset SRAM_BUFFER_PLAN_PATH; fi
  timeout 18000 python3 $BASE/tpu_validation/sim_bench.py --op $op --size $n --dtype fp16 > "$r/out.log" 2>&1
  emit "$op N=$n" "$r/out.log"
}

echo "===== VADD (nocache, autotune topk4) ====="
for n in 1048576 4194304 16777216 67108864; do run vadd $n $CFG_T4; done
echo
echo "===== GEMM (nocache, autotune topk4) ====="
for n in 512 1024 2048 4096 8192; do run gemm $n $CFG_T4 $PLAN; done
echo
echo "===== GEMV (nocache, autotune topk1) ====="
for n in 1024 2048 4096 8192 16384; do run gemv $n $CFG_GV; done
echo "ALL_NP_DONE"
