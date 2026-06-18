#!/bin/bash
# L2 BYPASS (l2d_type: nocache) sweep — vadd / gemm / gemv, HBM3 1638 GB/s.
# gemm/vadd: heuristic config; gemv: autotune topk=1 (heuristic gemv is buggy).
set -u
BASE=/workspace/PyTorchSim
FREQ=3502000000
PLAN=$BASE/tpu_validation/gemm_plan_v6e.py
CFG_HEUR=$BASE/configs/_v6e_nocache_heur.yml
CFG_AT=$BASE/configs/_v6e_nocache_at.yml

echo "############ CLEAR CACHE + L2=nocache ############"
rm -rf "$BASE/outputs" "$BASE/togsim_results" /tmp/torchinductor* /tmp/nc_* 2>/dev/null
echo "cleared. l2d_type check:"; grep -h l2d_type $CFG_HEUR $CFG_AT
echo

emit() { # $1=label $2=log
  local lbl=$1 log=$2
  local cyc=$(grep -oE "Total execution cycles: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
  local reads=$(grep -oE "[0-9]+ reads, [0-9]+ writes" "$log" | tail -32 | awk "{r+=\$1} END {print r}")
  local done=$(grep -cE "SIMBENCH_DONE|VMATH_DONE" "$log")
  local us="NA"; [ -n "${cyc:-}" ] && us=$(python3 -c "print(f'{$cyc/$FREQ*1e6:.1f}')")
  local mb=$(python3 -c "print(int(${reads:-0}*32/1e6))" 2>/dev/null)
  printf "NC %-16s done=%s cyc=%-9s sim_us=%-8s readMB=%s\n" "$lbl" "${done:-0}" "${cyc:-NA}" "$us" "$mb"
}

run() { # op size cfg [plan]
  local op=$1 n=$2 cfg=$3 plan=${4:-}
  local r=/tmp/nc_${op}_${n}; rm -rf "$r"; mkdir -p "$r/inductor"
  export TORCHSIM_DUMP_PATH="$r" TORCHINDUCTOR_CACHE_DIR="$r/inductor" TORCHSIM_DIR=$BASE TOGSIM_CONFIG=$cfg
  if [ -n "$plan" ]; then export SRAM_BUFFER_PLAN_PATH=$plan; else unset SRAM_BUFFER_PLAN_PATH; fi
  timeout 16000 python3 $BASE/tpu_validation/sim_bench.py --op $op --size $n --dtype fp16 > "$r/out.log" 2>&1
  emit "$op N=$n" "$r/out.log"
}

echo "===== VADD (heuristic, nocache) ====="
for n in 1048576 4194304 16777216 67108864; do run vadd $n $CFG_HEUR; done
echo
echo "===== GEMM (heuristic, nocache) ====="
for n in 512 1024 2048 4096 8192; do run gemm $n $CFG_HEUR $PLAN; done
echo
echo "===== GEMV (autotune topk=1, nocache) ====="
for n in 1024 2048 4096 8192 16384; do run gemv $n $CFG_AT; done
echo "ALL_NC_DONE"
