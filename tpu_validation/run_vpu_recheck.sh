#!/bin/bash
# Re-run sim VPU-peak microbench with FRESH cache per (kind,vlen) to verify the
# corrected conclusion: poly (pure fp16) ~ real 4.56; fma_relu slow (clamp f32 widen).
set -u
BASE=/workspace/PyTorchSim
SIZE=65536
ITERS=100
FREQ=3502000000   # core_freq Hz
declare -A CFG=( [v256]=systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
                 [v128]=systolic_ws_256x256_c1_simple_noc_tpuv6e_opt1.yml )

echo "size=$SIZE iters=$ITERS freq=$FREQ  flop=$(python3 -c "print(2*$SIZE*$ITERS)")"
for kind in poly fma_relu; do
  for vl in v256 v128; do
    run=/tmp/vpu_recheck_${kind}_${vl}
    rm -rf "$run"; mkdir -p "$run"
    export TORCHSIM_DUMP_PATH="$run"
    export TORCHINDUCTOR_CACHE_DIR="$run/inductor"
    export TOGSIM_CONFIG="$BASE/configs/${CFG[$vl]}"
    export TORCHSIM_DIR="$BASE"
    log="$run/out.log"
    echo "=== RUN kind=$kind vlen=$vl cfg=${CFG[$vl]} ==="
    python3 "$BASE/tpu_validation/vmath_bench.py" --size $SIZE --iters $ITERS --dtype fp16 --kind $kind > "$log" 2>&1
    cyc=$(grep -oE "Total cycle: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    vcyc=$(grep -oE "Vector active cycle [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    vutil=$(grep -oE "Vector Unit Utilization\(%\) [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
    done=$(grep -c VMATH_DONE "$log")
    echo "RESULT kind=$kind vlen=$vl done=$done total_cycle=${cyc:-NA} vector_cycle=${vcyc:-NA} vutil=${vutil:-NA}"
    if [ -n "${cyc:-}" ]; then
      python3 -c "c=$cyc; f=2*$SIZE*$ITERS; t=c/$FREQ; print(f'   -> time={t*1e6:.2f}us  TFLOP/s={f/t/1e12:.3f}')"
    fi
  done
done
echo "ALL_DONE"
