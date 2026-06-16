#!/bin/bash
# Llama 3.1 8B single decoder layer sweep on TPU v6e (run ON the TPU VM).
# prefill + decode, batch x seq grid, fp16. Large B x large S skipped (OOM).
set -u
cd ~
DT=bf16
# batch cap per seq (FFN intermediate B*S*14336*2*2 must fit ~16GB)
declare -A CAP=( [512]=128 [1024]=128 [2048]=128 [4096]=64 [8192]=32 )

run() {
  local mode=$1 S=$2 B=$3
  echo "=== $mode S=$S B=$B ==="
  python3 llama_layer_bench.py --mode "$mode" --seq "$S" --batch "$B" --dtype $DT 2>&1 \
    | grep -E "RESULT|Error|Traceback|RESOURCE_EXHAUSTED|XlaRuntimeError" | tail -3
}

for S in 512 1024 2048 4096 8192; do
  for B in 1 2 4 8 16 32 64 128; do
    [ "$B" -gt "${CAP[$S]}" ] && continue
    run prefill $S $B
    run decode  $S $B
  done
done
echo "ALL_LLAMA_DONE"
