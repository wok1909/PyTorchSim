#!/bin/bash
# Functional OFF latency sweep on v6e config. Prefill (causal flash template) +
# decode (test_gqa_decode). Reports Total execution cycles per point.
cd /workspace/PyTorchSim
export TORCHSIM_DIR=/workspace/PyTorchSim
export TOGSIM_CONFIG=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml
OUT=tpu_validation/sweep_latency_results.txt
: > $OUT
echo "=== PREFILL (causal flash template, fp32, B=1, Hq5/Hkv1/D128) ===" >> $OUT
for S in 512 1024 2048; do
  L=tpu_validation/_sweep_prefill_s${S}.log
  python3 tpu_validation/step0_routing.py --case causal_mha --seq $S --dtype fp32 > $L 2>&1
  CYC=$(grep -oE "Total execution cycles: [0-9]+" $L | tail -1 | grep -oE "[0-9]+")
  echo "prefill S=$S cycles=${CYC:-FAIL}" >> $OUT
done
echo "=== DECODE (test_gqa_decode, fp16, B=1, LLAMA4_TP8, tile=512) ===" >> $OUT
for S in 512 1024 2048 4096 8192; do
  L=tpu_validation/_sweep_decode_s${S}.log
  python3 tests/test_gqa_decode.py --model LLAMA4_TP8 --context_length $S --tile_size 512 > $L 2>&1
  CYC=$(grep -oE "Total execution cycles: [0-9]+" $L | tail -1 | grep -oE "[0-9]+")
  echo "decode S=$S cycles=${CYC:-FAIL}" >> $OUT
done
echo "SWEEP_DONE" >> $OUT
