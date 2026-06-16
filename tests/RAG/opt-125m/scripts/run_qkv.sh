#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TORCHSIM_DIR="${TORCHSIM_DIR:-/workspace/PyTorchSim}"

CONFIG_PATH="${RAG_DIR}/configs/hw_configs/sa1_128x128_upu.yml"

STAGE="${1:-prefill}"   # prefill | decode
BATCH_SIZE="${2:-1}"
SEQ_LEN="${3:-128}"
DTYPE="${4:-fp16}"
REPEATS="${5:-3}"
WARMUP="${6:-1}"

if [[ "$STAGE" != "prefill" && "$STAGE" != "decode" ]]; then
  echo "[ERROR] stage must be 'prefill' or 'decode'" >&2
  exit 1
fi

if [[ "$STAGE" == "decode" && "$SEQ_LEN" == "128" ]]; then
  SEQ_LEN=1
fi

TARGET_SCRIPT="${RAG_DIR}/opt-125m/${STAGE}/qkv.py"
if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "[ERROR] target script not found: $TARGET_SCRIPT" >&2
  exit 1
fi

export TORCHSIM_DIR
export TOGSIM_CONFIG="$CONFIG_PATH"

echo "[INFO] model=opt-125m op=qkv stage=${STAGE} batch=${BATCH_SIZE} seq=${SEQ_LEN} dtype=${DTYPE}"
echo "[INFO] config=${CONFIG_PATH}"

python "$TARGET_SCRIPT" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --dtype "$DTYPE" \
  --repeats "$REPEATS" \
  --warmup "$WARMUP" \
  --config-path "$CONFIG_PATH"
