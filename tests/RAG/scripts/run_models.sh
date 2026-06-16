#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

models_csv="opt-125m"
stage="both"
ops_csv="qkv,attn,oproj,ffn,lmhead,oproj_ffn_qkv,oproj_ffn_lmhead"
batch_range="1:1:1"
seq_range="128:128:1"
dtype="fp16"
repeats=3
warmup=1
query_len=1
output_dir=""

usage() {
  cat <<'EOF'
Usage:
  run_models.sh [options]

Options:
  --models <csv>                         Models to run (default: opt-125m)
                                         currently supported: opt-125m
  --stage <prefill|decode|both>          Stage selection (default: both)
  --ops <csv>                            Ops selection (default: all)
  --batch-range <start:end:step>         Batch sweep (default: 1:1:1)
  --seq-range <start:end:step>           Seq sweep (default: 128:128:1)
  --query-len <int>                      Query length for decode attn (default: 1)
  --dtype <fp16|bf16|fp32>               Dtype (default: fp16)
  --repeats <int>                        Repeats per point (default: 3)
  --warmup <int>                         Warmups per point (default: 1)
  --output-dir <path>                    Save per-model logs under this dir (optional)
  -h, --help                             Show help

Examples:
  ./run_models.sh --models opt-125m --stage prefill --ops qkv --batch-range 1:8:1 --seq-range 128:1024:128
  ./run_models.sh --models opt-125m --stage both --batch-range 1:4:1 --seq-range 128:512:128 --output-dir /tmp/rag_logs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      models_csv="$2"; shift 2 ;;
    --stage)
      stage="$2"; shift 2 ;;
    --ops)
      ops_csv="$2"; shift 2 ;;
    --batch-range)
      batch_range="$2"; shift 2 ;;
    --seq-range)
      seq_range="$2"; shift 2 ;;
    --query-len)
      query_len="$2"; shift 2 ;;
    --dtype)
      dtype="$2"; shift 2 ;;
    --repeats)
      repeats="$2"; shift 2 ;;
    --warmup)
      warmup="$2"; shift 2 ;;
    --output-dir)
      output_dir="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

IFS=',' read -r -a models_arr <<< "$models_csv"

if [[ -n "$output_dir" ]]; then
  mkdir -p "$output_dir"
fi

for model in "${models_arr[@]}"; do
  case "$model" in
    opt-125m)
      runner="${RAG_DIR}/opt-125m/scripts/run_opt-125m.sh"
      ;;
    *)
      echo "[WARN] Unsupported model '$model'. Skipping." >&2
      continue
      ;;
  esac

  if [[ ! -x "$runner" ]]; then
    echo "[ERROR] Runner not executable or missing: $runner" >&2
    exit 1
  fi

  echo "[INFO] Running model: $model"

  if [[ -n "$output_dir" ]]; then
    out_file="${output_dir}/${model}.log"
  else
    out_file=""
  fi

  cmd=(
    "$runner"
    --stage "$stage"
    --ops "$ops_csv"
    --batch-range "$batch_range"
    --seq-range "$seq_range"
    --query-len "$query_len"
    --dtype "$dtype"
    --repeats "$repeats"
    --warmup "$warmup"
  )

  if [[ -n "$out_file" ]]; then
    cmd+=(--output "$out_file")
  fi

  "${cmd[@]}"
done

echo "[INFO] run_models.sh done"
