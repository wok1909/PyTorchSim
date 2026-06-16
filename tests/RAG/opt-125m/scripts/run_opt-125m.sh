#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAG_DIR="$(cd "$MODEL_DIR/.." && pwd)"
TORCHSIM_DIR="${TORCHSIM_DIR:-/workspace/PyTorchSim}"
CONFIG_PATH="${RAG_DIR}/configs/hw_configs/sa1_128x128_upu.yml"

stage="both"  # prefill | decode | both
ops_csv="qkv,attn,oproj,ffn,lmhead,oproj_ffn_qkv,oproj_ffn_lmhead"
batch_range="1:1:1"
seq_range="128:128:1"
dtype="fp16"
repeats=3
warmup=1
query_len=1
output_file=""
results_root="${RAG_DIR}/results"

usage() {
  cat <<'EOF'
Usage:
  run_opt-125m.sh [options]

Options:
  --stage <prefill|decode|both>          Stage selection (default: both)
  --ops <csv>                            Ops to run (default: all)
                                         valid: qkv,attn,oproj,ffn,lmhead,oproj_ffn_qkv,oproj_ffn_lmhead
  --batch-range <start:end:step>         Batch sweep (default: 1:1:1)
  --seq-range <start:end:step>           Seq sweep (default: 128:128:1)
                                         decode+attn에서는 context_len으로 사용
  --query-len <int>                      Query length for decode attn (default: 1)
  --dtype <fp16|bf16|fp32>               Dtype (default: fp16)
  --repeats <int>                        Repeats per point (default: 3)
  --warmup <int>                         Warmups per point (default: 1)
  --output <path>                        Save all stdout to file (optional)
  --results-root <path>                  Result root dir (default: /workspace/PyTorchSim/tests/RAG/results)
  -h, --help                             Show help
EOF
}

parse_range() {
  local range="$1"
  local name="$2"
  IFS=':' read -r start end step <<< "$range"
  if [[ -z "${start:-}" || -z "${end:-}" || -z "${step:-}" ]]; then
    echo "[ERROR] Invalid ${name} range '${range}'. Expected start:end:step" >&2
    exit 1
  fi
  if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$step" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ${name} range must be non-negative integers: '${range}'" >&2
    exit 1
  fi
  if [[ "$step" -eq 0 ]]; then
    echo "[ERROR] ${name} step cannot be 0" >&2
    exit 1
  fi
  echo "$start" "$end" "$step"
}

is_valid_op() {
  case "$1" in
    qkv|attn|oproj|ffn|lmhead|oproj_ffn_qkv|oproj_ffn_lmhead) return 0 ;;
    *) return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --output)
      output_file="$2"; shift 2 ;;
    --results-root)
      results_root="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ "$stage" != "prefill" && "$stage" != "decode" && "$stage" != "both" ]]; then
  echo "[ERROR] stage must be prefill|decode|both" >&2
  exit 1
fi

if [[ "$dtype" != "fp16" && "$dtype" != "bf16" && "$dtype" != "fp32" ]]; then
  echo "[ERROR] dtype must be fp16|bf16|fp32" >&2
  exit 1
fi

if ! [[ "$query_len" =~ ^[0-9]+$ ]] || [[ "$query_len" -eq 0 ]]; then
  echo "[ERROR] query-len must be a positive integer" >&2
  exit 1
fi

read -r b_start b_end b_step <<< "$(parse_range "$batch_range" "batch")"
read -r s_start s_end s_step <<< "$(parse_range "$seq_range" "seq")"

IFS=',' read -r -a ops_arr <<< "$ops_csv"
for op in "${ops_arr[@]}"; do
  if ! is_valid_op "$op"; then
    echo "[ERROR] invalid op: $op" >&2
    exit 1
  fi
done

if [[ -n "$output_file" ]]; then
  mkdir -p "$(dirname "$output_file")"
  : > "$output_file"
fi
mkdir -p "$results_root"

export TORCHSIM_DIR
export TOGSIM_CONFIG="$CONFIG_PATH"

run_cmd() {
  if [[ -n "$output_file" ]]; then
    "$@" | tee -a "$output_file"
  else
    "$@"
  fi
}

run_point() {
  local stg="$1"
  local op="$2"
  local b="$3"
  local s="$4"
  local script="${MODEL_DIR}/${stg}/${op}.py"
  local effective_seq="$s"
  local context_len=""
  if [[ "$stg" == "decode" && "$op" == "attn" ]]; then
    context_len="$s"
  fi
  local ts
  ts="$(date +%Y%m%d_%H%M%S_%N)"
  local shape_tag="b${b}_s${effective_seq}"
  if [[ -n "$context_len" ]]; then
    shape_tag="b${b}_ctx${context_len}_q${query_len}"
  fi
  local run_dir="${results_root}/opt-125m/${stg}/${op}/${shape_tag}/${ts}"
  local run_logs_dir="${run_dir}/logs"
  local run_dump_dir="${run_dir}/dump"
  local run_stdout="${run_dir}/stdout.log"

  if [[ ! -f "$script" ]]; then
    echo "[WARN] skip (script not found): $script" >&2
    return 0
  fi

  mkdir -p "$run_logs_dir" "$run_dump_dir"

  cat > "${run_dir}/run_meta.txt" <<EOF
model=opt-125m
stage=${stg}
op=${op}
batch_size=${b}
seq_len=${effective_seq}
context_len=${context_len}
dtype=${dtype}
repeats=${repeats}
warmup=${warmup}
query_len=${query_len}
config_path=${CONFIG_PATH}
timestamp=${ts}
TORCHSIM_LOG_PATH=${run_logs_dir}
TORCHSIM_DUMP_PATH=${run_dump_dir}
EOF

  if [[ -n "$context_len" ]]; then
    echo "[INFO] model=opt-125m stage=${stg} op=${op} batch=${b} context_len=${context_len} query_len=${query_len} dtype=${dtype}"
  else
    echo "[INFO] model=opt-125m stage=${stg} op=${op} batch=${b} seq=${effective_seq} dtype=${dtype}"
  fi
  echo "[INFO] result_dir=${run_dir}"

  if [[ "$stg" == "decode" && "$op" == "attn" ]]; then
    run_cmd env TORCHSIM_LOG_PATH="$run_logs_dir" TORCHSIM_DUMP_PATH="$run_dump_dir" \
      python "$script" \
      --batch-size "$b" \
      --context-len "$context_len" \
      --query-len "$query_len" \
      --dtype "$dtype" \
      --repeats "$repeats" \
      --warmup "$warmup" \
      --config-path "$CONFIG_PATH" | tee "$run_stdout"
  else
    run_cmd env TORCHSIM_LOG_PATH="$run_logs_dir" TORCHSIM_DUMP_PATH="$run_dump_dir" \
      python "$script" \
      --batch-size "$b" \
      --seq-len "$effective_seq" \
      --dtype "$dtype" \
      --repeats "$repeats" \
      --warmup "$warmup" \
      --config-path "$CONFIG_PATH" | tee "$run_stdout"
  fi
}

echo "[INFO] model=opt-125m config=${CONFIG_PATH}"
echo "[INFO] stage=${stage} ops=${ops_csv} batch=${batch_range} seq=${seq_range} dtype=${dtype}"

for ((b=b_start; b<=b_end; b+=b_step)); do
  for ((s=s_start; s<=s_end; s+=s_step)); do
    if [[ "$stage" == "prefill" || "$stage" == "both" ]]; then
      for op in "${ops_arr[@]}"; do
        run_point "prefill" "$op" "$b" "$s"
      done
    fi

    if [[ "$stage" == "decode" || "$stage" == "both" ]]; then
      for op in "${ops_arr[@]}"; do
        run_point "decode" "$op" "$b" "$s"
      done
    fi
  done
done

echo "[INFO] model=opt-125m done"
if [[ -n "$output_file" ]]; then
  echo "[INFO] output saved to: $output_file"
fi
