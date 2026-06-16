#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXTERN_DIR="${ROOT_DIR}/TOGSim/extern"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
ALLOW_FALLBACK_PINNED="${ALLOW_FALLBACK_PINNED:-0}"

if [[ ! -d "${EXTERN_DIR}" ]]; then
  echo "[prepare_togsim_extern] Missing directory: ${EXTERN_DIR}" >&2
  exit 1
fi

has_required_paths() {
  for p in \
    "${EXTERN_DIR}/ramulator2/CMakeLists.txt" \
    "${EXTERN_DIR}/booksim/CMakeLists.txt" \
    "${EXTERN_DIR}/protobuf/cmake" \
    "${EXTERN_DIR}/stonneCore/CMakeLists.txt" \
    "${EXTERN_DIR}/onnx/CMakeLists.txt"
  do
    [[ -e "${p}" ]] || return 1
  done
  return 0
}

if [[ -d "${ROOT_DIR}/.git" ]]; then
  # Always align extern with the commit pinned by the current superproject branch.
  if git -C "${ROOT_DIR}" submodule sync --recursive && \
     git -C "${ROOT_DIR}" submodule update --init --recursive --checkout; then
    echo "[prepare_togsim_extern] Synced extern via git submodule update."
  fi
fi

if has_required_paths; then
  echo "[prepare_togsim_extern] TOGSim extern dependencies are present."
  exit 0
fi

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  echo "[prepare_togsim_extern] Missing TOGSim extern dependencies in verify-only mode." >&2
  echo "[prepare_togsim_extern] Run on host: git submodule update --init --recursive --checkout" >&2
  exit 1
fi

clone_at_commit() {
  local dst="$1"
  local repo="$2"
  local rev="$3"
  local recursive="${4:-0}"

  rm -rf "${dst}"
  git clone "${repo}" "${dst}"
  git -C "${dst}" checkout "${rev}"
  if [[ "${recursive}" == "1" ]]; then
    git -C "${dst}" submodule update --init --recursive
  fi
}

if [[ "${ALLOW_FALLBACK_PINNED}" != "1" ]]; then
  echo "[prepare_togsim_extern] Missing extern dependencies and no .git metadata to resolve pinned commits." >&2
  echo "[prepare_togsim_extern] Run on host before docker build: git submodule update --init --recursive --checkout" >&2
  echo "[prepare_togsim_extern] Or rerun with ALLOW_FALLBACK_PINNED=1 (legacy pinned fallback)." >&2
  exit 1
fi

echo "[prepare_togsim_extern] Falling back to direct clone for missing extern repos (legacy pinned commits)."
mkdir -p "${EXTERN_DIR}"

clone_at_commit \
  "${EXTERN_DIR}/ramulator2" \
  "https://github.com/PSAL-POSTECH/ramulator2" \
  "495561282d99f2ef2652618710e98c4a287025da"

clone_at_commit \
  "${EXTERN_DIR}/booksim" \
  "https://github.com/PSAL-POSTECH/booksim.git" \
  "1d69c9d9db190d130087471b373d12ed4605bf6e"

clone_at_commit \
  "${EXTERN_DIR}/protobuf" \
  "https://github.com/protocolbuffers/protobuf.git" \
  "0dab03ba7bc438d7ba3eac2b2c1eb39ed520f928" \
  "1"

clone_at_commit \
  "${EXTERN_DIR}/stonneCore" \
  "https://github.com/PSAL-POSTECH/stonne_core.git" \
  "97804185f00e98e56f74638e4282b9aecab8cfce"

clone_at_commit \
  "${EXTERN_DIR}/onnx" \
  "https://github.com/onnx/onnx.git" \
  "f7ee1ac60d06abe8e26c9b6bbe1e3db5286b614b" \
  "1"

echo "[prepare_togsim_extern] Extern repositories are prepared."
