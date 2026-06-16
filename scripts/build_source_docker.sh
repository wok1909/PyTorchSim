#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTORCH_IMAGE="${PYTORCH_IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel}"
BASE_IMAGE_TAG="${BASE_IMAGE_TAG:-pytorchsim-base-source:pt2.8}"
APP_IMAGE_TAG="${APP_IMAGE_TAG:-pytorchsim-source:pt2.8}"
CONTAINER_NAME="${CONTAINER_NAME:-torchsim-source}"
LLVM_BRANCH="${LLVM_BRANCH:-torchsim}"
SPIKE_BRANCH="${SPIKE_BRANCH:-TorchSim}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
PREPARE_EXTERN="${PREPARE_EXTERN:-1}"

for required in Dockerfile.base.source Dockerfile.source TOGSim; do
  if [[ ! -e "${ROOT_DIR}/${required}" ]]; then
    echo "[error] Missing required path: ${ROOT_DIR}/${required}" >&2
    exit 1
  fi
done

if [[ ! -d "${ROOT_DIR}/PyTorchSimDevice" ]]; then
  echo "[error] Missing ${ROOT_DIR}/PyTorchSimDevice in build context." >&2
  echo "[hint] Run this from the repository that contains PyTorchSimDevice, or mount the host repo correctly." >&2
  exit 1
fi

if [[ "${PREPARE_EXTERN}" == "1" ]]; then
  echo "[pre] Prepare TOGSim extern dependencies"
  bash "${ROOT_DIR}/scripts/prepare_togsim_extern.sh" "${ROOT_DIR}"

  if [[ -d "${ROOT_DIR}/.git" ]]; then
    EXPECTED_RAMULATOR_COMMIT="$(git -C "${ROOT_DIR}" ls-tree HEAD TOGSim/extern/ramulator2 | awk '{print $3}')"
    ACTUAL_RAMULATOR_COMMIT="$(git -C "${ROOT_DIR}/TOGSim/extern/ramulator2" rev-parse HEAD)"
    echo "[pre] ramulator2 expected=${EXPECTED_RAMULATOR_COMMIT} actual=${ACTUAL_RAMULATOR_COMMIT}"
  fi
fi

echo "[1/2] Build base image from source (gem5/llvm/spike)"
echo "[1/2] PYTORCH_IMAGE=${PYTORCH_IMAGE}"
docker build \
  -f "${ROOT_DIR}/Dockerfile.base.source" \
  --build-arg PYTORCH_IMAGE="${PYTORCH_IMAGE}" \
  --build-arg LLVM_BRANCH="${LLVM_BRANCH}" \
  --build-arg SPIKE_BRANCH="${SPIKE_BRANCH}" \
  --build-arg INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN}" \
  -t "${BASE_IMAGE_TAG}" \
  "${ROOT_DIR}"

echo "[2/2] Build PyTorchSim runtime image"
docker build \
  -f "${ROOT_DIR}/Dockerfile.source" \
  --build-arg BASE_IMAGE="${BASE_IMAGE_TAG}" \
  -t "${APP_IMAGE_TAG}" \
  "${ROOT_DIR}"

if [[ "${1:-}" == "run" ]]; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exec docker run -it --ipc=host \
    --name "${CONTAINER_NAME}" \
    -w /workspace/PyTorchSim \
    "${APP_IMAGE_TAG}" bash
fi

echo "Done. To run container:"
echo "docker run -it --ipc=host --name ${CONTAINER_NAME} -w /workspace/PyTorchSim ${APP_IMAGE_TAG} bash"
