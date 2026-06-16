#!/usr/bin/env bash
set -euo pipefail

home="${HOME_DIR:-/workspace}"
RISCV="${RISCV:-${home}/riscv}"
mkdir -p "${home}"
cd "${home}"

apt -y update
apt -y install scons ninja-build

# Gem5
git clone --depth 1 https://github.com/PSAL-POSTECH/gem5.git
cd gem5
scons build/RISCV/gem5.opt -j"$(nproc)"
export GEM5_PATH="${home}/gem5/build/RISCV/gem5.opt"
cd "${home}"

# LLVM
git clone --depth 1 --branch torchsim https://github.com/PSAL-POSTECH/llvm-project.git
cmake -S llvm-project/llvm -B llvm-project/build \
  -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/riscv-llvm \
  -DLLVM_TARGETS_TO_BUILD=RISCV
cmake --build llvm-project/build --parallel "$(nproc)"
cmake --install llvm-project/build
cd "${home}"

# Spike Simulator
git clone --depth 1 --branch TorchSim https://github.com/PSAL-POSTECH/riscv-isa-sim.git
mkdir -p riscv-isa-sim/build
cd riscv-isa-sim/build
../configure --prefix="${RISCV}"
make -j"$(nproc)"
make install
cd "${home}"
