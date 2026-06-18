#!/usr/bin/env python3
"""Single linear y = x @ W_o through PyTorchSim, prints the C++ sim's cycle count.
  output projection: x[1024,4096] @ W_o[4096,4096] -> y[1024,4096], bf16.
Run inside ok_torchsim container with TOGSIM_CONFIG set."""
import os, sys
base = os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim')
sys.path.insert(0, base)
import torch
from Simulator.simulator import TOGSimulator

config = os.environ['TOGSIM_CONFIG']
M = int(os.environ.get('LIN_M', 1024))
K = int(os.environ.get('LIN_K', 4096))
N = int(os.environ.get('LIN_N', 4096))
dt = torch.bfloat16
dev = torch.device('npu:0')
torch.manual_seed(0)

x = torch.randn(M, K).to(dtype=dt).to(device=dev)
W = torch.randn(K, N).to(dtype=dt).to(device=dev)
fn = lambda a, b: torch.matmul(a, b)
opt_fn = torch.compile(dynamic=False)(fn)
with TOGSimulator(config_path=config), torch.no_grad():
    torch.npu.launch_model(opt_fn, x, W, stream_index=0, timestamp=0)
    torch.npu.synchronize()
print(f"LINEAR_DONE M={M} K={K} N={N} dtype=bf16")
