#!/usr/bin/env python3
"""Run ONE op (gemm/gemv/vadd) through PyTorchSim and let the C++ sim print its
stats to stdout. Driver captures stdout per run. Meant to run INSIDE the
ok_torchsim container with TOGSIM_CONFIG set.

  python sim_bench.py --op gemm --size 2048 --dtype bf16
"""
import os, sys, argparse
base = os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim')
sys.path.insert(0, base)
import torch
from Simulator.simulator import TOGSimulator

config = os.environ['TOGSIM_CONFIG']
DT = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}

p = argparse.ArgumentParser()
p.add_argument('--op', required=True, choices=['gemm', 'gemv', 'vadd'])
p.add_argument('--size', nargs='+', type=int, required=True, help='square N (gemm/gemv) or length N (vadd)')
p.add_argument('--dtype', default='bf16', choices=list(DT))
args = p.parse_args()
dt = DT[args.dtype]
dev = torch.device('npu:0')
torch.manual_seed(0)

N = args.size[0]
if args.op == 'gemm':
    a = torch.randn(N, N).to(dtype=dt).to(device=dev)
    b = torch.randn(N, N).to(dtype=dt).to(device=dev)
    fn = lambda x, y: torch.matmul(x, y)
    inputs = (a, b)
elif args.op == 'gemv':
    a = torch.randn(N, N).to(dtype=dt).to(device=dev)
    x = torch.randn(N).to(dtype=dt).to(device=dev)
    fn = lambda A, v: torch.matmul(A, v)
    inputs = (a, x)
elif args.op == 'vadd':
    a = torch.randn(N).to(dtype=dt).to(device=dev)
    b = torch.randn(N).to(dtype=dt).to(device=dev)
    fn = lambda x, y: x + y
    inputs = (a, b)

opt_fn = torch.compile(dynamic=False)(fn)
with TOGSimulator(config_path=config), torch.no_grad():
    torch.npu.launch_model(opt_fn, *inputs, stream_index=0, timestamp=0)
    torch.npu.synchronize()
print(f"SIMBENCH_DONE op={args.op} N={N} dtype={args.dtype}")
