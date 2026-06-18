#!/usr/bin/env python3
"""VPU-compute-bound microbench for PyTorchSim (isolate vpu_vector_length_bits).

Unlike VADD (memory-bound), this keeps a SMALL array (SRAM-resident, low HBM
traffic) and applies a long chain of K elementwise transcendentals, so VPU
*compute* dominates latency. If vpu_vector_length_bits is modeled on the VPU
path, vlen-256 should be ~2x faster than vlen-128 here; if cycles stay equal,
vlen is not reaching the VPU codegen.

  python vmath_bench.py --size 65536 --iters 100 --dtype fp16

Run INSIDE the ok_torchsim container with TOGSIM_CONFIG set.
"""
import os, sys, argparse
base = os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim')
sys.path.insert(0, base)
import torch
from Simulator.simulator import TOGSimulator

config = os.environ['TOGSIM_CONFIG']
DT = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}

p = argparse.ArgumentParser()
p.add_argument('--size', type=int, nargs='+', required=True, help='shape (1 or 2 ints); keep small to stay SRAM-resident')
p.add_argument('--iters', type=int, required=True, help='K elementwise ops per element')
p.add_argument('--dtype', default='fp16', choices=list(DT))
p.add_argument('--kind', default='tanh', choices=['tanh', 'poly', 'softmax', 'fma_relu'],
               help='tanh = transcendental (fp32 only); poly = mul-add chain (fp16-safe); '
                    'softmax = repeated softmax (reduction path); '
                    'fma_relu = relu(y*c1+c2) chain — matches real-TPU vpu_peak_bench (VPU-peak probe)')
args = p.parse_args()
dt = DT[args.dtype]
dev = torch.device('npu:0')
torch.manual_seed(0)

shape, K = tuple(args.size), args.iters
a = torch.randn(*shape).to(dtype=dt).to(device=dev)

if args.kind == 'tanh':
    def fn(x):
        y = x
        for _ in range(K):
            y = torch.tanh(y)
        return y
elif args.kind == 'poly':  # pure mul-add, stays in dtype (no f32 widening) -> fp16-safe
    def fn(x):
        y = x
        for _ in range(K):
            y = y * 0.99 + 0.01
        return y
elif args.kind == 'fma_relu':  # matches real-TPU vpu_peak_bench: relu(y*c1+c2) chain (VPU-peak probe)
    def fn(x):
        y = x
        for _ in range(K):
            y = torch.clamp(y * 0.99 + 0.001, min=-1.0)
        return y
else:  # softmax: reduction path (max/exp/sum/div) -> wide vectors, VPU-compute-bound
    def fn(x):
        y = x
        for _ in range(K):
            y = torch.softmax(y, dim=-1)
        return y

opt_fn = torch.compile(dynamic=False)(fn)
with TOGSimulator(config_path=config), torch.no_grad():
    torch.npu.launch_model(opt_fn, a, stream_index=0, timestamp=0)
    torch.npu.synchronize()
print(f"VMATH_DONE shape={shape} iters={K} dtype={args.dtype} kind={args.kind}")
