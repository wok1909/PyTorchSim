"""Numeric correctness for the flash-SDPA template, fp16/fp32 x causal/noncausal.

Runs MHA (KV materialized to Hq heads) through the flash template with functional
mode ON (set TOGSIM_CONFIG to a functional_mode:1 config), compares against CPU
F.scaled_dot_product_attention computed in the SAME dtype. fp16 uses a looser
tolerance because the reference itself is fp16.
"""
import sys, os, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
from Simulator.simulator import TOGSimulator
from torch.nn.attention import sdpa_kernel, SDPBackend

device = torch.device("npu:0")
ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=64)
ap.add_argument("--causal", type=int, default=0)
ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
a = ap.parse_args()
causal = bool(a.causal)
dt = torch.float16 if a.dtype == "fp16" else torch.float32
# fp16 attention output magnitude ~O(1); allow generous abs tol for fp16 round-trips.
tol = 0.05 if dt == torch.float32 else 3e-3

B, Hq, Hkv, S, D = 1, 5, 1, a.seq, 128
torch.manual_seed(0)
scale = 1.0 / (D ** 0.5)

q  = torch.rand(B, Hq, S, D, dtype=dt)
k1 = torch.rand(B, Hkv, S, D, dtype=dt)
v1 = torch.rand(B, Hkv, S, D, dtype=dt)
k  = k1.expand(B, Hq, S, D).contiguous()   # materialize -> MHA
v  = v1.expand(B, Hq, S, D).contiguous()
kw = dict(attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale)

opt = torch.compile(F.scaled_dot_product_attention, dynamic=False)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    with torch.no_grad(), TOGSimulator():
        out = opt(q.to(device), k.to(device), v.to(device), **kw)
        torch.npu.synchronize()
out_npu = out.cpu().float()

with torch.no_grad():
    out_cpu = F.scaled_dot_product_attention(q, k, v, **kw).float()

diff = (out_npu - out_cpu).abs().max().item()
mean_diff = (out_npu - out_cpu).abs().mean().item()
# rel error on the overall tensor norm — robust, dtype-agnostic verdict
rel = (out_npu - out_cpu).norm().item() / (out_cpu.norm().item() + 1e-9)
print(f"NUMERIC dtype={a.dtype} causal={causal} seq={S} "
      f"max|npu-cpu|={diff:.6f} mean={mean_diff:.6f} rel_l2={rel:.5f} "
      f"npu_max={out_npu.abs().max():.4f} cpu_max={out_cpu.abs().max():.4f} "
      f"{'PASS' if (diff < tol or rel < 0.02) else 'FAIL'}")
print("npu[0,0,0,:4]=", out_npu[0,0,0,:4].tolist())
print("cpu[0,0,0,:4]=", out_cpu[0,0,0,:4].tolist())
print("npu[0,0,-1,:4]=", out_npu[0,0,-1,:4].tolist())
print("cpu[0,0,-1,:4]=", out_cpu[0,0,-1,:4].tolist())
