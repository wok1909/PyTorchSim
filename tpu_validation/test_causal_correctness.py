"""Numeric correctness check for the flash-SDPA template (causal or non-causal).
Runs MHA (KV materialized to Hq heads) through the template with functional mode
ON, compares to CPU F.scaled_dot_product_attention. Ground-truth check.
"""
import sys, os, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
from Simulator.simulator import TOGSimulator
from torch.nn.attention import sdpa_kernel, SDPBackend

device = torch.device("npu:0")
ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=64)
ap.add_argument("--causal", type=int, default=1)
a = ap.parse_args()
causal = bool(a.causal)

B, Hq, Hkv, S, D = 1, 5, 1, a.seq, 128
dt = torch.float32
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
out_npu = out.cpu()

with torch.no_grad():
    out_cpu = F.scaled_dot_product_attention(q, k, v, **kw)

diff = (out_npu - out_cpu).abs().max().item()
mean_diff = (out_npu - out_cpu).abs().mean().item()
print(f"NUMERIC causal={causal} seq={S} max|npu-cpu|={diff:.6f} mean={mean_diff:.6f} "
      f"npu_max={out_npu.abs().max():.4f} cpu_max={out_cpu.abs().max():.4f} "
      f"{'PASS' if diff < 0.05 else 'FAIL'}")
# diagnostics: head 0, query rows 0 and last, dims 0..3
print("npu[0,0,0,:4]=", out_npu[0,0,0,:4].tolist())
print("cpu[0,0,0,:4]=", out_cpu[0,0,0,:4].tolist())
print("npu[0,0,-1,:4]=", out_npu[0,0,-1,:4].tolist())
print("cpu[0,0,-1,:4]=", out_cpu[0,0,-1,:4].tolist())
