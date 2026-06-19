"""Prefill numeric check through the fused SDPA flash template (is_decode=False).
fp32, causal + non-causal, B=1, GQA Hq=5/Hkv=1, S=512. Compares sim vs CPU SDPA.
Run with TOGSIM_CONFIG=.../configs/_v6e_func512.yml (vlen512, functional_mode:1).
"""
import sys, os, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
from Simulator.simulator import TOGSimulator
from torch.nn.attention import sdpa_kernel, SDPBackend

device = torch.device("npu:0")
ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--kv", type=int, default=512)
ap.add_argument("--causal", type=int, default=1)
ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
a = ap.parse_args()

B, Hq, Hkv, D = a.batch, 5, 1, 128
dt = torch.float16 if a.dtype == "fp16" else torch.float32
L = a.kv                                   # prefill: query len == kv len
torch.manual_seed(0)
q = torch.rand(B, Hq, L, D, dtype=dt)
k = torch.rand(B, Hkv, a.kv, D, dtype=dt)
v = torch.rand(B, Hkv, a.kv, D, dtype=dt)
kw = dict(attn_mask=None, dropout_p=0.0, is_causal=bool(a.causal), enable_gqa=True)

opt = torch.compile(F.scaled_dot_product_attention, dynamic=False)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    with torch.no_grad(), TOGSimulator():
        out = opt(q.to(device), k.to(device), v.to(device), **kw)
        torch.npu.synchronize()

ref = F.scaled_dot_product_attention(q, k, v, **kw)
o, r = out.cpu().float(), ref.float()
md = (o - r).abs().max().item()
tol = 1e-2 if a.dtype == "fp32" else 5e-2
print(f"PREFILL_NUMERIC B={B} Hq={Hq} Hkv={Hkv} L={L} KV={a.kv} D={D} causal={a.causal} "
      f"dtype={a.dtype} maxdiff={md:.6f} {'PASS' if md < tol else 'FAIL'} "
      f"npu_absmax={o.abs().max().item():.5f} ref_absmax={r.abs().max().item():.5f}")
