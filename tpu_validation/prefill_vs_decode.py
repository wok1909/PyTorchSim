"""Prefill vs decode cycle comparison through the SAME SDPA flash template.
Same batch / KV length / heads / head-dim; only the query length differs
(prefill Lq=S, decode Lq=1). Timing config (functional_mode:0) -> TOGSim cycles.
"""
import sys, os, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
from Simulator.simulator import TOGSimulator
from torch.nn.attention import sdpa_kernel, SDPBackend

device = torch.device("npu:0")
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["prefill", "decode"], required=True)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--kv", type=int, default=512)     # KV-cache sequence length
ap.add_argument("--causal", type=int, default=0)
ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
a = ap.parse_args()

B, Hq, Hkv, D = a.batch, 5, 1, 128       # LLAMA4_TP8 GQA shape
dt = torch.float16 if a.dtype == "fp16" else torch.bfloat16
Lq = a.kv if a.mode == "prefill" else 1  # prefill: full seq; decode: single token
torch.manual_seed(0)

q = torch.rand(B, Hq, Lq, D, dtype=dt)
k = torch.rand(B, Hkv, a.kv, D, dtype=dt)
v = torch.rand(B, Hkv, a.kv, D, dtype=dt)
kw = dict(attn_mask=None, dropout_p=0.0, is_causal=bool(a.causal), enable_gqa=True)

opt = torch.compile(F.scaled_dot_product_attention, dynamic=False)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    with torch.no_grad(), TOGSimulator():
        out = opt(q.to(device), k.to(device), v.to(device), **kw)
        torch.npu.synchronize()
print(f"PVD_DONE mode={a.mode} B={B} Hq={Hq} Hkv={Hkv} Lq={Lq} KV={a.kv} D={D} "
      f"causal={a.causal} out={tuple(out.shape)}")
