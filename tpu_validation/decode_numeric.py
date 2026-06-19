"""Decode (L=1) GQA numeric check through the fused SDPA flash template.

Same shape/driver as prefill_vs_decode.py but Lq=1 (decode) and with a CPU
reference + maxdiff so we can verify the is_decode fused kernel's correctness.
fp32 by default to isolate the structural (lane-invariant) bug from fp16 noise.
Run with TOGSIM_CONFIG=configs/_v6e_func512.yml (vlen512, functional_mode:1).
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
ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
a = ap.parse_args()

B, Hq, Hkv, D = a.batch, 5, 1, 128       # LLAMA4_TP8 GQA shape (g = Hq/Hkv = 5)
dt = torch.float16 if a.dtype == "fp16" else torch.float32
Lq = 1                                    # decode: single query token
torch.manual_seed(0)

q = torch.rand(B, Hq, Lq, D, dtype=dt)
k = torch.rand(B, Hkv, a.kv, D, dtype=dt)
v = torch.rand(B, Hkv, a.kv, D, dtype=dt)
kw = dict(attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=True)

opt = torch.compile(F.scaled_dot_product_attention, dynamic=False)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    with torch.no_grad(), TOGSimulator():
        out = opt(q.to(device), k.to(device), v.to(device), **kw)
        torch.npu.synchronize()

ref = F.scaled_dot_product_attention(q, k, v, **kw)
o = out.cpu().float()      # (B, Hq, 1, D)
r = ref.float()
md = (o - r).abs().max().item()
tol = 1e-2 if a.dtype == "fp32" else 5e-2
print(f"DECODE_NUMERIC B={B} Hq={Hq} Hkv={Hkv} KV={a.kv} D={D} dtype={a.dtype} "
      f"maxdiff={md:.6f} {'PASS' if md < tol else 'FAIL'} "
      f"npu_absmax={o.abs().max().item():.5f} ref_absmax={r.abs().max().item():.5f}")
# full head0 pattern: how many near-zero, where (transpose signature)
h0 = o[0, 0, 0]
nz = (h0.abs() > 1e-3).sum().item()
print(f"  head0 full: nonzero={nz}/{D}  first16={[round(x,3) for x in h0[:16].tolist()]}")
print(f"  head0 idx of nonzero (first 20): {(h0.abs()>1e-3).nonzero().flatten()[:20].tolist()}")
# per query-head diagnostics (localize transpose vs scale vs partial)
for h in range(Hq):
    oh, rh = o[0, h, 0], r[0, h, 0]      # (D,)
    hd = (oh - rh).abs().max().item()
    print(f"  head{h}: maxdiff={hd:.4f} "
          f"npu[:4]={[round(x,4) for x in oh[:4].tolist()]} "
          f"ref[:4]={[round(x,4) for x in rh[:4].tolist()]} "
          f"npu_norm={oh.norm().item():.4f} ref_norm={rh.norm().item():.4f}")
