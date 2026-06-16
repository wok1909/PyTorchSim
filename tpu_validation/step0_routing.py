"""Step 0 de-risk: does the flash-SDPA MLIR template tile on v6e, and where does
a (KV-materialized) MHA causal SDPA call lower to?

Run with TOGSIM_CONFIG=<v6e> . We only care that it reaches the flash template +
compiles/sims (no 'Failed to find optimal tile size'); numerics are meaningless
under functional_mode:0.
"""
import sys, os, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
from Simulator.simulator import TOGSimulator
from torch.nn.attention import sdpa_kernel, SDPBackend

device = torch.device("npu:0")
ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["noncausal", "causal_mha"], default="noncausal")
ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
a = ap.parse_args()

B, Hq, Hkv, S, D = 1, 5, 1, a.seq, 128   # LLAMA4_TP8 shape
dt = torch.float16 if a.dtype == "fp16" else torch.float32

if a.case == "noncausal":
    # MHA non-causal: q/k/v [1,5,S,128]. Proven flash-template path → proves v6e tiling.
    q = torch.rand(B, Hq, S, D, dtype=dt)
    k = torch.rand(B, Hq, S, D, dtype=dt)
    v = torch.rand(B, Hq, S, D, dtype=dt)
    kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=False)
    tag = f"NONCAUSAL_MHA seq={S}"
else:
    # GQA -> materialize K/V to Hq heads (MHA), then causal. Tests causal routing.
    q = torch.rand(B, Hq, S, D, dtype=dt)
    k1 = torch.rand(B, Hkv, S, D, dtype=dt)
    v1 = torch.rand(B, Hkv, S, D, dtype=dt)
    k = k1.expand(B, Hq, S, D).contiguous()   # materialize repeat -> MHA
    v = v1.expand(B, Hq, S, D).contiguous()
    kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=True)
    tag = f"CAUSAL_MHA(matKV) seq={S}"

opt = torch.compile(F.scaled_dot_product_attention, dynamic=False)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    with torch.no_grad(), TOGSimulator():
        out = opt(q.to(device), k.to(device), v.to(device), **kwargs)
        torch.npu.synchronize()
print(f"STEP0_DONE {tag} out={tuple(out.shape)}")
