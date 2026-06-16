import torch, torch.nn as nn, torch.nn.functional as F, math, os, sys, argparse
sys.path.insert(0, os.environ.get("TORCHSIM_DIR","/workspace/PyTorchSim"))
from Simulator.simulator import TOGSimulator
device = torch.device("npu:0")
class M(nn.Module):
    def forward(self,q,k,v):
        return F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
ap=argparse.ArgumentParser(); ap.add_argument("--seq",type=int,default=512); a=ap.parse_args()
S=a.seq; Hq=5; Hkv=1; D=128; dt=torch.float16
q=torch.randn(1,Hq,S,D,dtype=dt).to(device); k=torch.randn(1,Hkv,S,D,dtype=dt).to(device); v=torch.randn(1,Hkv,S,D,dtype=dt).to(device)
comp=torch.compile(M().to(device),dynamic=False)
with torch.no_grad(), TOGSimulator():
    out=comp(q,k,v); torch.npu.synchronize()
print(f"SDPA_PREFILL_DONE S={S} out={tuple(out.shape)}")
