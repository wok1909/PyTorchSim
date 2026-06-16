import torch, torch.nn.functional as F, math, os, sys, argparse
sys.path.insert(0, os.environ.get("TORCHSIM_DIR","/workspace/PyTorchSim"))
from Simulator.simulator import TOGSimulator
from torch.nn.attention import sdpa_kernel, SDPBackend
device = torch.device("npu:0")
ap=argparse.ArgumentParser(); ap.add_argument("--seq",type=int,default=512); ap.add_argument("--causal",type=int,default=1); a=ap.parse_args()
S=a.seq; Hq=5; Hkv=1; D=128; dt=torch.float16; scale=1.0/math.sqrt(D)
q=torch.randn(1,Hq,S,D,dtype=dt).to(device); k=torch.randn(1,Hkv,S,D,dtype=dt).to(device); v=torch.randn(1,Hkv,S,D,dtype=dt).to(device)
opt=torch.compile(dynamic=False)(F.scaled_dot_product_attention)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    with torch.no_grad(), TOGSimulator():
        out=opt(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=bool(a.causal),enable_gqa=True)
        torch.npu.synchronize()
print(f"FLASH_PREFILL_DONE S={S} causal={a.causal} out={tuple(out.shape)}")
