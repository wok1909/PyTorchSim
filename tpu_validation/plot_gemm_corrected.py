#!/usr/bin/env python3
"""GEMM sim vs real, with CORRECTED real = module total (includes operand HBM
copy-done that the old 'fusion-only' real excluded). fp16, vlen-256."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FREQ=3502.0
N=[512,1024,2048,4096,8192]
sim_cyc=np.array([5827,23298,109602,653200,4428100])
sim=sim_cyc/FREQ
real_fusion=np.array([2.48,6.85,29.99,188.13,1593.11])   # OLD (compute only, excludes HBM operand copy)
real_module=np.array([3.52,8.92,36.31,211.51,1593.11])   # CORRECTED (full: copy-done + fusion)

x=np.arange(len(N))
fig,ax=plt.subplots(1,2,figsize=(16,6))
fig.suptitle("GEMM (fp16) sim vs real v6e — corrected real = module total (incl. operand HBM copy)",
             fontsize=14,fontweight="bold")

# Panel 1: absolute latency
a=ax[0]
a.plot(x, real_module,'o-',color="#c0392b",lw=2,label="real (module, CORRECTED)")
a.plot(x, real_fusion,'o--',color="#e59866",lw=1.5,label="real (fusion-only, old)")
a.plot(x, sim,'s-',color="#2e86c1",lw=2,label="sim (vlen-256)")
a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([f"{n:,}" for n in N])
a.set_xlabel("N"); a.set_ylabel("latency (us, log)"); a.set_title("Absolute latency")
a.legend(fontsize=9); a.grid(True,which="both",alpha=0.3)
for i,(s,r) in enumerate(zip(sim,real_module)):
    a.annotate(f"{s:.1f}",(x[i],s),textcoords="offset points",xytext=(0,-14),fontsize=7,color="#2e86c1",ha="center")
    a.annotate(f"{r:.1f}",(x[i],r),textcoords="offset points",xytext=(0,8),fontsize=7,color="#c0392b",ha="center")

# Panel 2: sim/real ratio (old vs corrected)
a=ax[1]
w=0.36
a.axhspan(0.9,1.1,color="green",alpha=0.12,label="±10%")
a.axhline(1.0,color="k",lw=0.8)
b1=a.bar(x-w/2, sim/real_fusion, w, color="#e59866", label="sim / real(fusion, old)")
b2=a.bar(x+w/2, sim/real_module, w, color="#2e86c1", label="sim / real(module, CORRECTED)")
for bars,vals in [(b1,sim/real_fusion),(b2,sim/real_module)]:
    for rect,v in zip(bars,vals):
        a.text(rect.get_x()+rect.get_width()/2, v+0.02, f"{v:.2f}",ha="center",va="bottom",fontsize=8)
a.set_xticks(x); a.set_xticklabels([f"{n:,}" for n in N])
a.set_xlabel("N"); a.set_ylabel("sim / real"); a.set_title("sim/real ratio: old vs corrected baseline")
a.set_ylim(0,1.3); a.legend(fontsize=9,loc="lower right"); a.grid(True,axis="y",alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.95])
out="/workspace/PyTorchSim/tpu_validation/gemm_corrected.png"
plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
