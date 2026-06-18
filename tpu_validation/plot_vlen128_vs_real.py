#!/usr/bin/env python3
"""GEMM/GEMV/VADD: sim(vlen-128) vs real v6e, normalized to real=1.0.
real = module total (corrected: GEMM includes operand HBM copy-done; GEMV/VADD
load is inline so module=op). sim_us = cycles/3502."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
FREQ=3502.0

DATA = {
 "GEMM": dict(N=[512,1024,2048,4096,8192],
              real=[3.52,8.92,36.31,211.51,1593.11],
              sim128=[6629,24194,110518,653714,4438947],
              xl=["512","1,024","2,048","4,096","8,192"]),
 "GEMV": dict(N=[1024,2048,4096,8192,16384],
              real=[3.06,7.83,24.60,93.13,362.42],
              sim128=[6944,27382,111195,495063,1950585],
              xl=["1,024","2,048","4,096","8,192","16,384"]),
 "VADD": dict(N=[1048576,4194304,16777216,67108864],
              real=[6.07,18.79,71.70,283.81],
              sim128=[17059,67361,268590,1073454],
              xl=["1M","4M","16M","64M"]),
}
TITLE={"GEMM":"GEMM (NxN x NxN)","GEMV":"GEMV (NxN x N)","VADD":"VADD (length N)"}

fig,axes=plt.subplots(1,3,figsize=(20,6))
fig.suptitle("PyTorchSim TPUv6e: sim vlen-128 vs real v6e  (normalized to real=1.0; real=module total)",
             fontsize=15,fontweight="bold")
for ax,op in zip(axes,["GEMM","GEMV","VADD"]):
    d=DATA[op]
    ratio=[(c/FREQ)/r for c,r in zip(d["sim128"],d["real"])]
    x=np.arange(len(d["N"])); w=0.34
    ax.axhspan(0.9,1.1,color="green",alpha=0.12)
    ax.axhline(1.0,color="k",lw=0.8)
    ax.bar(x-w/2,[1.0]*len(x),w,color="#c0392b",label="real v6e (=1.0)")
    b=ax.bar(x+w/2,ratio,w,color="#e67e22",label="sim vlen-128")
    for rect,v in zip(b,ratio):
        ax.text(rect.get_x()+rect.get_width()/2,v+0.02,f"{v:.2f}",ha="center",va="bottom",fontsize=8)
    ax.set_title(TITLE[op]); ax.set_xlabel("N"); ax.set_ylabel("sim / real")
    ax.set_xticks(x); ax.set_xticklabels(d["xl"]); ax.set_ylim(0,1.7)
    ax.legend(fontsize=9,loc="upper left"); ax.grid(True,axis="y",alpha=0.3)
plt.tight_layout(rect=[0,0,1,0.96])
out="/workspace/PyTorchSim/tpu_validation/vlen128_vs_real.png"
plt.savefig(out,dpi=110,bbox_inches="tight"); print("saved",out)
