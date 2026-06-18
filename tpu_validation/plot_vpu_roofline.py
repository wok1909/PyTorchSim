#!/usr/bin/env python3
"""VPU roofline (fp16): is real v6e's ~4.56 TFLOP/s plausible, and how much of the
VPU ceiling does the sim use? Ceilings: VPU (real measured 4.56), MXU (918), HBM (1759)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HBM_real = 1.759   # TB/s
VPU_real = 4.56    # TFLOP/s (measured fp16 peak, VPU-compute-bound)
VPU_sim_bench = 0.76   # sim achieved on same VPU-bound kernel
MXU = 918.0        # TFLOP/s (bf16 MXU peak, for scale)

# kernels: (label, OI flop/byte, achieved TFLOP/s, color, marker)
pts = [
    ("real VPU-bench (fma chain)", 50.0, 4.56, "C3", "*"),
    ("sim  VPU-bench",             25.0, 0.76, "C0", "*"),
    ("real GEMV (OI~1)",            1.0, 1.37, "C3", "s"),
    ("sim  GEMV",                   1.0, 1.25, "C0", "s"),
    ("real VADD (OI~0.17)",        0.167, 0.236, "C3", "^"),
    ("sim  VADD",                  0.167, 0.219, "C0", "^"),
]

fig, ax = plt.subplots(figsize=(11, 7.5))
oi = np.logspace(-2, 3, 400)
# ceilings
ax.plot(oi, np.minimum(HBM_real*oi, VPU_real), "k-", lw=2, label=f"VPU roofline (HBM {HBM_real} TB/s, VPU {VPU_real} TF)")
ax.axhline(VPU_real, color="C3", ls="--", lw=1.5, label=f"VPU compute ceiling = {VPU_real} TFLOP/s (real, measured)")
ax.axhline(MXU, color="gray", ls=":", lw=1, label=f"MXU ceiling = {MXU} TFLOP/s (for scale)")
ax.axhline(VPU_sim_bench, color="C0", ls="--", lw=1, label=f"sim VPU achieved = {VPU_sim_bench} ({100*VPU_sim_bench/VPU_real:.0f}% of real VPU)")
# ridge
ridge = VPU_real / HBM_real
ax.axvline(ridge, color="C3", ls=":", lw=0.8); ax.text(ridge*1.05, 0.02, f"VPU ridge\n{ridge:.1f} F/B", fontsize=8, color="C3")

for lbl, x, y, c, m in pts:
    ax.scatter([x], [y], color=c, marker=m, s=130, zorder=5, edgecolor="k", linewidth=0.5)
    ax.annotate(f"{lbl}\n{y:.2f} TF", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8, color=c)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.05, 1000); ax.set_ylim(0.05, 1500)
ax.set_xlabel("arithmetic intensity (FLOP / byte)"); ax.set_ylabel("TFLOP/s")
ax.set_title("TPU v6e VPU roofline (fp16): real measured vs sim\n"
             "real VPU-bench sits ON the 4.56 ceiling; sim reaches only 0.76 (17%)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
out = "/workspace/PyTorchSim/tpu_validation/vpu_roofline.png"
plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out)
