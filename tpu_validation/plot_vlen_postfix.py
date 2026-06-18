#!/usr/bin/env python3
"""Bar chart: PyTorchSim TPUv6e GEMM/GEMV/VADD, sim/real ratio for
real(=1.0) vs sim vlen-256 vs sim vlen-128 (POST-FIX: vlen reaches gem5).
sim_us = cycles / CORE_FREQ_MHZ ; ratio = sim_us / real_us.
vlen-256 is unchanged by the fix; only vlen-128 is updated with post-fix cycles."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FREQ = 3502.0  # MHz -> us = cycles/FREQ

# real v6e latency (us) — ground truth, unchanged (from plot_v6e_validation.py)
REAL = {
    "GEMM": ([512,1024,2048,4096,8192],        [2.49,6.85,29.89,187.75,1593.11]),
    "GEMV": ([1024,2048,4096,8192,16384],      [2.60,7.38,24.14,92.67,361.94]),
    "VADD": ([1048576,4194304,16777216,67108864], [6.07,18.78,71.69,283.81]),
}
# measured cycles {op: {N: (vlen256, vlen128)}}  (post-fix sweep)
CYC = {
 "GEMM": {512:(5827,6629), 1024:(23298,24194), 2048:(109602,110518), 4096:(652355,653714), 8192:(4428102,4438947)},
 "GEMV": {1024:(5753,6944), 2048:(23171,27382), 4096:(91336,111195), 8192:(423748,495063), 16384:(1399095,1950585)},
 "VADD": {1048576:(17067,17059), 4194304:(67361,67361), 16777216:(268590,268590), 67108864:(1073454,1073454)},
}
XLAB = {  # pretty x labels
 "GEMM": ["512","1,024","2,048","4,096","8,192"],
 "GEMV": ["1,024","2,048","4,096","8,192","16,384"],
 "VADD": ["1M","4M","16M","64M"],
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("PyTorchSim TPUv6e: vlen-256 vs vlen-128 vs real v6e  (normalized to real = 1.0, POST-FIX)",
             fontsize=15, fontweight="bold")
TITLE = {"GEMM":"GEMM (NxN x NxN)","GEMV":"GEMV (NxN x N)","VADD":"VADD (length N)"}

for ax, op in zip(axes, ["GEMM","GEMV","VADD"]):
    Ns, reals = REAL[op]
    r256, r128 = [], []
    for N, rl in zip(Ns, reals):
        c256, c128 = CYC[op][N]
        r256.append((c256/FREQ)/rl if c256 else np.nan)
        r128.append((c128/FREQ)/rl if c128 else np.nan)
    x = np.arange(len(Ns)); w = 0.27
    ax.axhspan(0.9, 1.1, color="green", alpha=0.12)
    ax.axhline(1.0, color="k", lw=0.8)
    b0 = ax.bar(x-w, [1.0]*len(Ns), w, color="#c0392b", label="real v6e (=1.0)")
    b1 = ax.bar(x,    r256, w, color="#2e86c1", label="sim vlen-256")
    b2 = ax.bar(x+w,  r128, w, color="#e67e22", label="sim vlen-128 (post-fix)")
    for bars, vals in [(b1,r256),(b2,r128)]:
        for rect, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(rect.get_x()+rect.get_width()/2, v+0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_title(TITLE[op]); ax.set_xlabel("N"); ax.set_ylabel("sim / real")
    ax.set_xticks(x); ax.set_xticklabels(XLAB[op]); ax.set_ylim(0, 1.7)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.96])
out = "/workspace/PyTorchSim/tpu_validation/vlen_postfix_bar.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
