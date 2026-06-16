"""vlen-256 vs vlen-128 vs ground-truth(real v6e), per op / per dim, normalized.
Sim cycles -> us via core_freq 3502 MHz (both configs share clock & vpu_num_lanes=256;
they differ only in vpu_vector_length_bits 256 vs 128). Bars normalized to real=1.0.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLK = 3502.0  # MHz -> cycles/us

# measured sim cycles (CMEM=0, fp16, functional-off)
gemm = dict(N=[512,1024,2048,4096,8192],
            c256=[5827,23298,109602,652355,4428100],
            c128=[5827,23298,109602,652354,4428100],
            real=[2.49,6.85,29.89,187.75,1593.11])  # real v6e bf16, us
gemv = dict(N=[1024,2048,4096,8192,16384],
            c256=[5756,23049,92712,405421,1399104],
            c128=[7189,27418,111193,405430,1399100],
            real=[2.60,7.38,24.14,92.67,361.94])
vadd = dict(N=[1048576,4194304,16777216,67108864],
            c256=[17066,67370,268590,1073454],
            c128=[17066,67370,268590,1073454],
            real=[6.07,18.78,71.69,283.81])

def norm(d):
    real = np.array(d['real'], float)
    s256 = np.array(d['c256'], float) / CLK / real
    s128 = np.array(d['c128'], float) / CLK / real
    return np.ones_like(real), s256, s128

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("PyTorchSim TPUv6e: vlen-256 vs vlen-128 vs real v6e  (normalized to real = 1.0)",
             fontsize=14, fontweight="bold")

for ax, (d, title, xlbl) in zip(axes, [
        (gemm, "GEMM (NxN x NxN)", "N"),
        (gemv, "GEMV (NxN x N)", "N"),
        (vadd, "VADD (length N)", "N")]):
    r, s256, s128 = norm(d)
    x = np.arange(len(d['N']))
    w = 0.27
    ax.bar(x - w, r,    w, label="real v6e (=1.0)", color="C3")
    ax.bar(x,     s256, w, label="sim vlen-256",    color="C0")
    ax.bar(x + w, s128, w, label="sim vlen-128",    color="C1")
    ax.axhspan(0.9, 1.1, color="green", alpha=0.10)
    ax.axhline(1.0, color="k", lw=0.8)
    for xi, (a, b) in enumerate(zip(s256, s128)):
        ax.text(xi,        a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=7, color="C0")
        ax.text(xi + w,    b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=7, color="C1")
    ax.set_xticks(x); ax.set_xticklabels([f"{n:,}" if n < 100000 else f"{n//1048576}M" for n in d['N']],
                                          rotation=30, fontsize=8)
    ax.set_title(title); ax.set_xlabel(xlbl); ax.set_ylabel("sim / real")
    ax.set_ylim(0, 1.5); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "/workspace/PyTorchSim/tpu_validation/vlen_compare_bar.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("SAVED", out)
