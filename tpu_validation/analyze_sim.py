#!/usr/bin/env python3
"""Parse PyTorchSim sweep logs (sim_logs/) -> sim metrics, merge with real-device
fp16 device-op data, emit comparison table + combined roofline.

Outputs:
  sim_vs_real_table.md
  roofline_sim_vs_real.png
"""
import os, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import load as load_real, op_intensity, PEAK_TFLOPS, PEAK_BW

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_FREQ_MHZ = 3502.0                         # v6e config core_freq -> cycles/us
SIM_LOGS = os.path.join(HERE, "sim_logs")

SIZES = {"gemm": [512, 1024, 2048, 4096, 8192],
         "gemv": [1024, 2048, 4096, 8192, 16384],
         "vadd": [1048576, 4194304, 16777216, 67108864]}


def theo_flop(op, N):
    return {"gemm": 2.0 * N**3, "gemv": 2.0 * N * N, "vadd": 1.0 * N}[op]


def parse_sim_log(path):
    txt = open(path).read()
    # Large runs print periodic stats; the FINAL summary is the last match.
    def g(pat):
        ms = re.findall(pat, txt)
        return float(ms[-1]) if ms else None
    cyc = g(r"Total execution cycles:\s*(\d+)")
    if cyc is None:
        return None
    return dict(cycles=cyc,
                sa_util=g(r"Systolic array \[0\] utilization\(%\)\s*([\d.]+)"),
                dram_bw=g(r"DRAM BW\s*([\d.]+)\s*GB/s"),
                vec_util=g(r"Vector unit [Uu]tilization\(%\)\s*([\d.]+)"))


def collect_sim():
    rows = []
    for op, ns in SIZES.items():
        for N in ns:
            p = os.path.join(SIM_LOGS, f"{op}_fp16_{N}.log")
            if not os.path.exists(p):
                continue
            r = parse_sim_log(p)
            if not r:
                rows.append(dict(op=op, N=N, failed=True))
                continue
            lat_us = r["cycles"] / CORE_FREQ_MHZ
            tflops = theo_flop(op, N) / (lat_us * 1e-6) / 1e12
            rows.append(dict(op=op, N=N, failed=False, sim_us=lat_us,
                             sim_tflops=tflops, ai=op_intensity(op, N, "fp16"), **r))
    return rows


def main():
    sim = collect_sim()
    real = {(r["op"], r["N"]): r for r in load_real() if r["dtype"] == "fp16"}

    lines = ["# PyTorchSim v6e vs real TPU v6e (fp16, device-op)", ""]
    lines.append(f"sim core_freq = {CORE_FREQ_MHZ:.0f} MHz · peak {PEAK_TFLOPS:.0f} TFLOPS / {PEAK_BW:.0f} GB/s")
    lines.append("")
    PK_BW = 1600.0  # official v6e HBM GB/s, for BW utilization %
    lines.append("| op | N | AI | sim us | real us | sim/real | sim TFLOPS | sim SA% "
                 "| sim BW | sim BW% | real BW | real BW% |")
    lines.append("|" + "---|" * 12)
    for s in sim:
        op, N = s["op"], s["N"]
        rr = real.get((op, N), {})
        ru = rr.get("dev_us"); rb = rr.get("dev_bw")
        if s.get("failed"):
            rbp = f"{100*rb/PK_BW:.1f}" if rb else "-"
            lines.append(f"| {op} | {N} | - | (running/failed) | {ru or '-'} | - | - | - "
                         f"| - | - | {rb or '-':.0f} | {rbp} |" if rb else
                         f"| {op} | {N} | - | (running/failed) | {ru or '-'} | - | - | - | - | - | - | - |")
            continue
        ratio = (s["sim_us"] / ru) if ru else None
        sbw = s["dram_bw"] or 0
        lines.append(
            f"| {op} | {N} | {s['ai']:.2f} | {s['sim_us']:.2f} | "
            f"{(ru or 0):.2f} | {(ratio or 0):.2f} | {s['sim_tflops']:.1f} | "
            f"{(s['sa_util'] or 0):.1f} | {sbw:.0f} | {100*sbw/PK_BW:.1f} | "
            f"{(rb or 0):.0f} | {(100*rb/PK_BW if rb else 0):.1f} |")
    out = os.path.join(HERE, "sim_vs_real_table.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {out}")

    # roofline: real (device-op) vs sim
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ai = np.logspace(-1.2, 3.7, 400)
    roof = np.minimum(ai * PEAK_BW * 1e9 / 1e12, PEAK_TFLOPS)
    ax.plot(ai, roof, "k-", lw=2, label=f"v6e roofline ({PEAK_TFLOPS:.0f} TF / {PEAK_BW:.0f} GB/s)")
    ax.axvline(PEAK_TFLOPS * 1e12 / (PEAK_BW * 1e9), ls=":", c="gray", lw=1)

    col = {"gemm": "#d62728", "gemv": "#1f77b4", "vadd": "#2ca02c"}
    for s in sim:
        if s.get("failed"):
            continue
        op = s["op"]
        ax.scatter(s["ai"], s["sim_tflops"], marker="x", s=80, c=col[op],
                   linewidths=2, zorder=6, label=f"sim {op}" if s["N"] == SIZES[op][0] else None)
        rr = real.get((op, s["N"]), {})
        if rr.get("dev_tflops"):
            ax.scatter(s["ai"], rr["dev_tflops"], marker="o", s=55, facecolors="none",
                       edgecolors=col[op], linewidths=1.5, zorder=5,
                       label=f"real {op}" if s["N"] == SIZES[op][0] else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Operational intensity (FLOP/byte)")
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_title("PyTorchSim v6e (x) vs real TPU v6e device-op (o), fp16")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_ylim(0.05, 1500)
    out2 = os.path.join(HERE, "roofline_sim_vs_real.png")
    fig.tight_layout(); fig.savefig(out2, dpi=130)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
