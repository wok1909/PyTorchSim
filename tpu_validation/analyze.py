#!/usr/bin/env python3
"""Aggregate gemm/gemv/vadd results + device-op trace metrics -> table + roofline.

Outputs:
  results_table.md   markdown table (wall-clock + device-op per op/size/dtype)
  roofline_v6e.png   roofline plot (achieved device-op throughput vs AI)
"""
import os, json, gzip, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_TFLOPS = 918.0                       # v6e bf16/fp16
PEAK_BW = 1638 * (2**30) / 1e9            # ~1759 GB/s
RIDGE_AI = PEAK_TFLOPS * 1e12 / (PEAK_BW * 1e9)   # ~522 FLOP/byte


def elem_bytes(dtype):
    return 2  # fp16/bf16 only


def op_intensity(op, N, dtype):
    """Theoretical operational intensity (FLOP / byte, minimal traffic)."""
    e = elem_bytes(dtype)
    if op == "gemm":
        return (2.0 * N**3) / (3.0 * N * N * e)        # = N/3 (bf16)
    if op == "gemv":
        return (2.0 * N * N) / ((N * N + 2.0 * N) * e)  # ~ 1/e
    if op == "vadd":
        return (1.0 * N) / (3.0 * N * e)                # = 1/(3e)
    raise ValueError(op)


def theo_flops(op, N):
    if op == "gemm":
        return 2.0 * N**3
    if op == "gemv":
        return 2.0 * N * N
    if op == "vadd":
        return 1.0 * N
    raise ValueError(op)


def device_metrics(run_dir):
    """Parse the trace under run_dir -> (per-iter dur us, dev_TFLOPS, dev_BW GB/s)
    for the dominant device op (pid=3,tid=3, max summed dur)."""
    traces = glob.glob(os.path.join(run_dir, "**", "*.trace.json.gz"), recursive=True)
    if not traces:
        return None
    data = json.load(gzip.open(traces[0], "rt"))
    ops = [e for e in data["traceEvents"]
           if e.get("ph") == "X" and e.get("pid") == 3 and e.get("tid") == 3]
    if not ops:
        return None
    agg = collections.defaultdict(lambda: {"dur": 0.0, "n": 0, "flops": 0, "bytes": 0})
    for e in ops:
        a = e.get("args", {})
        d = agg[e.get("name", "?")]
        d["dur"] += e.get("dur", 0.0)
        d["n"] += 1
        d["flops"] = max(d["flops"], int(a.get("model_flops", 0) or 0))
        d["bytes"] = max(d["bytes"], int(a.get("bytes_accessed", 0) or 0))
    name, d = max(agg.items(), key=lambda kv: kv[1]["dur"])
    per = d["dur"] / d["n"]                              # us per iter
    tflops = d["flops"] / (per * 1e-6) / 1e12 if d["flops"] else 0.0
    bw = d["bytes"] / (per * 1e-6) / 1e9 if d["bytes"] else 0.0
    return dict(name=name, dev_us=per, dev_tflops=tflops, dev_bw=bw)


def load():
    rows = []
    for op in ("gemm", "gemv", "vadd"):
        p = os.path.join(HERE, f"{op}_results.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            N, dt = r["N"], r["dtype"]
            run_dir = os.path.join(HERE, "tb_logs", f"{op}_{dt}_{N}")
            dev = device_metrics(run_dir) or {}
            rows.append(dict(
                op=op, dtype=dt, N=N,
                wall_us=r["avg_us"],
                wall_tflops=r.get("tflops", theo_flops(op, N) / (r["avg_us"] * 1e-6) / 1e12),
                wall_bw=r["bw_gbps"], wall_bw_pct=r["bw_pct"],
                mfu=r.get("mfu_pct", 0.0),
                ai=op_intensity(op, N, dt), pool=r.get("pool"),
                dev_us=dev.get("dev_us"), dev_tflops=dev.get("dev_tflops"),
                dev_bw=dev.get("dev_bw"),
            ))
    return rows


def write_table(rows):
    lines = []
    lines.append("# TPU v6e micro-benchmark results")
    lines.append("")
    lines.append(f"Peak: {PEAK_TFLOPS:.0f} TFLOPS (bf16/fp16) · {PEAK_BW:.0f} GB/s HBM · ridge AI {RIDGE_AI:.0f} FLOP/byte")
    lines.append("")
    hdr = ("| op | dtype | N | AI | wall us | wall TFLOPS | MFU% | wall BW | wall BW% "
           "| dev us | dev TFLOPS | dev BW | dev BW% | pool |")
    sep = "|" + "---|" * 14
    lines.append(hdr)
    lines.append(sep)
    for r in rows:
        dev_bw_pct = 100 * r["dev_bw"] / PEAK_BW if r["dev_bw"] else 0
        lines.append(
            f"| {r['op']} | {r['dtype']} | {r['N']} | {r['ai']:.2f} | "
            f"{r['wall_us']:.1f} | {r['wall_tflops']:.1f} | {r['mfu']:.1f} | "
            f"{r['wall_bw']:.0f} | {r['wall_bw_pct']:.1f} | "
            f"{(r['dev_us'] or 0):.1f} | {(r['dev_tflops'] or 0):.1f} | "
            f"{(r['dev_bw'] or 0):.0f} | {dev_bw_pct:.1f} | {r['pool']} |")
    out = os.path.join(HERE, "results_table.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {out}")


def plot_roofline(rows):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ai = np.logspace(-1.2, 3.7, 400)
    mem_roof = ai * PEAK_BW * 1e9 / 1e12               # TFLOPS = AI * BW
    roof = np.minimum(mem_roof, PEAK_TFLOPS)
    ax.plot(ai, roof, "k-", lw=2, label=f"v6e roofline ({PEAK_TFLOPS:.0f} TF, {PEAK_BW:.0f} GB/s)")
    ax.axvline(RIDGE_AI, ls=":", c="gray", lw=1)
    ax.text(RIDGE_AI*1.1, 2, f"ridge\nAI={RIDGE_AI:.0f}", fontsize=8, color="gray")

    style = {"gemm": ("o", "#d62728"), "gemv": ("s", "#1f77b4"), "vadd": ("^", "#2ca02c")}
    seen = set()
    for r in rows:
        if not r["dev_tflops"]:
            continue
        mk, col = style[r["op"]]
        face = col if r["dtype"] == "bf16" else "white"
        lbl = r["op"] if r["op"] not in seen else None
        seen.add(r["op"])
        ax.scatter(r["ai"], r["dev_tflops"], marker=mk, s=70,
                   facecolors=face, edgecolors=col, linewidths=1.5, label=lbl, zorder=5)
        ax.annotate(f"{r['N']}", (r["ai"], r["dev_tflops"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6.5, color=col)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Operational intensity (FLOP / byte)")
    ax.set_ylabel("Achieved throughput (TFLOPS, device-op)")
    ax.set_title("TPU v6e roofline — GEMM / GEMV / VADD (bf16 filled, fp16 hollow)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.05, 1500)
    out = os.path.join(HERE, "roofline_v6e.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    rows = load()
    write_table(rows)
    plot_roofline(rows)
