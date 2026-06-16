#!/usr/bin/env python3
"""Parse TPU profile trace.json.gz -> per-op latency / TFLOPS / BW.

Based on the handoff doc (section 9). Reads device XLA ops (pid=3,tid=3),
extracts dur / model_flops / bytes_accessed / source per op.

Usage:
    python3 parse_trace.py <tb_logs_dir>     # walks all *.trace.json.gz under it
e.g.
    python3 parse_trace.py tpu_validation/tb_logs
"""
import sys, os, gzip, json, glob, collections

PEAK_TFLOPS = 918.0                  # v6e BF16
HBM_GBPS = 1638 * (2**30) / 1e9      # ~1759 GB/s


def parse_one(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    # device XLA ops: pid=3, tid=3 (per handoff)
    ops = [e for e in data["traceEvents"]
           if e.get("ph") == "X" and e.get("pid") == 3 and e.get("tid") == 3]
    return ops


def summarize(ops, label):
    if not ops:
        print(f"  {label}: no device ops found")
        return
    # aggregate by op name (sum dur across the 5 captured iters, report per-iter via /n_iter best-effort)
    rows = []
    for e in ops:
        dur = e.get("dur", 0)  # us
        a = e.get("args", {})
        flops = int(a.get("model_flops", 0) or 0)
        bytes_ = int(a.get("bytes_accessed", 0) or 0)
        if dur <= 0:
            continue
        tflops = flops / (dur * 1e-6) / 1e12 if flops else 0
        bw = bytes_ / (dur * 1e-6) / 1e9 if bytes_ else 0
        rows.append((e.get("name", "?"), dur, tflops, bw, flops, bytes_,
                     a.get("hlo_category", ""), a.get("source", "")))
    # top by duration
    rows.sort(key=lambda r: -r[1])
    print(f"  {label}: {len(rows)} ops, top by dur:")
    print(f"    {'name':<22} {'dur(us)':>9} {'TFLOPS':>8} {'%pk':>5} {'GB/s':>8} {'%bw':>5}  {'cat'}")
    for name, dur, tflops, bw, fl, by, cat, src in rows[:6]:
        print(f"    {name[:22]:<22} {dur:>9.1f} {tflops:>8.1f} {100*tflops/PEAK_TFLOPS:>5.1f} "
              f"{bw:>8.1f} {100*bw/HBM_GBPS:>5.1f}  {cat}")


def main():
    if len(sys.argv) < 2:
        print("usage: python3 parse_trace.py <tb_logs_dir>")
        sys.exit(1)
    root = sys.argv[1]
    traces = sorted(glob.glob(os.path.join(root, "**", "*.trace.json.gz"), recursive=True))
    if not traces:
        print(f"no *.trace.json.gz under {root}")
        sys.exit(1)
    for t in traces:
        # label = run dir name (e.g. gemm_bf16_4096); path is
        # <run>/plugins/profile/<ts>/<file> -> 4x dirname to reach <run>
        label = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(t)))))
        if not label or label in ("profile", "plugins"):
            label = t
        try:
            ops = parse_one(t)
            summarize(ops, label)
        except Exception as e:
            print(f"  {t}: parse error {e}")


if __name__ == "__main__":
    main()
