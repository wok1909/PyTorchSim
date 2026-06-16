#!/usr/bin/env python3
"""Parse a PyTorchSim run log -> per-kernel cycles / SA util% / BW% (fine windows).
Usage: python parse_kernels.py <log> [label]
"""
import re, sys
FREQ = 3502.0; PK_BW = 1600.0; REQ = 32


def parse(path):
    L = open(path).read().splitlines()
    op_by = {}
    for ln in L:
        m = re.search(r"Kernel (\d+) has completed.*operation: (\S+)", ln)
        if m:
            op_by[int(m.group(1))] = m.group(2)
    kers = []
    for ln in L:
        m = re.search(r"Kernel (\d+) execution summary - Started at: (\d+) cycles, Total compute time: (\d+) cycles", ln)
        if m:
            kid, st, cp = int(m.group(1)), int(m.group(2)), int(m.group(3))
            kers.append((kid, op_by.get(kid, "?"), st, st + cp, cp))
    wins = []; sa = resp = None; prev = 0
    for ln in L:
        m = re.search(r"Systolic array \[0\] utilization\(%\)\s*[\d.]+, active_cycles (\d+)", ln)
        if m:
            sa = int(m.group(1))
        m = re.search(r"DRAM BW [\d.]+ GB/s \((\d+) responses\)", ln)
        if m:
            resp = int(m.group(1))
        m = re.search(r"Total_cycles (\d+)", ln)
        if m and resp is not None:
            e = int(m.group(1)); wins.append((prev, e, resp, sa or 0)); prev = e; resp = sa = None
    tot = re.findall(r"Total execution cycles:\s*(\d+)", "\n".join(L))
    return sorted(kers), wins, (int(tot[-1]) if tot else 0)


def main():
    path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path
    kers, wins, total = parse(path)
    print(f"\n===== {label} =====  kernels={len(kers)}  total={total} cyc = {total/FREQ:.2f} us")
    print(f"{'k':>2} {'op':34} {'cyc':>7} {'us':>6} {'SA%':>5} {'BW%':>5}")
    sa_w = bw_w = 0.0
    for kid, op, st, en, cp in kers:
        if cp == 0:
            print(f"{kid:>2} {op[:34]:34} {cp:>7} {'-':>6} {'-':>5} {'-':>5}"); continue
        b = saa = 0.0
        for ws, we, r, s in wins:
            ov = max(0, min(en, we) - max(st, ws)); span = we - ws
            if span > 0:
                b += r * ov / span; saa += s * ov / span
        us = cp / FREQ; bw = b * REQ / (us * 1e-6) / 1e9
        sa_pct = 100 * saa / cp; bw_pct = 100 * bw / PK_BW
        sa_w += saa; bw_w += b * REQ
        print(f"{kid:>2} {op[:34]:34} {cp:>7} {us:>6.2f} {sa_pct:>5.1f} {bw_pct:>5.1f}")
    # aggregate over full run
    if total:
        agg_sa = 100 * sa_w / total
        agg_bw = 100 * (bw_w / (total / FREQ * 1e-6) / 1e9) / PK_BW
        print(f"   {'AGGREGATE (whole run)':34} {total:>7} {total/FREQ:>6.2f} {agg_sa:>5.1f} {agg_bw:>5.1f}")


if __name__ == "__main__":
    main()
