#!/usr/bin/env python3
"""Independently re-extract real-v6e GEMM on-device kernel latency from the raw
TensorBoard trace.json.gz files, to verify the values used in the sim/real chart.
Reports per-iteration device-op time for the compute (matmul) op per size."""
import gzip, json, glob, collections

PLOT_REAL = {512:2.49, 1024:6.85, 2048:29.89, 4096:187.75, 8192:1593.11}  # values under verification

def device_ops(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return [e for e in data["traceEvents"]
            if e.get("ph") == "X" and e.get("pid") == 3 and e.get("tid") == 3 and e.get("dur",0) > 0]

print(f"{'N':>6} {'compute_op':>26} {'#inv':>5} {'sum_us':>10} {'per_iter_us':>12} {'plot_real':>10} {'match?':>8}")
for N in [512,1024,2048,4096,8192]:
    paths = sorted(glob.glob(f"tb_logs/gemm_fp16_{N}/plugins/profile/*/*.trace.json.gz"))
    # merge ops from all captures for this size
    ops = []
    for p in paths: ops += device_ops(p)
    # the compute op = the one with the largest total dur carrying model_flops (matmul/fusion)
    by_name = collections.defaultdict(lambda: [0.0,0])  # name -> [sum_dur, count]
    flops_names = set()
    for e in ops:
        nm = e.get("name","?"); a = e.get("args",{})
        by_name[nm][0] += e["dur"]; by_name[nm][1] += 1
        if int(a.get("model_flops",0) or 0) > 0: flops_names.add(nm)
    # pick compute op: prefer flops-bearing, else max total dur
    cand = [(nm,v[0],v[1]) for nm,v in by_name.items() if nm in flops_names] or \
           [(nm,v[0],v[1]) for nm,v in by_name.items()]
    cand.sort(key=lambda r:-r[1])
    nm,sd,cnt = cand[0]
    per_iter = sd/cnt if cnt else 0
    pr = PLOT_REAL[N]
    err = abs(per_iter-pr)/pr*100 if pr else 0
    print(f"{N:>6} {nm[:26]:>26} {cnt:>5} {sd:>10.2f} {per_iter:>12.2f} {pr:>10.2f} {('OK '+f'{err:.0f}%') if err<10 else ('DIFF '+f'{err:.0f}%'):>8}")
