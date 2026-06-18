#!/usr/bin/env python3
"""Full device-op breakdown per GEMM size: fusion vs copy-start/done vs module total.
Resolves whether the real 'fusion' op excludes operand HBM copy."""
import gzip, json, glob, collections

def dev_ops(path):
    with gzip.open(path,"rt") as f: d=json.load(f)
    return [e for e in d["traceEvents"] if e.get("ph")=="X" and e.get("pid")==3 and e.get("dur",0)>0]

for N in [512,2048,8192]:
    paths=sorted(glob.glob(f"tb_logs/gemm_fp16_{N}/plugins/profile/*/*.trace.json.gz"))
    ops=[]
    for p in paths: ops+=dev_ops(p)
    # group by (tid, name) summing dur + count; capture sample args
    agg=collections.defaultdict(lambda:[0.0,0,{}])
    for e in ops:
        k=(e.get("tid"), e.get("name","?"))
        agg[k][0]+=e["dur"]; agg[k][1]+=1
        if not agg[k][2]: agg[k][2]=e.get("args",{})
    print(f"\n===== GEMM fp16 N={N}  ({len(paths)} captures) =====")
    print(f"{'tid':>4} {'op name':>14} {'#':>4} {'sum_us':>9} {'per_iter_us':>11}  {'bytes_accessed':>14} {'flops':>10}")
    for (tid,nm),(sd,c,a) in sorted(agg.items(), key=lambda x:-x[1][0])[:10]:
        by=a.get("bytes_accessed",""); fl=a.get("model_flops","")
        print(f"{str(tid):>4} {nm[:14]:>14} {c:>4} {sd:>9.2f} {sd/c:>11.2f}  {str(by):>14} {str(fl):>10}")
