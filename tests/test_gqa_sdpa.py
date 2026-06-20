import os
import sys
import re
import json
import tempfile
import argparse

import torch
import torch.nn.functional as F
import torch._dynamo
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.append(os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
from Simulator.simulator import TOGSimulator

# GQA latency sweep used for the TPUv6e validation. For each (mode, batch, seq)
# it compiles a fused flash scaled_dot_product_attention on npu:0, runs it under
# TOGSim, and reports the device cycle count. Run with the v6e config to match the
# reference numbers:
#   TOGSIM_CONFIG=$TORCHSIM_DIR/configs/_v6e_vlen512.yml  (absolute path)
#   TORCHSIM_LLVM_PATH=/riscv-llvm/bin
# us = cycles / core_freq_mhz (3502 for _v6e_vlen512.yml).
#
# prefill: q_seq = kv_seq = S, is_causal=True   (causal block-skip in the timing graph)
# decode : q_seq = 1,          is_causal=False  (single decode token, KV length S)
# GQA shape: Hq=5 query heads, Hkv=1 KV head (group g=5), head_dim D=128, fp16.

device = torch.device("npu:0")
FREQ_MHZ = 3502.0
HQ, HKV, D = 5, 1, 128

# Allow many shapes in one process without dynamo recompile-limit fallbacks.
torch._dynamo.config.cache_size_limit = 256


def _cycles_from_log(text):
    m = re.findall(r"Total execution cycles:\s*(\d+)", text)
    return int(m[-1]) if m else None


def run_one(mode, batch, seq):
    """Compile + simulate one GQA attention shape; return device cycles (or None)."""
    q_seq = seq if mode == "prefill" else 1
    causal = (mode == "prefill")
    torch.manual_seed(0)
    q = torch.rand(batch, HQ, q_seq, D, dtype=torch.float16)
    k = torch.rand(batch, HKV, seq, D, dtype=torch.float16)
    v = torch.rand(batch, HKV, seq, D, dtype=torch.float16)

    opt = torch.compile(F.scaled_dot_product_attention, dynamic=False)
    # TOGSim prints "Total execution cycles" via spdlog to the real stderr fd, which
    # a Python-level contextlib.redirect_stdout does NOT capture. Redirect fds 1/2 to
    # a temp file so we catch the C++ simulator output, then restore.
    tf = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt")
    old1, old2 = os.dup(1), os.dup(2)
    try:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(tf.fileno(), 1); os.dup2(tf.fileno(), 2)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            with torch.no_grad(), TOGSimulator():
                out = opt(q.to(device), k.to(device), v.to(device),
                          attn_mask=None, dropout_p=0.0, is_causal=causal, enable_gqa=True)
                torch.npu.synchronize()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os.dup2(old1, 1); os.dup2(old2, 2); os.close(old1); os.close(old2)
    tf.flush(); tf.seek(0)
    log = open(tf.name).read()
    os.unlink(tf.name)
    cyc = _cycles_from_log(log)
    tag = f"{mode}_B{batch}_S{seq}"
    if cyc:
        print(f"  {tag:18} cycles={cyc:>12,}  ({cyc / FREQ_MHZ:8.2f} us)", flush=True)
    else:
        # surface the tail of the captured log so failures are diagnosable
        tail = "\n".join(log.splitlines()[-15:])
        print(f"  {tag:18} cycles=NA (no 'Total execution cycles')\n--- last log lines ---\n{tail}", flush=True)
    return cyc


def run_sweep(modes, batches, seqs, out_json=None):
    res = {}
    if out_json and os.path.exists(out_json):
        res = json.load(open(out_json))
    for mode in modes:
        print(f"=== GQA {mode.upper()} (Hq={HQ} Hkv={HKV} D={D} fp16) ===", flush=True)
        for b in batches:
            for s in seqs:
                key = f"{mode}_B{b}_S{s}"
                if key in res and res[key].get("cycles"):
                    continue
                cyc = run_one(mode, b, s)
                res[key] = {"cycles": cyc, "us": (cyc / FREQ_MHZ if cyc else None)}
                if out_json:
                    json.dump(res, open(out_json, "w"), indent=2)
    # summary table
    print("\n=== cycle table (us = cycles / %.0f) ===" % FREQ_MHZ, flush=True)
    for mode in modes:
        print(f"{mode:>8}" + "".join(f"{('S='+str(s)):>14}" for s in seqs))
        for b in batches:
            row = "".join(
                (f"{res.get(f'{mode}_B{b}_S{s}',{}).get('cycles'):>14,}"
                 if res.get(f"{mode}_B{b}_S{s}", {}).get("cycles") else f"{'--':>14}")
                for s in seqs)
            print(f"{('B='+str(b)):>8}" + row, flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GQA prefill/decode latency (cycle) sweep on npu:0")
    p.add_argument("--mode", type=str, default="both", choices=["prefill", "decode", "both"])
    p.add_argument("--batch", type=int, default=None, help="single batch (default: sweep 1,8,32,128)")
    p.add_argument("--seq_len", type=int, default=None, help="single seq len (default: sweep 512,1024,2048,4096)")
    p.add_argument("--out_json", type=str, default=None, help="optional path to dump per-shape cycles")
    args = p.parse_args()

    modes = ["prefill", "decode"] if args.mode == "both" else [args.mode]
    batches = [args.batch] if args.batch else [1, 8, 32, 128]
    seqs = [args.seq_len] if args.seq_len else [512, 1024, 2048, 4096]

    run_sweep(modes, batches, seqs, out_json=args.out_json)
    print("GQA latency sweep Done", flush=True)
