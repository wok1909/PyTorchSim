import argparse
import os
import sys

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common import OPT125M, add_common_bench_args, benchmark_compiled_module, parse_dtype, print_result, set_seed


class QKVOp(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.wq = nn.Linear(hidden_size, hidden_size, bias=True)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=True)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor):
        return self.wq(x), self.wk(x), self.wv(x)


def main():
    parser = add_common_bench_args(argparse.ArgumentParser(description="OPT-125M DECODE QKV latency"))
    parser.set_defaults(seq_len=1)
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    set_seed(args.seed)
    device = torch.device("npu:0")

    b, s, h = args.batch_size, args.seq_len, OPT125M["hidden_size"]
    x = torch.randn(b, s, h, dtype=dtype)

    module = QKVOp(hidden_size=h)
    stats = benchmark_compiled_module(
        module,
        [x],
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        config_path=args.config_path,
    )
    print_result("decode", "qkv", args, stats)


if __name__ == "__main__":
    main()
