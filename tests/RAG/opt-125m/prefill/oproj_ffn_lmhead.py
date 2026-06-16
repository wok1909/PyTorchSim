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


class OprojFfnLmheadOp(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int, vocab_size: int):
        super().__init__()
        self.oproj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc1 = nn.Linear(hidden_size, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, bias=True)
        self.act = nn.ReLU()
        self.lmhead = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.oproj(x)
        x = self.fc2(self.act(self.fc1(x)))
        return self.lmhead(x)


def main():
    parser = add_common_bench_args(argparse.ArgumentParser(description="OPT-125M PREFILL OPROJ_FFN_LMHEAD latency"))
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    set_seed(args.seed)
    device = torch.device("npu:0")

    b, s, h = args.batch_size, args.seq_len, OPT125M["hidden_size"]
    x = torch.randn(b, s, h, dtype=dtype)

    module = OprojFfnLmheadOp(hidden_size=h, ffn_dim=OPT125M["ffn_dim"], vocab_size=OPT125M["vocab_size"])
    stats = benchmark_compiled_module(
        module,
        [x],
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        config_path=args.config_path,
    )
    print_result("prefill", "oproj_ffn_lmhead", args, stats)


if __name__ == "__main__":
    main()
