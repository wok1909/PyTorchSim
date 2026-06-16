import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common import OPT125M, add_common_bench_args, benchmark_compiled_module, parse_dtype, print_result, set_seed


class DecodeAttnOp(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, is_causal: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.is_causal = is_causal

    def _to_bhsd(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        qh = self._to_bhsd(q)
        kh = self._to_bhsd(k_cache)
        vh = self._to_bhsd(v_cache)
        out = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=self.is_causal,
        )
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], self.hidden_size)


def main():
    parser = add_common_bench_args(argparse.ArgumentParser(description="OPT-125M DECODE ATTN latency"))
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--is-causal", action="store_true", default=True)
    parser.add_argument("--no-causal", action="store_false", dest="is_causal")
    parser.set_defaults(seq_len=1)
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    set_seed(args.seed)
    device = torch.device("npu:0")

    b, q_len, ctx, h = args.batch_size, args.query_len, args.context_len, OPT125M["hidden_size"]
    q = torch.randn(b, q_len, h, dtype=dtype)
    k = torch.randn(b, ctx, h, dtype=dtype)
    v = torch.randn(b, ctx, h, dtype=dtype)

    module = DecodeAttnOp(hidden_size=h, num_heads=OPT125M["num_attention_heads"], is_causal=args.is_causal)
    stats = benchmark_compiled_module(
        module,
        [q, k, v],
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        config_path=args.config_path,
    )
    print_result(
        "decode",
        "attn",
        args,
        stats,
        extra={"query_len": q_len, "context_len": ctx, "is_causal": args.is_causal},
    )


if __name__ == "__main__":
    main()
