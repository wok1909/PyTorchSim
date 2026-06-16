import argparse
import json
import math
import os
import sys
import time
from typing import Iterable, Sequence

import torch

BASE_DIR = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Simulator.simulator import TOGSimulator


OPT125M = {
    "hidden_size": 768,
    "num_attention_heads": 12,
    "head_dim": 64,
    "ffn_dim": 3072,
    "vocab_size": 50272,
    "num_hidden_layers": 12,
    "max_position_embeddings": 2048,
}

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{name}'. Use one of: {', '.join(DTYPE_MAP.keys())}")
    return DTYPE_MAP[key]


def add_common_bench_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-path", type=str, default=None)
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _to_device_args(inputs: Sequence[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [x.to(device=device) for x in inputs]


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_vals[low]
    frac = rank - low
    return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac


def summarize_ms(times_ms: Iterable[float]) -> dict:
    vals = list(times_ms)
    vals_sorted = sorted(vals)
    return {
        "count": len(vals),
        "mean_ms": sum(vals) / max(len(vals), 1),
        "min_ms": vals_sorted[0] if vals_sorted else float("nan"),
        "max_ms": vals_sorted[-1] if vals_sorted else float("nan"),
        "p50_ms": _percentile(vals_sorted, 0.50),
        "p90_ms": _percentile(vals_sorted, 0.90),
    }


def benchmark_compiled_module(
    module: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    config_path: str | None,
) -> dict:
    module_dtype = None
    for x in inputs:
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            module_dtype = x.dtype
            break

    if module_dtype is not None:
        module = module.to(device=device, dtype=module_dtype).eval()
    else:
        module = module.to(device=device).eval()
    compiled = torch.compile(module, dynamic=False)
    dev_inputs = _to_device_args(inputs, device)

    times_ms = []
    with torch.no_grad():
        with TOGSimulator(config_path=config_path):
            for _ in range(max(warmup, 0)):
                torch.npu.launch_model(compiled, *dev_inputs, stream_index=0, timestamp=0)
                torch.npu.synchronize()

            for _ in range(max(repeats, 1)):
                t0 = time.perf_counter()
                torch.npu.launch_model(compiled, *dev_inputs, stream_index=0, timestamp=0)
                torch.npu.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    return summarize_ms(times_ms)


def print_result(stage: str, op: str, args: argparse.Namespace, stats: dict, extra: dict | None = None) -> None:
    payload = {
        "model": "opt-125m",
        "stage": stage,
        "op": op,
        "batch_size": args.batch_size,
        "seq_len": getattr(args, "seq_len", None),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "stats": stats,
    }
    if extra:
        payload.update(extra)
    print(json.dumps(payload, indent=2))
