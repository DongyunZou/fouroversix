from __future__ import annotations

import argparse

import torch

from fouroversix import QuantizationConfig, QuantizeBackend, quantize
from fouroversix.utils import DataType, ScaleRule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["triton", "cute_sm120"], required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--scale-rule", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--block-scale-2d", action="store_true")
    parser.add_argument("--pseudo-quantize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    x = torch.randn(args.rows, args.cols, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend(args.backend),
        dtype=DataType(args.dtype),
        scale_rule=ScaleRule(args.scale_rule),
        transpose=args.transpose,
        block_scale_2d=args.block_scale_2d,
        pseudo_quantize=args.pseudo_quantize,
    )

    for _ in range(8):
        quantize(x, config)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    quantize(x, config)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
