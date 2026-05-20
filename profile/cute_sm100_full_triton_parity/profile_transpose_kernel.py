from __future__ import annotations

import argparse

import torch
from fouroversix import QuantizationConfig, QuantizeBackend, quantize
from fouroversix.utils import DataType, ScaleRule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["triton", "cute_sm100"], required=True)
    parser.add_argument("--dtype", default="mxfp4")
    parser.add_argument("--scale-rule", default="static_6")
    parser.add_argument("--shape", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--transpose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pseudo-quantize", action="store_true")
    parser.add_argument("--block-scale-2d", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(
        args.shape,
        args.shape,
        dtype=torch.bfloat16,
        device="cuda",
    )
    config = QuantizationConfig(
        backend=QuantizeBackend(args.backend),
        dtype=DataType(args.dtype),
        scale_rule=ScaleRule(args.scale_rule),
        transpose=args.transpose,
        pseudo_quantize=args.pseudo_quantize,
        block_scale_2d=args.block_scale_2d,
    )

    for _ in range(10):
        quantize(x, config)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        quantize(x, config)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
