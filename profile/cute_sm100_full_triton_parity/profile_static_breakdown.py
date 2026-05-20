from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
from fouroversix import QuantizationConfig, QuantizeBackend, quantize
from fouroversix.kernels import triton as triton_kernels
from fouroversix.kernels.cute_sm100 import ops as cute_ops
from fouroversix.quantize import QuantizedTensor
from fouroversix.utils import DataType, RoundStyle, ScaleRule


def time_ms(fn: Callable[[], object], *, iters: int, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def median_ms(fn: Callable[[], object], *, iters: int, repeats: int) -> tuple[float, list[float]]:
    samples = [time_ms(fn, iters=iters, warmup=10) for _ in range(repeats)]
    return statistics.median(samples), samples


def quantized_tensor(
    values: torch.Tensor,
    scale_factors: torch.Tensor,
    amax: torch.Tensor,
    dtype: DataType,
    shape: tuple[int, int],
    *,
    blackwell_layout: bool,
) -> QuantizedTensor:
    return QuantizedTensor(
        values,
        scale_factors,
        amax,
        dtype,
        shape,
        ScaleRule.static_6,
        RoundStyle.nearest,
        scale_factors_are_in_blackwell_layout=blackwell_layout,
    )


def cute_static(
    x: torch.Tensor,
    dtype: DataType,
    x_amax: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype == DataType.nvfp4:
        return cute_ops.quantize_nvfp4_static(
            x,
            max_quantized_value=6,
            x_amax=x_amax,
        )
    if dtype == DataType.nvfp6_e2m3:
        return cute_ops.quantize_nvfp6_static(
            x,
            max_quantized_value=7.5,
            use_e3m2=False,
            x_amax=x_amax,
        )
    if dtype == DataType.nvfp6_e3m2:
        return cute_ops.quantize_nvfp6_static(
            x,
            max_quantized_value=28.0,
            use_e3m2=True,
            x_amax=x_amax,
        )
    if dtype == DataType.nvfp3:
        return cute_ops.quantize_nvfp3_static(
            x,
            scale_block_size=dtype.block_size,
            x_amax=x_amax,
        )
    msg = f"unsupported dtype for this breakdown: {dtype.value}"
    raise ValueError(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="nvfp4")
    parser.add_argument("--shape", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    dtype = DataType(args.dtype)
    torch.manual_seed(0)
    x = torch.randn(args.shape, args.shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )

    x_amax = x.abs().max().float()
    cute_values, cute_scales, cute_amax = cute_static(x, dtype, x_amax)
    triton_values, triton_scales, triton_amax = triton_kernels.quantize(
        x,
        x_amax=x_amax,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )
    torch.cuda.synchronize()

    cases: list[tuple[str, Callable[[], object]]] = [
        ("amax_only", lambda: x.abs().max().float()),
        ("triton_lowlevel_with_amax", lambda: triton_kernels.quantize(
            x,
            x_amax=x_amax,
            dtype=dtype,
            scale_rule=ScaleRule.static_6,
        )),
        ("cute_lowlevel_with_amax", lambda: cute_static(x, dtype, x_amax)),
        ("triton_lowlevel_auto_amax", lambda: triton_kernels.quantize(
            x,
            dtype=dtype,
            scale_rule=ScaleRule.static_6,
        )),
        ("cute_lowlevel_auto_amax", lambda: cute_static(x, dtype, None)),
        ("triton_quantized_tensor", lambda: quantized_tensor(
            triton_values,
            triton_scales,
            triton_amax,
            dtype,
            x.shape,
            blackwell_layout=True,
        )),
        ("cute_quantized_tensor_rowmajor", lambda: quantized_tensor(
            cute_values,
            cute_scales,
            cute_amax,
            dtype,
            x.shape,
            blackwell_layout=False,
        )),
        ("triton_frontend", lambda: quantize(x, config_triton)),
        ("cute_frontend", lambda: quantize(x, config_cute)),
    ]

    print(f"dtype={dtype.value} shape={x.shape} iters={args.iters} repeats={args.repeats}")
    for name, fn in cases:
        median, samples = median_ms(fn, iters=args.iters, repeats=args.repeats)
        sample_text = ", ".join(f"{sample:.5f}" for sample in samples)
        print(f"{name}: median={median:.5f}ms samples=[{sample_text}]")

    print(
        "shapes:",
        f"triton_values={tuple(triton_values.shape)}",
        f"triton_scales={tuple(triton_scales.shape)}",
        f"cute_values={tuple(cute_values.shape)}",
        f"cute_scales={tuple(cute_scales.shape)}",
    )


if __name__ == "__main__":
    main()
