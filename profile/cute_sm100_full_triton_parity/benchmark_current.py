from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from fouroversix import QuantizationConfig, QuantizeBackend, quantize
from fouroversix.quantize.frontend import AVAILABLE_BACKENDS
from fouroversix.utils import DataType, ScaleRule


SHAPES = [(128, 256), (1024, 1024), (4096, 4096)]
DTypeScaleRules = {
    DataType.if3: [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
    ],
    DataType.if3_bs8: [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
    ],
    DataType.if4: [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
    ],
    DataType.if4_bs8: [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
    ],
    DataType.if6_e2m3: [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
    ],
    DataType.if6_e3m2: [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
    ],
    DataType.nvfp4: [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
    DataType.nvfp4_bs8: [ScaleRule.static_4, ScaleRule.static_6],
    DataType.nvfp3: [ScaleRule.static_6],
    DataType.nvfp3_bs8: [ScaleRule.static_6],
    DataType.mxfp3: [ScaleRule.static_4, ScaleRule.static_6],
    DataType.mxfp3_bs8: [ScaleRule.static_4, ScaleRule.static_6],
    DataType.mxfp4: [ScaleRule.static_4, ScaleRule.static_6],
    DataType.mxfp4_bs8: [ScaleRule.static_4, ScaleRule.static_6],
    DataType.mxfp6_e2m3: [ScaleRule.static_4, ScaleRule.static_6],
    DataType.mxfp6_e3m2: [ScaleRule.static_4, ScaleRule.static_6],
    DataType.nvint3: [ScaleRule.static_6],
    DataType.nvint3_bs8: [ScaleRule.static_6],
    DataType.nvint4: [ScaleRule.static_6],
    DataType.nvint4_bs8: [ScaleRule.static_6],
    DataType.nvint6: [ScaleRule.static_6],
    DataType.nvfp6_e2m3: [ScaleRule.static_6],
    DataType.nvfp6_e3m2: [ScaleRule.static_6],
}
FEATURES = [
    {},
    {"transpose": True},
    {"pseudo_quantize": True},
    {"block_scale_2d": True},
]
REPEATS = 9


def time_ms(fn, *, iters: int, warmup: int = 20) -> float:
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


def compare_ms(
    triton_fn,
    cute_fn,
    *,
    iters: int,
    repeats: int = REPEATS,
) -> tuple[float, float, list[float], list[float]]:
    for _ in range(3):
        triton_fn()
        cute_fn()
    torch.cuda.synchronize()

    triton_samples = []
    cute_samples = []
    for repeat_idx in range(repeats):
        if repeat_idx % 2 == 0:
            triton_samples.append(time_ms(triton_fn, iters=iters, warmup=10))
            cute_samples.append(time_ms(cute_fn, iters=iters, warmup=10))
        else:
            cute_samples.append(time_ms(cute_fn, iters=iters, warmup=10))
            triton_samples.append(time_ms(triton_fn, iters=iters, warmup=10))

    return (
        statistics.median(triton_samples),
        statistics.median(cute_samples),
        triton_samples,
        cute_samples,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CuTe sm100 benchmark")

    torch.manual_seed(0)
    rows = []
    triton_backend = AVAILABLE_BACKENDS[QuantizeBackend.triton]
    cute_backend = AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100]
    for shape in SHAPES:
        x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        iters = 200 if shape[0] <= 1024 else 100
        for dtype, scale_rules in DTypeScaleRules.items():
            for scale_rule in scale_rules:
                for feature_kwargs in FEATURES:
                    if feature_kwargs.get("block_scale_2d") and shape[0] % 16 != 0:
                        continue
                    if feature_kwargs.get("transpose") and shape[0] != shape[1]:
                        continue

                    config_triton = QuantizationConfig(
                        backend=QuantizeBackend.triton,
                        dtype=dtype,
                        scale_rule=scale_rule,
                        **feature_kwargs,
                    )
                    config_cute = QuantizationConfig(
                        backend=QuantizeBackend.cute_sm100,
                        dtype=dtype,
                        scale_rule=scale_rule,
                        **feature_kwargs,
                    )

                    if not triton_backend.can_quantize(x, config_triton):
                        continue
                    if not cute_backend.can_quantize(x, config_cute):
                        continue

                    triton_fn = lambda: quantize(x, config_triton)
                    cute_fn = lambda: quantize(x, config_cute)
                    triton_ms, cute_ms, triton_samples, cute_samples = compare_ms(
                        triton_fn,
                        cute_fn,
                        iters=iters,
                    )
                    speedup = triton_ms / cute_ms
                    required_speedup = (
                        1.0 if feature_kwargs.get("pseudo_quantize") else 1.2
                    )
                    rows.append(
                        {
                            "shape": shape,
                            "dtype": dtype.value,
                            "scale_rule": scale_rule.value,
                            "features": feature_kwargs,
                            "triton_ms": triton_ms,
                            "cute_sm100_ms": cute_ms,
                            "triton_samples_ms": triton_samples,
                            "cute_sm100_samples_ms": cute_samples,
                            "speedup_vs_triton": speedup,
                            "required_speedup": required_speedup,
                            "meets_1_2x": speedup >= 1.2,
                            "meets_required_target": speedup >= required_speedup,
                        },
                    )

    output = {
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing": {
            "method": "median of alternating-order repeats",
            "repeats": REPEATS,
            "target": "non-pseudo rows require >=1.2x; pseudo_quantize rows require >1.0x",
        },
        "rows": rows,
    }
    out_path = Path(__file__).with_name("benchmark_current.json")
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(out_path)
    failures = [row for row in rows if not row["meets_required_target"]]
    print(f"{len(rows) - len(failures)}/{len(rows)} workloads meet required target")
    for row in failures:
        print(
            row["shape"],
            row["dtype"],
            row["scale_rule"],
            row["features"],
            f"triton={row['triton_ms']:.4f}ms",
            f"cute={row['cute_sm100_ms']:.4f}ms",
            f"speedup={row['speedup_vs_triton']:.3f}x",
        )


if __name__ == "__main__":
    main()
