from __future__ import annotations

import json
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CuTe sm100 benchmark")

    torch.manual_seed(0)
    rows = []
    triton_backend = AVAILABLE_BACKENDS[QuantizeBackend.triton]
    cute_backend = AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100]
    for shape in SHAPES:
        x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        iters = 200 if shape[0] <= 1024 else 50
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

                    triton_ms = time_ms(
                        lambda: quantize(x, config_triton),
                        iters=iters,
                    )
                    cute_ms = time_ms(
                        lambda: quantize(x, config_cute),
                        iters=iters,
                    )
                    rows.append(
                        {
                            "shape": shape,
                            "dtype": dtype.value,
                            "scale_rule": scale_rule.value,
                            "features": feature_kwargs,
                            "triton_ms": triton_ms,
                            "cute_sm100_ms": cute_ms,
                            "speedup_vs_triton": triton_ms / cute_ms,
                            "meets_1_2x": triton_ms / cute_ms >= 1.2,
                        },
                    )

    output = {
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": rows,
    }
    out_path = Path(__file__).with_name("benchmark_current.json")
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(out_path)
    failures = [row for row in rows if not row["meets_1_2x"]]
    print(f"{len(rows) - len(failures)}/{len(rows)} workloads meet 1.2x")
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
