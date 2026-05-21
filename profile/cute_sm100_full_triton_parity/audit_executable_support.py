from __future__ import annotations

import json
from pathlib import Path

import torch
from fouroversix import QuantizationConfig, QuantizeBackend, quantize
from fouroversix.quantize.frontend import AVAILABLE_BACKENDS
from fouroversix.utils import DataType, RoundStyle


SHAPE = (128, 256)
FEATURES = [
    {},
    {"transpose": True},
    {"pseudo_quantize": True},
    {"block_scale_2d": True},
]
ROUND_STYLES = [
    RoundStyle.nearest,
    RoundStyle.stochastic,
    RoundStyle.stochastic_unbiased,
]


def _runnable(
    x: torch.Tensor,
    backend: QuantizeBackend,
    config: QuantizationConfig,
) -> tuple[bool, str | None]:
    if not AVAILABLE_BACKENDS[backend].can_quantize(x, config):
        return False, None
    try:
        quantize(x, config)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - this is an executable support audit.
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return True, None


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the support audit")

    torch.manual_seed(0)
    x = torch.randn(*SHAPE, dtype=torch.bfloat16, device="cuda")
    triton_predicate_failures = []
    cute_missing_runnable = []
    cute_predicate_failures = []
    triton_predicate_count = 0
    triton_runnable_count = 0
    cute_predicate_count = 0
    cute_runnable_count = 0

    for dtype in [dtype for dtype in DataType if dtype.torch_dtype is None]:
        for scale_rule in sorted(dtype.supported_scale_rules, key=lambda rule: rule.value):
            for feature_kwargs in FEATURES:
                if feature_kwargs.get("transpose") and SHAPE[0] != SHAPE[1]:
                    continue
                if feature_kwargs.get("block_scale_2d") and SHAPE[0] % 16 != 0:
                    continue
                for round_style in ROUND_STYLES:
                    configs = {
                        backend: QuantizationConfig(
                            backend=backend,
                            dtype=dtype,
                            scale_rule=scale_rule,
                            round_style=round_style,
                            **feature_kwargs,
                        )
                        for backend in (
                            QuantizeBackend.triton,
                            QuantizeBackend.cute_sm100,
                        )
                    }
                    predicates = {
                        backend: AVAILABLE_BACKENDS[backend].can_quantize(x, config)
                        for backend, config in configs.items()
                    }
                    triton_predicate_count += int(predicates[QuantizeBackend.triton])
                    cute_predicate_count += int(predicates[QuantizeBackend.cute_sm100])
                    triton_runs, triton_error = _runnable(
                        x,
                        QuantizeBackend.triton,
                        configs[QuantizeBackend.triton],
                    )
                    cute_runs, cute_error = _runnable(
                        x,
                        QuantizeBackend.cute_sm100,
                        configs[QuantizeBackend.cute_sm100],
                    )
                    triton_runnable_count += int(triton_runs)
                    cute_runnable_count += int(cute_runs)
                    row = {
                        "dtype": dtype.value,
                        "scale_rule": scale_rule.value,
                        "round_style": round_style.value,
                        "features": feature_kwargs,
                    }
                    if predicates[QuantizeBackend.triton] and not triton_runs:
                        triton_predicate_failures.append({**row, "error": triton_error})
                    if triton_runs and not predicates[QuantizeBackend.cute_sm100]:
                        cute_missing_runnable.append(row)
                    if predicates[QuantizeBackend.cute_sm100] and not cute_runs:
                        cute_predicate_failures.append({**row, "error": cute_error})

    output = {
        "shape": SHAPE,
        "triton_predicate_count": triton_predicate_count,
        "triton_runnable_count": triton_runnable_count,
        "cute_predicate_count": cute_predicate_count,
        "cute_runnable_count": cute_runnable_count,
        "triton_predicate_failures": triton_predicate_failures,
        "cute_missing_runnable": cute_missing_runnable,
        "cute_predicate_failures": cute_predicate_failures,
    }
    out_path = Path(__file__).with_name("executable_support_current.json")
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(out_path)
    print(
        "triton_predicate",
        triton_predicate_count,
        "triton_runnable",
        triton_runnable_count,
        "cute_predicate",
        cute_predicate_count,
        "cute_runnable",
        cute_runnable_count,
    )
    print("triton_predicate_failures", len(triton_predicate_failures))
    print("cute_missing_runnable", len(cute_missing_runnable))
    print("cute_predicate_failures", len(cute_predicate_failures))


if __name__ == "__main__":
    main()
