from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from fouroversix import (
    DataType,
    QuantizationConfig,
    QuantizeBackend,
    RoundStyle,
    ScaleRule,
    quantize,
)
from fouroversix.quantize import from_blocked
from fouroversix.quantize.frontend import AVAILABLE_BACKENDS
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.utils import SM_100


FOUR_OVER_SIX_DTYPES = [DataType.nvfp4, DataType.nvfp4_bs8]
SCALE_RULES = [
    ScaleRule.abs_max,
    ScaleRule.mae,
    ScaleRule.mse,
    ScaleRule.static_4,
    ScaleRule.static_6,
]
ADAPTIVE_SCALE_RULES = {
    ScaleRule.abs_max,
    ScaleRule.mae,
    ScaleRule.mse,
}
ROUND_STYLES = [
    RoundStyle.nearest,
    RoundStyle.stochastic,
    RoundStyle.stochastic_unbiased,
]
SHAPES = [(128, 256), (1024, 1024)]
INPUT_KINDS = ["zeros", "ones", "rand01", "randn"]


@dataclass(frozen=True)
class Case:
    dtype: str
    scale_rule: str
    round_style: str
    block_scale_2d: bool
    transpose: bool
    rht: bool
    pseudo_quantize: bool
    shape: tuple[int, int]
    input_kind: str
    seed: int


def make_input(shape: tuple[int, int], input_kind: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if input_kind == "zeros":
        return torch.zeros(*shape, dtype=torch.bfloat16, device="cuda")
    if input_kind == "ones":
        return torch.ones(*shape, dtype=torch.bfloat16, device="cuda")
    if input_kind == "rand01":
        return torch.randint(0, 2, shape, dtype=torch.int32, device="cuda").to(
            torch.bfloat16,
        )
    if input_kind == "randn":
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    msg = f"unknown input kind {input_kind}"
    raise ValueError(msg)


def normalized_scale_bytes(tensor: QuantizedTensor) -> torch.Tensor:
    if tensor.scale_factors is None:
        return torch.empty(0, dtype=torch.uint8, device=tensor.values.device)
    rows = tensor.padded_shape[0]
    cols = tensor.scale_factors.numel() // rows
    if tensor.scale_factors_are_in_blackwell_layout:
        scales = from_blocked(tensor.scale_factors, (rows, cols))
    else:
        scales = tensor.scale_factors.reshape(rows, cols)
    useful_rows = tensor.original_shape[0]
    useful_cols = min(tensor.original_shape[1] // tensor.dtype.block_size, cols)
    return scales[:useful_rows, :useful_cols].contiguous().view(torch.uint8)


def useful_value_bytes(tensor: QuantizedTensor) -> torch.Tensor:
    cols = tensor.original_shape[1] // tensor.dtype.quantized_value_type.packing_factor
    return tensor.values[: tensor.original_shape[0], :cols].contiguous()


def first_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape)}
    mismatch = a != b
    if not mismatch.any().item():
        return None
    flat_idx = int(torch.nonzero(mismatch.flatten(), as_tuple=False)[0].item())
    idx = list(torch.unravel_index(torch.tensor(flat_idx, device=a.device), a.shape))
    idx = [int(i.item()) for i in idx]
    return {
        "index": idx,
        "a": int(a[tuple(idx)].item()),
        "b": int(b[tuple(idx)].item()),
    }


def tensor_equal_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return 0.0
    if a.numel() == 0:
        return 1.0
    return float((a == b).float().mean().item())


def compare_quantized(
    cute: QuantizedTensor,
    expected: QuantizedTensor,
) -> dict[str, Any]:
    cute_values = useful_value_bytes(cute)
    expected_values = useful_value_bytes(expected)
    cute_scales = normalized_scale_bytes(cute)
    expected_scales = normalized_scale_bytes(expected)

    cute_amax = (
        cute.amax.reshape(1).view(torch.uint8) if cute.amax is not None else None
    )
    expected_amax = (
        expected.amax.reshape(1).view(torch.uint8)
        if expected.amax is not None
        else None
    )
    if cute_amax is None and expected_amax is None:
        amax_equal = True
        amax_first_mismatch = None
    elif cute_amax is None or expected_amax is None:
        amax_equal = False
        amax_first_mismatch = {
            "cute": None if cute_amax is None else cute_amax.cpu().tolist(),
            "expected": None
            if expected_amax is None
            else expected_amax.cpu().tolist(),
        }
    else:
        amax_equal = torch.equal(cute_amax, expected_amax)
        amax_first_mismatch = first_mismatch(cute_amax, expected_amax)

    values_dtype_equal = cute.values.dtype == expected.values.dtype
    if cute.scale_factors is None or expected.scale_factors is None:
        scales_dtype_equal = (
            cute.scale_factors is None and expected.scale_factors is None
        )
    else:
        scales_dtype_equal = cute.scale_factors.dtype == expected.scale_factors.dtype
    if cute.amax is None or expected.amax is None:
        amax_dtype_equal = cute.amax is None and expected.amax is None
    else:
        amax_dtype_equal = cute.amax.dtype == expected.amax.dtype

    values_equal = torch.equal(cute_values, expected_values)
    scales_equal = torch.equal(cute_scales, expected_scales)
    return {
        "bitwise_equal": bool(
            amax_equal
            and amax_dtype_equal
            and values_equal
            and values_dtype_equal
            and scales_equal
            and scales_dtype_equal
        ),
        "amax_equal": bool(amax_equal),
        "amax_dtype_equal": bool(amax_dtype_equal),
        "values_equal": bool(values_equal),
        "values_dtype_equal": bool(values_dtype_equal),
        "scales_equal": bool(scales_equal),
        "scales_dtype_equal": bool(scales_dtype_equal),
        "values_equal_ratio": tensor_equal_ratio(cute_values, expected_values),
        "scales_equal_ratio": tensor_equal_ratio(cute_scales, expected_scales),
        "values_first_mismatch": first_mismatch(cute_values, expected_values),
        "scales_first_mismatch": first_mismatch(cute_scales, expected_scales),
        "amax_first_mismatch": amax_first_mismatch,
    }


def unsupported_reason(
    x: torch.Tensor,
    cute_config: QuantizationConfig,
    expected_config: QuantizationConfig,
) -> str | None:
    cute_backend = AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100]
    expected_backend = AVAILABLE_BACKENDS[expected_config.backend]
    if not cute_backend.can_quantize(x, cute_config):
        return "cute_sm100_not_supported"
    if not expected_backend.can_quantize(x, expected_config):
        return f"{expected_config.backend.value}_not_supported"
    return None


def expected_backend_for(dtype: DataType, scale_rule: ScaleRule) -> QuantizeBackend:
    if dtype == DataType.nvfp4_bs8 and scale_rule in ADAPTIVE_SCALE_RULES:
        return QuantizeBackend.pytorch
    return QuantizeBackend.triton


def iter_cases(*, quick: bool, include_stochastic: bool) -> list[Case]:
    dtypes = FOUR_OVER_SIX_DTYPES
    shapes = [(128, 256)] if quick else SHAPES
    input_kinds = ["randn"] if quick else INPUT_KINDS
    round_styles = ROUND_STYLES if include_stochastic else [RoundStyle.nearest]
    cases = []
    for dtype in dtypes:
        for scale_rule in SCALE_RULES:
            for round_style in round_styles:
                for block_scale_2d in [False, True]:
                    for transpose in [False, True]:
                        for rht in [False, True]:
                            if block_scale_2d and transpose:
                                continue
                            for shape in shapes:
                                for input_kind in input_kinds:
                                    cases.append(
                                        Case(
                                            dtype=dtype.value,
                                            scale_rule=scale_rule.value,
                                            round_style=round_style.value,
                                            block_scale_2d=block_scale_2d,
                                            transpose=transpose,
                                            rht=rht,
                                            pseudo_quantize=False,
                                            shape=shape,
                                            input_kind=input_kind,
                                            seed=0,
                                        )
                                    )
    return cases


def run_case(case: Case) -> dict[str, Any]:
    dtype = DataType(case.dtype)
    scale_rule = ScaleRule(case.scale_rule)
    round_style = RoundStyle(case.round_style)
    x = make_input(case.shape, case.input_kind, case.seed)
    expected_backend = expected_backend_for(dtype, scale_rule)
    result: dict[str, Any] = {
        "case": asdict(case),
        "comparison_backend": expected_backend.value,
    }
    if expected_backend == QuantizeBackend.pytorch and case.rht:
        result["status"] = "skipped"
        result["reason"] = "rht_preprocess_is_outside_four_over_six_quant_dequant"
        return result

    expected_x = x.clone()
    expected_transpose = case.transpose
    expected_rht = case.rht
    if expected_backend == QuantizeBackend.pytorch:
        expected_x = x.T.contiguous() if case.transpose else expected_x
        expected_transpose = False
        expected_rht = False

    cute_config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        block_scale_2d=case.block_scale_2d,
        dtype=dtype,
        rht=case.rht,
        round_style=round_style,
        scale_rule=scale_rule,
        transpose=case.transpose,
    )
    expected_config = QuantizationConfig(
        backend=expected_backend,
        block_scale_2d=case.block_scale_2d,
        dtype=dtype,
        rht=expected_rht,
        round_style=round_style,
        scale_rule=scale_rule,
        transpose=expected_transpose,
    )

    reason = unsupported_reason(x, cute_config, expected_config)
    if reason is not None:
        result["status"] = "unsupported"
        result["reason"] = reason
        return result

    if round_style.is_stochastic:
        result["status"] = "skipped"
        result["reason"] = "stochastic_round_style_not_bitwise_comparable_without_shared_random_stream"
        return result

    try:
        cute = quantize(x.clone(), cute_config)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "exception"
        result["backend"] = "cute_sm100"
        result["exception_type"] = type(exc).__name__
        result["exception"] = str(exc)
        return result
    try:
        expected = quantize(expected_x, expected_config)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "exception"
        result["backend"] = expected_backend.value
        result["exception_type"] = type(exc).__name__
        result["exception"] = str(exc)
        return result
    cute_values = useful_value_bytes(cute)
    expected_values = useful_value_bytes(expected)
    if cute_values.shape != expected_values.shape:
        result["status"] = "skipped"
        result["reason"] = "raw_value_layout_not_bitwise_comparable"
        result["cute_values_shape"] = list(cute_values.shape)
        result["expected_values_shape"] = list(expected_values.shape)
        return result
    result["status"] = "compared"
    result.update(compare_quantized(cute, expected))
    return result


def write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    compared = [r for r in results if r["status"] == "compared"]
    triton_compared = [
        r
        for r in compared
        if r.get("comparison_backend") == QuantizeBackend.triton.value
    ]
    reference_compared = [
        r
        for r in compared
        if r.get("comparison_backend") == QuantizeBackend.pytorch.value
    ]
    mismatches = [r for r in compared if not r["bitwise_equal"]]
    unsupported = [r for r in results if r["status"] == "unsupported"]
    skipped = [r for r in results if r["status"] == "skipped"]
    exceptions = [r for r in results if r["status"] == "exception"]
    lines = [
        "# SM100 CuTe Four-Over-Six Bitwise Parity Probe",
        "",
        f"- Compared cases: {len(compared)}",
        f"- Compared against unmodified Triton: {len(triton_compared)}",
        f"- Compared against PyTorch reference: {len(reference_compared)}",
        f"- Bitwise equal: {len(compared) - len(mismatches)}",
        f"- Bitwise mismatches: {len(mismatches)}",
        f"- Unsupported by one backend: {len(unsupported)}",
        f"- Skipped: {len(skipped)}",
        f"- Exceptions: {len(exceptions)}",
        "",
    ]
    if mismatches:
        lines.extend(["## First Mismatches", ""])
        for result in mismatches[:50]:
            case = result["case"]
            lines.append(
                "- "
                + ", ".join(
                    [
                        f"dtype={case['dtype']}",
                        f"backend={result['comparison_backend']}",
                        f"scale_rule={case['scale_rule']}",
                        f"round_style={case['round_style']}",
                        f"block_scale_2d={case['block_scale_2d']}",
                        f"transpose={case['transpose']}",
                        f"rht={case['rht']}",
                        f"shape={tuple(case['shape'])}",
                        f"input={case['input_kind']}",
                        f"values={result['values_equal_ratio']:.9f}",
                        f"scales={result['scales_equal_ratio']:.9f}",
                        f"values_first={result['values_first_mismatch']}",
                        f"scales_first={result['scales_first_mismatch']}",
                        f"amax_first={result['amax_first_mismatch']}",
                    ]
                )
            )
        lines.append("")
    if exceptions:
        lines.extend(["## Exceptions", ""])
        for result in exceptions[:50]:
            case = result["case"]
            lines.append(
                "- "
                + ", ".join(
                    [
                        f"backend={result['backend']}",
                        f"exception={result['exception_type']}",
                        f"dtype={case['dtype']}",
                        f"scale_rule={case['scale_rule']}",
                        f"round_style={case['round_style']}",
                        f"block_scale_2d={case['block_scale_2d']}",
                        f"transpose={case['transpose']}",
                        f"rht={case['rht']}",
                        f"shape={tuple(case['shape'])}",
                        f"input={case['input_kind']}",
                    ]
                )
            )
        lines.append("")
    if skipped:
        lines.extend(["## Skipped", ""])
        for result in skipped[:50]:
            case = result["case"]
            lines.append(
                "- "
                + ", ".join(
                    [
                        f"reason={result['reason']}",
                        f"dtype={case['dtype']}",
                        f"scale_rule={case['scale_rule']}",
                        f"round_style={case['round_style']}",
                        f"block_scale_2d={case['block_scale_2d']}",
                        f"transpose={case['transpose']}",
                        f"rht={case['rht']}",
                        f"shape={tuple(case['shape'])}",
                        f"input={case['input_kind']}",
                    ]
                )
            )
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-stochastic", action="store_true")
    parser.add_argument("--json-out", type=Path, default=Path("latest.json"))
    parser.add_argument("--md-out", type=Path, default=Path("latest.md"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] != SM_100:
        raise RuntimeError(
            f"SM100 is required, got {torch.cuda.get_device_capability()}"
        )

    if not AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].is_available():
        raise RuntimeError("cute_sm100 backend is not available")
    if not AVAILABLE_BACKENDS[QuantizeBackend.triton].is_available():
        raise RuntimeError("triton backend is not available")

    results = []
    for idx, case in enumerate(
        iter_cases(quick=args.quick, include_stochastic=args.include_stochastic),
        start=1,
    ):
        result = run_case(case)
        results.append(result)
        if result["status"] == "compared" and not result["bitwise_equal"]:
            print(f"mismatch {idx}: {result['case']} {result['values_equal_ratio']:.6f}")
        elif result["status"] == "exception":
            print(
                f"exception {idx}: {result['backend']} {result['exception_type']} "
                f"{result['case']}"
            )
        elif idx % 100 == 0:
            print(f"processed {idx}")

    payload = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(results, args.md_out)

    compared = [r for r in results if r["status"] == "compared"]
    mismatches = [r for r in compared if not r["bitwise_equal"]]
    print(
        f"compared={len(compared)} mismatches={len(mismatches)} "
        f"triton={sum(r.get('comparison_backend') == QuantizeBackend.triton.value for r in compared)} "
        f"reference={sum(r.get('comparison_backend') == QuantizeBackend.pytorch.value for r in compared)} "
        f"unsupported={sum(r['status'] == 'unsupported' for r in results)} "
        f"skipped={sum(r['status'] == 'skipped' for r in results)} "
        f"exceptions={sum(r['status'] == 'exception' for r in results)}"
    )


if __name__ == "__main__":
    main()
