from __future__ import annotations

import datetime as dt
import json
import platform
import socket
from pathlib import Path
from typing import Any

import torch
from fouroversix import (
    DataType,
    QuantizationConfig,
    QuantizeBackend,
    RoundStyle,
    ScaleRule,
    dequantize,
    quantize,
)
from fouroversix.quantize import from_blocked
from fouroversix.quantize.frontend import AVAILABLE_BACKENDS


SHAPES = [
    (1, 576),
    (6, 1536),
    (128, 256),
    (1024, 1024),
    (4096, 4096),
    (8192, 4096),
]
SCALE_RULES = [
    ScaleRule.static_4,
    ScaleRule.static_6,
    ScaleRule.abs_max,
    ScaleRule.mae,
    ScaleRule.mse,
]
DTYPE = DataType.nvfp4
OUTPUT_DIR = Path(__file__).resolve().parent
SAMPLE_LIMIT = 8


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _values_for_shape(tensor: Any, shape: tuple[int, int]) -> torch.Tensor:
    packed_cols = shape[1] // DTYPE.quantized_value_type.packing_factor
    return tensor.values[: shape[0], :packed_cols].contiguous()


def _scale_matrix_for_shape(tensor: Any, shape: tuple[int, int]) -> torch.Tensor:
    scale_cols = shape[1] // DTYPE.block_size
    if tensor.scale_factors_are_in_blackwell_layout:
        scales = from_blocked(
            tensor.scale_factors,
            (tensor.padded_shape[0], tensor.padded_shape[1] // DTYPE.block_size),
        )
        return scales[: shape[0], :scale_cols].contiguous()

    rows = tensor.scale_factors.numel() // scale_cols
    return tensor.scale_factors.reshape(rows, scale_cols)[: shape[0]].contiguous()


def _sample_u8_mismatches(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    index_names: tuple[str, str],
) -> list[dict[str, int]]:
    diff = left != right
    coords = diff.nonzero(as_tuple=False)[:SAMPLE_LIMIT].detach().cpu().tolist()
    samples = []
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    for row, col in coords:
        samples.append(
            {
                index_names[0]: int(row),
                index_names[1]: int(col),
                "triton_u8": int(left_cpu[row, col].item()),
                "cute_sm120_u8": int(right_cpu[row, col].item()),
            },
        )
    return samples


def _equality_stats(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    index_names: tuple[str, str],
) -> dict[str, Any]:
    left_u8 = left.view(torch.uint8)
    right_u8 = right.view(torch.uint8)
    mismatch_count = int((left_u8 != right_u8).sum().item())
    total = left_u8.numel()
    return {
        "total": total,
        "mismatch_count": mismatch_count,
        "equal_ratio": 1.0 if total == 0 else 1.0 - mismatch_count / total,
        "sample_mismatches": _sample_u8_mismatches(
            left_u8,
            right_u8,
            index_names=index_names,
        ),
    }


def _diff_stats(diff: torch.Tensor) -> dict[str, Any]:
    diff = diff.float()
    abs_diff = diff.abs()
    finite = torch.isfinite(diff)
    return {
        "mae": float(abs_diff.mean().item()),
        "mse": float((diff * diff).mean().item()),
        "max_error": float(abs_diff.max().item()),
        "finite": bool(finite.all().item()),
    }


def _quantize_pair(
    x: torch.Tensor,
    scale_rule: ScaleRule,
) -> tuple[Any, Any]:
    configs = {
        backend: QuantizationConfig(
            backend=backend,
            dtype=DTYPE,
            scale_rule=scale_rule,
            round_style=RoundStyle.nearest,
            kwargs=(
                {"match_triton_reduction": True}
                if backend == QuantizeBackend.cute_sm120
                else {}
            ),
        )
        for backend in (QuantizeBackend.triton, QuantizeBackend.cute_sm120)
    }
    for backend, config in configs.items():
        backend_cls = AVAILABLE_BACKENDS[backend]
        if not backend_cls.is_available():
            raise RuntimeError(f"{backend.value} backend is not available")
        if not backend_cls.can_quantize(x, config):
            raise RuntimeError(
                f"{backend.value} cannot quantize shape={tuple(x.shape)} "
                f"scale_rule={scale_rule.value}",
            )

    triton = quantize(x.clone(), configs[QuantizeBackend.triton])
    cute = quantize(x.clone(), configs[QuantizeBackend.cute_sm120])
    torch.cuda.synchronize()
    return triton, cute


def _patch_amax(row: dict[str, Any], triton: Any, cute: Any) -> None:
    triton_u8 = triton.amax.detach().reshape(1).view(torch.uint8).cpu().tolist()
    cute_u8 = cute.amax.detach().reshape(1).view(torch.uint8).cpu().tolist()
    row["amax"] = {
        "triton": float(triton.amax.item()),
        "cute_sm120": float(cute.amax.item()),
        "bitwise_equal": triton_u8 == cute_u8,
        "triton_u8": [int(v) for v in triton_u8],
        "cute_sm120_u8": [int(v) for v in cute_u8],
    }


def _run_case_with_amax(
    shape: tuple[int, int], scale_rule: ScaleRule
) -> dict[str, Any]:
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    triton, cute = _quantize_pair(x, scale_rule)

    values_triton = _values_for_shape(triton, shape)
    values_cute = _values_for_shape(cute, shape)
    scales_triton = _scale_matrix_for_shape(triton, shape)
    scales_cute = _scale_matrix_for_shape(cute, shape)
    dequant_triton = dequantize(
        triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequant_cute = dequantize(
        cute,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )

    row = {
        "shape": list(shape),
        "scale_rule": scale_rule.value,
        "amax": {},
        "values": _equality_stats(
            values_triton,
            values_cute,
            index_names=("row", "packed_col"),
        ),
        "scale_factors_e4m3": _equality_stats(
            scales_triton,
            scales_cute,
            index_names=("row", "scale_block_col"),
        ),
        "dequant_cute_minus_triton": _diff_stats(dequant_cute - dequant_triton),
        "input_error": {
            "triton": _diff_stats(dequant_triton - x.float()),
            "cute_sm120": _diff_stats(dequant_cute - x.float()),
        },
    }
    _patch_amax(row, triton, cute)

    del x, triton, cute, dequant_triton, dequant_cute
    torch.cuda.empty_cache()
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    value_mismatches = sum(row["values"]["mismatch_count"] for row in rows)
    scale_mismatches = sum(row["scale_factors_e4m3"]["mismatch_count"] for row in rows)
    max_dequant_error = max(
        row["dequant_cute_minus_triton"]["max_error"] for row in rows
    )
    max_dequant_mae = max(row["dequant_cute_minus_triton"]["mae"] for row in rows)
    max_dequant_mse = max(row["dequant_cute_minus_triton"]["mse"] for row in rows)
    return {
        "cases": len(rows),
        "value_mismatch_count": value_mismatches,
        "scale_mismatch_count": scale_mismatches,
        "amax_bitwise_equal_cases": sum(
            1 for row in rows if row["amax"]["bitwise_equal"]
        ),
        "max_dequant_mae": max_dequant_mae,
        "max_dequant_mse": max_dequant_mse,
        "max_dequant_error": max_dequant_error,
        "all_bitwise_equal": value_mismatches == 0
        and scale_mismatches == 0
        and all(row["amax"]["bitwise_equal"] for row in rows)
        and max_dequant_error == 0.0,
    }


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# SM120 NVFP4 Triton Parity",
        "",
        f"Generated: `{output['generated_at']}`",
        f"Device: `{output['device']['name']}`, capability `{output['device']['capability']}`",
        f"CuTe SM120 match Triton reduction: `{output['config']['cute_sm120_match_triton_reduction']}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{output['summary']['cases']}`",
        f"- All bitwise equal: `{output['summary']['all_bitwise_equal']}`",
        f"- Value mismatches: `{output['summary']['value_mismatch_count']}`",
        f"- E4M3 scale mismatches: `{output['summary']['scale_mismatch_count']}`",
        f"- Max dequant MAE/MSE/error: "
        f"`{output['summary']['max_dequant_mae']:.8e}` / "
        f"`{output['summary']['max_dequant_mse']:.8e}` / "
        f"`{output['summary']['max_dequant_error']:.8e}`",
        "",
        "## Cases",
        "",
        "| shape | scale_rule | values_equal | scale_equal | dequant_mae | dequant_mse | dequant_max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["rows"]:
        lines.append(
            "| "
            f"{tuple(row['shape'])} | "
            f"{row['scale_rule']} | "
            f"{row['values']['equal_ratio']:.6f} | "
            f"{row['scale_factors_e4m3']['equal_ratio']:.6f} | "
            f"{row['dequant_cute_minus_triton']['mae']:.8e} | "
            f"{row['dequant_cute_minus_triton']['mse']:.8e} | "
            f"{row['dequant_cute_minus_triton']['max_error']:.8e} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rows = []
    for shape in SHAPES:
        for scale_rule in SCALE_RULES:
            row = _run_case_with_amax(shape, scale_rule)
            rows.append(row)
            print(
                shape,
                scale_rule.value,
                f"values={row['values']['equal_ratio']:.6f}",
                f"scales={row['scale_factors_e4m3']['equal_ratio']:.6f}",
                f"max={row['dequant_cute_minus_triton']['max_error']:.8e}",
            )

    output = {
        "generated_at": _utc_now().isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "config": {
            "dtype": DTYPE.value,
            "round_style": RoundStyle.nearest.value,
            "backends": [
                QuantizeBackend.triton.value,
                QuantizeBackend.cute_sm120.value,
            ],
            "shapes": [list(shape) for shape in SHAPES],
            "scale_rules": [rule.value for rule in SCALE_RULES],
            "cute_sm120_match_triton_reduction": True,
        },
        "summary": _summarize(rows),
        "rows": rows,
    }

    json_path = OUTPUT_DIR / "nvfp4_sm120_triton_parity_latest.json"
    md_path = OUTPUT_DIR / "nvfp4_sm120_triton_parity_latest.md"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    _write_markdown(output, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
