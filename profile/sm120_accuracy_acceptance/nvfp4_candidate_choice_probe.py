from __future__ import annotations

import datetime as dt
import json
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


DTYPE = DataType.nvfp4
OUTPUT_DIR = Path(__file__).resolve().parent
SHAPES = [(1024, 1024), (4096, 4096), (8192, 4096)]
SCALE_RULES = [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse]
SAMPLE_LIMIT = 12


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


def _quantize(
    x: torch.Tensor,
    backend: QuantizeBackend,
    scale_rule: ScaleRule,
    *,
    x_amax: torch.Tensor | None = None,
) -> Any:
    kwargs = {"x_amax": x_amax} if x_amax is not None else {}
    if backend == QuantizeBackend.cute_sm120:
        kwargs["match_triton_reduction"] = True
    config = QuantizationConfig(
        backend=backend,
        dtype=DTYPE,
        scale_rule=scale_rule,
        round_style=RoundStyle.nearest,
        kwargs=kwargs,
    )
    return quantize(x.clone(), config)


def _choice_from_candidates(
    adaptive_scales: torch.Tensor,
    candidate4_scales: torch.Tensor,
    candidate6_scales: torch.Tensor,
) -> torch.Tensor:
    match4 = adaptive_scales == candidate4_scales
    match6 = adaptive_scales == candidate6_scales
    choice = torch.full(
        adaptive_scales.shape,
        -1,
        dtype=torch.int8,
        device=adaptive_scales.device,
    )
    choice = torch.where(match6, torch.zeros((), dtype=torch.int8, device=choice.device), choice)
    choice = torch.where(match4, torch.ones((), dtype=torch.int8, device=choice.device), choice)
    both = match4 & match6
    choice = torch.where(both, torch.full((), 2, dtype=torch.int8, device=choice.device), choice)
    return choice


def _fouroversix_candidate_scales(
    x: torch.Tensor,
    amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    block_max = x.float().abs().reshape(m, k // DTYPE.block_size, DTYPE.block_size).amax(dim=-1)
    global_scale = torch.zeros((), dtype=torch.float32, device=x.device)
    if float(amax.item()) != 0.0:
        global_scale = torch.tensor(1536.0, dtype=torch.float32, device=x.device) / amax.float()
    scale6_hp = (block_max / torch.tensor(6.0, dtype=torch.float32, device=x.device)) * global_scale
    scale4_hp = scale6_hp * torch.tensor(1.5, dtype=torch.float32, device=x.device)
    scale6 = scale6_hp.to(torch.float8_e4m3fn).view(torch.uint8)
    scale4 = scale4_hp.to(torch.float8_e4m3fn).view(torch.uint8)
    return scale4, scale6


def _round_e2m1_rn(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    mag = torch.where(
        ax <= 0.25,
        torch.zeros_like(ax),
        torch.where(
            ax < 0.75,
            torch.full_like(ax, 0.5),
            torch.where(
                ax <= 1.25,
                torch.full_like(ax, 1.0),
                torch.where(
                    ax < 1.75,
                    torch.full_like(ax, 1.5),
                    torch.where(
                        ax <= 2.5,
                        torch.full_like(ax, 2.0),
                        torch.where(
                            ax < 3.5,
                            torch.full_like(ax, 3.0),
                            torch.where(
                                ax <= 5.0,
                                torch.full_like(ax, 4.0),
                                torch.full_like(ax, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return torch.copysign(mag, x)


def _candidate_error_margin(
    x: torch.Tensor,
    amax: torch.Tensor,
    scale4_u8: torch.Tensor,
    scale6_u8: torch.Tensor,
    scale_rule: ScaleRule,
) -> torch.Tensor:
    m, k = x.shape
    xb = x.float().reshape(m, k // DTYPE.block_size, DTYPE.block_size)
    denominator = torch.tensor(1536.0, dtype=torch.float32, device=x.device)
    amax_f32 = amax.float()

    scale4 = scale4_u8.contiguous().view(torch.float8_e4m3fn).float()
    scale6 = scale6_u8.contiguous().view(torch.float8_e4m3fn).float()
    out4 = torch.where(scale4 != 0, denominator / (amax_f32 * scale4), 0.0)
    out6 = torch.where(scale6 != 0, denominator / (amax_f32 * scale6), 0.0)
    q4 = _round_e2m1_rn(xb * out4.unsqueeze(-1))
    q6 = _round_e2m1_rn(xb * out6.unsqueeze(-1))
    d4 = ((q4 * scale4.unsqueeze(-1) * amax_f32) / denominator - xb).abs()
    d6 = ((q6 * scale6.unsqueeze(-1) * amax_f32) / denominator - xb).abs()

    if scale_rule == ScaleRule.abs_max:
        err4 = d4.amax(dim=-1)
        err6 = d6.amax(dim=-1)
    elif scale_rule == ScaleRule.mae:
        err4 = d4.sum(dim=-1)
        err6 = d6.sum(dim=-1)
    else:
        err4 = (d4 * d4).sum(dim=-1)
        err6 = (d6 * d6).sum(dim=-1)
    return err4 - err6


def _sample_choice_mismatches(
    triton_choice: torch.Tensor,
    cute_choice: torch.Tensor,
    py_margin: torch.Tensor,
    values_triton: torch.Tensor,
    values_cute: torch.Tensor,
    scales_triton: torch.Tensor,
    scales_cute: torch.Tensor,
) -> list[dict[str, int]]:
    diff = triton_choice != cute_choice
    coords = diff.nonzero(as_tuple=False)[:SAMPLE_LIMIT].detach().cpu().tolist()
    samples = []
    for row, col in coords:
        samples.append(
            {
                "row": int(row),
                "scale_block_col": int(col),
                "triton_choice": int(triton_choice[row, col].item()),
                "cute_sm120_choice": int(cute_choice[row, col].item()),
                "triton_scale_u8": int(scales_triton[row, col].item()),
                "cute_sm120_scale_u8": int(scales_cute[row, col].item()),
                "triton_first_packed_u8": int(values_triton[row, col * 8].item()),
                "cute_sm120_first_packed_u8": int(values_cute[row, col * 8].item()),
                "py_error4_minus_error6": float(py_margin[row, col].item()),
            }
        )
    return samples


def _run_case(shape: tuple[int, int], scale_rule: ScaleRule) -> dict[str, Any]:
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

    triton = _quantize(x, QuantizeBackend.triton, scale_rule)
    cute = _quantize(x, QuantizeBackend.cute_sm120, scale_rule, x_amax=triton.amax)
    torch.cuda.synchronize()

    triton_values = _values_for_shape(triton, shape)
    cute_values = _values_for_shape(cute, shape)
    triton_scales = _scale_matrix_for_shape(triton, shape).view(torch.uint8)
    cute_scales = _scale_matrix_for_shape(cute, shape).view(torch.uint8)
    candidate4_scales, candidate6_scales = _fouroversix_candidate_scales(x, triton.amax)
    py_margin = _candidate_error_margin(
        x,
        triton.amax,
        candidate4_scales,
        candidate6_scales,
        scale_rule,
    )
    py_choice = torch.where(
        py_margin < 0,
        torch.ones((), dtype=torch.int8, device=x.device),
        torch.zeros((), dtype=torch.int8, device=x.device),
    )

    triton_choice = _choice_from_candidates(
        triton_scales.unsqueeze(-1).expand(-1, -1, 8),
        candidate4_scales.unsqueeze(-1).expand(-1, -1, 8),
        candidate6_scales.unsqueeze(-1).expand(-1, -1, 8),
    ).amin(dim=-1)
    cute_choice = _choice_from_candidates(
        cute_scales.unsqueeze(-1).expand(-1, -1, 8),
        candidate4_scales.unsqueeze(-1).expand(-1, -1, 8),
        candidate6_scales.unsqueeze(-1).expand(-1, -1, 8),
    ).amin(dim=-1)

    value_mismatch_by_block = (triton_values.reshape(shape[0], -1, 8) != cute_values.reshape(shape[0], -1, 8)).any(dim=-1)
    scale_mismatch = triton_scales != cute_scales
    choice_mismatch = triton_choice != cute_choice
    py_triton_choice_mismatch = py_choice != triton_choice
    py_cute_choice_mismatch = py_choice != cute_choice
    mismatch_margin = py_margin[choice_mismatch].abs()

    total_blocks = triton_choice.numel()
    row = {
        "shape": list(shape),
        "scale_rule": scale_rule.value,
        "amax": {
            "triton": float(triton.amax.item()),
            "cute_sm120": float(cute.amax.item()),
            "bitwise_equal": triton.amax.detach().reshape(1).view(torch.uint8).cpu().tolist()
            == cute.amax.detach().reshape(1).view(torch.uint8).cpu().tolist(),
        },
        "total_scale_blocks": int(total_blocks),
        "value_mismatch_blocks": int(value_mismatch_by_block.sum().item()),
        "scale_mismatch_blocks": int(scale_mismatch.sum().item()),
        "choice_mismatch_blocks": int(choice_mismatch.sum().item()),
        "triton_unclassified_blocks": int((triton_choice < 0).sum().item()),
        "cute_unclassified_blocks": int((cute_choice < 0).sum().item()),
        "py_choice_mismatches_triton": int(py_triton_choice_mismatch.sum().item()),
        "py_choice_mismatches_cute_sm120": int(py_cute_choice_mismatch.sum().item()),
        "choice_mismatch_py_margin_abs_max": (
            0.0 if mismatch_margin.numel() == 0 else float(mismatch_margin.max().item())
        ),
        "choice_mismatch_py_margin_abs_mean": (
            0.0 if mismatch_margin.numel() == 0 else float(mismatch_margin.mean().item())
        ),
        "choice_mismatch_with_value_or_scale_mismatch": int(
            (choice_mismatch & (value_mismatch_by_block | scale_mismatch)).sum().item()
        ),
        "value_or_scale_mismatch_without_choice_mismatch": int(
            ((value_mismatch_by_block | scale_mismatch) & ~choice_mismatch).sum().item()
        ),
        "sample_choice_mismatches": _sample_choice_mismatches(
            triton_choice,
            cute_choice,
            py_margin,
            triton_values,
            cute_values,
            triton_scales,
            cute_scales,
        ),
    }

    del x, triton, cute
    torch.cuda.empty_cache()
    return row


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# SM120 NVFP4 Candidate Choice Probe",
        "",
        f"Generated: `{output['generated_at']}`",
        f"Device: `{output['device']}`",
        "",
        "| shape | scale_rule | value_mismatch_blocks | scale_mismatch_blocks | choice_mismatch_blocks | py_vs_triton | py_vs_cute | margin_abs_max | same_choice_mismatch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["rows"]:
        lines.append(
            "| "
            f"{tuple(row['shape'])} | "
            f"{row['scale_rule']} | "
            f"{row['value_mismatch_blocks']} | "
            f"{row['scale_mismatch_blocks']} | "
            f"{row['choice_mismatch_blocks']} | "
            f"{row['py_choice_mismatches_triton']} | "
            f"{row['py_choice_mismatches_cute_sm120']} | "
            f"{row['choice_mismatch_py_margin_abs_max']:.8e} | "
            f"{row['value_or_scale_mismatch_without_choice_mismatch']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rows = []
    for shape in SHAPES:
        for scale_rule in SCALE_RULES:
            row = _run_case(shape, scale_rule)
            rows.append(row)
            print(
                shape,
                scale_rule.value,
                f"choice_mismatch={row['choice_mismatch_blocks']}",
                f"same_choice_mismatch={row['value_or_scale_mismatch_without_choice_mismatch']}",
            )

    output = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
    }
    json_path = OUTPUT_DIR / "nvfp4_candidate_choice_probe_latest.json"
    md_path = OUTPUT_DIR / "nvfp4_candidate_choice_probe_latest.md"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    _write_markdown(output, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
