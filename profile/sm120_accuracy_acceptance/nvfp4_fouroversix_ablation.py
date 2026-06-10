from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from fouroversix import (
    DataType,
    QuantizationConfig,
    QuantizeBackend,
    RoundStyle,
    ScaleRule,
    quantize,
)
from fouroversix.kernels.triton.constants import (
    E2M1_MAX_VALUE,
    E4M3_MAX_FOUROVERSIX,
    ROUND_STYLE_NEAREST,
    SCALE_RULE_ABS_MAX,
    SCALE_RULE_MAE,
    SCALE_TYPE_NV_IF,
)
from fouroversix.kernels.triton.fp4 import convert_to_e2m1x2_and_quantized_fp16
from fouroversix.kernels.triton.fp8 import convert_e4m3_to_high_precision
from fouroversix.kernels.triton.quantize import compute_scale_factors_kernel
from fouroversix.quantize import from_blocked


DTYPE = DataType.nvfp4
OUTPUT_DIR = Path(__file__).resolve().parent
SHAPES = [(1024, 1024), (4096, 4096), (8192, 4096)]
SCALE_RULES = [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse]


@triton.jit
def _fouroversix_candidates_kernel(
    x_ptr,
    amax_ptr,
    rbits_ptr,
    values4_ptr,
    values6_ptr,
    scale4_ptr,
    scale6_ptr,
    choice_ptr,
    K: tl.constexpr,
    PACKED_COLS: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    x_block = tl.load(x_ptr + offs_m[:, None] * K + offs_n[None, :]).to(tl.float32)

    x_scaled6, scale6_u8, x_amax = compute_scale_factors_kernel(
        x_block,
        amax_ptr,
        128,
        64,
        False,
        E2M1_MAX_VALUE,
        E4M3_MAX_FOUROVERSIX,
        ROUND_STYLE_NEAREST,
        SCALE_TYPE_NV_IF,
        16,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )
    x6, q6_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_scaled6,
        rbits_ptr,
        128,
        64,
        ROUND_STYLE_NEAREST,
        16,
        MAJOR_COMPUTE_CAPABILITY,
    )
    scale6_f32 = convert_e4m3_to_high_precision(
        scale6_u8,
        tl.float32,
        MAJOR_COMPUTE_CAPABILITY,
    )
    deq6 = tl.div_rn(
        q6_fp16.to(tl.float32) * scale6_f32.expand_dims(2) * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX,
    )

    x_scaled4, scale4_u8, _ = compute_scale_factors_kernel(
        x_block,
        amax_ptr,
        128,
        64,
        False,
        E2M1_MAX_VALUE,
        E4M3_MAX_FOUROVERSIX,
        ROUND_STYLE_NEAREST,
        SCALE_TYPE_NV_IF,
        16,
        1.5,
        MAJOR_COMPUTE_CAPABILITY,
    )
    x4, q4_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_scaled4,
        rbits_ptr,
        128,
        64,
        ROUND_STYLE_NEAREST,
        16,
        MAJOR_COMPUTE_CAPABILITY,
    )
    scale4_f32 = convert_e4m3_to_high_precision(
        scale4_u8,
        tl.float32,
        MAJOR_COMPUTE_CAPABILITY,
    )
    deq4 = tl.div_rn(
        q4_fp16.to(tl.float32) * scale4_f32.expand_dims(2) * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX,
    )

    x_grouped = x_block.reshape(128, 4, 16)
    diff4 = deq4 - x_grouped
    diff6 = deq6 - x_grouped
    if SCALE_RULE == SCALE_RULE_ABS_MAX:
        err4 = tl.max(tl.abs(diff4), axis=-1)
        err6 = tl.max(tl.abs(diff6), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MAE:
        err4 = tl.sum(tl.abs(diff4), axis=-1)
        err6 = tl.sum(tl.abs(diff6), axis=-1)
    else:
        err4 = tl.sum(diff4 * diff4, axis=-1)
        err6 = tl.sum(diff6 * diff6, axis=-1)

    packed_cols = pid_n * 32 + tl.arange(0, 32)
    tl.store(
        values4_ptr + offs_m[:, None] * PACKED_COLS + packed_cols[None, :],
        x4.reshape(128, 32),
    )
    tl.store(
        values6_ptr + offs_m[:, None] * PACKED_COLS + packed_cols[None, :],
        x6.reshape(128, 32),
    )

    scale_cols = pid_n * 4 + tl.arange(0, 4)
    tl.store(
        scale4_ptr + offs_m[:, None] * SCALE_COLS + scale_cols[None, :],
        scale4_u8,
    )
    tl.store(
        scale6_ptr + offs_m[:, None] * SCALE_COLS + scale_cols[None, :],
        scale6_u8,
    )
    tl.store(
        choice_ptr + offs_m[:, None] * SCALE_COLS + scale_cols[None, :],
        (err4 < err6).to(tl.int8),
    )


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
    match_triton_reduction: bool = False,
) -> Any:
    kwargs = {"x_amax": x_amax} if x_amax is not None else {}
    if backend == QuantizeBackend.cute_sm120:
        kwargs["match_triton_reduction"] = match_triton_reduction
    config = QuantizationConfig(
        backend=backend,
        dtype=DTYPE,
        scale_rule=scale_rule,
        round_style=RoundStyle.nearest,
        kwargs=kwargs,
    )
    return quantize(x.clone(), config)


def _candidate_match(
    values: torch.Tensor,
    scales: torch.Tensor,
    values4: torch.Tensor,
    values6: torch.Tensor,
    scale4: torch.Tensor,
    scale6: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_blocks = values.reshape(values.shape[0], -1, 8)
    value4_blocks = values4.reshape(values.shape[0], -1, 8)
    value6_blocks = values6.reshape(values.shape[0], -1, 8)
    match4 = (scales == scale4) & (value_blocks == value4_blocks).all(dim=-1)
    match6 = (scales == scale6) & (value_blocks == value6_blocks).all(dim=-1)
    classified = match4 | match6
    choice = torch.full(scales.shape, -1, dtype=torch.int8, device=scales.device)
    choice = torch.where(match6, torch.zeros((), dtype=torch.int8, device=scales.device), choice)
    choice = torch.where(match4, torch.ones((), dtype=torch.int8, device=scales.device), choice)
    return classified, choice, match4 & match6


def _run_case(shape: tuple[int, int], scale_rule: ScaleRule) -> dict[str, Any]:
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

    triton_out = _quantize(x, QuantizeBackend.triton, scale_rule)
    cute_default_out = _quantize(
        x,
        QuantizeBackend.cute_sm120,
        scale_rule,
        x_amax=triton_out.amax,
        match_triton_reduction=False,
    )
    cute_matched_out = _quantize(
        x,
        QuantizeBackend.cute_sm120,
        scale_rule,
        x_amax=triton_out.amax,
        match_triton_reduction=True,
    )
    torch.cuda.synchronize()

    m, k = shape
    packed_cols = k // DTYPE.quantized_value_type.packing_factor
    scale_cols = k // DTYPE.block_size
    values4 = torch.empty((m, packed_cols), dtype=torch.uint8, device="cuda")
    values6 = torch.empty((m, packed_cols), dtype=torch.uint8, device="cuda")
    scale4 = torch.empty((m, scale_cols), dtype=torch.uint8, device="cuda")
    scale6 = torch.empty((m, scale_cols), dtype=torch.uint8, device="cuda")
    official_choice = torch.empty((m, scale_cols), dtype=torch.int8, device="cuda")
    rbits = torch.empty(1, dtype=torch.uint32, device="cuda")

    _fouroversix_candidates_kernel[(m // 128, k // 64)](
        x,
        triton_out.amax,
        rbits,
        values4,
        values6,
        scale4,
        scale6,
        official_choice,
        K=k,
        PACKED_COLS=packed_cols,
        SCALE_COLS=scale_cols,
        SCALE_RULE=scale_rule.value,
        MAJOR_COMPUTE_CAPABILITY=torch.cuda.get_device_capability()[0],
    )
    torch.cuda.synchronize()

    triton_values = _values_for_shape(triton_out, shape)
    triton_scales = _scale_matrix_for_shape(triton_out, shape).view(torch.uint8)

    triton_classified, triton_choice, triton_ambiguous = _candidate_match(
        triton_values,
        triton_scales,
        values4,
        values6,
        scale4,
        scale6,
    )

    def summarize_cute(tensor: Any) -> dict[str, Any]:
        cute_values = _values_for_shape(tensor, shape)
        cute_scales = _scale_matrix_for_shape(tensor, shape).view(torch.uint8)
        cute_classified, cute_choice, cute_ambiguous = _candidate_match(
            cute_values,
            cute_scales,
            values4,
            values6,
            scale4,
            scale6,
        )
        final_block_mismatch = (
            (triton_scales != cute_scales)
            | (triton_values.reshape(m, -1, 8) != cute_values.reshape(m, -1, 8)).any(
                dim=-1,
            )
        )
        choice_mismatch = cute_choice != official_choice
        valid_choice_mismatch = choice_mismatch & cute_classified & triton_classified
        unexplained_final = final_block_mismatch & ~valid_choice_mismatch
        return {
            "candidate_payload_unmatched": int((~cute_classified).sum().item()),
            "choice_vs_official_mismatch": int(
                ((cute_choice != official_choice) & cute_classified).sum().item()
            ),
            "final_value_or_scale_mismatch_blocks": int(
                final_block_mismatch.sum().item()
            ),
            "final_mismatch_explained_by_choice": int(
                (final_block_mismatch & valid_choice_mismatch).sum().item()
            ),
            "final_mismatch_unexplained_by_choice": int(
                unexplained_final.sum().item()
            ),
            "ambiguous_candidate_blocks": int(cute_ambiguous.sum().item()),
            "amax_bitwise_equal_after_force": triton_out.amax.detach()
            .reshape(1)
            .view(torch.uint8)
            .cpu()
            .tolist()
            == tensor.amax.detach().reshape(1).view(torch.uint8).cpu().tolist(),
        }

    row = {
        "shape": list(shape),
        "scale_rule": scale_rule.value,
        "total_blocks": int(official_choice.numel()),
        "triton_candidate_payload_unmatched": int((~triton_classified).sum().item()),
        "triton_choice_vs_official_mismatch": int(
            ((triton_choice != official_choice) & triton_classified).sum().item()
        ),
        "triton_ambiguous_candidate_blocks": int(triton_ambiguous.sum().item()),
        "cute_sm120_default": summarize_cute(cute_default_out),
        "cute_sm120_match_triton": summarize_cute(cute_matched_out),
    }

    del (
        x,
        triton_out,
        cute_default_out,
        cute_matched_out,
        values4,
        values6,
        scale4,
        scale6,
        official_choice,
    )
    torch.cuda.empty_cache()
    return row


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# SM120 NVFP4 FourOverSix Ablation",
        "",
        f"Generated: `{output['generated_at']}`",
        f"Device: `{output['device']}`",
        "",
        "## Default CuTe Reduction",
        "",
        "| shape | scale_rule | triton_unmatched | cute_unmatched | triton_choice_diff | cute_choice_diff | final_mismatch | explained_by_choice | unexplained |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["rows"]:
        cute = row["cute_sm120_default"]
        lines.append(
            "| "
            f"{tuple(row['shape'])} | "
            f"{row['scale_rule']} | "
            f"{row['triton_candidate_payload_unmatched']} | "
            f"{cute['candidate_payload_unmatched']} | "
            f"{row['triton_choice_vs_official_mismatch']} | "
            f"{cute['choice_vs_official_mismatch']} | "
            f"{cute['final_value_or_scale_mismatch_blocks']} | "
            f"{cute['final_mismatch_explained_by_choice']} | "
            f"{cute['final_mismatch_unexplained_by_choice']} |"
        )
    lines.extend(
        [
            "",
            "## Triton-Matched CuTe Reduction",
            "",
            "| shape | scale_rule | triton_unmatched | cute_unmatched | triton_choice_diff | cute_choice_diff | final_mismatch | explained_by_choice | unexplained |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output["rows"]:
        cute = row["cute_sm120_match_triton"]
        lines.append(
            "| "
            f"{tuple(row['shape'])} | "
            f"{row['scale_rule']} | "
            f"{row['triton_candidate_payload_unmatched']} | "
            f"{cute['candidate_payload_unmatched']} | "
            f"{row['triton_choice_vs_official_mismatch']} | "
            f"{cute['choice_vs_official_mismatch']} | "
            f"{cute['final_value_or_scale_mismatch_blocks']} | "
            f"{cute['final_mismatch_explained_by_choice']} | "
            f"{cute['final_mismatch_unexplained_by_choice']} |"
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
            default = row["cute_sm120_default"]
            matched = row["cute_sm120_match_triton"]
            print(
                shape,
                scale_rule.value,
                f"default_final={default['final_value_or_scale_mismatch_blocks']}",
                f"default_unexplained={default['final_mismatch_unexplained_by_choice']}",
                f"matched_final={matched['final_value_or_scale_mismatch_blocks']}",
                f"matched_unexplained={matched['final_mismatch_unexplained_by_choice']}",
            )

    output = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
    }
    json_path = OUTPUT_DIR / "nvfp4_fouroversix_ablation_latest.json"
    md_path = OUTPUT_DIR / "nvfp4_fouroversix_ablation_latest.md"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    _write_markdown(output, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
