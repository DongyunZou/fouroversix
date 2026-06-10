from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from fouroversix import DataType, RoundStyle, ScaleRule
from fouroversix.utils import ScaleType
from fouroversix.kernels.triton.fp4 import convert_to_e2m1x2_and_quantized_fp16
from fouroversix.kernels.triton.fp8 import (
    convert_e4m3_to_high_precision,
)
from fouroversix.kernels.triton.constants import SCALE_RULE_MAE
from fouroversix.kernels.triton.quantize import compute_scale_factors_kernel


DTYPE = DataType.nvfp4
OUTPUT_DIR = Path(__file__).resolve().parent
INPUT_PROBE = OUTPUT_DIR / "nvfp4_candidate_choice_probe_latest.json"
SAMPLE_LIMIT = 12


@triton.jit
def _triton_block_error_probe_kernel(
    x_ptr,
    amax_ptr,
    rbits_ptr,
    rows_ptr,
    cols_ptr,
    err4_ptr,
    err6_ptr,
    scale4_ptr,
    scale6_ptr,
    choice_ptr,
    K: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    NUM_SAMPLES: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    row = tl.load(rows_ptr + pid)
    col = tl.load(cols_ptr + pid)
    tile_m = row // 128
    tile_n = (col * 16) // 64
    local_m = row - tile_m * 128
    local_c = col - tile_n * 4
    offs_m = tile_m * 128 + tl.arange(0, 128)
    offs_n = tile_n * 64 + tl.arange(0, 64)
    x_block = tl.load(x_ptr + offs_m[:, None] * K + offs_n[None, :]).to(tl.float32)
    amax = tl.load(amax_ptr)

    x_scaled6, scale6_u8_block, _ = compute_scale_factors_kernel(
        x_block,
        amax_ptr,
        128,
        64,
        False,
        6,
        256,
        ROUND_STYLE,
        SCALE_TYPE,
        16,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )
    scale6_f32 = convert_e4m3_to_high_precision(
        scale6_u8_block,
        tl.float32,
        MAJOR_COMPUTE_CAPABILITY,
    )
    _, q6_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_scaled6,
        rbits_ptr,
        128,
        64,
        ROUND_STYLE,
        16,
        MAJOR_COMPUTE_CAPABILITY,
    )
    deq6 = tl.div_rn(
        q6_fp16.to(tl.float32) * scale6_f32.expand_dims(2) * amax,
        1536.0,
    )

    x_scaled4, scale4_u8_block, _ = compute_scale_factors_kernel(
        x_block,
        amax_ptr,
        128,
        64,
        False,
        6,
        256,
        ROUND_STYLE,
        SCALE_TYPE,
        16,
        1.5,
        MAJOR_COMPUTE_CAPABILITY,
    )
    scale4_f32 = convert_e4m3_to_high_precision(
        scale4_u8_block,
        tl.float32,
        MAJOR_COMPUTE_CAPABILITY,
    )
    _, q4_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_scaled4,
        rbits_ptr,
        128,
        64,
        ROUND_STYLE,
        16,
        MAJOR_COMPUTE_CAPABILITY,
    )
    deq4 = tl.div_rn(
        q4_fp16.to(tl.float32) * scale4_f32.expand_dims(2) * amax,
        1536.0,
    )

    x_grouped = x_block.reshape(128, 4, 16)
    diff4 = deq4 - x_grouped
    diff6 = deq6 - x_grouped
    if SCALE_RULE == SCALE_RULE_MAE:
        err4_all = tl.sum(tl.abs(diff4), axis=-1)
        err6_all = tl.sum(tl.abs(diff6), axis=-1)
    else:
        err4_all = tl.sum(diff4 * diff4, axis=-1)
        err6_all = tl.sum(diff6 * diff6, axis=-1)

    row_mask = tl.arange(0, 128)[:, None] == local_m
    col_mask = tl.arange(0, 4)[None, :] == local_c
    sample_mask = row_mask & col_mask
    err4 = tl.sum(tl.sum(tl.where(sample_mask, err4_all, 0.0), axis=0), axis=0)
    err6 = tl.sum(tl.sum(tl.where(sample_mask, err6_all, 0.0), axis=0), axis=0)
    scale4_u8 = tl.sum(
        tl.sum(
            tl.where(sample_mask, scale4_u8_block.cast(tl.uint32, bitcast=False), 0),
            axis=0,
        ),
        axis=0,
    ).to(tl.uint8)
    scale6_u8 = tl.sum(
        tl.sum(
            tl.where(sample_mask, scale6_u8_block.cast(tl.uint32, bitcast=False), 0),
            axis=0,
        ),
        axis=0,
    ).to(tl.uint8)

    tl.store(err4_ptr + pid, err4)
    tl.store(err6_ptr + pid, err6)
    tl.store(scale4_ptr + pid, scale4_u8)
    tl.store(scale6_ptr + pid, scale6_u8)
    tl.store(choice_ptr + pid, (err4 < err6).to(tl.int8))


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


def _sum_fp32(values: torch.Tensor, order: str) -> torch.Tensor:
    vals = values.float()
    if order == "left":
        out = vals[0]
        for val in vals[1:]:
            out = out + val
        return out
    if order == "lane_even_odd":
        even = vals[0]
        for val in vals[2::2]:
            even = even + val
        odd = vals[1]
        for val in vals[3::2]:
            odd = odd + val
        return even + odd
    if order == "lane_odd_even":
        odd = vals[1]
        for val in vals[3::2]:
            odd = odd + val
        even = vals[0]
        for val in vals[2::2]:
            even = even + val
        return odd + even
    if order == "lane_even_odd_rev":
        even = vals[14]
        for val in vals[12::-2]:
            even = even + val
        odd = vals[15]
        for val in vals[13::-2]:
            odd = odd + val
        return even + odd
    if order == "lane_odd_even_rev":
        odd = vals[15]
        for val in vals[13::-2]:
            odd = odd + val
        even = vals[14]
        for val in vals[12::-2]:
            even = even + val
        return odd + even
    if order == "pair4":
        pair = vals.reshape(8, 2).sum(dim=1)
        return (pair[0] + pair[1] + pair[2] + pair[3]) + (
            pair[4] + pair[5] + pair[6] + pair[7]
        )
    if order == "tree":
        pair = vals.reshape(8, 2).sum(dim=1)
        quad = pair.reshape(4, 2).sum(dim=1)
        return quad.reshape(2, 2).sum(dim=1).sum()
    raise ValueError(order)


def _python_candidate_errors(
    x_block: torch.Tensor,
    amax: torch.Tensor,
    scale_rule: ScaleRule,
) -> dict[str, float]:
    block_max = x_block.float().abs().amax()
    global_scale = torch.tensor(1536.0, dtype=torch.float32, device=x_block.device) / amax.float()
    scale6_u8 = ((block_max / 6.0) * global_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    scale4_u8 = ((block_max / 6.0) * global_scale * 1.5).to(torch.float8_e4m3fn).view(torch.uint8)
    scale6 = scale6_u8.view(torch.float8_e4m3fn).float()
    scale4 = scale4_u8.view(torch.float8_e4m3fn).float()

    out6 = torch.where(scale6 != 0, torch.tensor(1536.0, device=x_block.device) / (amax.float() * scale6), 0.0)
    out4 = torch.where(scale4 != 0, torch.tensor(1536.0, device=x_block.device) / (amax.float() * scale4), 0.0)
    q6 = _round_e2m1_rn(x_block.float() * out6)
    q4 = _round_e2m1_rn(x_block.float() * out4)
    diff6 = (q6 * scale6 * amax.float()) / 1536.0 - x_block.float()
    diff4 = (q4 * scale4 * amax.float()) / 1536.0 - x_block.float()
    if scale_rule == ScaleRule.mae:
        vals4 = diff4.abs()
        vals6 = diff6.abs()
    else:
        vals4 = diff4 * diff4
        vals6 = diff6 * diff6

    result: dict[str, float] = {}
    for order in (
        "left",
        "lane_even_odd",
        "lane_odd_even",
        "lane_even_odd_rev",
        "lane_odd_even_rev",
        "pair4",
        "tree",
    ):
        result[f"py_{order}_err4"] = float(_sum_fp32(vals4, order).item())
        result[f"py_{order}_err6"] = float(_sum_fp32(vals6, order).item())
        result[f"py_{order}_choice"] = float(
            (_sum_fp32(vals4, order) < _sum_fp32(vals6, order)).item()
        )
    return result


def _run_samples(
    shape: tuple[int, int],
    scale_rule: ScaleRule,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    amax = torch.linalg.vector_norm(x, ord=float("inf"), dtype=torch.float32)

    rows = torch.tensor([sample["row"] for sample in samples], dtype=torch.int64, device="cuda")
    cols = torch.tensor([sample["scale_block_col"] for sample in samples], dtype=torch.int64, device="cuda")
    count = rows.numel()
    err4 = torch.empty(count, dtype=torch.float32, device="cuda")
    err6 = torch.empty(count, dtype=torch.float32, device="cuda")
    scale4 = torch.empty(count, dtype=torch.uint8, device="cuda")
    scale6 = torch.empty(count, dtype=torch.uint8, device="cuda")
    choice = torch.empty(count, dtype=torch.int8, device="cuda")
    rbits = torch.empty(1, dtype=torch.uint32, device="cuda")

    _triton_block_error_probe_kernel[(count,)](
        x,
        amax,
        rbits,
        rows,
        cols,
        err4,
        err6,
        scale4,
        scale6,
        choice,
        K=shape[1],
        ROUND_STYLE=RoundStyle.nearest.value,
        SCALE_TYPE=ScaleType.nv.value,
        SCALE_RULE=scale_rule.value,
        NUM_SAMPLES=count,
        MAJOR_COMPUTE_CAPABILITY=torch.cuda.get_device_capability()[0],
    )
    torch.cuda.synchronize()

    out = []
    for idx, sample in enumerate(samples):
        row = sample["row"]
        col = sample["scale_block_col"]
        x_block = x[row, col * 16 : (col + 1) * 16]
        record = dict(sample)
        record.update(
            {
                "triton_probe_err4": float(err4[idx].item()),
                "triton_probe_err6": float(err6[idx].item()),
                "triton_probe_margin": float((err4[idx] - err6[idx]).item()),
                "triton_probe_choice": int(choice[idx].item()),
                "triton_probe_scale4_u8": int(scale4[idx].item()),
                "triton_probe_scale6_u8": int(scale6[idx].item()),
            }
        )
        record.update(_python_candidate_errors(x_block, amax, scale_rule))
        out.append(record)
    return out


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# SM120 NVFP4 Triton Error Probe",
        "",
        f"Generated: `{output['generated_at']}`",
        f"Device: `{output['device']}`",
        "",
        "| shape | scale_rule | row | col | triton_choice | cute_choice | probe_choice | probe_margin | py_left_choice | py_pair4_choice | py_tree_choice |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["rows"]:
        for sample in row["samples"]:
            lines.append(
                "| "
                f"{tuple(row['shape'])} | "
                f"{row['scale_rule']} | "
                f"{sample['row']} | "
                f"{sample['scale_block_col']} | "
                f"{sample['triton_choice']} | "
                f"{sample['cute_sm120_choice']} | "
                f"{sample['triton_probe_choice']} | "
                f"{sample['triton_probe_margin']:.8e} | "
                f"{int(sample['py_left_choice'])} | "
                f"{int(sample['py_pair4_choice'])} | "
                f"{int(sample['py_tree_choice'])} |"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    source = json.loads(INPUT_PROBE.read_text())
    rows = []
    for row in source["rows"]:
        if row["scale_rule"] == ScaleRule.abs_max.value:
            continue
        samples = row.get("sample_choice_mismatches", [])[:SAMPLE_LIMIT]
        if not samples:
            continue
        shape = tuple(row["shape"])
        scale_rule = ScaleRule(row["scale_rule"])
        rows.append(
            {
                "shape": list(shape),
                "scale_rule": scale_rule.value,
                "samples": _run_samples(shape, scale_rule, samples),
            }
        )

    output = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
    }
    json_path = OUTPUT_DIR / "nvfp4_triton_error_probe_latest.json"
    md_path = OUTPUT_DIR / "nvfp4_triton_error_probe_latest.md"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    _write_markdown(output, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
