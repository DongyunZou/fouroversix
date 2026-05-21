from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass import Float32, Int32, Uint8, Uint32

from fouroversix.kernels.cute_sm100.fp4_common import (
    bfloat2_hmax_reduce_to_f32,
    bfloat2_hmax2,
    bfloat2_int4_absmax_error,
    bfloat2_int4_mae_error,
    bfloat2_int4_mse_error,
    bfloat2_fp6_absmax_error,
    bfloat2_fp6_mae_error,
    bfloat2_fp6_mse_error,
    bfloat2_int6_absmax_error,
    bfloat2_int6_mae_error,
    bfloat2_int6_mse_error,
    bfloat2_max_abs_8,
    bfloat2_nvfp4_absmax_error,
    bfloat2_nvfp4_dequant_bfloat2,
    bfloat2_nvfp4_mae_error,
    bfloat2_nvfp4_mse_error,
    bfloat2_to_float2_scaled,
    bfloat2x4_to_e2m0x8_values,
    bfloat2x4_to_e2m1x8_packed,
    bfloat2x4_to_int3x8_values,
    bfloat2x8_to_e2m0x16_values,
    bfloat2x4_to_int4x8_packed,
    bfloat2x8_to_int3x16_values,
    bfloat2x8_to_int4x16_packed,
    bfloat2x8_to_int4x16_packed_stochastic,
    bfloat2x8_to_int6x16_packed,
    bfloat2x8_to_e2m3x16_packed,
    bfloat2x8_to_e3m2x16_packed,
    bfloat2x8_to_e2m1x16_packed,
    bfloat2x8_to_e2m1x16_packed_stochastic,
    cvt_e2m0x4_f32_values,
    cvt_e2m0x4_to_f32,
    cvt_e2m3x4_to_f32,
    cvt_e3m2x4_to_f32,
    cvt_f32_to_e4m3,
    cvt_int3x4_f32_values,
    cvt_int3x4_to_f32,
    cvt_int4x4_to_f32,
    cvt_int6x4_to_f32,
    float2_to_bfloat2,
    float_to_ue8m0_ceil,
    get_ptr_as_int64,
    ld_global_u16,
    ld_global_v4_u32,
    nvfp4_compute_dequant_scale,
    nvfp4_compute_output_scale,
    nvfp4_compute_quant_scale_exact,
    rcp_approx_ftz,
    stochastic_round_e2m1_value,
    stochastic_round_int4_value,
    st_global_u32,
    st_global_v4_u32,
    st_global_v4_f32,
    st_global_u64,
    ue8m0_to_inv_scale,
    ue8m0_to_scale,
)


NVFP4_SCALE_BLOCK_SIZE = 16
MXFP4_SCALE_BLOCK_SIZE = 32
THREADS_PER_BLOCK = 256
PSEUDO_THREADS_PER_BLOCK = 128
MX_STATIC_THREADS_PER_BLOCK = 128
NVFP4_BASE_THREADS_PER_BLOCK = 128
STATIC_2D_THREADS_PER_BLOCK = 32
MXFP3_2D_WARPS_PER_BLOCK = 4
MXFP3_2D_THREADS_PER_BLOCK = 128
IF3_2D_GROUP_SIZE = 16
IF3_2D_GROUPS_PER_BLOCK = 16
MXFP4_BS8_2D_GROUP_SIZE = 8
MXFP4_BS8_2D_GROUPS_PER_BLOCK = THREADS_PER_BLOCK // MXFP4_BS8_2D_GROUP_SIZE
BLOCKS_PER_SM = 8
MAX_THREADS_PER_BLOCK = 1024
E4M3_STATIC_MAX = 448.0
E4M3_FOUROVERSIX_MAX = 256.0
E2M1_MAX = 6.0
E2M0_MAX = 4.0
INT3_MAX = 3.0
IF3_INT_EXPANSION_FACTOR = 1.3333333333
IF3_INT_EXPANSION_FACTOR_RCP = 0.75
INT4_MAX = 7.0
IF4_INT_EXPANSION_FACTOR = 0.8571428571
IF4_INT_EXPANSION_FACTOR_RCP = 1.16666666
IF6_E2M3_INT_EXPANSION_FACTOR = 0.2419355
IF6_E2M3_INT_EXPANSION_FACTOR_RCP = 4.1333333333
IF6_E3M2_INT_EXPANSION_FACTOR = 0.9032258065
IF6_E3M2_INT_EXPANSION_FACTOR_RCP = 1.1071428571
E2M3_MAX = 7.5
E3M2_MAX = 28.0
SCALE_RULE_MAE = 2
SCALE_RULE_MSE = 3
SCALE_RULE_ABS_MAX = 4


@cute.jit
def _compute_global_scale(
    amax_tensor: cute.Tensor,
    global_scale_multiplier: cutlass.Constexpr[float],
) -> Float32:
    amax = Float32(amax_tensor[Int32(0)])
    global_scale = Float32(0.0)
    if amax != Float32(0.0):
        global_scale = Float32(float(global_scale_multiplier)) / amax
    return global_scale


@cute.jit
def _blackwell_scale_index(
    row_idx: Int32,
    col_idx: Int32,
    scale_blocks_per_row: cutlass.Constexpr[int],
) -> Int32:
    row_group = row_idx // Int32(128)
    row_in_group = row_idx % Int32(128)
    col_group = col_idx // Int32(4)
    col_in_group = col_idx % Int32(4)
    return (
        (
            (
                row_group * Int32(scale_blocks_per_row // 4)
                + col_group
            )
            * Int32(32)
            + (row_in_group % Int32(32))
        )
        * Int32(16)
        + (row_in_group // Int32(32)) * Int32(4)
        + col_in_group
    )


@cute.jit
def _process_nvfp4_static_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    max_quantized_value: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: cutlass.Constexpr[bool] = False,
    seed_base: Uint32 = Uint32(0),
) -> tuple[Uint8, cutlass.Uint64]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4 = Uint32(0)
    h5 = Uint32(0)
    h6 = Uint32(0)
    h7 = Uint32(0)

    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(
            h0,
            h1,
            h2,
            h3,
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    fp4_max_rcp = rcp_approx_ftz(Float32(float(max_quantized_value)))
    scale_float = global_scale * (block_max * fp4_max_rcp)
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))

    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    if cutlass.const_expr(stochastic_rounding):
        packed64 = bfloat2x8_to_e2m1x16_packed_stochastic(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
            seed_base,
        )
    elif cutlass.const_expr(scale_block_size == 8):
        packed64 = cutlass.Uint64(
            bfloat2x4_to_e2m1x8_packed(h0, h1, h2, h3, output_scale),
        )
    else:
        packed64 = bfloat2x8_to_e2m1x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    return scale_fp8, packed64


@cute.jit
def _process_nvfp4_static_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
    max_quantized_value: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: cutlass.Constexpr[bool] = False,
    seed_base: Uint32 = Uint32(0),
) -> tuple[Uint8, cutlass.Uint64]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(2),
        elem_base + Int32(3),
        col_idx,
    )
    h2 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(4),
        elem_base + Int32(5),
        col_idx,
    )
    h3 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(6),
        elem_base + Int32(7),
        col_idx,
    )
    h4 = Uint32(0)
    h5 = Uint32(0)
    h6 = Uint32(0)
    h7 = Uint32(0)

    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(
            h0,
            h1,
            h2,
            h3,
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
    else:
        h4 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(8),
            elem_base + Int32(9),
            col_idx,
        )
        h5 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(10),
            elem_base + Int32(11),
            col_idx,
        )
        h6 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(12),
            elem_base + Int32(13),
            col_idx,
        )
        h7 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(14),
            elem_base + Int32(15),
            col_idx,
        )
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    fp4_max_rcp = rcp_approx_ftz(Float32(float(max_quantized_value)))
    scale_float = global_scale * (block_max * fp4_max_rcp)
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))

    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    if cutlass.const_expr(stochastic_rounding):
        packed64 = bfloat2x8_to_e2m1x16_packed_stochastic(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
            seed_base,
        )
    elif cutlass.const_expr(scale_block_size == 8):
        packed64 = cutlass.Uint64(
            bfloat2x4_to_e2m1x8_packed(h0, h1, h2, h3, output_scale),
        )
    else:
        packed64 = bfloat2x8_to_e2m1x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    return scale_fp8, packed64


@cute.jit
def _process_nvfp4_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    max_quantized_value: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4 = Uint32(0)
    h5 = Uint32(0)
    h6 = Uint32(0)
    h7 = Uint32(0)
    if cutlass.const_expr(scale_block_size != 8):
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    fp4_max_rcp = rcp_approx_ftz(Float32(float(max_quantized_value)))
    scale_float = global_scale * (block_max * fp4_max_rcp)
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    return (
        bfloat2_nvfp4_dequant_bfloat2(h0, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h1, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h2, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h3, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h4, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h5, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h6, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h7, output_scale, dequant_scale),
    )


@cute.jit
def _ld_transposed_bfloat2(
    x: cute.Tensor,
    row0: Int32,
    row1: Int32,
    col_idx: Int32,
) -> Uint32:
    ptr0 = get_ptr_as_int64(x[row0, None], col_idx)
    ptr1 = get_ptr_as_int64(x[row1, None], col_idx)
    lo = Uint32(ld_global_u16(ptr0))
    hi = Uint32(ld_global_u16(ptr1))
    return lo | (hi << Uint32(16))


@cute.jit
def _process_mxfp4_static_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    max_quantized_value: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = MXFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: cutlass.Constexpr[bool] = False,
    seed_base: Uint32 = Uint32(0),
) -> tuple[Uint8, cutlass.Uint64, cutlass.Uint64]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(2),
        elem_base + Int32(3),
        col_idx,
    )
    h2 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(4),
        elem_base + Int32(5),
        col_idx,
    )
    h3 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(6),
        elem_base + Int32(7),
        col_idx,
    )

    if cutlass.const_expr(scale_block_size == 8):
        max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3)
        block_max = bfloat2_hmax_reduce_to_f32(max0)
        normalized_max = block_max * rcp_approx_ftz(
            Float32(float(max_quantized_value)),
        )
        scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
        scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
        inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
        packed32 = bfloat2x4_to_e2m1x8_packed(h0, h1, h2, h3, inv_scale)
        return scale_ue8m0, cutlass.Uint64(packed32), cutlass.Uint64(0)
    else:
        h4 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(8),
            elem_base + Int32(9),
            col_idx,
        )
        h5 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(10),
            elem_base + Int32(11),
            col_idx,
        )
        h6 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(12),
            elem_base + Int32(13),
            col_idx,
        )
        h7 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(14),
            elem_base + Int32(15),
            col_idx,
        )
        h8 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(16),
            elem_base + Int32(17),
            col_idx,
        )
        h9 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(18),
            elem_base + Int32(19),
            col_idx,
        )
        h10 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(20),
            elem_base + Int32(21),
            col_idx,
        )
        h11 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(22),
            elem_base + Int32(23),
            col_idx,
        )
        h12 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(24),
            elem_base + Int32(25),
            col_idx,
        )
        h13 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(26),
            elem_base + Int32(27),
            col_idx,
        )
        h14 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(28),
            elem_base + Int32(29),
            col_idx,
        )
        h15 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(30),
            elem_base + Int32(31),
            col_idx,
        )
        max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
        max1 = bfloat2_max_abs_8(h8, h9, h10, h11, h12, h13, h14, h15)
        block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))

    normalized_max = block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
    scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
    inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

    if cutlass.const_expr(stochastic_rounding):
        packed64_0 = bfloat2x8_to_e2m1x16_packed_stochastic(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            inv_scale,
            seed_base,
        )
        packed64_1 = bfloat2x8_to_e2m1x16_packed_stochastic(
            h8,
            h9,
            h10,
            h11,
            h12,
            h13,
            h14,
            h15,
            inv_scale,
            seed_base + Uint32(16),
        )
    else:
        packed64_0 = bfloat2x8_to_e2m1x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            inv_scale,
        )
        packed64_1 = bfloat2x8_to_e2m1x16_packed(
            h8,
            h9,
            h10,
            h11,
            h12,
            h13,
            h14,
            h15,
            inv_scale,
        )
    return scale_ue8m0, packed64_0, packed64_1


@cute.jit
def _process_mxfp4_static_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    max_quantized_value: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = MXFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: cutlass.Constexpr[bool] = False,
    seed_base: Uint32 = Uint32(0),
) -> tuple[Uint8, cutlass.Uint64, cutlass.Uint64]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    if cutlass.const_expr(scale_block_size == 8):
        max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3)
        block_max = bfloat2_hmax_reduce_to_f32(max0)
        normalized_max = block_max * rcp_approx_ftz(
            Float32(float(max_quantized_value)),
        )
        scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
        scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
        inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
        packed32 = bfloat2x4_to_e2m1x8_packed(h0, h1, h2, h3, inv_scale)
        return scale_ue8m0, cutlass.Uint64(packed32), cutlass.Uint64(0)
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        ptr2 = get_ptr_as_int64(row_tensor, elem_base + Int32(16))
        ptr3 = get_ptr_as_int64(row_tensor, elem_base + Int32(24))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
        h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
        h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
        max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
        max1 = bfloat2_max_abs_8(h8, h9, h10, h11, h12, h13, h14, h15)
        block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))

    normalized_max = block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
    scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
    inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

    if cutlass.const_expr(stochastic_rounding):
        packed64_0 = bfloat2x8_to_e2m1x16_packed_stochastic(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            inv_scale,
            seed_base,
        )
        packed64_1 = bfloat2x8_to_e2m1x16_packed_stochastic(
            h8,
            h9,
            h10,
            h11,
            h12,
            h13,
            h14,
            h15,
            inv_scale,
            seed_base + Uint32(16),
        )
    else:
        packed64_0 = bfloat2x8_to_e2m1x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            inv_scale,
        )
        packed64_1 = bfloat2x8_to_e2m1x16_packed(
            h8,
            h9,
            h10,
            h11,
            h12,
            h13,
            h14,
            h15,
            inv_scale,
        )
    return scale_ue8m0, packed64_0, packed64_1


@cute.jit
def _process_mxfp4_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    max_quantized_value: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = MXFP4_SCALE_BLOCK_SIZE,
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    if cutlass.const_expr(scale_block_size == 8):
        max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3)
        block_max = bfloat2_hmax_reduce_to_f32(max0)
        normalized_max = block_max * rcp_approx_ftz(
            Float32(float(max_quantized_value)),
        )
        scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
        inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
        dequant_scale = ue8m0_to_scale(scale_ue8m0_u32)
        z = cutlass.Uint32(0)
        return (
            bfloat2_nvfp4_dequant_bfloat2(h0, inv_scale, dequant_scale),
            bfloat2_nvfp4_dequant_bfloat2(h1, inv_scale, dequant_scale),
            bfloat2_nvfp4_dequant_bfloat2(h2, inv_scale, dequant_scale),
            bfloat2_nvfp4_dequant_bfloat2(h3, inv_scale, dequant_scale),
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
        )

    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
    ptr2 = get_ptr_as_int64(row_tensor, elem_base + Int32(16))
    ptr3 = get_ptr_as_int64(row_tensor, elem_base + Int32(24))
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
    h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
    h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
    max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    max1 = bfloat2_max_abs_8(h8, h9, h10, h11, h12, h13, h14, h15)
    block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))
    normalized_max = block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
    inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
    dequant_scale = ue8m0_to_scale(scale_ue8m0_u32)

    return (
        bfloat2_nvfp4_dequant_bfloat2(h0, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h1, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h2, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h3, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h4, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h5, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h6, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h7, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h8, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h9, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h10, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h11, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h12, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h13, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h14, inv_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h15, inv_scale, dequant_scale),
    )


@cute.jit
def _dequant_e2m0x4_to_bfloat2x2(
    packed: Uint32,
    dequant_scale: Float32,
) -> tuple[Uint32, Uint32]:
    f0, f1, f2, f3 = cvt_e2m0x4_to_f32(packed)
    return (
        float2_to_bfloat2(f0 * dequant_scale, f1 * dequant_scale),
        float2_to_bfloat2(f2 * dequant_scale, f3 * dequant_scale),
    )


@cute.jit
def _process_nvfp3_static_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
    scale_block_size: cutlass.Constexpr[int],
) -> tuple[Uint8, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(x, elem_base + Int32(2), elem_base + Int32(3), col_idx)
    h2 = _ld_transposed_bfloat2(x, elem_base + Int32(4), elem_base + Int32(5), col_idx)
    h3 = _ld_transposed_bfloat2(x, elem_base + Int32(6), elem_base + Int32(7), col_idx)

    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3)
    else:
        h4 = _ld_transposed_bfloat2(x, elem_base + Int32(8), elem_base + Int32(9), col_idx)
        h5 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(10),
            elem_base + Int32(11),
            col_idx,
        )
        h6 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(12),
            elem_base + Int32(13),
            col_idx,
        )
        h7 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(14),
            elem_base + Int32(15),
            col_idx,
        )
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)

    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(4.0)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(scale_block_size == 8):
        v0, v1 = bfloat2x4_to_e2m0x8_values(h0, h1, h2, h3, output_scale)
        return scale_fp8, v0, v1, Uint32(0), Uint32(0)

    v0, v1, v2, v3 = bfloat2x8_to_e2m0x16_values(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale,
    )
    return scale_fp8, v0, v1, v2, v3


@cute.jit
def _process_nvfp3_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_block_size: cutlass.Constexpr[int],
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    h4 = Uint32(0)
    h5 = Uint32(0)
    h6 = Uint32(0)
    h7 = Uint32(0)
    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3)
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)

    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M0_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(scale_block_size == 8):
        v0, v1 = bfloat2x4_to_e2m0x8_values(
            h0,
            h1,
            h2,
            h3,
            output_scale,
        )
        out0, out1 = _dequant_e2m0x4_to_bfloat2x2(v0, dequant_scale)
        out2, out3 = _dequant_e2m0x4_to_bfloat2x2(v1, dequant_scale)
        z = cutlass.Uint32(0)
        return (out0, out1, out2, out3, z, z, z, z)

    v0, v1, v2, v3 = bfloat2x8_to_e2m0x16_values(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale,
    )
    out0, out1 = _dequant_e2m0x4_to_bfloat2x2(v0, dequant_scale)
    out2, out3 = _dequant_e2m0x4_to_bfloat2x2(v1, dequant_scale)
    out4, out5 = _dequant_e2m0x4_to_bfloat2x2(v2, dequant_scale)
    out6, out7 = _dequant_e2m0x4_to_bfloat2x2(v3, dequant_scale)
    return (out0, out1, out2, out3, out4, out5, out6, out7)


@cute.jit
def _process_mxfp3_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    scale_block_size: cutlass.Constexpr[int],
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    if cutlass.const_expr(scale_block_size == 8):
        block_max = bfloat2_hmax_reduce_to_f32(
            bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3),
        )
        normalized_max = block_max * rcp_approx_ftz(Float32(4.0))
        scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
        inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
        dequant_scale = ue8m0_to_scale(scale_ue8m0_u32)
        v0, v1 = bfloat2x4_to_e2m0x8_values(
            h0,
            h1,
            h2,
            h3,
            inv_scale,
        )
        out0, out1 = _dequant_e2m0x4_to_bfloat2x2(v0, dequant_scale)
        out2, out3 = _dequant_e2m0x4_to_bfloat2x2(v1, dequant_scale)
        z = cutlass.Uint32(0)
        return (
            out0,
            out1,
            out2,
            out3,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
        )

    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
    ptr2 = get_ptr_as_int64(row_tensor, elem_base + Int32(16))
    ptr3 = get_ptr_as_int64(row_tensor, elem_base + Int32(24))
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
    h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
    h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
    max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    max1 = bfloat2_max_abs_8(h8, h9, h10, h11, h12, h13, h14, h15)
    block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))
    normalized_max = block_max * rcp_approx_ftz(Float32(4.0))
    scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
    inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
    dequant_scale = ue8m0_to_scale(scale_ue8m0_u32)
    v0, v1, v2, v3 = bfloat2x8_to_e2m0x16_values(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        inv_scale,
    )
    v4, v5, v6, v7 = bfloat2x8_to_e2m0x16_values(
        h8,
        h9,
        h10,
        h11,
        h12,
        h13,
        h14,
        h15,
        inv_scale,
    )
    out0, out1 = _dequant_e2m0x4_to_bfloat2x2(v0, dequant_scale)
    out2, out3 = _dequant_e2m0x4_to_bfloat2x2(v1, dequant_scale)
    out4, out5 = _dequant_e2m0x4_to_bfloat2x2(v2, dequant_scale)
    out6, out7 = _dequant_e2m0x4_to_bfloat2x2(v3, dequant_scale)
    out8, out9 = _dequant_e2m0x4_to_bfloat2x2(v4, dequant_scale)
    out10, out11 = _dequant_e2m0x4_to_bfloat2x2(v5, dequant_scale)
    out12, out13 = _dequant_e2m0x4_to_bfloat2x2(v6, dequant_scale)
    out14, out15 = _dequant_e2m0x4_to_bfloat2x2(v7, dequant_scale)
    return (
        out0,
        out1,
        out2,
        out3,
        out4,
        out5,
        out6,
        out7,
        out8,
        out9,
        out10,
        out11,
        out12,
        out13,
        out14,
        out15,
    )


@cute.jit
def _dequant_fp6x4_to_bfloat2x2(
    packed: Uint32,
    dequant_scale: Float32,
    use_e3m2: cutlass.Constexpr[bool],
) -> tuple[Uint32, Uint32]:
    if cutlass.const_expr(use_e3m2):
        f0, f1, f2, f3 = cvt_e3m2x4_to_f32(packed)
    else:
        f0, f1, f2, f3 = cvt_e2m3x4_to_f32(packed)
    return (
        float2_to_bfloat2(f0 * dequant_scale, f1 * dequant_scale),
        float2_to_bfloat2(f2 * dequant_scale, f3 * dequant_scale),
    )


@cute.jit
def _dequant_int6x4_to_bfloat2x2(
    packed: Uint32,
    dequant_scale: Float32,
) -> tuple[Uint32, Uint32]:
    f0, f1, f2, f3 = cvt_int6x4_to_f32(packed, dequant_scale)
    return (
        float2_to_bfloat2(f0, f1),
        float2_to_bfloat2(f2, f3),
    )


@cute.jit
def _dequant_int3x4_to_bfloat2x2(
    packed: Uint32,
    dequant_scale: Float32,
) -> tuple[Uint32, Uint32]:
    f0, f1, f2, f3 = cvt_int3x4_to_f32(packed)
    return (
        float2_to_bfloat2(f0 * dequant_scale, f1 * dequant_scale),
        float2_to_bfloat2(f2 * dequant_scale, f3 * dequant_scale),
    )


@cute.jit
def _dequant_int4x4_to_bfloat2x2(
    packed: Uint32,
    dequant_scale: Float32,
) -> tuple[Uint32, Uint32]:
    f0, f1, f2, f3 = cvt_int4x4_to_f32(packed)
    return (
        float2_to_bfloat2(f0 * dequant_scale, f1 * dequant_scale),
        float2_to_bfloat2(f2 * dequant_scale, f3 * dequant_scale),
    )


def _process_mxfp6_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    max_quantized_value: cutlass.Constexpr[float],
    use_e3m2: cutlass.Constexpr[bool],
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
    ptr2 = get_ptr_as_int64(row_tensor, elem_base + Int32(16))
    ptr3 = get_ptr_as_int64(row_tensor, elem_base + Int32(24))
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
    h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
    h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
    max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    max1 = bfloat2_max_abs_8(h8, h9, h10, h11, h12, h13, h14, h15)
    block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))
    normalized_max = block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
    inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
    dequant_scale = ue8m0_to_scale(scale_ue8m0_u32)

    if cutlass.const_expr(use_e3m2):
        packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            inv_scale,
        )
        packed64_2, packed64_3 = bfloat2x8_to_e3m2x16_packed(
            h8,
            h9,
            h10,
            h11,
            h12,
            h13,
            h14,
            h15,
            inv_scale,
        )
    else:
        packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            inv_scale,
        )
        packed64_2, packed64_3 = bfloat2x8_to_e2m3x16_packed(
            h8,
            h9,
            h10,
            h11,
            h12,
            h13,
            h14,
            h15,
            inv_scale,
        )

    p0 = Uint32(packed64_0 & cutlass.Uint64(0xFFFFFFFF))
    p1 = Uint32(packed64_0 >> cutlass.Uint64(32))
    p2 = Uint32(packed64_1 & cutlass.Uint64(0xFFFFFFFF))
    p3 = Uint32(packed64_1 >> cutlass.Uint64(32))
    p4 = Uint32(packed64_2 & cutlass.Uint64(0xFFFFFFFF))
    p5 = Uint32(packed64_2 >> cutlass.Uint64(32))
    p6 = Uint32(packed64_3 & cutlass.Uint64(0xFFFFFFFF))
    p7 = Uint32(packed64_3 >> cutlass.Uint64(32))

    out0, out1 = _dequant_fp6x4_to_bfloat2x2(p0, dequant_scale, use_e3m2)
    out2, out3 = _dequant_fp6x4_to_bfloat2x2(p1, dequant_scale, use_e3m2)
    out4, out5 = _dequant_fp6x4_to_bfloat2x2(p2, dequant_scale, use_e3m2)
    out6, out7 = _dequant_fp6x4_to_bfloat2x2(p3, dequant_scale, use_e3m2)
    out8, out9 = _dequant_fp6x4_to_bfloat2x2(p4, dequant_scale, use_e3m2)
    out10, out11 = _dequant_fp6x4_to_bfloat2x2(p5, dequant_scale, use_e3m2)
    out12, out13 = _dequant_fp6x4_to_bfloat2x2(p6, dequant_scale, use_e3m2)
    out14, out15 = _dequant_fp6x4_to_bfloat2x2(p7, dequant_scale, use_e3m2)
    return (
        out0,
        out1,
        out2,
        out3,
        out4,
        out5,
        out6,
        out7,
        out8,
        out9,
        out10,
        out11,
        out12,
        out13,
        out14,
        out15,
    )


@cute.jit
def _process_nvint3_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_block_size: cutlass.Constexpr[int],
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    h4 = Uint32(0)
    h5 = Uint32(0)
    h6 = Uint32(0)
    h7 = Uint32(0)
    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(
            h0,
            h1,
            h2,
            h3,
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)

    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(3.0)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(scale_block_size == 8):
        v0, v1 = bfloat2x4_to_int3x8_values(
            h0,
            h1,
            h2,
            h3,
            output_scale,
        )
        out0, out1 = _dequant_int3x4_to_bfloat2x2(v0, dequant_scale)
        out2, out3 = _dequant_int3x4_to_bfloat2x2(v1, dequant_scale)
        z = cutlass.Uint32(0)
        return (out0, out1, out2, out3, z, z, z, z)

    v0, v1, v2, v3 = bfloat2x8_to_int3x16_values(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale,
    )
    out0, out1 = _dequant_int3x4_to_bfloat2x2(v0, dequant_scale)
    out2, out3 = _dequant_int3x4_to_bfloat2x2(v1, dequant_scale)
    out4, out5 = _dequant_int3x4_to_bfloat2x2(v2, dequant_scale)
    out6, out7 = _dequant_int3x4_to_bfloat2x2(v3, dequant_scale)
    return (out0, out1, out2, out3, out4, out5, out6, out7)


@cute.jit
def _process_nvint6_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(31.0)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    packed64_0, packed64_1 = bfloat2x8_to_int6x16_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale,
    )
    p0 = Uint32(packed64_0 & cutlass.Uint64(0xFFFFFFFF))
    p1 = Uint32(packed64_0 >> cutlass.Uint64(32))
    p2 = Uint32(packed64_1 & cutlass.Uint64(0xFFFFFFFF))
    p3 = Uint32(packed64_1 >> cutlass.Uint64(32))
    out0, out1 = _dequant_int6x4_to_bfloat2x2(p0, dequant_scale)
    out2, out3 = _dequant_int6x4_to_bfloat2x2(p1, dequant_scale)
    out4, out5 = _dequant_int6x4_to_bfloat2x2(p2, dequant_scale)
    out6, out7 = _dequant_int6x4_to_bfloat2x2(p3, dequant_scale)
    return (out0, out1, out2, out3, out4, out5, out6, out7)


@cute.jit
def _process_nvint6_static_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
) -> tuple[Uint8, cutlass.Uint64, cutlass.Uint64]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(x, elem_base + Int32(2), elem_base + Int32(3), col_idx)
    h2 = _ld_transposed_bfloat2(x, elem_base + Int32(4), elem_base + Int32(5), col_idx)
    h3 = _ld_transposed_bfloat2(x, elem_base + Int32(6), elem_base + Int32(7), col_idx)
    h4 = _ld_transposed_bfloat2(x, elem_base + Int32(8), elem_base + Int32(9), col_idx)
    h5 = _ld_transposed_bfloat2(x, elem_base + Int32(10), elem_base + Int32(11), col_idx)
    h6 = _ld_transposed_bfloat2(x, elem_base + Int32(12), elem_base + Int32(13), col_idx)
    h7 = _ld_transposed_bfloat2(x, elem_base + Int32(14), elem_base + Int32(15), col_idx)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(31.0)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

    packed64_0, packed64_1 = bfloat2x8_to_int6x16_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale,
    )
    return scale_fp8, packed64_0, packed64_1


@cute.jit
def _process_nvint3_static_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
    scale_block_size: cutlass.Constexpr[int],
) -> tuple[Uint8, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(x, elem_base + Int32(2), elem_base + Int32(3), col_idx)
    h2 = _ld_transposed_bfloat2(x, elem_base + Int32(4), elem_base + Int32(5), col_idx)
    h3 = _ld_transposed_bfloat2(x, elem_base + Int32(6), elem_base + Int32(7), col_idx)

    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(
            h0,
            h1,
            h2,
            h3,
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
    else:
        h4 = _ld_transposed_bfloat2(x, elem_base + Int32(8), elem_base + Int32(9), col_idx)
        h5 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(10),
            elem_base + Int32(11),
            col_idx,
        )
        h6 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(12),
            elem_base + Int32(13),
            col_idx,
        )
        h7 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(14),
            elem_base + Int32(15),
            col_idx,
        )
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)

    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(3.0)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(scale_block_size == 8):
        v0, v1 = bfloat2x4_to_int3x8_values(h0, h1, h2, h3, output_scale)
        return scale_fp8, v0, v1, Uint32(0), Uint32(0)

    v0, v1, v2, v3 = bfloat2x8_to_int3x16_values(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale,
    )
    return scale_fp8, v0, v1, v2, v3


@cute.jit
def _process_nvint4_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_block_size: cutlass.Constexpr[int],
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    h4 = Uint32(0)
    h5 = Uint32(0)
    h6 = Uint32(0)
    h7 = Uint32(0)
    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(
            h0,
            h1,
            h2,
            h3,
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)

    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(7.0)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(scale_block_size == 8):
        packed32 = bfloat2x4_to_int4x8_packed(
            h0,
            h1,
            h2,
            h3,
            output_scale,
        )
        p0 = Uint32(packed32 & Uint32(0xFFFF))
        p1 = Uint32(packed32 >> Uint32(16))
        out0, out1 = _dequant_int4x4_to_bfloat2x2(p0, dequant_scale)
        out2, out3 = _dequant_int4x4_to_bfloat2x2(p1, dequant_scale)
        z = cutlass.Uint32(0)
        return (out0, out1, out2, out3, z, z, z, z)

    packed64 = bfloat2x8_to_int4x16_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale,
    )
    p0 = Uint32(packed64 & cutlass.Uint64(0xFFFF))
    p1 = Uint32((packed64 >> cutlass.Uint64(16)) & cutlass.Uint64(0xFFFF))
    p2 = Uint32((packed64 >> cutlass.Uint64(32)) & cutlass.Uint64(0xFFFF))
    p3 = Uint32((packed64 >> cutlass.Uint64(48)) & cutlass.Uint64(0xFFFF))
    out0, out1 = _dequant_int4x4_to_bfloat2x2(p0, dequant_scale)
    out2, out3 = _dequant_int4x4_to_bfloat2x2(p1, dequant_scale)
    out4, out5 = _dequant_int4x4_to_bfloat2x2(p2, dequant_scale)
    out6, out7 = _dequant_int4x4_to_bfloat2x2(p3, dequant_scale)
    return (out0, out1, out2, out3, out4, out5, out6, out7)


@cute.jit
def _process_nvfp6_static_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    max_quantized_value: cutlass.Constexpr[float],
    use_e3m2: cutlass.Constexpr[bool],
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (
        block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    )
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(use_e3m2):
        packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    else:
        packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )

    p0 = Uint32(packed64_0 & cutlass.Uint64(0xFFFFFFFF))
    p1 = Uint32(packed64_0 >> cutlass.Uint64(32))
    p2 = Uint32(packed64_1 & cutlass.Uint64(0xFFFFFFFF))
    p3 = Uint32(packed64_1 >> cutlass.Uint64(32))
    out0, out1 = _dequant_fp6x4_to_bfloat2x2(p0, dequant_scale, use_e3m2)
    out2, out3 = _dequant_fp6x4_to_bfloat2x2(p1, dequant_scale, use_e3m2)
    out4, out5 = _dequant_fp6x4_to_bfloat2x2(p2, dequant_scale, use_e3m2)
    out6, out7 = _dequant_fp6x4_to_bfloat2x2(p3, dequant_scale, use_e3m2)
    return (out0, out1, out2, out3, out4, out5, out6, out7)


@cute.jit
def _process_nvint4_static_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: cutlass.Constexpr[bool] = False,
    seed_base: Uint32 = Uint32(0),
) -> tuple[Uint8, cutlass.Uint64]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(
            h0,
            h1,
            h2,
            h3,
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)

    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(INT4_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))

    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    if cutlass.const_expr(scale_block_size == 8):
        packed32 = bfloat2x4_to_int4x8_packed(
            h0,
            h1,
            h2,
            h3,
            output_scale,
        )
        packed64 = cutlass.Uint64(packed32)
    else:
        if cutlass.const_expr(stochastic_rounding):
            packed64 = bfloat2x8_to_int4x16_packed_stochastic(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale,
                seed_base,
            )
        else:
            packed64 = bfloat2x8_to_int4x16_packed(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale,
            )
    return scale_fp8, packed64


@cute.jit
def _process_nvint4_static_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: cutlass.Constexpr[bool] = False,
    seed_base: Uint32 = Uint32(0),
) -> tuple[Uint8, cutlass.Uint64]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(x, elem_base + Int32(2), elem_base + Int32(3), col_idx)
    h2 = _ld_transposed_bfloat2(x, elem_base + Int32(4), elem_base + Int32(5), col_idx)
    h3 = _ld_transposed_bfloat2(x, elem_base + Int32(6), elem_base + Int32(7), col_idx)

    if cutlass.const_expr(scale_block_size == 8):
        block_max_h2 = bfloat2_max_abs_8(
            h0,
            h1,
            h2,
            h3,
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
    else:
        h4 = _ld_transposed_bfloat2(x, elem_base + Int32(8), elem_base + Int32(9), col_idx)
        h5 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(10),
            elem_base + Int32(11),
            col_idx,
        )
        h6 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(12),
            elem_base + Int32(13),
            col_idx,
        )
        h7 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(14),
            elem_base + Int32(15),
            col_idx,
        )
        block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)

    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(INT4_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    if cutlass.const_expr(scale_block_size == 8):
        packed32 = bfloat2x4_to_int4x8_packed(h0, h1, h2, h3, output_scale)
        packed64 = cutlass.Uint64(packed32)
    else:
        if cutlass.const_expr(stochastic_rounding):
            packed64 = bfloat2x8_to_int4x16_packed_stochastic(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale,
                seed_base,
            )
        else:
            packed64 = bfloat2x8_to_int4x16_packed(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale,
            )
    return scale_fp8, packed64


@cute.jit
def _process_nvfp6_static_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    max_quantized_value: cutlass.Constexpr[float],
    use_e3m2: cutlass.Constexpr[bool],
) -> tuple[Uint8, cutlass.Uint64, cutlass.Uint64]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float = global_scale * (
        block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    )
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(use_e3m2):
        packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    else:
        packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    return scale_fp8, packed64_0, packed64_1


@cute.jit
def _process_nvfp6_static_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
    max_quantized_value: cutlass.Constexpr[float],
    use_e3m2: cutlass.Constexpr[bool],
) -> tuple[Uint8, cutlass.Uint64, cutlass.Uint64]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(2),
        elem_base + Int32(3),
        col_idx,
    )
    h2 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(4),
        elem_base + Int32(5),
        col_idx,
    )
    h3 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(6),
        elem_base + Int32(7),
        col_idx,
    )
    h4 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(8),
        elem_base + Int32(9),
        col_idx,
    )
    h5 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(10),
        elem_base + Int32(11),
        col_idx,
    )
    h6 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(12),
        elem_base + Int32(13),
        col_idx,
    )
    h7 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(14),
        elem_base + Int32(15),
        col_idx,
    )

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float = global_scale * (
        block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    )
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(use_e3m2):
        packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    else:
        packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    return scale_fp8, packed64_0, packed64_1


@cute.jit
def _nvfp4_block_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    h2: cutlass.Uint32,
    h3: cutlass.Uint32,
    h4: cutlass.Uint32,
    h5: cutlass.Uint32,
    h6: cutlass.Uint32,
    h7: cutlass.Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err01 = bfloat2_nvfp4_absmax_error(h0, output_scale, dequant_scale)
        err23 = bfloat2_nvfp4_absmax_error(h1, output_scale, dequant_scale)
        err45 = bfloat2_nvfp4_absmax_error(h2, output_scale, dequant_scale)
        err67 = bfloat2_nvfp4_absmax_error(h3, output_scale, dequant_scale)
        err89 = bfloat2_nvfp4_absmax_error(h4, output_scale, dequant_scale)
        err1011 = bfloat2_nvfp4_absmax_error(h5, output_scale, dequant_scale)
        err1213 = bfloat2_nvfp4_absmax_error(h6, output_scale, dequant_scale)
        err1415 = bfloat2_nvfp4_absmax_error(h7, output_scale, dequant_scale)
        err = cutlass.max(err01, err23)
        err = cutlass.max(err, err45)
        err = cutlass.max(err, err67)
        err = cutlass.max(err, err89)
        err = cutlass.max(err, err1011)
        err = cutlass.max(err, err1213)
        return cutlass.max(err, err1415)
    elif cutlass.const_expr(scale_rule_id == SCALE_RULE_MAE):
        return (
            bfloat2_nvfp4_mae_error(h0, output_scale, dequant_scale)
            + bfloat2_nvfp4_mae_error(h1, output_scale, dequant_scale)
            + bfloat2_nvfp4_mae_error(h2, output_scale, dequant_scale)
            + bfloat2_nvfp4_mae_error(h3, output_scale, dequant_scale)
            + bfloat2_nvfp4_mae_error(h4, output_scale, dequant_scale)
            + bfloat2_nvfp4_mae_error(h5, output_scale, dequant_scale)
            + bfloat2_nvfp4_mae_error(h6, output_scale, dequant_scale)
            + bfloat2_nvfp4_mae_error(h7, output_scale, dequant_scale)
        )
    else:
        return (
            bfloat2_nvfp4_mse_error(h0, output_scale, dequant_scale)
            + bfloat2_nvfp4_mse_error(h1, output_scale, dequant_scale)
            + bfloat2_nvfp4_mse_error(h2, output_scale, dequant_scale)
            + bfloat2_nvfp4_mse_error(h3, output_scale, dequant_scale)
            + bfloat2_nvfp4_mse_error(h4, output_scale, dequant_scale)
            + bfloat2_nvfp4_mse_error(h5, output_scale, dequant_scale)
            + bfloat2_nvfp4_mse_error(h6, output_scale, dequant_scale)
            + bfloat2_nvfp4_mse_error(h7, output_scale, dequant_scale)
        )


@cute.jit
def _abs_f32(x: Float32) -> Float32:
    out = x
    if out < Float32(0.0):
        out = -out
    return out


@cute.jit
def _nvfp4_stochastic_pair_error_bfloat(
    h2: cutlass.Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    seed_base: Uint32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    x0, x1 = bfloat2_to_float2_scaled(h2, Float32(1.0))
    s0, s1 = bfloat2_to_float2_scaled(h2, output_scale)
    q0 = stochastic_round_e2m1_value(s0, seed_base)
    q1 = stochastic_round_e2m1_value(s1, seed_base + Uint32(1))
    d0 = _abs_f32(q0 * dequant_scale - x0)
    d1 = _abs_f32(q1 * dequant_scale - x1)

    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        return cutlass.max(d0, d1)
    elif cutlass.const_expr(scale_rule_id == SCALE_RULE_MAE):
        return d0 + d1
    else:
        return d0 * d0 + d1 * d1


@cute.jit
def _nvfp4_stochastic_block_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    h2: cutlass.Uint32,
    h3: cutlass.Uint32,
    h4: cutlass.Uint32,
    h5: cutlass.Uint32,
    h6: cutlass.Uint32,
    h7: cutlass.Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    seed_base: Uint32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    err01 = _nvfp4_stochastic_pair_error_bfloat(
        h0, output_scale, dequant_scale, seed_base + Uint32(0), scale_rule_id
    )
    err23 = _nvfp4_stochastic_pair_error_bfloat(
        h1, output_scale, dequant_scale, seed_base + Uint32(2), scale_rule_id
    )
    err45 = _nvfp4_stochastic_pair_error_bfloat(
        h2, output_scale, dequant_scale, seed_base + Uint32(4), scale_rule_id
    )
    err67 = _nvfp4_stochastic_pair_error_bfloat(
        h3, output_scale, dequant_scale, seed_base + Uint32(6), scale_rule_id
    )
    err89 = _nvfp4_stochastic_pair_error_bfloat(
        h4, output_scale, dequant_scale, seed_base + Uint32(8), scale_rule_id
    )
    err1011 = _nvfp4_stochastic_pair_error_bfloat(
        h5, output_scale, dequant_scale, seed_base + Uint32(10), scale_rule_id
    )
    err1213 = _nvfp4_stochastic_pair_error_bfloat(
        h6, output_scale, dequant_scale, seed_base + Uint32(12), scale_rule_id
    )
    err1415 = _nvfp4_stochastic_pair_error_bfloat(
        h7, output_scale, dequant_scale, seed_base + Uint32(14), scale_rule_id
    )

    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err = cutlass.max(err01, err23)
        err = cutlass.max(err, err45)
        err = cutlass.max(err, err67)
        err = cutlass.max(err, err89)
        err = cutlass.max(err, err1011)
        err = cutlass.max(err, err1213)
        return cutlass.max(err, err1415)
    return err01 + err23 + err45 + err67 + err89 + err1011 + err1213 + err1415


@cute.jit
def _int4_block_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    h2: cutlass.Uint32,
    h3: cutlass.Uint32,
    h4: cutlass.Uint32,
    h5: cutlass.Uint32,
    h6: cutlass.Uint32,
    h7: cutlass.Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err01 = bfloat2_int4_absmax_error(h0, output_scale, dequant_scale)
        err23 = bfloat2_int4_absmax_error(h1, output_scale, dequant_scale)
        err45 = bfloat2_int4_absmax_error(h2, output_scale, dequant_scale)
        err67 = bfloat2_int4_absmax_error(h3, output_scale, dequant_scale)
        err89 = bfloat2_int4_absmax_error(h4, output_scale, dequant_scale)
        err1011 = bfloat2_int4_absmax_error(h5, output_scale, dequant_scale)
        err1213 = bfloat2_int4_absmax_error(h6, output_scale, dequant_scale)
        err1415 = bfloat2_int4_absmax_error(h7, output_scale, dequant_scale)
        err = cutlass.max(err01, err23)
        err = cutlass.max(err, err45)
        err = cutlass.max(err, err67)
        err = cutlass.max(err, err89)
        err = cutlass.max(err, err1011)
        err = cutlass.max(err, err1213)
        return cutlass.max(err, err1415)
    elif cutlass.const_expr(scale_rule_id == SCALE_RULE_MAE):
        return (
            bfloat2_int4_mae_error(h0, output_scale, dequant_scale)
            + bfloat2_int4_mae_error(h1, output_scale, dequant_scale)
            + bfloat2_int4_mae_error(h2, output_scale, dequant_scale)
            + bfloat2_int4_mae_error(h3, output_scale, dequant_scale)
            + bfloat2_int4_mae_error(h4, output_scale, dequant_scale)
            + bfloat2_int4_mae_error(h5, output_scale, dequant_scale)
            + bfloat2_int4_mae_error(h6, output_scale, dequant_scale)
            + bfloat2_int4_mae_error(h7, output_scale, dequant_scale)
        )
    else:
        return (
            bfloat2_int4_mse_error(h0, output_scale, dequant_scale)
            + bfloat2_int4_mse_error(h1, output_scale, dequant_scale)
            + bfloat2_int4_mse_error(h2, output_scale, dequant_scale)
            + bfloat2_int4_mse_error(h3, output_scale, dequant_scale)
            + bfloat2_int4_mse_error(h4, output_scale, dequant_scale)
            + bfloat2_int4_mse_error(h5, output_scale, dequant_scale)
            + bfloat2_int4_mse_error(h6, output_scale, dequant_scale)
            + bfloat2_int4_mse_error(h7, output_scale, dequant_scale)
        )


@cute.jit
def _int4_stochastic_pair_error_bfloat(
    h2: cutlass.Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    seed_base: Uint32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    x0, x1 = bfloat2_to_float2_scaled(h2, Float32(1.0))
    s0, s1 = bfloat2_to_float2_scaled(h2, output_scale)
    q0 = stochastic_round_int4_value(s0, seed_base)
    q1 = stochastic_round_int4_value(s1, seed_base + Uint32(1))
    d0 = _abs_f32(q0 * dequant_scale - x0)
    d1 = _abs_f32(q1 * dequant_scale - x1)

    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        return cutlass.max(d0, d1)
    elif cutlass.const_expr(scale_rule_id == SCALE_RULE_MAE):
        return d0 + d1
    else:
        return d0 * d0 + d1 * d1


@cute.jit
def _int4_stochastic_block_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    h2: cutlass.Uint32,
    h3: cutlass.Uint32,
    h4: cutlass.Uint32,
    h5: cutlass.Uint32,
    h6: cutlass.Uint32,
    h7: cutlass.Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    seed_base: Uint32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    err01 = _int4_stochastic_pair_error_bfloat(
        h0, output_scale, dequant_scale, seed_base + Uint32(0), scale_rule_id
    )
    err23 = _int4_stochastic_pair_error_bfloat(
        h1, output_scale, dequant_scale, seed_base + Uint32(2), scale_rule_id
    )
    err45 = _int4_stochastic_pair_error_bfloat(
        h2, output_scale, dequant_scale, seed_base + Uint32(4), scale_rule_id
    )
    err67 = _int4_stochastic_pair_error_bfloat(
        h3, output_scale, dequant_scale, seed_base + Uint32(6), scale_rule_id
    )
    err89 = _int4_stochastic_pair_error_bfloat(
        h4, output_scale, dequant_scale, seed_base + Uint32(8), scale_rule_id
    )
    err1011 = _int4_stochastic_pair_error_bfloat(
        h5, output_scale, dequant_scale, seed_base + Uint32(10), scale_rule_id
    )
    err1213 = _int4_stochastic_pair_error_bfloat(
        h6, output_scale, dequant_scale, seed_base + Uint32(12), scale_rule_id
    )
    err1415 = _int4_stochastic_pair_error_bfloat(
        h7, output_scale, dequant_scale, seed_base + Uint32(14), scale_rule_id
    )

    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err = cutlass.max(err01, err23)
        err = cutlass.max(err, err45)
        err = cutlass.max(err, err67)
        err = cutlass.max(err, err89)
        err = cutlass.max(err, err1011)
        err = cutlass.max(err, err1213)
        return cutlass.max(err, err1415)
    return err01 + err23 + err45 + err67 + err89 + err1011 + err1213 + err1415


@cute.jit
def _fp6_block_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    h2: cutlass.Uint32,
    h3: cutlass.Uint32,
    h4: cutlass.Uint32,
    h5: cutlass.Uint32,
    h6: cutlass.Uint32,
    h7: cutlass.Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    use_e3m2: cutlass.Constexpr[bool],
) -> Float32:
    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err01 = bfloat2_fp6_absmax_error(h0, output_scale, dequant_scale, use_e3m2)
        err23 = bfloat2_fp6_absmax_error(h1, output_scale, dequant_scale, use_e3m2)
        err45 = bfloat2_fp6_absmax_error(h2, output_scale, dequant_scale, use_e3m2)
        err67 = bfloat2_fp6_absmax_error(h3, output_scale, dequant_scale, use_e3m2)
        err89 = bfloat2_fp6_absmax_error(h4, output_scale, dequant_scale, use_e3m2)
        err1011 = bfloat2_fp6_absmax_error(h5, output_scale, dequant_scale, use_e3m2)
        err1213 = bfloat2_fp6_absmax_error(h6, output_scale, dequant_scale, use_e3m2)
        err1415 = bfloat2_fp6_absmax_error(h7, output_scale, dequant_scale, use_e3m2)
        err = cutlass.max(err01, err23)
        err = cutlass.max(err, err45)
        err = cutlass.max(err, err67)
        err = cutlass.max(err, err89)
        err = cutlass.max(err, err1011)
        err = cutlass.max(err, err1213)
        return cutlass.max(err, err1415)
    elif cutlass.const_expr(scale_rule_id == SCALE_RULE_MAE):
        return (
            bfloat2_fp6_mae_error(h0, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mae_error(h1, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mae_error(h2, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mae_error(h3, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mae_error(h4, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mae_error(h5, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mae_error(h6, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mae_error(h7, output_scale, dequant_scale, use_e3m2)
        )
    else:
        return (
            bfloat2_fp6_mse_error(h0, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mse_error(h1, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mse_error(h2, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mse_error(h3, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mse_error(h4, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mse_error(h5, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mse_error(h6, output_scale, dequant_scale, use_e3m2)
            + bfloat2_fp6_mse_error(h7, output_scale, dequant_scale, use_e3m2)
        )


@cute.jit
def _int6_block_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    h2: cutlass.Uint32,
    h3: cutlass.Uint32,
    h4: cutlass.Uint32,
    h5: cutlass.Uint32,
    h6: cutlass.Uint32,
    h7: cutlass.Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err01 = bfloat2_int6_absmax_error(h0, output_scale, dequant_scale)
        err23 = bfloat2_int6_absmax_error(h1, output_scale, dequant_scale)
        err45 = bfloat2_int6_absmax_error(h2, output_scale, dequant_scale)
        err67 = bfloat2_int6_absmax_error(h3, output_scale, dequant_scale)
        err89 = bfloat2_int6_absmax_error(h4, output_scale, dequant_scale)
        err1011 = bfloat2_int6_absmax_error(h5, output_scale, dequant_scale)
        err1213 = bfloat2_int6_absmax_error(h6, output_scale, dequant_scale)
        err1415 = bfloat2_int6_absmax_error(h7, output_scale, dequant_scale)
        err = cutlass.max(err01, err23)
        err = cutlass.max(err, err45)
        err = cutlass.max(err, err67)
        err = cutlass.max(err, err89)
        err = cutlass.max(err, err1011)
        err = cutlass.max(err, err1213)
        return cutlass.max(err, err1415)
    elif cutlass.const_expr(scale_rule_id == SCALE_RULE_MAE):
        return (
            bfloat2_int6_mae_error(h0, output_scale, dequant_scale)
            + bfloat2_int6_mae_error(h1, output_scale, dequant_scale)
            + bfloat2_int6_mae_error(h2, output_scale, dequant_scale)
            + bfloat2_int6_mae_error(h3, output_scale, dequant_scale)
            + bfloat2_int6_mae_error(h4, output_scale, dequant_scale)
            + bfloat2_int6_mae_error(h5, output_scale, dequant_scale)
            + bfloat2_int6_mae_error(h6, output_scale, dequant_scale)
            + bfloat2_int6_mae_error(h7, output_scale, dequant_scale)
        )
    else:
        return (
            bfloat2_int6_mse_error(h0, output_scale, dequant_scale)
            + bfloat2_int6_mse_error(h1, output_scale, dequant_scale)
            + bfloat2_int6_mse_error(h2, output_scale, dequant_scale)
            + bfloat2_int6_mse_error(h3, output_scale, dequant_scale)
            + bfloat2_int6_mse_error(h4, output_scale, dequant_scale)
            + bfloat2_int6_mse_error(h5, output_scale, dequant_scale)
            + bfloat2_int6_mse_error(h6, output_scale, dequant_scale)
            + bfloat2_int6_mse_error(h7, output_scale, dequant_scale)
        )


@cute.jit
def _candidate4_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    q0: Float32,
    q1: Float32,
    q2: Float32,
    q3: Float32,
    dequant_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    x0, x1 = bfloat2_to_float2_scaled(h0, Float32(1.0))
    x2, x3 = bfloat2_to_float2_scaled(h1, Float32(1.0))
    d0 = _abs_f32(q0 * dequant_scale - x0)
    d1 = _abs_f32(q1 * dequant_scale - x1)
    d2 = _abs_f32(q2 * dequant_scale - x2)
    d3 = _abs_f32(q3 * dequant_scale - x3)

    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err = cutlass.max(d0, d1)
        err = cutlass.max(err, d2)
        return cutlass.max(err, d3)
    elif cutlass.const_expr(scale_rule_id == SCALE_RULE_MAE):
        return d0 + d1 + d2 + d3
    return d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3


@cute.jit
def _fp3_quant4_bfloat_values(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    output_scale: Float32,
) -> tuple[Float32, Float32, Float32, Float32]:
    x0, x1 = bfloat2_to_float2_scaled(h0, output_scale)
    x2, x3 = bfloat2_to_float2_scaled(h1, output_scale)
    return cvt_e2m0x4_f32_values(x0, x1, x2, x3)


@cute.jit
def _int3_quant4_bfloat_values(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    output_scale: Float32,
) -> tuple[Float32, Float32, Float32, Float32]:
    x0, x1 = bfloat2_to_float2_scaled(h0, output_scale)
    x2, x3 = bfloat2_to_float2_scaled(h1, output_scale)
    return cvt_int3x4_f32_values(x0, x1, x2, x3)


@cute.jit
def _dequant_values4_to_bfloat2x2(
    q0: Float32,
    q1: Float32,
    q2: Float32,
    q3: Float32,
    dequant_scale: Float32,
) -> tuple[Uint32, Uint32]:
    return (
        float2_to_bfloat2(q0 * dequant_scale, q1 * dequant_scale),
        float2_to_bfloat2(q2 * dequant_scale, q3 * dequant_scale),
    )


@cute.jit
def _fp3_block_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    h2: cutlass.Uint32,
    h3: cutlass.Uint32,
    h4: cutlass.Uint32,
    h5: cutlass.Uint32,
    h6: cutlass.Uint32,
    h7: cutlass.Uint32,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    dequant_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    q0, q1, q2, q3 = cvt_e2m0x4_to_f32(v0)
    q4, q5, q6, q7 = cvt_e2m0x4_to_f32(v1)
    q8, q9, q10, q11 = cvt_e2m0x4_to_f32(v2)
    q12, q13, q14, q15 = cvt_e2m0x4_to_f32(v3)
    err0 = _candidate4_error_bfloat(
        h0, h1, q0, q1, q2, q3, dequant_scale, scale_rule_id
    )
    err1 = _candidate4_error_bfloat(
        h2, h3, q4, q5, q6, q7, dequant_scale, scale_rule_id
    )
    err2 = _candidate4_error_bfloat(
        h4, h5, q8, q9, q10, q11, dequant_scale, scale_rule_id
    )
    err3 = _candidate4_error_bfloat(
        h6, h7, q12, q13, q14, q15, dequant_scale, scale_rule_id
    )
    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err = cutlass.max(err0, err1)
        err = cutlass.max(err, err2)
        return cutlass.max(err, err3)
    return err0 + err1 + err2 + err3


@cute.jit
def _int3_block_error_bfloat(
    h0: cutlass.Uint32,
    h1: cutlass.Uint32,
    h2: cutlass.Uint32,
    h3: cutlass.Uint32,
    h4: cutlass.Uint32,
    h5: cutlass.Uint32,
    h6: cutlass.Uint32,
    h7: cutlass.Uint32,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    dequant_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
) -> Float32:
    q0, q1, q2, q3 = cvt_int3x4_to_f32(v0)
    q4, q5, q6, q7 = cvt_int3x4_to_f32(v1)
    q8, q9, q10, q11 = cvt_int3x4_to_f32(v2)
    q12, q13, q14, q15 = cvt_int3x4_to_f32(v3)
    int_dequant_scale = dequant_scale * Float32(IF3_INT_EXPANSION_FACTOR)
    err0 = _candidate4_error_bfloat(
        h0, h1, q0, q1, q2, q3, int_dequant_scale, scale_rule_id
    )
    err1 = _candidate4_error_bfloat(
        h2, h3, q4, q5, q6, q7, int_dequant_scale, scale_rule_id
    )
    err2 = _candidate4_error_bfloat(
        h4, h5, q8, q9, q10, q11, int_dequant_scale, scale_rule_id
    )
    err3 = _candidate4_error_bfloat(
        h6, h7, q12, q13, q14, q15, int_dequant_scale, scale_rule_id
    )
    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        err = cutlass.max(err0, err1)
        err = cutlass.max(err, err2)
        return cutlass.max(err, err3)
    return err0 + err1 + err2 + err3


@cute.jit
def _process_nvfp4_adaptive_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    stochastic_rounding: cutlass.Constexpr[bool],
    seed_base: Uint32,
) -> tuple[Uint8, cutlass.Uint64]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float_6 = global_scale * (block_max * rcp_approx_ftz(Float32(6.0)))
    scale_fp8_u32_6 = cvt_f32_to_e4m3(scale_float_6)
    scale_fp8_6 = Uint8(scale_fp8_u32_6 & cutlass.Uint32(0xFF))
    output_scale_6 = nvfp4_compute_output_scale(scale_fp8_u32_6, global_scale)
    selection_scale_6 = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32_6,
        global_scale,
    )
    dequant_scale_6 = nvfp4_compute_dequant_scale(scale_fp8_u32_6, global_scale)
    if cutlass.const_expr(stochastic_rounding):
        packed64_6 = bfloat2x8_to_e2m1x16_packed_stochastic(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale_6,
            seed_base,
        )
        error_6 = _nvfp4_stochastic_block_error_bfloat(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            selection_scale_6,
            dequant_scale_6,
            seed_base,
            scale_rule_id,
        )
    else:
        packed64_6 = bfloat2x8_to_e2m1x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale_6,
        )
        error_6 = _nvfp4_block_error_bfloat(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            selection_scale_6,
            dequant_scale_6,
            scale_rule_id,
        )

    scale_float_4 = global_scale * (block_max * rcp_approx_ftz(Float32(4.0)))
    scale_fp8_u32_4 = cvt_f32_to_e4m3(scale_float_4)
    scale_fp8_4 = Uint8(scale_fp8_u32_4 & cutlass.Uint32(0xFF))
    output_scale_4 = nvfp4_compute_output_scale(scale_fp8_u32_4, global_scale)
    selection_scale_4 = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32_4,
        global_scale,
    )
    dequant_scale_4 = nvfp4_compute_dequant_scale(scale_fp8_u32_4, global_scale)
    if cutlass.const_expr(stochastic_rounding):
        packed64_4 = bfloat2x8_to_e2m1x16_packed_stochastic(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale_4,
            seed_base,
        )
        error_4 = _nvfp4_stochastic_block_error_bfloat(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            selection_scale_4,
            dequant_scale_4,
            seed_base,
            scale_rule_id,
        )
    else:
        packed64_4 = bfloat2x8_to_e2m1x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale_4,
        )
        error_4 = _nvfp4_block_error_bfloat(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            selection_scale_4,
            dequant_scale_4,
            scale_rule_id,
        )

    scale_fp8 = scale_fp8_6
    packed64 = packed64_6
    if error_4 < error_6:
        scale_fp8 = scale_fp8_4
        packed64 = packed64_4

    return scale_fp8, packed64


@cute.jit
def _process_nvfp4_adaptive_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
) -> tuple[Uint8, cutlass.Uint64]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(x, elem_base + Int32(2), elem_base + Int32(3), col_idx)
    h2 = _ld_transposed_bfloat2(x, elem_base + Int32(4), elem_base + Int32(5), col_idx)
    h3 = _ld_transposed_bfloat2(x, elem_base + Int32(6), elem_base + Int32(7), col_idx)
    h4 = _ld_transposed_bfloat2(x, elem_base + Int32(8), elem_base + Int32(9), col_idx)
    h5 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(10),
        elem_base + Int32(11),
        col_idx,
    )
    h6 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(12),
        elem_base + Int32(13),
        col_idx,
    )
    h7 = _ld_transposed_bfloat2(
        x,
        elem_base + Int32(14),
        elem_base + Int32(15),
        col_idx,
    )

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float_6 = global_scale * (block_max * rcp_approx_ftz(Float32(6.0)))
    scale_fp8_u32_6 = cvt_f32_to_e4m3(scale_float_6)
    scale_fp8_6 = Uint8(scale_fp8_u32_6 & cutlass.Uint32(0xFF))
    output_scale_6 = nvfp4_compute_output_scale(scale_fp8_u32_6, global_scale)
    selection_scale_6 = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32_6,
        global_scale,
    )
    dequant_scale_6 = nvfp4_compute_dequant_scale(scale_fp8_u32_6, global_scale)
    packed64_6 = bfloat2x8_to_e2m1x16_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale_6,
    )
    error_6 = _nvfp4_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale_6,
        dequant_scale_6,
        scale_rule_id,
    )

    scale_float_4 = global_scale * (block_max * rcp_approx_ftz(Float32(4.0)))
    scale_fp8_u32_4 = cvt_f32_to_e4m3(scale_float_4)
    scale_fp8_4 = Uint8(scale_fp8_u32_4 & cutlass.Uint32(0xFF))
    output_scale_4 = nvfp4_compute_output_scale(scale_fp8_u32_4, global_scale)
    selection_scale_4 = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32_4,
        global_scale,
    )
    dequant_scale_4 = nvfp4_compute_dequant_scale(scale_fp8_u32_4, global_scale)
    packed64_4 = bfloat2x8_to_e2m1x16_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale_4,
    )
    error_4 = _nvfp4_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale_4,
        dequant_scale_4,
        scale_rule_id,
    )

    scale_fp8 = scale_fp8_6
    packed64 = packed64_6
    if error_4 < error_6:
        scale_fp8 = scale_fp8_4
        packed64 = packed64_4

    return scale_fp8, packed64


@cute.jit
def _process_if4_adaptive_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    stochastic_rounding: cutlass.Constexpr[bool],
    seed_base: Uint32,
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
) -> tuple[Uint8, cutlass.Uint64]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    if cutlass.const_expr(scale_block_size == 8):
        h4, h5, h6, h7 = h0, h1, h2, h3
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M1_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    selection_scale = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32,
        global_scale,
    )
    dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(stochastic_rounding):
        packed_fp = bfloat2x8_to_e2m1x16_packed_stochastic(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
            seed_base,
        )
        error_fp = _nvfp4_stochastic_block_error_bfloat(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            selection_scale,
            dequant_scale,
            seed_base,
            scale_rule_id,
        )
    else:
        if cutlass.const_expr(scale_block_size == 8):
            packed_fp = cutlass.Uint64(
                bfloat2x4_to_e2m1x8_packed(h0, h1, h2, h3, output_scale),
            )
        else:
            packed_fp = bfloat2x8_to_e2m1x16_packed(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale,
            )
        error_fp = _nvfp4_block_error_bfloat(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            selection_scale,
            dequant_scale,
            scale_rule_id,
        )

    output_scale_int = output_scale * Float32(IF4_INT_EXPANSION_FACTOR_RCP)
    selection_scale_int = selection_scale * Float32(IF4_INT_EXPANSION_FACTOR_RCP)
    dequant_scale_int = dequant_scale * Float32(IF4_INT_EXPANSION_FACTOR)
    if cutlass.const_expr(stochastic_rounding):
        packed_int = bfloat2x8_to_int4x16_packed_stochastic(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale_int,
            seed_base,
        )
        error_int = _int4_stochastic_block_error_bfloat(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            selection_scale_int,
            dequant_scale_int,
            seed_base,
            scale_rule_id,
        )
    else:
        if cutlass.const_expr(scale_block_size == 8):
            packed_int = cutlass.Uint64(
                bfloat2x4_to_int4x8_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale_int,
                ),
            )
        else:
            packed_int = bfloat2x8_to_int4x16_packed(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale_int,
            )
        error_int = _int4_block_error_bfloat(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            selection_scale_int,
            dequant_scale_int,
            scale_rule_id,
        )

    packed64 = packed_fp
    if error_int < error_fp:
        scale_fp8 = scale_fp8 + Uint8(128)
        packed64 = packed_int

    return scale_fp8, packed64


@cute.jit
def _process_if4_adaptive_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
) -> tuple[Uint8, cutlass.Uint64]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(x, elem_base + Int32(2), elem_base + Int32(3), col_idx)
    h2 = _ld_transposed_bfloat2(x, elem_base + Int32(4), elem_base + Int32(5), col_idx)
    h3 = _ld_transposed_bfloat2(x, elem_base + Int32(6), elem_base + Int32(7), col_idx)

    if cutlass.const_expr(scale_block_size == 8):
        h4, h5, h6, h7 = h0, h1, h2, h3
    else:
        h4 = _ld_transposed_bfloat2(x, elem_base + Int32(8), elem_base + Int32(9), col_idx)
        h5 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(10),
            elem_base + Int32(11),
            col_idx,
        )
        h6 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(12),
            elem_base + Int32(13),
            col_idx,
        )
        h7 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(14),
            elem_base + Int32(15),
            col_idx,
        )

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M1_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    selection_scale = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32,
        global_scale,
    )
    dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(scale_block_size == 8):
        packed_fp = cutlass.Uint64(
            bfloat2x4_to_e2m1x8_packed(h0, h1, h2, h3, output_scale),
        )
    else:
        packed_fp = bfloat2x8_to_e2m1x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    error_fp = _nvfp4_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale,
        dequant_scale,
        scale_rule_id,
    )

    output_scale_int = output_scale * Float32(IF4_INT_EXPANSION_FACTOR_RCP)
    selection_scale_int = selection_scale * Float32(IF4_INT_EXPANSION_FACTOR_RCP)
    dequant_scale_int = dequant_scale * Float32(IF4_INT_EXPANSION_FACTOR)
    if cutlass.const_expr(scale_block_size == 8):
        packed_int = cutlass.Uint64(
            bfloat2x4_to_int4x8_packed(
                h0,
                h1,
                h2,
                h3,
                output_scale_int,
            ),
        )
    else:
        packed_int = bfloat2x8_to_int4x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale_int,
        )
    error_int = _int4_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale_int,
        dequant_scale_int,
        scale_rule_id,
    )

    packed64 = packed_fp
    if error_int < error_fp:
        scale_fp8 = scale_fp8 + Uint8(128)
        packed64 = packed_int

    return scale_fp8, packed64


@cute.jit
def _process_if4_adaptive_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    if cutlass.const_expr(scale_block_size == 8):
        h4, h5, h6, h7 = h0, h1, h2, h3
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M1_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    selection_scale = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32,
        global_scale,
    )
    dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    error_fp = _nvfp4_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale,
        dequant_scale,
        scale_rule_id,
    )

    output_scale_int = output_scale * Float32(IF4_INT_EXPANSION_FACTOR_RCP)
    selection_scale_int = selection_scale * Float32(IF4_INT_EXPANSION_FACTOR_RCP)
    dequant_scale_int = dequant_scale * Float32(IF4_INT_EXPANSION_FACTOR)
    if cutlass.const_expr(scale_block_size == 8):
        packed_int = cutlass.Uint64(
            bfloat2x4_to_int4x8_packed(
                h0,
                h1,
                h2,
                h3,
                output_scale_int,
            ),
        )
    else:
        packed_int = bfloat2x8_to_int4x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale_int,
        )
    error_int = _int4_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale_int,
        dequant_scale_int,
        scale_rule_id,
    )

    z = cutlass.Uint32(0)
    out0 = bfloat2_nvfp4_dequant_bfloat2(h0, output_scale, dequant_scale)
    out1 = bfloat2_nvfp4_dequant_bfloat2(h1, output_scale, dequant_scale)
    out2 = bfloat2_nvfp4_dequant_bfloat2(h2, output_scale, dequant_scale)
    out3 = bfloat2_nvfp4_dequant_bfloat2(h3, output_scale, dequant_scale)
    out4 = z
    out5 = z
    out6 = z
    out7 = z
    if cutlass.const_expr(scale_block_size != 8):
        out4 = bfloat2_nvfp4_dequant_bfloat2(h4, output_scale, dequant_scale)
        out5 = bfloat2_nvfp4_dequant_bfloat2(h5, output_scale, dequant_scale)
        out6 = bfloat2_nvfp4_dequant_bfloat2(h6, output_scale, dequant_scale)
        out7 = bfloat2_nvfp4_dequant_bfloat2(h7, output_scale, dequant_scale)

    if error_int < error_fp:
        p0 = Uint32(packed_int & cutlass.Uint64(0xFFFF))
        p1 = Uint32((packed_int >> cutlass.Uint64(16)) & cutlass.Uint64(0xFFFF))
        out0, out1 = _dequant_int4x4_to_bfloat2x2(p0, dequant_scale_int)
        out2, out3 = _dequant_int4x4_to_bfloat2x2(p1, dequant_scale_int)
        if cutlass.const_expr(scale_block_size != 8):
            p2 = Uint32((packed_int >> cutlass.Uint64(32)) & cutlass.Uint64(0xFFFF))
            p3 = Uint32((packed_int >> cutlass.Uint64(48)) & cutlass.Uint64(0xFFFF))
            out4, out5 = _dequant_int4x4_to_bfloat2x2(p2, dequant_scale_int)
            out6, out7 = _dequant_int4x4_to_bfloat2x2(p3, dequant_scale_int)

    return (out0, out1, out2, out3, out4, out5, out6, out7)


@cute.jit
def _process_if3_adaptive_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
) -> tuple[Uint8, Uint32, Uint32, Uint32, Uint32]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    if cutlass.const_expr(scale_block_size == 8):
        h4, h5, h6, h7 = h0, h1, h2, h3
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M0_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(scale_block_size == 8):
        fp0, fp1 = bfloat2x4_to_e2m0x8_values(
            h0,
            h1,
            h2,
            h3,
            output_scale,
        )
        fp2, fp3 = fp0, fp1
        int0, int1 = bfloat2x4_to_int3x8_values(
            h0,
            h1,
            h2,
            h3,
            output_scale * Float32(IF3_INT_EXPANSION_FACTOR_RCP),
        )
        int2, int3 = int0, int1
    else:
        fp0, fp1, fp2, fp3 = bfloat2x8_to_e2m0x16_values(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
        int0, int1, int2, int3 = bfloat2x8_to_int3x16_values(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale * Float32(IF3_INT_EXPANSION_FACTOR_RCP),
        )

    error_fp = _fp3_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        fp0,
        fp1,
        fp2,
        fp3,
        dequant_scale,
        scale_rule_id,
    )
    error_int = _int3_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        int0,
        int1,
        int2,
        int3,
        dequant_scale,
        scale_rule_id,
    )

    out0, out1, out2, out3 = fp0, fp1, fp2, fp3
    if error_int < error_fp:
        scale_fp8 = scale_fp8 + Uint8(128)
        out0, out1, out2, out3 = int0, int1, int2, int3

    return scale_fp8, out0, out1, out2, out3


@cute.jit
def _process_if3_adaptive_block_bfloat_transposed(
    x: cute.Tensor,
    col_idx: Int32,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
) -> tuple[Uint8, Uint32, Uint32, Uint32, Uint32]:
    h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), col_idx)
    h1 = _ld_transposed_bfloat2(x, elem_base + Int32(2), elem_base + Int32(3), col_idx)
    h2 = _ld_transposed_bfloat2(x, elem_base + Int32(4), elem_base + Int32(5), col_idx)
    h3 = _ld_transposed_bfloat2(x, elem_base + Int32(6), elem_base + Int32(7), col_idx)

    if cutlass.const_expr(scale_block_size == 8):
        h4, h5, h6, h7 = h0, h1, h2, h3
    else:
        h4 = _ld_transposed_bfloat2(x, elem_base + Int32(8), elem_base + Int32(9), col_idx)
        h5 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(10),
            elem_base + Int32(11),
            col_idx,
        )
        h6 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(12),
            elem_base + Int32(13),
            col_idx,
        )
        h7 = _ld_transposed_bfloat2(
            x,
            elem_base + Int32(14),
            elem_base + Int32(15),
            col_idx,
        )

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M0_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(scale_block_size == 8):
        fp0, fp1 = bfloat2x4_to_e2m0x8_values(
            h0,
            h1,
            h2,
            h3,
            output_scale,
        )
        fp2, fp3 = fp0, fp1
        int0, int1 = bfloat2x4_to_int3x8_values(
            h0,
            h1,
            h2,
            h3,
            output_scale * Float32(IF3_INT_EXPANSION_FACTOR_RCP),
        )
        int2, int3 = int0, int1
    else:
        fp0, fp1, fp2, fp3 = bfloat2x8_to_e2m0x16_values(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
        int0, int1, int2, int3 = bfloat2x8_to_int3x16_values(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale * Float32(IF3_INT_EXPANSION_FACTOR_RCP),
        )

    error_fp = _fp3_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        fp0,
        fp1,
        fp2,
        fp3,
        dequant_scale,
        scale_rule_id,
    )
    error_int = _int3_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        int0,
        int1,
        int2,
        int3,
        dequant_scale,
        scale_rule_id,
    )

    out0, out1, out2, out3 = fp0, fp1, fp2, fp3
    if error_int < error_fp:
        scale_fp8 = scale_fp8 + Uint8(128)
        out0, out1, out2, out3 = int0, int1, int2, int3

    return scale_fp8, out0, out1, out2, out3


@cute.jit
def _process_if3_adaptive_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    scale_block_size: cutlass.Constexpr[int] = NVFP4_SCALE_BLOCK_SIZE,
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

    if cutlass.const_expr(scale_block_size == 8):
        h4, h5, h6, h7 = h0, h1, h2, h3
    else:
        ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
        h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
    scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M0_MAX)))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    fpq0, fpq1, fpq2, fpq3 = _fp3_quant4_bfloat_values(h0, h1, output_scale)
    fpq4, fpq5, fpq6, fpq7 = _fp3_quant4_bfloat_values(h2, h3, output_scale)
    int_output_scale = output_scale * Float32(IF3_INT_EXPANSION_FACTOR_RCP)
    intq0, intq1, intq2, intq3 = _int3_quant4_bfloat_values(
        h0, h1, int_output_scale
    )
    intq4, intq5, intq6, intq7 = _int3_quant4_bfloat_values(
        h2, h3, int_output_scale
    )

    if cutlass.const_expr(scale_block_size == 8):
        fpq8, fpq9, fpq10, fpq11 = fpq0, fpq1, fpq2, fpq3
        fpq12, fpq13, fpq14, fpq15 = fpq4, fpq5, fpq6, fpq7
        intq8, intq9, intq10, intq11 = intq0, intq1, intq2, intq3
        intq12, intq13, intq14, intq15 = intq4, intq5, intq6, intq7
    else:
        fpq8, fpq9, fpq10, fpq11 = _fp3_quant4_bfloat_values(
            h4, h5, output_scale
        )
        fpq12, fpq13, fpq14, fpq15 = _fp3_quant4_bfloat_values(
            h6, h7, output_scale
        )
        intq8, intq9, intq10, intq11 = _int3_quant4_bfloat_values(
            h4, h5, int_output_scale
        )
        intq12, intq13, intq14, intq15 = _int3_quant4_bfloat_values(
            h6, h7, int_output_scale
        )

    error_fp0 = _candidate4_error_bfloat(
        h0, h1, fpq0, fpq1, fpq2, fpq3, dequant_scale, scale_rule_id
    )
    error_fp1 = _candidate4_error_bfloat(
        h2, h3, fpq4, fpq5, fpq6, fpq7, dequant_scale, scale_rule_id
    )
    error_fp2 = _candidate4_error_bfloat(
        h4, h5, fpq8, fpq9, fpq10, fpq11, dequant_scale, scale_rule_id
    )
    error_fp3 = _candidate4_error_bfloat(
        h6, h7, fpq12, fpq13, fpq14, fpq15, dequant_scale, scale_rule_id
    )
    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        error_fp = cutlass.max(error_fp0, error_fp1)
        error_fp = cutlass.max(error_fp, error_fp2)
        error_fp = cutlass.max(error_fp, error_fp3)
    else:
        error_fp = error_fp0 + error_fp1 + error_fp2 + error_fp3

    int_dequant_scale = dequant_scale * Float32(IF3_INT_EXPANSION_FACTOR)
    error_int0 = _candidate4_error_bfloat(
        h0, h1, intq0, intq1, intq2, intq3, int_dequant_scale, scale_rule_id
    )
    error_int1 = _candidate4_error_bfloat(
        h2, h3, intq4, intq5, intq6, intq7, int_dequant_scale, scale_rule_id
    )
    error_int2 = _candidate4_error_bfloat(
        h4, h5, intq8, intq9, intq10, intq11, int_dequant_scale, scale_rule_id
    )
    error_int3 = _candidate4_error_bfloat(
        h6, h7, intq12, intq13, intq14, intq15, int_dequant_scale, scale_rule_id
    )
    if cutlass.const_expr(scale_rule_id == SCALE_RULE_ABS_MAX):
        error_int = cutlass.max(error_int0, error_int1)
        error_int = cutlass.max(error_int, error_int2)
        error_int = cutlass.max(error_int, error_int3)
    else:
        error_int = error_int0 + error_int1 + error_int2 + error_int3

    z = cutlass.Uint32(0)
    out0 = z
    out1 = z
    out2 = z
    out3 = z
    out4 = z
    out5 = z
    out6 = z
    out7 = z
    use_int = error_int < error_fp

    if not use_int:
        out0, out1 = _dequant_values4_to_bfloat2x2(
            fpq0, fpq1, fpq2, fpq3, dequant_scale
        )
        out2, out3 = _dequant_values4_to_bfloat2x2(
            fpq4, fpq5, fpq6, fpq7, dequant_scale
        )
        if cutlass.const_expr(scale_block_size != 8):
            out4, out5 = _dequant_values4_to_bfloat2x2(
                fpq8, fpq9, fpq10, fpq11, dequant_scale
            )
            out6, out7 = _dequant_values4_to_bfloat2x2(
                fpq12, fpq13, fpq14, fpq15, dequant_scale
            )

    if use_int:
        out0, out1 = _dequant_values4_to_bfloat2x2(
            intq0, intq1, intq2, intq3, int_dequant_scale
        )
        out2, out3 = _dequant_values4_to_bfloat2x2(
            intq4, intq5, intq6, intq7, int_dequant_scale
        )
        if cutlass.const_expr(scale_block_size != 8):
            out4, out5 = _dequant_values4_to_bfloat2x2(
                intq8, intq9, intq10, intq11, int_dequant_scale
            )
            out6, out7 = _dequant_values4_to_bfloat2x2(
                intq12, intq13, intq14, intq15, int_dequant_scale
            )

    return (out0, out1, out2, out3, out4, out5, out6, out7)


@cute.jit
def _process_if6_adaptive_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    max_quantized_value: cutlass.Constexpr[float],
    int_expansion_factor: cutlass.Constexpr[float],
    int_expansion_factor_rcp: cutlass.Constexpr[float],
    use_e3m2: cutlass.Constexpr[bool],
) -> tuple[Uint8, cutlass.Uint64, cutlass.Uint64]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float = global_scale * (
        block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    )
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    selection_scale = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32,
        global_scale,
    )
    dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(use_e3m2):
        packed_fp_0, packed_fp_1 = bfloat2x8_to_e3m2x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    else:
        packed_fp_0, packed_fp_1 = bfloat2x8_to_e2m3x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    error_fp = _fp6_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale,
        dequant_scale,
        scale_rule_id,
        use_e3m2,
    )

    output_scale_int = output_scale * Float32(float(int_expansion_factor_rcp))
    selection_scale_int = selection_scale * Float32(float(int_expansion_factor_rcp))
    dequant_scale_int = dequant_scale * Float32(float(int_expansion_factor))
    packed_int_0, packed_int_1 = bfloat2x8_to_int6x16_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale_int,
    )
    error_int = _int6_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale_int,
        dequant_scale_int,
        scale_rule_id,
    )
    packed0 = packed_fp_0
    packed1 = packed_fp_1
    if error_int < error_fp:
        scale_fp8 = scale_fp8 + Uint8(128)
        packed0 = packed_int_0
        packed1 = packed_int_1

    return scale_fp8, packed0, packed1


@cute.jit
def _process_if6_adaptive_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
    max_quantized_value: cutlass.Constexpr[float],
    int_expansion_factor: cutlass.Constexpr[float],
    int_expansion_factor_rcp: cutlass.Constexpr[float],
    use_e3m2: cutlass.Constexpr[bool],
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float = global_scale * (
        block_max * rcp_approx_ftz(Float32(float(max_quantized_value)))
    )
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    selection_scale = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32,
        global_scale,
    )
    dequant_scale = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)

    if cutlass.const_expr(use_e3m2):
        packed_fp_0, packed_fp_1 = bfloat2x8_to_e3m2x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    else:
        packed_fp_0, packed_fp_1 = bfloat2x8_to_e2m3x16_packed(
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            output_scale,
        )
    error_fp = _fp6_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale,
        dequant_scale,
        scale_rule_id,
        use_e3m2,
    )

    output_scale_int = output_scale * Float32(float(int_expansion_factor_rcp))
    selection_scale_int = selection_scale * Float32(float(int_expansion_factor_rcp))
    dequant_scale_int = dequant_scale * Float32(float(int_expansion_factor))
    packed_int_0, packed_int_1 = bfloat2x8_to_int6x16_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        output_scale_int,
    )
    error_int = _int6_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale_int,
        dequant_scale_int,
        scale_rule_id,
    )

    p0 = Uint32(packed_fp_0 & cutlass.Uint64(0xFFFFFFFF))
    p1 = Uint32(packed_fp_0 >> cutlass.Uint64(32))
    p2 = Uint32(packed_fp_1 & cutlass.Uint64(0xFFFFFFFF))
    p3 = Uint32(packed_fp_1 >> cutlass.Uint64(32))
    out0, out1 = _dequant_fp6x4_to_bfloat2x2(p0, dequant_scale, use_e3m2)
    out2, out3 = _dequant_fp6x4_to_bfloat2x2(p1, dequant_scale, use_e3m2)
    out4, out5 = _dequant_fp6x4_to_bfloat2x2(p2, dequant_scale, use_e3m2)
    out6, out7 = _dequant_fp6x4_to_bfloat2x2(p3, dequant_scale, use_e3m2)
    if error_int < error_fp:
        p0 = Uint32(packed_int_0 & cutlass.Uint64(0xFFFFFFFF))
        p1 = Uint32(packed_int_0 >> cutlass.Uint64(32))
        p2 = Uint32(packed_int_1 & cutlass.Uint64(0xFFFFFFFF))
        p3 = Uint32(packed_int_1 >> cutlass.Uint64(32))
        out0, out1 = _dequant_int6x4_to_bfloat2x2(p0, dequant_scale_int)
        out2, out3 = _dequant_int6x4_to_bfloat2x2(p1, dequant_scale_int)
        out4, out5 = _dequant_int6x4_to_bfloat2x2(p2, dequant_scale_int)
        out6, out7 = _dequant_int6x4_to_bfloat2x2(p3, dequant_scale_int)

    return (out0, out1, out2, out3, out4, out5, out6, out7)


@cute.jit
def _process_nvfp4_adaptive_pseudo_block_bfloat(
    row_tensor: cute.Tensor,
    elem_base: Int32,
    global_scale: Float32,
    scale_rule_id: cutlass.Constexpr[int],
) -> tuple[
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
    cutlass.Uint32,
]:
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    scale_float_6 = global_scale * (block_max * rcp_approx_ftz(Float32(6.0)))
    scale_fp8_u32_6 = cvt_f32_to_e4m3(scale_float_6)
    output_scale_6 = nvfp4_compute_output_scale(scale_fp8_u32_6, global_scale)
    selection_scale_6 = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32_6,
        global_scale,
    )
    dequant_scale_6 = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale_6 = nvfp4_compute_dequant_scale(scale_fp8_u32_6, global_scale)
    error_6 = _nvfp4_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale_6,
        dequant_scale_6,
        scale_rule_id,
    )

    scale_float_4 = global_scale * (block_max * rcp_approx_ftz(Float32(4.0)))
    scale_fp8_u32_4 = cvt_f32_to_e4m3(scale_float_4)
    output_scale_4 = nvfp4_compute_output_scale(scale_fp8_u32_4, global_scale)
    selection_scale_4 = nvfp4_compute_quant_scale_exact(
        scale_fp8_u32_4,
        global_scale,
    )
    dequant_scale_4 = Float32(0.0)
    if global_scale != Float32(0.0):
        dequant_scale_4 = nvfp4_compute_dequant_scale(scale_fp8_u32_4, global_scale)
    error_4 = _nvfp4_block_error_bfloat(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        selection_scale_4,
        dequant_scale_4,
        scale_rule_id,
    )

    output_scale = output_scale_6
    dequant_scale = dequant_scale_6
    if error_4 < error_6:
        output_scale = output_scale_4
        dequant_scale = dequant_scale_4

    return (
        bfloat2_nvfp4_dequant_bfloat2(h0, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h1, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h2, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h3, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h4, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h5, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h6, output_scale, dequant_scale),
        bfloat2_nvfp4_dequant_bfloat2(h7, output_scale, dequant_scale),
    )


class Sm100NVFP4StaticQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
        stochastic_rounding: bool = False,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_block_size = scale_block_size
        self.stochastic_rounding = stochastic_rounding
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[NVFP4_BASE_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=16,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * NVFP4_BASE_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * NVFP4_BASE_THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value)
            * E4M3_STATIC_MAX
            * float(self.adjustment_factor),
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            scale_fp8, packed64 = _process_nvfp4_static_block_bfloat(
                x[row_idx, None],
                elem_base,
                global_scale,
                self.max_quantized_value,
                self.scale_block_size,
                self.stochastic_rounding,
                Uint32(row_idx * self.k + elem_base),
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * (self.scale_block_size // 2)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                st_global_u32(output_ptr, Uint32(packed64 & cutlass.Uint64(0xFFFFFFFF)))
            else:
                st_global_u64(output_ptr, packed64)

            sf_idx = sf_idx + stride


class Sm100NVFP4StaticTransposeQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
        stochastic_rounding: bool = False,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_block_size = scale_block_size
        self.stochastic_rounding = stochastic_rounding
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value)
            * E4M3_STATIC_MAX
            * float(self.adjustment_factor),
        )

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * self.scale_block_size

            scale_fp8, packed64 = _process_nvfp4_static_block_bfloat_transposed(
                x,
                row_idx,
                elem_base,
                global_scale,
                self.max_quantized_value,
                self.scale_block_size,
                self.stochastic_rounding,
                Uint32(row_idx * self.k + elem_base),
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * (self.scale_block_size // 2)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                st_global_u32(output_ptr, Uint32(packed64 & cutlass.Uint64(0xFFFFFFFF)))
            else:
                st_global_u64(output_ptr, packed64)

            work_idx = work_idx + stride


class Sm100NVFP4AdaptiveQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        stochastic_rounding: bool = False,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.stochastic_rounding = stochastic_rounding
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[NVFP4_BASE_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * NVFP4_BASE_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * NVFP4_BASE_THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_FOUROVERSIX_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            scale_fp8, packed64 = _process_nvfp4_adaptive_block_bfloat(
                x[row_idx, None],
                elem_base,
                global_scale,
                self.scale_rule_id,
                self.stochastic_rounding,
                Uint32(row_idx * self.k + elem_base),
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * (NVFP4_SCALE_BLOCK_SIZE // 2)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            st_global_u64(output_ptr, packed64)

            sf_idx = sf_idx + stride


class Sm100NVFP4AdaptiveTransposeQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_FOUROVERSIX_MAX,
        )

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            scale_fp8, packed64 = _process_nvfp4_adaptive_block_bfloat_transposed(
                x,
                row_idx,
                elem_base,
                global_scale,
                self.scale_rule_id,
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * (NVFP4_SCALE_BLOCK_SIZE // 2)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            st_global_u64(output_ptr, packed64)

            work_idx = work_idx + stride


class Sm100NVFP4StaticQuantize2D:
    def __init__(
        self,
        k: int,
        max_quantized_value: int,
        stochastic_rounding: bool = False,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.stochastic_rounding = stochastic_rounding
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value) * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_max = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            fp4_max_rcp = rcp_approx_ftz(Float32(float(self.max_quantized_value)))
            scale_float = global_scale * (block_max * fp4_max_rcp)
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                if cutlass.const_expr(self.stochastic_rounding):
                    packed64 = bfloat2x8_to_e2m1x16_packed_stochastic(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        output_scale,
                        Uint32(row_idx * self.k + elem_base),
                    )
                else:
                    packed64 = bfloat2x8_to_e2m1x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        output_scale,
                    )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_offset = col_idx * (NVFP4_SCALE_BLOCK_SIZE // 2)
                output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
                st_global_u64(output_ptr, packed64)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVFP4BS8StaticQuantize2D:
    def __init__(self, k: int, max_quantized_value: int):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_blocks_per_row = k // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value) * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * Int32(8)

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                row_max = bfloat2_max_abs_8(
                    h0,
                    h1,
                    h2,
                    h3,
                    h0,
                    h1,
                    h2,
                    h3,
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            fp4_max_rcp = rcp_approx_ftz(Float32(float(self.max_quantized_value)))
            scale_float = global_scale * (block_max * fp4_max_rcp)
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                packed32 = bfloat2x4_to_e2m1x8_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_offset = col_idx * Int32(4)
                output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
                st_global_u32(output_ptr, packed32)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVFP4AdaptiveQuantize2D:
    def __init__(self, k: int, scale_rule_id: int):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_FOUROVERSIX_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_max = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

            scale_float_6 = global_scale * (block_max * rcp_approx_ftz(Float32(6.0)))
            scale_fp8_u32_6 = cvt_f32_to_e4m3(scale_float_6)
            scale_fp8_6 = Uint8(scale_fp8_u32_6 & cutlass.Uint32(0xFF))
            output_scale_6 = nvfp4_compute_output_scale(
                scale_fp8_u32_6,
                global_scale,
            )
            selection_scale_6 = nvfp4_compute_quant_scale_exact(
                scale_fp8_u32_6,
                global_scale,
            )
            dequant_scale_6 = nvfp4_compute_dequant_scale(
                scale_fp8_u32_6,
                global_scale,
            )

            scale_float_4 = global_scale * (block_max * rcp_approx_ftz(Float32(4.0)))
            scale_fp8_u32_4 = cvt_f32_to_e4m3(scale_float_4)
            scale_fp8_4 = Uint8(scale_fp8_u32_4 & cutlass.Uint32(0xFF))
            output_scale_4 = nvfp4_compute_output_scale(
                scale_fp8_u32_4,
                global_scale,
            )
            selection_scale_4 = nvfp4_compute_quant_scale_exact(
                scale_fp8_u32_4,
                global_scale,
            )
            dequant_scale_4 = nvfp4_compute_dequant_scale(
                scale_fp8_u32_4,
                global_scale,
            )

            error_6 = Float32(0.0)
            error_4 = Float32(0.0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_error_6 = _nvfp4_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    selection_scale_6,
                    dequant_scale_6,
                    self.scale_rule_id,
                )
                row_error_4 = _nvfp4_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    selection_scale_4,
                    dequant_scale_4,
                    self.scale_rule_id,
                )
                if cutlass.const_expr(self.scale_rule_id == SCALE_RULE_ABS_MAX):
                    error_6 = cutlass.max(error_6, row_error_6)
                    error_4 = cutlass.max(error_4, row_error_4)
                else:
                    error_6 = error_6 + row_error_6
                    error_4 = error_4 + row_error_4
                row_offset = row_offset + Int32(1)

            scale_fp8 = scale_fp8_6
            output_scale = output_scale_6
            if error_4 < error_6:
                scale_fp8 = scale_fp8_4
                output_scale = output_scale_4

            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                packed64 = bfloat2x8_to_e2m1x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    output_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_offset = col_idx * (NVFP4_SCALE_BLOCK_SIZE // 2)
                output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
                st_global_u64(output_ptr, packed64)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100IF4AdaptiveQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        stochastic_rounding: bool = False,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.stochastic_rounding = stochastic_rounding
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            scale_fp8, packed64 = _process_if4_adaptive_block_bfloat(
                x[row_idx, None],
                elem_base,
                global_scale,
                self.scale_rule_id,
                self.stochastic_rounding,
                Uint32(row_idx * self.k + elem_base),
                self.scale_block_size,
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * (self.scale_block_size // 2)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                st_global_u32(output_ptr, Uint32(packed64 & cutlass.Uint64(0xFFFFFFFF)))
            else:
                st_global_u64(output_ptr, packed64)

            sf_idx = sf_idx + stride


class Sm100IF3AdaptiveQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M0_MAX * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            scale_fp8, v0, v1, v2, v3 = _process_if3_adaptive_block_bfloat(
                x[row_idx, None],
                elem_base,
                global_scale,
                self.scale_rule_id,
                self.scale_block_size,
            )

            scales[sf_idx] = scale_fp8
            output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
            if cutlass.const_expr(self.scale_block_size == 8):
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
            else:
                st_global_v4_u32(output_ptr, v0, v1, v2, v3)

            sf_idx = sf_idx + stride


class Sm100IF4AdaptiveTransposeQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_FOUROVERSIX_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            scale_fp8, packed64 = _process_if4_adaptive_block_bfloat_transposed(
                x,
                row_idx,
                elem_base,
                global_scale,
                self.scale_rule_id,
                self.scale_block_size,
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * (self.scale_block_size // 2)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                st_global_u32(output_ptr, Uint32(packed64 & cutlass.Uint64(0xFFFFFFFF)))
            else:
                st_global_u64(output_ptr, packed64)

            sf_idx = sf_idx + stride


class Sm100IF3AdaptiveTransposeQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M0_MAX * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            scale_fp8, v0, v1, v2, v3 = _process_if3_adaptive_block_bfloat_transposed(
                x,
                row_idx,
                elem_base,
                global_scale,
                self.scale_rule_id,
                self.scale_block_size,
            )

            scales[sf_idx] = scale_fp8
            output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
            if cutlass.const_expr(self.scale_block_size == 8):
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
            else:
                st_global_v4_u32(output_ptr, v0, v1, v2, v3)

            sf_idx = sf_idx + stride


class Sm100IF3AdaptiveQuantize2D:
    def __init__(self, k: int, scale_rule_id: int):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        lane_idx = tidx % Int32(IF3_2D_GROUP_SIZE)
        group_idx = tidx // Int32(IF3_2D_GROUP_SIZE)
        tile_idx = bidx * IF3_2D_GROUPS_PER_BLOCK + group_idx
        stride = grid_dim_x * IF3_2D_GROUPS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M0_MAX * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + lane_idx
            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
            h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
            row_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
            row_max = bfloat2_hmax_reduce_to_f32(row_max_h2)
            block_max = cute.arch.warp_reduction_max(
                row_max,
                threads_in_group=IF3_2D_GROUP_SIZE,
            )
            scale_float = global_scale * (
                block_max * rcp_approx_ftz(Float32(E2M0_MAX))
            )
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
            dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)
            output_scale_int = output_scale * Float32(IF3_INT_EXPANSION_FACTOR_RCP)

            fp0, fp1, fp2, fp3 = bfloat2x8_to_e2m0x16_values(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale,
            )
            int0, int1, int2, int3 = bfloat2x8_to_int3x16_values(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale_int,
            )
            row_error_fp = _fp3_block_error_bfloat(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                fp0,
                fp1,
                fp2,
                fp3,
                dequant_scale,
                self.scale_rule_id,
            )
            row_error_int = _int3_block_error_bfloat(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                int0,
                int1,
                int2,
                int3,
                dequant_scale,
                self.scale_rule_id,
            )
            if cutlass.const_expr(self.scale_rule_id == SCALE_RULE_ABS_MAX):
                error_fp = cute.arch.warp_reduction_max(
                    row_error_fp,
                    threads_in_group=IF3_2D_GROUP_SIZE,
                )
                error_int = cute.arch.warp_reduction_max(
                    row_error_int,
                    threads_in_group=IF3_2D_GROUP_SIZE,
                )
            else:
                error_fp = cute.arch.warp_reduction_sum(
                    row_error_fp,
                    threads_in_group=IF3_2D_GROUP_SIZE,
                )
                error_int = cute.arch.warp_reduction_sum(
                    row_error_int,
                    threads_in_group=IF3_2D_GROUP_SIZE,
                )

            output_scale_selected = output_scale
            scale_fp8_selected = scale_fp8
            if error_int < error_fp:
                output_scale_selected = output_scale_int
                scale_fp8_selected = scale_fp8 + Uint8(128)

            v0, v1, v2, v3 = fp0, fp1, fp2, fp3
            if error_int < error_fp:
                v0, v1, v2, v3 = bfloat2x8_to_int3x16_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    output_scale_selected,
                )

            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            scales[sf_idx] = scale_fp8_selected
            output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr, v0, v1, v2, v3)

            tile_idx = tile_idx + stride


class Sm100IF3BS8AdaptiveQuantize2D:
    def __init__(self, k: int, scale_rule_id: int):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_blocks_per_row = k // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M0_MAX * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * Int32(8)

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                row_max = bfloat2_max_abs_8(
                    h0,
                    h1,
                    h2,
                    h3,
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (
                block_max * rcp_approx_ftz(Float32(E2M0_MAX))
            )
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
            dequant_scale = nvfp4_compute_dequant_scale(scale_fp8_u32, global_scale)
            output_scale_int = output_scale * Float32(IF3_INT_EXPANSION_FACTOR_RCP)

            error_fp = Float32(0.0)
            error_int = Float32(0.0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                fp0, fp1 = bfloat2x4_to_e2m0x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale,
                )
                int0, int1 = bfloat2x4_to_int3x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale_int,
                )
                row_error_fp = _fp3_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h0,
                    h1,
                    h2,
                    h3,
                    fp0,
                    fp1,
                    fp0,
                    fp1,
                    dequant_scale,
                    self.scale_rule_id,
                )
                row_error_int = _int3_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h0,
                    h1,
                    h2,
                    h3,
                    int0,
                    int1,
                    int0,
                    int1,
                    dequant_scale,
                    self.scale_rule_id,
                )
                if cutlass.const_expr(self.scale_rule_id == SCALE_RULE_ABS_MAX):
                    error_fp = cutlass.max(error_fp, row_error_fp)
                    error_int = cutlass.max(error_int, row_error_int)
                else:
                    error_fp = error_fp + row_error_fp
                    error_int = error_int + row_error_int
                row_offset = row_offset + Int32(1)

            output_scale_selected = output_scale
            scale_fp8_selected = scale_fp8
            if error_int < error_fp:
                output_scale_selected = output_scale_int
                scale_fp8_selected = scale_fp8 + Uint8(128)

            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                v0, v1 = bfloat2x4_to_e2m0x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale_selected,
                )
                if error_int < error_fp:
                    v0, v1 = bfloat2x4_to_int3x8_values(
                        h0,
                        h1,
                        h2,
                        h3,
                        output_scale_selected,
                    )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8_selected
                output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100IF6AdaptiveQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        max_quantized_value: float,
        int_expansion_factor: float,
        int_expansion_factor_rcp: float,
        use_e3m2: bool,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.max_quantized_value = max_quantized_value
        self.int_expansion_factor = int_expansion_factor
        self.int_expansion_factor_rcp = int_expansion_factor_rcp
        self.use_e3m2 = use_e3m2
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value)
            * E4M3_STATIC_MAX
            * float(self.adjustment_factor),
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            scale_fp8, packed64_0, packed64_1 = _process_if6_adaptive_block_bfloat(
                x[row_idx, None],
                elem_base,
                global_scale,
                self.scale_rule_id,
                self.max_quantized_value,
                self.int_expansion_factor,
                self.int_expansion_factor_rcp,
                self.use_e3m2,
            )

            scale_idx = _blackwell_scale_index(
                row_idx,
                col_idx,
                self.scale_blocks_per_row,
            )
            scales[scale_idx] = scale_fp8
            output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            output_ptr1 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(8),
            )
            st_global_u64(output_ptr0, packed64_0)
            st_global_u64(output_ptr1, packed64_1)

            sf_idx = sf_idx + stride


class Sm100IF6AdaptivePseudoQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        max_quantized_value: float,
        int_expansion_factor: float,
        int_expansion_factor_rcp: float,
        use_e3m2: bool,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.max_quantized_value = max_quantized_value
        self.int_expansion_factor = int_expansion_factor
        self.int_expansion_factor_rcp = int_expansion_factor_rcp
        self.use_e3m2 = use_e3m2
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[PSEUDO_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * PSEUDO_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * PSEUDO_THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value) * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_if6_adaptive_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.scale_rule_id,
                    self.max_quantized_value,
                    self.int_expansion_factor,
                    self.int_expansion_factor_rcp,
                    self.use_e3m2,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100IF4AdaptiveQuantize2D:
    def __init__(self, k: int, scale_rule_id: int):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_max = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (
                block_max * rcp_approx_ftz(Float32(E2M1_MAX))
            )
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(
                scale_fp8_u32,
                global_scale,
            )
            selection_scale = nvfp4_compute_quant_scale_exact(
                scale_fp8_u32,
                global_scale,
            )
            dequant_scale = nvfp4_compute_dequant_scale(
                scale_fp8_u32,
                global_scale,
            )

            output_scale_int = output_scale * Float32(IF4_INT_EXPANSION_FACTOR_RCP)
            selection_scale_int = selection_scale * Float32(
                IF4_INT_EXPANSION_FACTOR_RCP,
            )
            dequant_scale_int = dequant_scale * Float32(IF4_INT_EXPANSION_FACTOR)

            error_fp = Float32(0.0)
            error_int = Float32(0.0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_error_fp = _nvfp4_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    selection_scale,
                    dequant_scale,
                    self.scale_rule_id,
                )
                row_error_int = _int4_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    selection_scale_int,
                    dequant_scale_int,
                    self.scale_rule_id,
                )
                if cutlass.const_expr(self.scale_rule_id == SCALE_RULE_ABS_MAX):
                    error_fp = cutlass.max(error_fp, row_error_fp)
                    error_int = cutlass.max(error_int, row_error_int)
                else:
                    error_fp = error_fp + row_error_fp
                    error_int = error_int + row_error_int
                row_offset = row_offset + Int32(1)

            output_scale_selected = output_scale
            scale_fp8_selected = scale_fp8
            if error_int < error_fp:
                output_scale_selected = output_scale_int
                scale_fp8_selected = scale_fp8 + Uint8(128)

            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                packed64 = bfloat2x8_to_e2m1x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    output_scale_selected,
                )
                if error_int < error_fp:
                    packed64 = bfloat2x8_to_int4x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        output_scale_selected,
                    )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8_selected
                output_offset = col_idx * (NVFP4_SCALE_BLOCK_SIZE // 2)
                output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
                st_global_u64(output_ptr, packed64)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100IF4BS8AdaptiveQuantize2D:
    def __init__(self, k: int, scale_rule_id: int):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_block_size = 8
        self.scale_blocks_per_row = k // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * Int32(8)

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                row_max = bfloat2_max_abs_8(
                    h0,
                    h1,
                    h2,
                    h3,
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (
                block_max * rcp_approx_ftz(Float32(E2M1_MAX))
            )
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(
                scale_fp8_u32,
                global_scale,
            )
            selection_scale = nvfp4_compute_quant_scale_exact(
                scale_fp8_u32,
                global_scale,
            )
            dequant_scale = nvfp4_compute_dequant_scale(
                scale_fp8_u32,
                global_scale,
            )
            output_scale_int = output_scale * Float32(IF4_INT_EXPANSION_FACTOR_RCP)
            selection_scale_int = selection_scale * Float32(
                IF4_INT_EXPANSION_FACTOR_RCP,
            )
            dequant_scale_int = dequant_scale * Float32(IF4_INT_EXPANSION_FACTOR)

            error_fp = Float32(0.0)
            error_int = Float32(0.0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                row_error_fp = _nvfp4_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h0,
                    h1,
                    h2,
                    h3,
                    selection_scale,
                    dequant_scale,
                    self.scale_rule_id,
                )
                row_error_int = _int4_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h0,
                    h1,
                    h2,
                    h3,
                    selection_scale_int,
                    dequant_scale_int,
                    self.scale_rule_id,
                )
                if cutlass.const_expr(self.scale_rule_id == SCALE_RULE_ABS_MAX):
                    error_fp = cutlass.max(error_fp, row_error_fp)
                    error_int = cutlass.max(error_int, row_error_int)
                else:
                    error_fp = error_fp + row_error_fp
                    error_int = error_int + row_error_int
                row_offset = row_offset + Int32(1)

            output_scale_selected = output_scale
            scale_fp8_selected = scale_fp8
            if error_int < error_fp:
                output_scale_selected = output_scale_int
                scale_fp8_selected = scale_fp8 + Uint8(128)

            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                packed64 = cutlass.Uint64(
                    bfloat2x4_to_e2m1x8_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        output_scale_selected,
                    ),
                )
                if error_int < error_fp:
                    packed64 = cutlass.Uint64(
                        bfloat2x4_to_int4x8_packed(
                            h0,
                            h1,
                            h2,
                            h3,
                            output_scale_selected,
                        ),
                    )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8_selected
                output_offset = col_idx * Int32(4)
                output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
                st_global_u32(output_ptr, Uint32(packed64 & cutlass.Uint64(0xFFFFFFFF)))
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100IF6AdaptiveQuantize2D:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        max_quantized_value: float,
        int_expansion_factor: float,
        int_expansion_factor_rcp: float,
        use_e3m2: bool,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.max_quantized_value = max_quantized_value
        self.int_expansion_factor = int_expansion_factor
        self.int_expansion_factor_rcp = int_expansion_factor_rcp
        self.use_e3m2 = use_e3m2
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value) * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_max = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            scale_float = global_scale * (
                bfloat2_hmax_reduce_to_f32(block_max_h2)
                * rcp_approx_ftz(Float32(float(self.max_quantized_value)))
            )
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(
                scale_fp8_u32,
                global_scale,
            )
            selection_scale = nvfp4_compute_quant_scale_exact(
                scale_fp8_u32,
                global_scale,
            )
            dequant_scale = nvfp4_compute_dequant_scale(
                scale_fp8_u32,
                global_scale,
            )
            output_scale_int = output_scale * Float32(
                float(self.int_expansion_factor_rcp),
            )
            selection_scale_int = selection_scale * Float32(
                float(self.int_expansion_factor_rcp),
            )
            dequant_scale_int = dequant_scale * Float32(
                float(self.int_expansion_factor),
            )

            error_fp = Float32(0.0)
            error_int = Float32(0.0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_error_fp = _fp6_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    selection_scale,
                    dequant_scale,
                    self.scale_rule_id,
                    self.use_e3m2,
                )
                row_error_int = _int6_block_error_bfloat(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    selection_scale_int,
                    dequant_scale_int,
                    self.scale_rule_id,
                )
                if cutlass.const_expr(self.scale_rule_id == SCALE_RULE_ABS_MAX):
                    error_fp = cutlass.max(error_fp, row_error_fp)
                    error_int = cutlass.max(error_int, row_error_int)
                else:
                    error_fp = error_fp + row_error_fp
                    error_int = error_int + row_error_int
                row_offset = row_offset + Int32(1)

            output_scale_selected = output_scale
            scale_fp8_selected = scale_fp8
            if error_int < error_fp:
                output_scale_selected = output_scale_int
                scale_fp8_selected = scale_fp8 + Uint8(128)

            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                if cutlass.const_expr(self.use_e3m2):
                    packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        output_scale_selected,
                    )
                else:
                    packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        output_scale_selected,
                    )
                if error_int < error_fp:
                    packed64_0, packed64_1 = bfloat2x8_to_int6x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        output_scale_selected,
                    )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8_selected
                output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
                output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(8),
                )
                st_global_u64(output_ptr0, packed64_0)
                st_global_u64(output_ptr1, packed64_1)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100MXFP4StaticQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: int,
        scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
        stochastic_rounding: bool = False,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_block_size = scale_block_size
        self.stochastic_rounding = stochastic_rounding
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[MX_STATIC_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * MX_STATIC_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * MX_STATIC_THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            scale_ue8m0, packed64_0, packed64_1 = (
                _process_mxfp4_static_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    self.max_quantized_value,
                    self.scale_block_size,
                    self.stochastic_rounding,
                    Uint32(row_idx * self.k + elem_base),
                )
            )

            scales[sf_idx] = scale_ue8m0
            output_offset = col_idx * (self.scale_block_size // 2)
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                st_global_u32(output_ptr0, Uint32(packed64_0 & cutlass.Uint64(0xFFFFFFFF)))
            else:
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(8),
                )
                st_global_u64(output_ptr0, packed64_0)
                st_global_u64(output_ptr1, packed64_1)

            sf_idx = sf_idx + stride


class Sm100MXFP4StaticTransposeQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: int,
        scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
        stochastic_rounding: bool = False,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_block_size = scale_block_size
        self.stochastic_rounding = stochastic_rounding
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * self.scale_block_size

            scale_ue8m0, packed64_0, packed64_1 = (
                _process_mxfp4_static_block_bfloat_transposed(
                    x,
                    row_idx,
                    elem_base,
                    self.max_quantized_value,
                    self.scale_block_size,
                    self.stochastic_rounding,
                    Uint32(row_idx * self.k + elem_base),
                )
            )

            scales[sf_idx] = scale_ue8m0
            output_offset = col_idx * (self.scale_block_size // 2)
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                st_global_u32(output_ptr0, Uint32(packed64_0 & cutlass.Uint64(0xFFFFFFFF)))
            else:
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(8),
                )
                st_global_u64(output_ptr0, packed64_0)
                st_global_u64(output_ptr1, packed64_1)

            work_idx = work_idx + stride


class Sm100MXFP3StaticQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[MX_STATIC_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * MX_STATIC_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * MX_STATIC_THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

            if cutlass.const_expr(self.scale_block_size == 8):
                block_max = bfloat2_hmax_reduce_to_f32(
                    bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3),
                )
                normalized_max = block_max * rcp_approx_ftz(Float32(4.0))
                scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
                scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
                inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
                v0, v1 = bfloat2x4_to_e2m0x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    inv_scale,
                )
                output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
                st_global_u64(
                    output_ptr,
                    (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0),
                )
            else:
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                ptr2 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(16))
                ptr3 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(24))
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
                h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
                max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                max1 = bfloat2_max_abs_8(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                )
                block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))
                normalized_max = block_max * rcp_approx_ftz(Float32(4.0))
                scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
                scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
                inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
                v0, v1, v2, v3 = bfloat2x8_to_e2m0x16_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    inv_scale,
                )
                v4, v5, v6, v7 = bfloat2x8_to_e2m0x16_values(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                    inv_scale,
                )
                output_ptr0 = get_ptr_as_int64(values[row_idx, None], elem_base)
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    elem_base + Int32(16),
                )
                st_global_v4_u32(output_ptr0, v0, v1, v2, v3)
                st_global_v4_u32(output_ptr1, v4, v5, v6, v7)

            scales[sf_idx] = scale_ue8m0
            sf_idx = sf_idx + stride


class Sm100MXFP3StaticTransposeQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * self.scale_block_size

            h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), row_idx)
            h1 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(2),
                elem_base + Int32(3),
                row_idx,
            )
            h2 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(4),
                elem_base + Int32(5),
                row_idx,
            )
            h3 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(6),
                elem_base + Int32(7),
                row_idx,
            )

            if cutlass.const_expr(self.scale_block_size == 8):
                block_max = bfloat2_hmax_reduce_to_f32(
                    bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3),
                )
                normalized_max = block_max * rcp_approx_ftz(Float32(4.0))
                scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
                scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
                inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
                v0, v1 = bfloat2x4_to_e2m0x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    inv_scale,
                )
                output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
                st_global_u64(
                    output_ptr,
                    (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0),
                )
            else:
                h4 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(8),
                    elem_base + Int32(9),
                    row_idx,
                )
                h5 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(10),
                    elem_base + Int32(11),
                    row_idx,
                )
                h6 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(12),
                    elem_base + Int32(13),
                    row_idx,
                )
                h7 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(14),
                    elem_base + Int32(15),
                    row_idx,
                )
                h8 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(16),
                    elem_base + Int32(17),
                    row_idx,
                )
                h9 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(18),
                    elem_base + Int32(19),
                    row_idx,
                )
                h10 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(20),
                    elem_base + Int32(21),
                    row_idx,
                )
                h11 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(22),
                    elem_base + Int32(23),
                    row_idx,
                )
                h12 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(24),
                    elem_base + Int32(25),
                    row_idx,
                )
                h13 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(26),
                    elem_base + Int32(27),
                    row_idx,
                )
                h14 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(28),
                    elem_base + Int32(29),
                    row_idx,
                )
                h15 = _ld_transposed_bfloat2(
                    x,
                    elem_base + Int32(30),
                    elem_base + Int32(31),
                    row_idx,
                )
                max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                max1 = bfloat2_max_abs_8(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                )
                block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))
                normalized_max = block_max * rcp_approx_ftz(Float32(4.0))
                scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
                scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
                inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
                v0, v1, v2, v3 = bfloat2x8_to_e2m0x16_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    inv_scale,
                )
                v4, v5, v6, v7 = bfloat2x8_to_e2m0x16_values(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                    inv_scale,
                )
                output_ptr0 = get_ptr_as_int64(values[row_idx, None], elem_base)
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    elem_base + Int32(16),
                )
                st_global_v4_u32(output_ptr0, v0, v1, v2, v3)
                st_global_v4_u32(output_ptr1, v4, v5, v6, v7)

            scales[sf_idx] = scale_ue8m0
            work_idx = work_idx + stride


class Sm100MXFP3StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // MXFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles).launch(
            grid=[num_blocks, 1, 1],
            block=[MXFP3_2D_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        lane_idx = tidx % Int32(MXFP4_SCALE_BLOCK_SIZE)
        warp_idx = tidx // Int32(MXFP4_SCALE_BLOCK_SIZE)
        tile_idx = bidx * MXFP3_2D_WARPS_PER_BLOCK + warp_idx
        stride = grid_dim_x * MXFP3_2D_WARPS_PER_BLOCK

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * MXFP4_SCALE_BLOCK_SIZE

            row_idx = row_group * MXFP4_SCALE_BLOCK_SIZE + lane_idx
            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
            ptr2 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(16))
            ptr3 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(24))
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
            h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
            h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
            h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
            max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
            max1 = bfloat2_max_abs_8(
                h8,
                h9,
                h10,
                h11,
                h12,
                h13,
                h14,
                h15,
            )
            row_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))
            block_max = cute.arch.warp_reduction_max(row_max)
            normalized_max = block_max * rcp_approx_ftz(Float32(4.0))
            scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
            scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
            inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

            v0, v1, v2, v3 = bfloat2x8_to_e2m0x16_values(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                inv_scale,
            )
            v4, v5, v6, v7 = bfloat2x8_to_e2m0x16_values(
                h8,
                h9,
                h10,
                h11,
                h12,
                h13,
                h14,
                h15,
                inv_scale,
            )

            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            scales[sf_idx] = scale_ue8m0
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], elem_base)
            output_ptr1 = get_ptr_as_int64(
                values[row_idx, None],
                elem_base + Int32(16),
            )
            st_global_v4_u32(output_ptr0, v0, v1, v2, v3)
            st_global_v4_u32(output_ptr1, v4, v5, v6, v7)

            tile_idx = tile_idx + stride


class Sm100MXFP6StaticQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: float,
        use_e3m2: bool,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.use_e3m2 = use_e3m2
        self.scale_blocks_per_row = k // MXFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[MX_STATIC_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * MX_STATIC_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * MX_STATIC_THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * MXFP4_SCALE_BLOCK_SIZE

            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
            ptr2 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(16))
            ptr3 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(24))
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
            h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
            h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
            h12, h13, h14, h15 = ld_global_v4_u32(ptr3)

            max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
            max1 = bfloat2_max_abs_8(
                h8,
                h9,
                h10,
                h11,
                h12,
                h13,
                h14,
                h15,
            )
            block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))
            normalized_max = block_max * rcp_approx_ftz(
                Float32(float(self.max_quantized_value)),
            )
            scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
            scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
            inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

            if cutlass.const_expr(self.use_e3m2):
                packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    inv_scale,
                )
                packed64_2, packed64_3 = bfloat2x8_to_e3m2x16_packed(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                    inv_scale,
                )
            else:
                packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    inv_scale,
                )
                packed64_2, packed64_3 = bfloat2x8_to_e2m3x16_packed(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                    inv_scale,
                )

            scales[sf_idx] = scale_ue8m0
            output_offset = col_idx * MXFP4_SCALE_BLOCK_SIZE
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            output_ptr1 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(8),
            )
            output_ptr2 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(16),
            )
            output_ptr3 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(24),
            )
            st_global_u64(output_ptr0, packed64_0)
            st_global_u64(output_ptr1, packed64_1)
            st_global_u64(output_ptr2, packed64_2)
            st_global_u64(output_ptr3, packed64_3)

            sf_idx = sf_idx + stride


class Sm100MXFP6StaticTransposeQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: float,
        use_e3m2: bool,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.use_e3m2 = use_e3m2
        self.scale_blocks_per_row = k // MXFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * MXFP4_SCALE_BLOCK_SIZE

            h0 = _ld_transposed_bfloat2(x, elem_base, elem_base + Int32(1), row_idx)
            h1 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(2),
                elem_base + Int32(3),
                row_idx,
            )
            h2 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(4),
                elem_base + Int32(5),
                row_idx,
            )
            h3 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(6),
                elem_base + Int32(7),
                row_idx,
            )
            h4 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(8),
                elem_base + Int32(9),
                row_idx,
            )
            h5 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(10),
                elem_base + Int32(11),
                row_idx,
            )
            h6 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(12),
                elem_base + Int32(13),
                row_idx,
            )
            h7 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(14),
                elem_base + Int32(15),
                row_idx,
            )
            h8 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(16),
                elem_base + Int32(17),
                row_idx,
            )
            h9 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(18),
                elem_base + Int32(19),
                row_idx,
            )
            h10 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(20),
                elem_base + Int32(21),
                row_idx,
            )
            h11 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(22),
                elem_base + Int32(23),
                row_idx,
            )
            h12 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(24),
                elem_base + Int32(25),
                row_idx,
            )
            h13 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(26),
                elem_base + Int32(27),
                row_idx,
            )
            h14 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(28),
                elem_base + Int32(29),
                row_idx,
            )
            h15 = _ld_transposed_bfloat2(
                x,
                elem_base + Int32(30),
                elem_base + Int32(31),
                row_idx,
            )

            max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
            max1 = bfloat2_max_abs_8(
                h8,
                h9,
                h10,
                h11,
                h12,
                h13,
                h14,
                h15,
            )
            block_max = bfloat2_hmax_reduce_to_f32(bfloat2_hmax2(max0, max1))
            normalized_max = block_max * rcp_approx_ftz(
                Float32(float(self.max_quantized_value)),
            )
            scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
            scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
            inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

            if cutlass.const_expr(self.use_e3m2):
                packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    inv_scale,
                )
                packed64_2, packed64_3 = bfloat2x8_to_e3m2x16_packed(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                    inv_scale,
                )
            else:
                packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    inv_scale,
                )
                packed64_2, packed64_3 = bfloat2x8_to_e2m3x16_packed(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                    inv_scale,
                )

            scales[sf_idx] = scale_ue8m0
            output_offset = col_idx * MXFP4_SCALE_BLOCK_SIZE
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            output_ptr1 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(8),
            )
            output_ptr2 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(16),
            )
            output_ptr3 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(24),
            )
            st_global_u64(output_ptr0, packed64_0)
            st_global_u64(output_ptr1, packed64_1)
            st_global_u64(output_ptr2, packed64_2)
            st_global_u64(output_ptr3, packed64_3)

            work_idx = work_idx + stride


class Sm100MXFP3BS8StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * Int32(8)

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                row_max = bfloat2_max_abs_8(
                    h0,
                    h1,
                    h2,
                    h3,
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            normalized_max = block_max * rcp_approx_ftz(Float32(4.0))
            scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
            scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
            inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                v0, v1 = bfloat2x4_to_e2m0x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    inv_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_ue8m0
                output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100MXFP4StaticQuantize2D:
    def __init__(self, k: int, max_quantized_value: int):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_blocks_per_row = k // MXFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * MXFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(MXFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * MXFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                ptr2 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(16))
                ptr3 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(24))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
                h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
                max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                max1 = bfloat2_max_abs_8(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, bfloat2_hmax2(max0, max1))
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            normalized_max = block_max * rcp_approx_ftz(
                Float32(float(self.max_quantized_value)),
            )
            scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
            scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
            inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

            row_offset = Int32(0)
            while row_offset < Int32(MXFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * MXFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                ptr2 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(16))
                ptr3 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(24))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
                h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
                packed64_0 = bfloat2x8_to_e2m1x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    inv_scale,
                )
                packed64_1 = bfloat2x8_to_e2m1x16_packed(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                    inv_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_ue8m0
                output_offset = col_idx * (MXFP4_SCALE_BLOCK_SIZE // 2)
                output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(8),
                )
                st_global_u64(output_ptr0, packed64_0)
                st_global_u64(output_ptr1, packed64_1)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100MXFP4BS8StaticQuantize2D:
    def __init__(self, k: int, max_quantized_value: int):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_blocks_per_row = k // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        lane_idx = tidx % Int32(MXFP4_BS8_2D_GROUP_SIZE)
        group_idx = tidx // Int32(MXFP4_BS8_2D_GROUP_SIZE)
        tile_idx = bidx * MXFP4_BS8_2D_GROUPS_PER_BLOCK + group_idx
        stride = grid_dim_x * MXFP4_BS8_2D_GROUPS_PER_BLOCK

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * Int32(8)

            row_idx = row_group * Int32(8) + lane_idx
            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
            row_max_h2 = bfloat2_max_abs_8(
                h0,
                h1,
                h2,
                h3,
                cutlass.Uint32(0),
                cutlass.Uint32(0),
                cutlass.Uint32(0),
                cutlass.Uint32(0),
            )
            row_max = bfloat2_hmax_reduce_to_f32(row_max_h2)
            block_max = cute.arch.warp_reduction_max(
                row_max,
                threads_in_group=MXFP4_BS8_2D_GROUP_SIZE,
            )
            normalized_max = block_max * rcp_approx_ftz(
                Float32(float(self.max_quantized_value)),
            )
            scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
            scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
            inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

            packed32 = bfloat2x4_to_e2m1x8_packed(
                h0,
                h1,
                h2,
                h3,
                inv_scale,
            )

            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            scales[sf_idx] = scale_ue8m0
            output_offset = col_idx * Int32(4)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            st_global_u32(output_ptr, packed32)

            tile_idx = tile_idx + stride


class Sm100MXFP6StaticQuantize2D:
    def __init__(self, k: int, max_quantized_value: float, use_e3m2: bool):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.use_e3m2 = use_e3m2
        self.scale_blocks_per_row = k // MXFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles).launch(
            grid=[num_blocks, 1, 1],
            block=[STATIC_2D_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * STATIC_2D_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * STATIC_2D_THREADS_PER_BLOCK

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * MXFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(MXFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * MXFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                ptr2 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(16))
                ptr3 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(24))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
                h12, h13, h14, h15 = ld_global_v4_u32(ptr3)
                max0 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                max1 = bfloat2_max_abs_8(
                    h8,
                    h9,
                    h10,
                    h11,
                    h12,
                    h13,
                    h14,
                    h15,
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, bfloat2_hmax2(max0, max1))
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            normalized_max = block_max * rcp_approx_ftz(
                Float32(float(self.max_quantized_value)),
            )
            scale_ue8m0_u32 = float_to_ue8m0_ceil(normalized_max)
            scale_ue8m0 = Uint8(scale_ue8m0_u32 & cutlass.Uint32(0xFF))
            inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

            row_offset = Int32(0)
            while row_offset < Int32(MXFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * MXFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                ptr2 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(16))
                ptr3 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(24))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
                h12, h13, h14, h15 = ld_global_v4_u32(ptr3)

                if cutlass.const_expr(self.use_e3m2):
                    packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        inv_scale,
                    )
                    packed64_2, packed64_3 = bfloat2x8_to_e3m2x16_packed(
                        h8,
                        h9,
                        h10,
                        h11,
                        h12,
                        h13,
                        h14,
                        h15,
                        inv_scale,
                    )
                else:
                    packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        inv_scale,
                    )
                    packed64_2, packed64_3 = bfloat2x8_to_e2m3x16_packed(
                        h8,
                        h9,
                        h10,
                        h11,
                        h12,
                        h13,
                        h14,
                        h15,
                        inv_scale,
                    )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_ue8m0
                output_offset = col_idx * MXFP4_SCALE_BLOCK_SIZE
                output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(8),
                )
                output_ptr2 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(16),
                )
                output_ptr3 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(24),
                )
                st_global_u64(output_ptr0, packed64_0)
                st_global_u64(output_ptr1, packed64_1)
                st_global_u64(output_ptr2, packed64_2)
                st_global_u64(output_ptr3, packed64_3)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVINT4StaticQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
        stochastic_rounding: bool = False,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.stochastic_rounding = stochastic_rounding
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            INT4_MAX * E4M3_STATIC_MAX * float(self.adjustment_factor),
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            scale_fp8, packed64 = _process_nvint4_static_block_bfloat(
                x[row_idx, None],
                elem_base,
                global_scale,
                self.scale_block_size,
                self.stochastic_rounding,
                Uint32(row_idx * self.k + elem_base),
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * (self.scale_block_size // 2)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                st_global_u32(output_ptr, Uint32(packed64 & cutlass.Uint64(0xFFFFFFFF)))
            else:
                st_global_u64(output_ptr, packed64)

            sf_idx = sf_idx + stride


class Sm100NVINT4StaticTransposeQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
        stochastic_rounding: bool = False,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.stochastic_rounding = stochastic_rounding
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row
        global_scale = _compute_global_scale(
            amax_tensor,
            INT4_MAX * E4M3_STATIC_MAX * float(self.adjustment_factor),
        )

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * self.scale_block_size

            scale_fp8, packed64 = _process_nvint4_static_block_bfloat_transposed(
                x,
                row_idx,
                elem_base,
                global_scale,
                self.scale_block_size,
                self.stochastic_rounding,
                Uint32(row_idx * self.k + elem_base),
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * (self.scale_block_size // 2)
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                st_global_u32(output_ptr, Uint32(packed64 & cutlass.Uint64(0xFFFFFFFF)))
            else:
                st_global_u64(output_ptr, packed64)

            work_idx = work_idx + stride


class Sm100NVINT6StaticQuantize:
    def __init__(self, k: int, adjustment_factor: float = 1.0):
        self.k = k
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            31.0 * E4M3_STATIC_MAX * float(self.adjustment_factor),
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
            h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

            block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(31.0)))
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            packed64_0, packed64_1 = bfloat2x8_to_int6x16_packed(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale,
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            output_ptr1 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(8),
            )
            st_global_u64(output_ptr0, packed64_0)
            st_global_u64(output_ptr1, packed64_1)

            sf_idx = sf_idx + stride


class Sm100NVINT6StaticTransposeQuantize:
    def __init__(self, k: int, adjustment_factor: float = 1.0):
        self.k = k
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row
        global_scale = _compute_global_scale(
            amax_tensor,
            31.0 * E4M3_STATIC_MAX * float(self.adjustment_factor),
        )

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            scale_fp8, packed64_0, packed64_1 = (
                _process_nvint6_static_block_bfloat_transposed(
                    x,
                    row_idx,
                    elem_base,
                    global_scale,
                )
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            output_ptr1 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(8),
            )
            st_global_u64(output_ptr0, packed64_0)
            st_global_u64(output_ptr1, packed64_1)

            work_idx = work_idx + stride


class Sm100NVINT6StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            31.0 * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                block_max_h2 = bfloat2_hmax2(
                    block_max_h2,
                    bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7),
                )
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(31.0)))
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                packed64_0, packed64_1 = bfloat2x8_to_int6x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    output_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
                output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(8),
                )
                st_global_u64(output_ptr0, packed64_0)
                st_global_u64(output_ptr1, packed64_1)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVINT3StaticQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            3.0 * E4M3_STATIC_MAX * float(self.adjustment_factor),
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

            if cutlass.const_expr(self.scale_block_size == 8):
                block_max_h2 = bfloat2_max_abs_8(
                    h0,
                    h1,
                    h2,
                    h3,
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                )
            else:
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(3.0)))
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * self.scale_block_size
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                v0, v1 = bfloat2x4_to_int3x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale,
                )
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
            else:
                v0, v1, v2, v3 = bfloat2x8_to_int3x16_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    output_scale,
                )
                st_global_v4_u32(output_ptr, v0, v1, v2, v3)

            sf_idx = sf_idx + stride


class Sm100NVINT3StaticTransposeQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row
        global_scale = _compute_global_scale(
            amax_tensor,
            3.0 * E4M3_STATIC_MAX * float(self.adjustment_factor),
        )

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * self.scale_block_size

            scale_fp8, v0, v1, v2, v3 = _process_nvint3_static_block_bfloat_transposed(
                x,
                row_idx,
                elem_base,
                global_scale,
                self.scale_block_size,
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * self.scale_block_size
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
            else:
                st_global_v4_u32(output_ptr, v0, v1, v2, v3)

            work_idx = work_idx + stride


class Sm100NVINT3StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            3.0 * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_max = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(3.0)))
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                v0, v1, v2, v3 = bfloat2x8_to_int3x16_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    output_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
                output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
                st_global_v4_u32(output_ptr, v0, v1, v2, v3)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVINT3BS8StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            3.0 * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * Int32(8)

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                row_max = bfloat2_max_abs_8(
                    h0,
                    h1,
                    h2,
                    h3,
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(3.0)))
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                v0, v1 = bfloat2x4_to_int3x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVFP3StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        lane_idx = tidx % Int32(IF3_2D_GROUP_SIZE)
        group_idx = tidx // Int32(IF3_2D_GROUP_SIZE)
        tile_idx = bidx * IF3_2D_GROUPS_PER_BLOCK + group_idx
        stride = grid_dim_x * IF3_2D_GROUPS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M0_MAX * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + lane_idx
            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
            h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
            row_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
            row_max = bfloat2_hmax_reduce_to_f32(row_max_h2)
            block_max = cute.arch.warp_reduction_max(
                row_max,
                threads_in_group=IF3_2D_GROUP_SIZE,
            )
            scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M0_MAX)))
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            v0, v1, v2, v3 = bfloat2x8_to_e2m0x16_values(
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                output_scale,
            )

            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            scales[sf_idx] = scale_fp8
            output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            st_global_v4_u32(output_ptr, v0, v1, v2, v3)

            tile_idx = tile_idx + stride


class Sm100NVFP3BS8StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M0_MAX * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * Int32(8)

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                row_max = bfloat2_max_abs_8(
                    h0,
                    h1,
                    h2,
                    h3,
                    h0,
                    h1,
                    h2,
                    h3,
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(E2M0_MAX)))
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                v0, v1 = bfloat2x4_to_e2m0x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVFP6StaticQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: float,
        use_e3m2: bool,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.use_e3m2 = use_e3m2
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value)
            * E4M3_STATIC_MAX
            * float(self.adjustment_factor),
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            scale_fp8, packed64_0, packed64_1 = _process_nvfp6_static_block_bfloat(
                x[row_idx, None],
                elem_base,
                global_scale,
                self.max_quantized_value,
                self.use_e3m2,
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            output_ptr1 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(8),
            )
            st_global_u64(output_ptr0, packed64_0)
            st_global_u64(output_ptr1, packed64_1)

            sf_idx = sf_idx + stride


class Sm100NVFP6StaticTransposeQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: float,
        use_e3m2: bool,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.use_e3m2 = use_e3m2
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value)
            * E4M3_STATIC_MAX
            * float(self.adjustment_factor),
        )

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            scale_fp8, packed64_0, packed64_1 = (
                _process_nvfp6_static_block_bfloat_transposed(
                    x,
                    row_idx,
                    elem_base,
                    global_scale,
                    self.max_quantized_value,
                    self.use_e3m2,
                )
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
            output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
            output_ptr1 = get_ptr_as_int64(
                values[row_idx, None],
                output_offset + Int32(8),
            )
            st_global_u64(output_ptr0, packed64_0)
            st_global_u64(output_ptr1, packed64_1)

            work_idx = work_idx + stride


class Sm100NVFP3StaticQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=16,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            4.0 * E4M3_STATIC_MAX * float(self.adjustment_factor),
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)

            if cutlass.const_expr(self.scale_block_size == 8):
                block_max = bfloat2_hmax_reduce_to_f32(
                    bfloat2_max_abs_8(h0, h1, h2, h3, h0, h1, h2, h3),
                )
                output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
            else:
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                block_max = bfloat2_hmax_reduce_to_f32(
                    bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7),
                )
                output_ptr = get_ptr_as_int64(values[row_idx, None], elem_base)

            scale_float = global_scale * (block_max * rcp_approx_ftz(Float32(4.0)))
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            if cutlass.const_expr(self.scale_block_size == 8):
                v0, v1 = bfloat2x4_to_e2m0x8_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale,
                )
                st_global_u64(
                    output_ptr,
                    (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0),
                )
            else:
                v0, v1, v2, v3 = bfloat2x8_to_e2m0x16_values(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    output_scale,
                )
                st_global_v4_u32(output_ptr, v0, v1, v2, v3)

            scales[sf_idx] = scale_fp8
            sf_idx = sf_idx + stride


class Sm100NVFP3StaticTransposeQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        work_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        output_rows = total_scale_blocks // self.scale_blocks_per_row
        global_scale = _compute_global_scale(
            amax_tensor,
            4.0 * E4M3_STATIC_MAX * float(self.adjustment_factor),
        )

        while work_idx < total_scale_blocks:
            row_idx = work_idx % output_rows
            col_idx = work_idx // output_rows
            sf_idx = row_idx * self.scale_blocks_per_row + col_idx
            elem_base = col_idx * self.scale_block_size

            scale_fp8, v0, v1, v2, v3 = _process_nvfp3_static_block_bfloat_transposed(
                x,
                row_idx,
                elem_base,
                global_scale,
                self.scale_block_size,
            )

            scales[sf_idx] = scale_fp8
            output_offset = col_idx * self.scale_block_size
            output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
            if cutlass.const_expr(self.scale_block_size == 8):
                packed64 = (cutlass.Uint64(v1) << cutlass.Uint64(32)) | cutlass.Uint64(v0)
                st_global_u64(output_ptr, packed64)
            else:
                st_global_v4_u32(output_ptr, v0, v1, v2, v3)

            work_idx = work_idx + stride


class Sm100NVINT4StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            INT4_MAX * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_max = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (
                block_max * rcp_approx_ftz(Float32(INT4_MAX))
            )
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                packed64 = bfloat2x8_to_int4x16_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    h7,
                    output_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_offset = col_idx * (NVFP4_SCALE_BLOCK_SIZE // 2)
                output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
                st_global_u64(output_ptr, packed64)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVINT4BS8StaticQuantize2D:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            INT4_MAX * E4M3_STATIC_MAX,
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * Int32(8)

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                row_max = bfloat2_max_abs_8(
                    h0,
                    h1,
                    h2,
                    h3,
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                )
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (
                block_max * rcp_approx_ftz(Float32(INT4_MAX))
            )
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(8):
                row_idx = row_group * Int32(8) + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                packed32 = bfloat2x4_to_int4x8_packed(
                    h0,
                    h1,
                    h2,
                    h3,
                    output_scale,
                )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_offset = col_idx * Int32(4)
                output_ptr = get_ptr_as_int64(values[row_idx, None], output_offset)
                st_global_u32(output_ptr, packed32)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVFP6StaticQuantize2D:
    def __init__(
        self,
        k: int,
        max_quantized_value: float,
        use_e3m2: bool,
        adjustment_factor: float = 1.0,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.use_e3m2 = use_e3m2
        self.adjustment_factor = adjustment_factor
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, values, scales, total_scale_tiles, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        values: cute.Tensor,
        scales: cute.Tensor,
        total_scale_tiles: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        tile_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value)
            * E4M3_STATIC_MAX
            * float(self.adjustment_factor),
        )

        while tile_idx < total_scale_tiles:
            row_group = tile_idx // self.scale_blocks_per_row
            col_idx = tile_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            block_max_h2 = cutlass.Uint32(0)
            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                row_max = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
                block_max_h2 = bfloat2_hmax2(block_max_h2, row_max)
                row_offset = row_offset + Int32(1)

            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)
            scale_float = global_scale * (
                block_max * rcp_approx_ftz(Float32(float(self.max_quantized_value)))
            )
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))
            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

            row_offset = Int32(0)
            while row_offset < Int32(NVFP4_SCALE_BLOCK_SIZE):
                row_idx = row_group * NVFP4_SCALE_BLOCK_SIZE + row_offset
                ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
                ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
                h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
                h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
                if cutlass.const_expr(self.use_e3m2):
                    packed64_0, packed64_1 = bfloat2x8_to_e3m2x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        output_scale,
                    )
                else:
                    packed64_0, packed64_1 = bfloat2x8_to_e2m3x16_packed(
                        h0,
                        h1,
                        h2,
                        h3,
                        h4,
                        h5,
                        h6,
                        h7,
                        output_scale,
                    )

                sf_idx = row_idx * self.scale_blocks_per_row + col_idx
                scales[sf_idx] = scale_fp8
                output_offset = col_idx * NVFP4_SCALE_BLOCK_SIZE
                output_ptr0 = get_ptr_as_int64(values[row_idx, None], output_offset)
                output_ptr1 = get_ptr_as_int64(
                    values[row_idx, None],
                    output_offset + Int32(8),
                )
                st_global_u64(output_ptr0, packed64_0)
                st_global_u64(output_ptr1, packed64_1)
                row_offset = row_offset + Int32(1)

            tile_idx = tile_idx + stride


class Sm100NVFP4StaticPseudoQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[PSEUDO_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * PSEUDO_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * PSEUDO_THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value) * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_nvfp4_static_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.max_quantized_value,
                    self.scale_block_size,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            if cutlass.const_expr(self.scale_block_size != 8):
                output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
                st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100NVFP3StaticPseudoQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M0_MAX * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_nvfp3_static_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.scale_block_size,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            if cutlass.const_expr(self.scale_block_size != 8):
                output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
                st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100NVFP6StaticPseudoQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: float,
        use_e3m2: bool,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.use_e3m2 = use_e3m2
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            float(self.max_quantized_value) * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_nvfp6_static_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.max_quantized_value,
                    self.use_e3m2,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100NVINT6StaticPseudoQuantize:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            31.0 * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_nvint6_static_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100NVINT3StaticPseudoQuantize:
    def __init__(self, k: int, scale_block_size: int):
        self.k = k
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            3.0 * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_nvint3_static_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.scale_block_size,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            if cutlass.const_expr(self.scale_block_size != 8):
                output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
                st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100NVINT4StaticPseudoQuantize:
    def __init__(self, k: int, scale_block_size: int):
        self.k = k
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            7.0 * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_nvint4_static_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.scale_block_size,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            if cutlass.const_expr(self.scale_block_size != 8):
                output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
                st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100MXFP4StaticPseudoQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: int,
        scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[PSEUDO_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * PSEUDO_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * PSEUDO_THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            (
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
                h8,
                h9,
                h10,
                h11,
                h12,
                h13,
                h14,
                h15,
            ) = _process_mxfp4_static_pseudo_block_bfloat(
                x[row_idx, None],
                elem_base,
                self.max_quantized_value,
                self.scale_block_size,
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            if cutlass.const_expr(self.scale_block_size != 8):
                output_ptr1 = get_ptr_as_int64(
                    out[row_idx, None],
                    elem_base + Int32(8),
                )
                output_ptr2 = get_ptr_as_int64(
                    out[row_idx, None],
                    elem_base + Int32(16),
                )
                output_ptr3 = get_ptr_as_int64(
                    out[row_idx, None],
                    elem_base + Int32(24),
                )
                st_global_v4_u32(output_ptr1, h4, h5, h6, h7)
                st_global_v4_u32(output_ptr2, h8, h9, h10, h11)
                st_global_v4_u32(output_ptr3, h12, h13, h14, h15)

            sf_idx = sf_idx + stride


class Sm100MXFP3StaticPseudoQuantize:
    def __init__(
        self,
        k: int,
        scale_block_size: int,
    ):
        self.k = k
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[PSEUDO_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * PSEUDO_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * PSEUDO_THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            (
                out0,
                out1,
                out2,
                out3,
                out4,
                out5,
                out6,
                out7,
                out8,
                out9,
                out10,
                out11,
                out12,
                out13,
                out14,
                out15,
            ) = _process_mxfp3_static_pseudo_block_bfloat(
                x[row_idx, None],
                elem_base,
                self.scale_block_size,
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr0, out0, out1, out2, out3)
            if cutlass.const_expr(self.scale_block_size != 8):
                output_ptr1 = get_ptr_as_int64(
                    out[row_idx, None],
                    elem_base + Int32(8),
                )
                output_ptr2 = get_ptr_as_int64(
                    out[row_idx, None],
                    elem_base + Int32(16),
                )
                output_ptr3 = get_ptr_as_int64(
                    out[row_idx, None],
                    elem_base + Int32(24),
                )
                st_global_v4_u32(output_ptr1, out4, out5, out6, out7)
                st_global_v4_u32(output_ptr2, out8, out9, out10, out11)
                st_global_v4_u32(output_ptr3, out12, out13, out14, out15)

            sf_idx = sf_idx + stride


class Sm100MXFP6StaticPseudoQuantize:
    def __init__(
        self,
        k: int,
        max_quantized_value: float,
        use_e3m2: bool,
    ):
        self.k = k
        self.max_quantized_value = max_quantized_value
        self.use_e3m2 = use_e3m2
        self.scale_blocks_per_row = k // MXFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[PSEUDO_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * PSEUDO_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * PSEUDO_THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * MXFP4_SCALE_BLOCK_SIZE

            (
                out0,
                out1,
                out2,
                out3,
                out4,
                out5,
                out6,
                out7,
                out8,
                out9,
                out10,
                out11,
                out12,
                out13,
                out14,
                out15,
            ) = _process_mxfp6_static_pseudo_block_bfloat(
                x[row_idx, None],
                elem_base,
                self.max_quantized_value,
                self.use_e3m2,
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            output_ptr1 = get_ptr_as_int64(
                out[row_idx, None],
                elem_base + Int32(8),
            )
            output_ptr2 = get_ptr_as_int64(
                out[row_idx, None],
                elem_base + Int32(16),
            )
            output_ptr3 = get_ptr_as_int64(
                out[row_idx, None],
                elem_base + Int32(24),
            )
            st_global_v4_u32(output_ptr0, out0, out1, out2, out3)
            st_global_v4_u32(output_ptr1, out4, out5, out6, out7)
            st_global_v4_u32(output_ptr2, out8, out9, out10, out11)
            st_global_v4_u32(output_ptr3, out12, out13, out14, out15)

            sf_idx = sf_idx + stride


class Sm100NVFP4AdaptivePseudoQuantize:
    def __init__(self, k: int, scale_rule_id: int):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_FOUROVERSIX_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_nvfp4_adaptive_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.scale_rule_id,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100IF4AdaptivePseudoQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[PSEUDO_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * PSEUDO_THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * PSEUDO_THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M1_MAX * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_if4_adaptive_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.scale_rule_id,
                    self.scale_block_size,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            if cutlass.const_expr(self.scale_block_size != 8):
                output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
                st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100IF3AdaptivePseudoQuantize:
    def __init__(
        self,
        k: int,
        scale_rule_id: int,
        scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    ):
        self.k = k
        self.scale_rule_id = scale_rule_id
        self.scale_block_size = scale_block_size
        self.scale_blocks_per_row = k // scale_block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        amax_tensor: cute.Tensor,
        stream,
    ):
        self.kernel(x, out, total_scale_blocks, amax_tensor).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        amax_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK
        global_scale = _compute_global_scale(
            amax_tensor,
            E2M0_MAX * E4M3_STATIC_MAX,
        )

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * self.scale_block_size

            h0, h1, h2, h3, h4, h5, h6, h7 = (
                _process_if3_adaptive_pseudo_block_bfloat(
                    x[row_idx, None],
                    elem_base,
                    global_scale,
                    self.scale_rule_id,
                    self.scale_block_size,
                )
            )

            output_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            st_global_v4_u32(output_ptr0, h0, h1, h2, h3)
            if cutlass.const_expr(self.scale_block_size != 8):
                output_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
                st_global_v4_u32(output_ptr1, h4, h5, h6, h7)

            sf_idx = sf_idx + stride


class Sm100RHTTransform:
    def __init__(self, k: int):
        self.k = k
        self.blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(x, out, total_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        total_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        block_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK

        while block_idx < total_blocks:
            row_idx = block_idx // self.blocks_per_row
            col_idx = block_idx % self.blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            ptr0 = get_ptr_as_int64(x[row_idx, None], elem_base)
            ptr1 = get_ptr_as_int64(x[row_idx, None], elem_base + Int32(8))
            h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
            h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

            x0, x1 = bfloat2_to_float2_scaled(h0, Float32(1.0))
            x2, x3 = bfloat2_to_float2_scaled(h1, Float32(1.0))
            x4, x5 = bfloat2_to_float2_scaled(h2, Float32(1.0))
            x6, x7 = bfloat2_to_float2_scaled(h3, Float32(1.0))
            x8, x9 = bfloat2_to_float2_scaled(h4, Float32(1.0))
            x10, x11 = bfloat2_to_float2_scaled(h5, Float32(1.0))
            x12, x13 = bfloat2_to_float2_scaled(h6, Float32(1.0))
            x14, x15 = bfloat2_to_float2_scaled(h7, Float32(1.0))

            x3 = -x3
            x5 = -x5
            x6 = -x6
            x7 = -x7
            x8 = -x8
            x9 = -x9
            x10 = -x10
            x12 = -x12
            x14 = -x14
            x15 = -x15

            a0 = x0 + x1
            a1 = x0 - x1
            a2 = x2 + x3
            a3 = x2 - x3
            a4 = x4 + x5
            a5 = x4 - x5
            a6 = x6 + x7
            a7 = x6 - x7
            a8 = x8 + x9
            a9 = x8 - x9
            a10 = x10 + x11
            a11 = x10 - x11
            a12 = x12 + x13
            a13 = x12 - x13
            a14 = x14 + x15
            a15 = x14 - x15

            b0 = a0 + a2
            b1 = a1 + a3
            b2 = a0 - a2
            b3 = a1 - a3
            b4 = a4 + a6
            b5 = a5 + a7
            b6 = a4 - a6
            b7 = a5 - a7
            b8 = a8 + a10
            b9 = a9 + a11
            b10 = a8 - a10
            b11 = a9 - a11
            b12 = a12 + a14
            b13 = a13 + a15
            b14 = a12 - a14
            b15 = a13 - a15

            c0 = b0 + b4
            c1 = b1 + b5
            c2 = b2 + b6
            c3 = b3 + b7
            c4 = b0 - b4
            c5 = b1 - b5
            c6 = b2 - b6
            c7 = b3 - b7
            c8 = b8 + b12
            c9 = b9 + b13
            c10 = b10 + b14
            c11 = b11 + b15
            c12 = b8 - b12
            c13 = b9 - b13
            c14 = b10 - b14
            c15 = b11 - b15

            scale = Float32(0.25)
            y0 = (c0 + c8) * scale
            y1 = (c1 + c9) * scale
            y2 = (c2 + c10) * scale
            y3 = (c3 + c11) * scale
            y4 = (c4 + c12) * scale
            y5 = (c5 + c13) * scale
            y6 = (c6 + c14) * scale
            y7 = (c7 + c15) * scale
            y8 = (c0 - c8) * scale
            y9 = (c1 - c9) * scale
            y10 = (c2 - c10) * scale
            y11 = (c3 - c11) * scale
            y12 = (c4 - c12) * scale
            y13 = (c5 - c13) * scale
            y14 = (c6 - c14) * scale
            y15 = (c7 - c15) * scale

            out0 = float2_to_bfloat2(y0, y1)
            out1 = float2_to_bfloat2(y2, y3)
            out2 = float2_to_bfloat2(y4, y5)
            out3 = float2_to_bfloat2(y6, y7)
            out4 = float2_to_bfloat2(y8, y9)
            out5 = float2_to_bfloat2(y10, y11)
            out6 = float2_to_bfloat2(y12, y13)
            out7 = float2_to_bfloat2(y14, y15)

            out_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            out_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            st_global_v4_u32(out_ptr0, out0, out1, out2, out3)
            st_global_v4_u32(out_ptr1, out4, out5, out6, out7)

            block_idx = block_idx + stride


class Sm100FP6DequantizeValues:
    def __init__(self, k: int, use_e3m2: bool, is_if6: bool):
        self.k = k
        self.use_e3m2 = use_e3m2
        self.is_if6 = is_if6
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(values, scales, out, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
            p0, p1, p2, p3 = ld_global_v4_u32(ptr)

            f00 = Float32(0.0)
            f01 = Float32(0.0)
            f02 = Float32(0.0)
            f03 = Float32(0.0)
            f10 = Float32(0.0)
            f11 = Float32(0.0)
            f12 = Float32(0.0)
            f13 = Float32(0.0)
            f20 = Float32(0.0)
            f21 = Float32(0.0)
            f22 = Float32(0.0)
            f23 = Float32(0.0)
            f30 = Float32(0.0)
            f31 = Float32(0.0)
            f32 = Float32(0.0)
            f33 = Float32(0.0)

            if cutlass.const_expr(self.is_if6):
                scale_byte = Uint8(scales[sf_idx])
                use_int = scale_byte >= Uint8(128)
                if use_int:
                    factor = Float32(0.241943359375)
                    if cutlass.const_expr(self.use_e3m2):
                        factor = Float32(0.9033203125)
                    f00, f01, f02, f03 = cvt_int6x4_to_f32(p0, factor)
                    f10, f11, f12, f13 = cvt_int6x4_to_f32(p1, factor)
                    f20, f21, f22, f23 = cvt_int6x4_to_f32(p2, factor)
                    f30, f31, f32, f33 = cvt_int6x4_to_f32(p3, factor)
                elif cutlass.const_expr(self.use_e3m2):
                    f00, f01, f02, f03 = cvt_e3m2x4_to_f32(p0)
                    f10, f11, f12, f13 = cvt_e3m2x4_to_f32(p1)
                    f20, f21, f22, f23 = cvt_e3m2x4_to_f32(p2)
                    f30, f31, f32, f33 = cvt_e3m2x4_to_f32(p3)
                else:
                    f00, f01, f02, f03 = cvt_e2m3x4_to_f32(p0)
                    f10, f11, f12, f13 = cvt_e2m3x4_to_f32(p1)
                    f20, f21, f22, f23 = cvt_e2m3x4_to_f32(p2)
                    f30, f31, f32, f33 = cvt_e2m3x4_to_f32(p3)
            elif cutlass.const_expr(self.use_e3m2):
                f00, f01, f02, f03 = cvt_e3m2x4_to_f32(p0)
                f10, f11, f12, f13 = cvt_e3m2x4_to_f32(p1)
                f20, f21, f22, f23 = cvt_e3m2x4_to_f32(p2)
                f30, f31, f32, f33 = cvt_e3m2x4_to_f32(p3)
            else:
                f00, f01, f02, f03 = cvt_e2m3x4_to_f32(p0)
                f10, f11, f12, f13 = cvt_e2m3x4_to_f32(p1)
                f20, f21, f22, f23 = cvt_e2m3x4_to_f32(p2)
                f30, f31, f32, f33 = cvt_e2m3x4_to_f32(p3)

            out_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            out_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(4))
            out_ptr2 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            out_ptr3 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(12))
            st_global_v4_f32(out_ptr0, f00, f01, f02, f03)
            st_global_v4_f32(out_ptr1, f10, f11, f12, f13)
            st_global_v4_f32(out_ptr2, f20, f21, f22, f23)
            st_global_v4_f32(out_ptr3, f30, f31, f32, f33)

            sf_idx = sf_idx + stride


class Sm100INT6DequantizeValues:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(values, out, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
            p0, p1, p2, p3 = ld_global_v4_u32(ptr)
            f00, f01, f02, f03 = cvt_int6x4_to_f32(p0, Float32(1.0))
            f10, f11, f12, f13 = cvt_int6x4_to_f32(p1, Float32(1.0))
            f20, f21, f22, f23 = cvt_int6x4_to_f32(p2, Float32(1.0))
            f30, f31, f32, f33 = cvt_int6x4_to_f32(p3, Float32(1.0))

            out_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            out_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(4))
            out_ptr2 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            out_ptr3 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(12))
            st_global_v4_f32(out_ptr0, f00, f01, f02, f03)
            st_global_v4_f32(out_ptr1, f10, f11, f12, f13)
            st_global_v4_f32(out_ptr2, f20, f21, f22, f23)
            st_global_v4_f32(out_ptr3, f30, f31, f32, f33)

            sf_idx = sf_idx + stride


class Sm100INT3DequantizeValues:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(values, out, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
            p0, p1, p2, p3 = ld_global_v4_u32(ptr)
            f00, f01, f02, f03 = cvt_int3x4_to_f32(p0)
            f10, f11, f12, f13 = cvt_int3x4_to_f32(p1)
            f20, f21, f22, f23 = cvt_int3x4_to_f32(p2)
            f30, f31, f32, f33 = cvt_int3x4_to_f32(p3)

            out_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            out_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(4))
            out_ptr2 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            out_ptr3 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(12))
            st_global_v4_f32(out_ptr0, f00, f01, f02, f03)
            st_global_v4_f32(out_ptr1, f10, f11, f12, f13)
            st_global_v4_f32(out_ptr2, f20, f21, f22, f23)
            st_global_v4_f32(out_ptr3, f30, f31, f32, f33)

            sf_idx = sf_idx + stride


class Sm100FP3DequantizeValues:
    def __init__(self, k: int):
        self.k = k
        self.scale_blocks_per_row = k // NVFP4_SCALE_BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(values, out, total_scale_blocks).launch(
            grid=[num_blocks, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        out: cute.Tensor,
        total_scale_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        sf_idx = bidx * THREADS_PER_BLOCK + tidx
        stride = grid_dim_x * THREADS_PER_BLOCK

        while sf_idx < total_scale_blocks:
            row_idx = sf_idx // self.scale_blocks_per_row
            col_idx = sf_idx % self.scale_blocks_per_row
            elem_base = col_idx * NVFP4_SCALE_BLOCK_SIZE

            ptr = get_ptr_as_int64(values[row_idx, None], elem_base)
            p0, p1, p2, p3 = ld_global_v4_u32(ptr)
            f00, f01, f02, f03 = cvt_e2m0x4_to_f32(p0)
            f10, f11, f12, f13 = cvt_e2m0x4_to_f32(p1)
            f20, f21, f22, f23 = cvt_e2m0x4_to_f32(p2)
            f30, f31, f32, f33 = cvt_e2m0x4_to_f32(p3)

            out_ptr0 = get_ptr_as_int64(out[row_idx, None], elem_base)
            out_ptr1 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(4))
            out_ptr2 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(8))
            out_ptr3 = get_ptr_as_int64(out[row_idx, None], elem_base + Int32(12))
            st_global_v4_f32(out_ptr0, f00, f01, f02, f03)
            st_global_v4_f32(out_ptr1, f10, f11, f12, f13)
            st_global_v4_f32(out_ptr2, f20, f21, f22, f23)
            st_global_v4_f32(out_ptr3, f30, f31, f32, f33)

            sf_idx = sf_idx + stride


@functools.cache
def _compile_rht_transform(k: int):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100RHTTransform(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_static_quantize(
    k: int,
    max_quantized_value: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4StaticQuantize(
        k,
        max_quantized_value,
        scale_block_size,
        stochastic_rounding,
        adjustment_factor,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_static_transpose_quantize(
    k: int,
    n: int,
    max_quantized_value: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4StaticTransposeQuantize(
        k,
        max_quantized_value,
        scale_block_size,
        stochastic_rounding,
        adjustment_factor,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_static_quantize_2d(
    k: int,
    max_quantized_value: int,
    stochastic_rounding: bool = False,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4StaticQuantize2D(
        k,
        max_quantized_value,
        stochastic_rounding,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_static_bs8_quantize_2d(k: int, max_quantized_value: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4BS8StaticQuantize2D(k, max_quantized_value)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_static_pseudo_quantize(
    k: int,
    max_quantized_value: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4StaticPseudoQuantize(k, max_quantized_value, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp3_static_pseudo_quantize(k: int, scale_block_size: int):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP3StaticPseudoQuantize(k, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp6_static_pseudo_quantize(
    k: int,
    max_quantized_value: float,
    use_e3m2: bool,
):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP6StaticPseudoQuantize(k, max_quantized_value, use_e3m2)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint6_static_pseudo_quantize(k: int):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT6StaticPseudoQuantize(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint3_static_pseudo_quantize(k: int, scale_block_size: int):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT3StaticPseudoQuantize(k, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint4_static_pseudo_quantize(k: int, scale_block_size: int):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT4StaticPseudoQuantize(k, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp4_static_pseudo_quantize(
    k: int,
    max_quantized_value: int,
    scale_block_size: int,
):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP4StaticPseudoQuantize(
        k,
        max_quantized_value,
        scale_block_size,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp3_static_pseudo_quantize(k: int, scale_block_size: int):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP3StaticPseudoQuantize(k, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp6_static_pseudo_quantize(
    k: int,
    max_quantized_value: float,
    use_e3m2: bool,
):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP6StaticPseudoQuantize(k, max_quantized_value, use_e3m2)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_adaptive_quantize(
    k: int,
    scale_rule_id: int,
    stochastic_rounding: bool = False,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4AdaptiveQuantize(k, scale_rule_id, stochastic_rounding)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_adaptive_transpose_quantize(
    k: int,
    n: int,
    scale_rule_id: int,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4AdaptiveTransposeQuantize(k, scale_rule_id)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_adaptive_quantize_2d(k: int, scale_rule_id: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4AdaptiveQuantize2D(k, scale_rule_id)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if3_adaptive_quantize(
    k: int,
    scale_rule_id: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF3AdaptiveQuantize(k, scale_rule_id, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if3_adaptive_transpose_quantize(
    k: int,
    n: int,
    scale_rule_id: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF3AdaptiveTransposeQuantize(k, scale_rule_id, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if4_adaptive_quantize(
    k: int,
    scale_rule_id: int,
    stochastic_rounding: bool = False,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF4AdaptiveQuantize(
        k,
        scale_rule_id,
        stochastic_rounding,
        scale_block_size,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if4_adaptive_transpose_quantize(
    k: int,
    n: int,
    scale_rule_id: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF4AdaptiveTransposeQuantize(
        k,
        scale_rule_id,
        scale_block_size,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if3_adaptive_quantize_2d(k: int, scale_rule_id: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF3AdaptiveQuantize2D(k, scale_rule_id)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if3_bs8_adaptive_quantize_2d(k: int, scale_rule_id: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF3BS8AdaptiveQuantize2D(k, scale_rule_id)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if6_adaptive_quantize(
    k: int,
    scale_rule_id: int,
    max_quantized_value: float,
    int_expansion_factor: float,
    int_expansion_factor_rcp: float,
    use_e3m2: bool,
    adjustment_factor: float = 1.0,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF6AdaptiveQuantize(
        k,
        scale_rule_id,
        max_quantized_value,
        int_expansion_factor,
        int_expansion_factor_rcp,
        use_e3m2,
        adjustment_factor,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if4_adaptive_quantize_2d(k: int, scale_rule_id: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF4AdaptiveQuantize2D(k, scale_rule_id)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if4_bs8_adaptive_quantize_2d(k: int, scale_rule_id: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF4BS8AdaptiveQuantize2D(k, scale_rule_id)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if6_adaptive_pseudo_quantize(
    k: int,
    scale_rule_id: int,
    max_quantized_value: float,
    int_expansion_factor: float,
    int_expansion_factor_rcp: float,
    use_e3m2: bool,
):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF6AdaptivePseudoQuantize(
        k,
        scale_rule_id,
        max_quantized_value,
        int_expansion_factor,
        int_expansion_factor_rcp,
        use_e3m2,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if6_adaptive_quantize_2d(
    k: int,
    scale_rule_id: int,
    max_quantized_value: float,
    int_expansion_factor: float,
    int_expansion_factor_rcp: float,
    use_e3m2: bool,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF6AdaptiveQuantize2D(
        k,
        scale_rule_id,
        max_quantized_value,
        int_expansion_factor,
        int_expansion_factor_rcp,
        use_e3m2,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp4_static_quantize(
    k: int,
    max_quantized_value: int,
    scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP4StaticQuantize(
        k,
        max_quantized_value,
        scale_block_size,
        stochastic_rounding,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp4_static_transpose_quantize(
    k: int,
    n: int,
    max_quantized_value: int,
    scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP4StaticTransposeQuantize(
        k,
        max_quantized_value,
        scale_block_size,
        stochastic_rounding,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp3_static_quantize(
    k: int,
    scale_block_size: int,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP3StaticQuantize(k, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp3_static_transpose_quantize(
    k: int,
    n: int,
    scale_block_size: int,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP3StaticTransposeQuantize(k, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp3_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP3StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp3_bs8_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP3BS8StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp6_static_quantize(
    k: int,
    max_quantized_value: float,
    use_e3m2: bool,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP6StaticQuantize(k, max_quantized_value, use_e3m2)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp6_static_transpose_quantize(
    k: int,
    n: int,
    max_quantized_value: float,
    use_e3m2: bool,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP6StaticTransposeQuantize(k, max_quantized_value, use_e3m2)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp6_static_quantize_2d(
    k: int,
    max_quantized_value: float,
    use_e3m2: bool,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP6StaticQuantize2D(k, max_quantized_value, use_e3m2)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp4_static_quantize_2d(k: int, max_quantized_value: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP4StaticQuantize2D(k, max_quantized_value)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_mxfp4_bs8_static_quantize_2d(k: int, max_quantized_value: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100MXFP4BS8StaticQuantize2D(k, max_quantized_value)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint4_static_quantize(
    k: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT4StaticQuantize(
        k,
        scale_block_size,
        stochastic_rounding,
        adjustment_factor,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint4_static_transpose_quantize(
    k: int,
    n: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT4StaticTransposeQuantize(
        k,
        scale_block_size,
        stochastic_rounding,
        adjustment_factor,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp3_static_quantize(
    k: int,
    scale_block_size: int,
    adjustment_factor: float = 1.0,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP3StaticQuantize(k, scale_block_size, adjustment_factor)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp3_static_transpose_quantize(
    k: int,
    n: int,
    scale_block_size: int,
    adjustment_factor: float = 1.0,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP3StaticTransposeQuantize(k, scale_block_size, adjustment_factor)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint3_static_quantize(
    k: int,
    scale_block_size: int,
    adjustment_factor: float = 1.0,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT3StaticQuantize(k, scale_block_size, adjustment_factor)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint3_static_transpose_quantize(
    k: int,
    n: int,
    scale_block_size: int,
    adjustment_factor: float = 1.0,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT3StaticTransposeQuantize(k, scale_block_size, adjustment_factor)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint3_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT3StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint3_bs8_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT3BS8StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp3_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP3StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp3_bs8_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP3BS8StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint6_static_quantize(
    k: int,
    adjustment_factor: float = 1.0,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT6StaticQuantize(k, adjustment_factor)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint6_static_transpose_quantize(
    k: int,
    n: int,
    adjustment_factor: float = 1.0,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT6StaticTransposeQuantize(k, adjustment_factor)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint6_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT6StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint4_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT4StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvint4_bs8_static_quantize_2d(k: int):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k // 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVINT4BS8StaticQuantize2D(k)
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp6_static_quantize(
    k: int,
    max_quantized_value: float,
    use_e3m2: bool,
    adjustment_factor: float = 1.0,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP6StaticQuantize(
        k,
        max_quantized_value,
        use_e3m2,
        adjustment_factor,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp6_static_transpose_quantize(
    k: int,
    n: int,
    max_quantized_value: float,
    use_e3m2: bool,
    adjustment_factor: float = 1.0,
):
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (k, n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (n, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP6StaticTransposeQuantize(
        k,
        max_quantized_value,
        use_e3m2,
        adjustment_factor,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_nvfp6_static_quantize_2d(
    k: int,
    max_quantized_value: float,
    use_e3m2: bool,
    adjustment_factor: float = 1.0,
):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP6StaticQuantize2D(
        k,
        max_quantized_value,
        use_e3m2,
        adjustment_factor,
    )
    compiled = cute.compile(
        kernel,
        x_fake,
        values_fake,
        scales_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_adaptive_pseudo_quantize(k: int, scale_rule_id: int):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100NVFP4AdaptivePseudoQuantize(k, scale_rule_id)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if4_adaptive_pseudo_quantize(
    k: int,
    scale_rule_id: int,
    scale_block_size: int,
):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF4AdaptivePseudoQuantize(k, scale_rule_id, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_if3_adaptive_pseudo_quantize(
    k: int,
    scale_rule_id: int,
    scale_block_size: int,
):
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100IF3AdaptivePseudoQuantize(k, scale_rule_id, scale_block_size)
    compiled = cute.compile(
        kernel,
        x_fake,
        out_fake,
        Int32(1),
        Int32(1),
        amax_fake,
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_fp6_dequantize_values(k: int, use_e3m2: bool, is_if6: bool):
    sym_m = cute.sym_int()
    sym_scale_blocks = cute.sym_int()

    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scale_blocks,),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100FP6DequantizeValues(k, use_e3m2, is_if6)
    compiled = cute.compile(
        kernel,
        values_fake,
        scales_fake,
        out_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_int3_dequantize_values(k: int):
    sym_m = cute.sym_int()

    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100INT3DequantizeValues(k)
    compiled = cute.compile(
        kernel,
        values_fake,
        out_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_fp3_dequantize_values(k: int):
    sym_m = cute.sym_int()

    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100FP3DequantizeValues(k)
    compiled = cute.compile(
        kernel,
        values_fake,
        out_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


@functools.cache
def _compile_int6_dequantize_values(k: int):
    sym_m = cute.sym_int()

    values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream()

    kernel = Sm100INT6DequantizeValues(k)
    compiled = cute.compile(
        kernel,
        values_fake,
        out_fake,
        Int32(1),
        Int32(1),
        stream_fake,
    )
    return compiled


def _validate_quantize_input(
    x: torch.Tensor,
    max_quantized_value: int,
) -> tuple[int, int]:
    if x.device.type != "cuda":
        msg = "CuTe sm100 NVFP4 quantize requires a CUDA tensor"
        raise ValueError(msg)
    if x.dtype != torch.bfloat16:
        msg = f"CuTe sm100 NVFP4 quantize requires bfloat16 input, got {x.dtype}"
        raise ValueError(msg)
    if x.ndim != 2:
        msg = f"CuTe sm100 NVFP4 quantize requires a 2D tensor, got {x.ndim}D"
        raise ValueError(msg)
    if x.shape[1] % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = (
            f"last dimension must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, "
            f"got {x.shape[1]}"
        )
        raise ValueError(msg)
    if max_quantized_value not in {4, 6}:
        msg = f"unsupported static NVFP4 max quantized value: {max_quantized_value}"
        raise ValueError(msg)
    return x.shape


def _validate_mxfp4_input(
    x: torch.Tensor,
    max_quantized_value: int,
    scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
) -> tuple[int, int]:
    if x.device.type != "cuda":
        msg = "CuTe sm100 MXFP4 quantize requires a CUDA tensor"
        raise ValueError(msg)
    if x.dtype != torch.bfloat16:
        msg = f"CuTe sm100 MXFP4 quantize requires bfloat16 input, got {x.dtype}"
        raise ValueError(msg)
    if x.ndim != 2:
        msg = f"CuTe sm100 MXFP4 quantize requires a 2D tensor, got {x.ndim}D"
        raise ValueError(msg)
    if scale_block_size not in {8, MXFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {MXFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if x.shape[1] % scale_block_size != 0:
        msg = (
            f"last dimension must be divisible by {scale_block_size}, "
            f"got {x.shape[1]}"
        )
        raise ValueError(msg)
    if max_quantized_value not in {4, 6}:
        msg = f"unsupported static MXFP4 max quantized value: {max_quantized_value}"
        raise ValueError(msg)
    return x.shape


def _launch_grid(
    total_scale_blocks: int,
    device: torch.device,
    *,
    threads_per_block: int = THREADS_PER_BLOCK,
) -> int:
    target_grid = torch.cuda.get_device_properties(device).multi_processor_count
    target_grid *= BLOCKS_PER_SM
    return min(
        (total_scale_blocks + threads_per_block - 1) // threads_per_block,
        target_grid,
    )


def _uncapped_launch_grid(
    total_scale_blocks: int,
    *,
    threads_per_block: int,
) -> int:
    return (total_scale_blocks + threads_per_block - 1) // threads_per_block


def _resolve_amax(
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    if x_amax is None:
        return torch.linalg.vector_norm(x, ord=float("inf"), dtype=torch.float32)

    if x_amax.numel() != 1:
        msg = f"x_amax must contain exactly one element, got {x_amax.numel()}"
        raise ValueError(msg)

    return x_amax.to(device=x.device, dtype=torch.float32)


def quantize_nvfp4_static(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _quantize_nvfp4_candidate(
        x,
        max_quantized_value=max_quantized_value,
        scale_block_size=scale_block_size,
        stochastic_rounding=stochastic_rounding,
        adjustment_factor=adjustment_factor,
        x_amax=x_amax,
    )


def quantize_nvfp4_static_transpose(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, max_quantized_value)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if m % scale_block_size != 0:
        msg = f"rows must be divisible by {scale_block_size}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = k * (m // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_static_transpose_quantize(
        m,
        k,
        max_quantized_value,
        scale_block_size,
        stochastic_rounding,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def _quantize_nvfp4_candidate(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, max_quantized_value)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = (
        total_scale_blocks + NVFP4_BASE_THREADS_PER_BLOCK - 1
    ) // NVFP4_BASE_THREADS_PER_BLOCK

    kernel = _compile_static_quantize(
        k,
        max_quantized_value,
        scale_block_size,
        stochastic_rounding,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp4_static_2d(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    stochastic_rounding: bool = False,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, max_quantized_value)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_static_quantize_2d(
        k,
        max_quantized_value,
        stochastic_rounding,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp4_bs8_static_2d(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, max_quantized_value)
    if m % 8 != 0:
        msg = f"rows must be divisible by 8, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // 8),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_tiles = (m // 8) * (k // 8)
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_static_bs8_quantize_2d(k, max_quantized_value)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def pseudo_quantize_nvfp4_static(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = _validate_quantize_input(x, max_quantized_value)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_static_pseudo_quantize(k, max_quantized_value, scale_block_size)
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_nvfp3_static(
    x: torch.Tensor,
    *,
    scale_block_size: int,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = _validate_quantize_input(x, 4)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_nvfp3_static_pseudo_quantize(k, scale_block_size)
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_nvfp6_static(
    x: torch.Tensor,
    *,
    max_quantized_value: float,
    use_e3m2: bool,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = _validate_quantize_input(x, 6)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_nvfp6_static_pseudo_quantize(
        k,
        max_quantized_value,
        use_e3m2,
    )
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_nvint6_static(
    x: torch.Tensor,
    *,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = _validate_quantize_input(x, 6)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_nvint6_static_pseudo_quantize(k)
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_nvint3_static(
    x: torch.Tensor,
    *,
    scale_block_size: int,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = _validate_quantize_input(x, 6)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_nvint3_static_pseudo_quantize(k, scale_block_size)
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_nvint4_static(
    x: torch.Tensor,
    *,
    scale_block_size: int,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = _validate_quantize_input(x, 6)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_nvint4_static_pseudo_quantize(k, scale_block_size)
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_mxfp4_static(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
) -> torch.Tensor:
    m, k = _validate_mxfp4_input(x, max_quantized_value, scale_block_size)
    x = x.contiguous()

    out = torch.empty_like(x)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp4_static_pseudo_quantize(
        k,
        max_quantized_value,
        scale_block_size,
    )
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_mxfp3_static(
    x: torch.Tensor,
    *,
    scale_block_size: int,
) -> torch.Tensor:
    m, k = _validate_mxfp4_input(x, 4, scale_block_size)
    x = x.contiguous()

    out = torch.empty_like(x)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp3_static_pseudo_quantize(k, scale_block_size)
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_mxfp6_static(
    x: torch.Tensor,
    *,
    max_quantized_value: float,
    use_e3m2: bool,
) -> torch.Tensor:
    m, k = _validate_quantize_input(x, 6)
    if k % MXFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"columns must be divisible by {MXFP4_SCALE_BLOCK_SIZE}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    out = torch.empty_like(x)
    total_scale_blocks = m * (k // MXFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp6_static_pseudo_quantize(
        k,
        max_quantized_value,
        use_e3m2,
    )
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return out


def quantize_mxfp4_static(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = _validate_mxfp4_input(x, max_quantized_value, scale_block_size)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=MX_STATIC_THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp4_static_quantize(
        k,
        max_quantized_value,
        scale_block_size,
        stochastic_rounding,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp4_static_transpose(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
    scale_block_size: int = MXFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = _validate_mxfp4_input(x, max_quantized_value, scale_block_size)
    if m % scale_block_size != 0:
        msg = f"rows must be divisible by {scale_block_size}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((n, m // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (n, m // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_blocks = n * (m // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp4_static_transpose_quantize(
        m,
        n,
        max_quantized_value,
        scale_block_size,
        stochastic_rounding,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp3_static(
    x: torch.Tensor,
    *,
    scale_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = _validate_mxfp4_input(x, 4, scale_block_size)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=MX_STATIC_THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp3_static_quantize(k, scale_block_size)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp3_static_transpose(
    x: torch.Tensor,
    *,
    scale_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = _validate_mxfp4_input(x, 4, scale_block_size)
    if m % scale_block_size != 0:
        msg = f"rows must be divisible by {scale_block_size}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((n, m), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (n, m // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_blocks = n * (m // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp3_static_transpose_quantize(m, n, scale_block_size)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp3_static_2d(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = _validate_mxfp4_input(x, 4, MXFP4_SCALE_BLOCK_SIZE)
    if m % MXFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {MXFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // MXFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_tiles = (m // MXFP4_SCALE_BLOCK_SIZE) * (
        k // MXFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(
        total_scale_tiles,
        x.device,
        threads_per_block=MXFP3_2D_WARPS_PER_BLOCK,
    )

    kernel = _compile_mxfp3_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp6_static(
    x: torch.Tensor,
    *,
    max_quantized_value: float,
    use_e3m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if k % MXFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"columns must be divisible by {MXFP4_SCALE_BLOCK_SIZE}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // MXFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_blocks = m * (k // MXFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=MX_STATIC_THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp6_static_quantize(
        k,
        max_quantized_value,
        use_e3m2,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp6_static_transpose(
    x: torch.Tensor,
    *,
    max_quantized_value: float,
    use_e3m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = _validate_quantize_input(x, 6)
    if m % MXFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {MXFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((n, m), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (n, m // MXFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_blocks = n * (m // MXFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp6_static_transpose_quantize(
        m,
        n,
        max_quantized_value,
        use_e3m2,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp3_bs8_static_2d(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 4)
    if m % 8 != 0:
        msg = f"rows must be divisible by 8, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // 8),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_tiles = (m // 8) * (k // 8)
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_mxfp3_bs8_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp4_static_2d(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = _validate_mxfp4_input(x, max_quantized_value)
    if m % MXFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {MXFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // MXFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_tiles = (m // MXFP4_SCALE_BLOCK_SIZE) * (
        k // MXFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_mxfp4_static_quantize_2d(k, max_quantized_value)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp6_static_2d(
    x: torch.Tensor,
    *,
    max_quantized_value: float,
    use_e3m2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if k % MXFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"columns must be divisible by {MXFP4_SCALE_BLOCK_SIZE}, got {k}"
        raise ValueError(msg)
    if m % MXFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {MXFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // MXFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_tiles = (m // MXFP4_SCALE_BLOCK_SIZE) * (
        k // MXFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(
        total_scale_tiles,
        x.device,
        threads_per_block=STATIC_2D_THREADS_PER_BLOCK,
    )

    kernel = _compile_mxfp6_static_quantize_2d(
        k,
        max_quantized_value,
        use_e3m2,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_mxfp4_bs8_static_2d(
    x: torch.Tensor,
    *,
    max_quantized_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, max_quantized_value)
    if m % 8 != 0:
        msg = f"rows must be divisible by 8, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // 8),
        dtype=torch.uint8,
        device=x.device,
    )
    total_scale_tiles = (m // 8) * (k // 8)
    num_blocks = _launch_grid(
        total_scale_tiles,
        x.device,
        threads_per_block=MXFP4_BS8_2D_GROUPS_PER_BLOCK,
    )

    kernel = _compile_mxfp4_bs8_static_quantize_2d(k, max_quantized_value)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return values, scale_factors


def quantize_nvint4_static(
    x: torch.Tensor,
    *,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_nvint4_static_quantize(
        k,
        scale_block_size,
        stochastic_rounding,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint4_static_transpose(
    x: torch.Tensor,
    *,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    stochastic_rounding: bool = False,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if m % scale_block_size != 0:
        msg = f"rows must be divisible by {scale_block_size}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = k * (m // scale_block_size)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_nvint4_static_transpose_quantize(
        m,
        k,
        scale_block_size,
        stochastic_rounding,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint6_static(
    x: torch.Tensor,
    *,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_nvint6_static_quantize(k, adjustment_factor)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint6_static_transpose(
    x: torch.Tensor,
    *,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = k * (m // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_nvint6_static_transpose_quantize(m, k, adjustment_factor)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint6_static_2d(
    x: torch.Tensor,
    *,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_nvint6_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint3_static_2d(
    x: torch.Tensor,
    *,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_nvint3_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint3_bs8_static_2d(
    x: torch.Tensor,
    *,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if m % 8 != 0:
        msg = f"rows must be divisible by 8, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // 8),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // 8) * (k // 8)
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_nvint3_bs8_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp3_static_2d(
    x: torch.Tensor,
    *,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 4)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(
        total_scale_tiles,
        x.device,
        threads_per_block=IF3_2D_GROUPS_PER_BLOCK,
    )

    kernel = _compile_nvfp3_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp3_bs8_static_2d(
    x: torch.Tensor,
    *,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 4)
    if m % 8 != 0:
        msg = f"rows must be divisible by 8, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // 8),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // 8) * (k // 8)
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_nvfp3_bs8_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint3_static(
    x: torch.Tensor,
    *,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_nvint3_static_quantize(k, scale_block_size, adjustment_factor)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp3_static(
    x: torch.Tensor,
    *,
    scale_block_size: int,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 4)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_nvfp3_static_quantize(k, scale_block_size, adjustment_factor)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp3_static_transpose(
    x: torch.Tensor,
    *,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 4)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if m % scale_block_size != 0:
        msg = f"rows must be divisible by {scale_block_size}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = k * (m // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_nvfp3_static_transpose_quantize(
        m,
        k,
        scale_block_size,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint4_static_2d(
    x: torch.Tensor,
    *,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_nvint4_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint4_bs8_static_2d(
    x: torch.Tensor,
    *,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if m % 8 != 0:
        msg = f"rows must be divisible by 8, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // 8),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // 8) * (k // 8)
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_nvint4_bs8_static_quantize_2d(k)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp6_static(
    x: torch.Tensor,
    *,
    max_quantized_value: float,
    use_e3m2: bool,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_nvfp6_static_quantize(
        k,
        max_quantized_value,
        use_e3m2,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp6_static_transpose(
    x: torch.Tensor,
    *,
    max_quantized_value: float,
    use_e3m2: bool,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = k * (m // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_nvfp6_static_transpose_quantize(
        m,
        k,
        max_quantized_value,
        use_e3m2,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp6_static_2d(
    x: torch.Tensor,
    *,
    max_quantized_value: float,
    use_e3m2: bool,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_nvfp6_static_quantize_2d(
        k,
        max_quantized_value,
        use_e3m2,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp4_adaptive(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    stochastic_rounding: bool = False,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_adaptive_quantize(k, scale_rule_id, stochastic_rounding)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp4_adaptive_transpose(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = k * (m // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_adaptive_transpose_quantize(m, k, scale_rule_id)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvfp4_adaptive_2d(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_adaptive_quantize_2d(k, scale_rule_id)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if4_adaptive(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    stochastic_rounding: bool = False,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if scale_block_size == 8 and stochastic_rounding:
        msg = "IF4_BS8 stochastic rounding is not supported by the CuTe sm100 backend"
        raise ValueError(msg)
    m, k = _validate_quantize_input(x, 6)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_if4_adaptive_quantize(
        k,
        scale_rule_id,
        stochastic_rounding,
        scale_block_size,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_nvint3_static_transpose(
    x: torch.Tensor,
    *,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = _validate_quantize_input(x, 6)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)
    if m % scale_block_size != 0:
        msg = f"rows must be divisible by {scale_block_size}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = k * (m // scale_block_size)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_nvint3_static_transpose_quantize(
        m,
        k,
        scale_block_size,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if3_adaptive(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive three-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 4)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_if3_adaptive_quantize(k, scale_rule_id, scale_block_size)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if4_adaptive_transpose(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive IF4 scale rule id {scale_rule_id}"
        raise ValueError(msg)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    if m % scale_block_size != 0:
        msg = f"rows must be divisible by {scale_block_size}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = k * (m // scale_block_size)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_if4_adaptive_transpose_quantize(
        m,
        k,
        scale_rule_id,
        scale_block_size,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if3_adaptive_transpose(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive three-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 4)
    if m % scale_block_size != 0:
        msg = f"rows must be divisible by {scale_block_size}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((k, m), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (k, m // scale_block_size),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = k * (m // scale_block_size)
    num_blocks = _launch_grid(total_scale_blocks, x.device)

    kernel = _compile_if3_adaptive_transpose_quantize(
        m,
        k,
        scale_rule_id,
        scale_block_size,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if3_adaptive_2d(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive three-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 4)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(
        total_scale_tiles,
        x.device,
        threads_per_block=IF3_2D_GROUPS_PER_BLOCK,
    )

    kernel = _compile_if3_adaptive_quantize_2d(k, scale_rule_id)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if3_bs8_adaptive_2d(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive three-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 4)
    if m % 8 != 0:
        msg = f"rows must be divisible by 8, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // 8),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // 8) * (k // 8)
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_if3_bs8_adaptive_quantize_2d(k, scale_rule_id)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if4_adaptive_2d(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_if4_adaptive_quantize_2d(k, scale_rule_id)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if4_bs8_adaptive_2d(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    if m % 8 != 0:
        msg = f"rows must be divisible by 8, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // 8),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // 8) * (k // 8)
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_if4_bs8_adaptive_quantize_2d(k, scale_rule_id)
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def quantize_if6_adaptive(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    max_quantized_value: float,
    int_expansion_factor: float,
    int_expansion_factor_rcp: float,
    use_e3m2: bool,
    adjustment_factor: float = 1.0,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    scale_factors = torch.empty(
        (total_scale_blocks,),
        dtype=torch.uint8,
        device=x.device,
    )
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_if6_adaptive_quantize(
        k,
        scale_rule_id,
        max_quantized_value,
        int_expansion_factor,
        int_expansion_factor_rcp,
        use_e3m2,
        adjustment_factor,
    )
    kernel(
        x,
        values,
        scale_factors,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def pseudo_quantize_if6_adaptive(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    max_quantized_value: float,
    int_expansion_factor: float,
    int_expansion_factor_rcp: float,
    use_e3m2: bool,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)

    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_if6_adaptive_pseudo_quantize(
        k,
        scale_rule_id,
        max_quantized_value,
        int_expansion_factor,
        int_expansion_factor_rcp,
        use_e3m2,
    )
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def quantize_if6_adaptive_2d(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    max_quantized_value: float,
    int_expansion_factor: float,
    int_expansion_factor_rcp: float,
    use_e3m2: bool,
    x_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    if m % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"rows must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {m}"
        raise ValueError(msg)
    x = x.contiguous()

    values = torch.empty((m, k), dtype=torch.uint8, device=x.device)
    scale_factors = torch.empty(
        (m, k // NVFP4_SCALE_BLOCK_SIZE),
        dtype=torch.uint8,
        device=x.device,
    )
    amax = _resolve_amax(x, x_amax)
    total_scale_tiles = (m // NVFP4_SCALE_BLOCK_SIZE) * (
        k // NVFP4_SCALE_BLOCK_SIZE
    )
    num_blocks = _launch_grid(total_scale_tiles, x.device)

    kernel = _compile_if6_adaptive_quantize_2d(
        k,
        scale_rule_id,
        max_quantized_value,
        int_expansion_factor,
        int_expansion_factor_rcp,
        use_e3m2,
    )
    kernel(
        x,
        values,
        scale_factors.reshape(-1),
        total_scale_tiles,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return values, scale_factors, amax


def pseudo_quantize_nvfp4_adaptive(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive four-over-six scale rule id {scale_rule_id}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_adaptive_pseudo_quantize(k, scale_rule_id)
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_if4_adaptive(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive IF4 scale rule id {scale_rule_id}"
        raise ValueError(msg)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=PSEUDO_THREADS_PER_BLOCK,
    )

    kernel = _compile_if4_adaptive_pseudo_quantize(
        k,
        scale_rule_id,
        scale_block_size,
    )
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def pseudo_quantize_if3_adaptive(
    x: torch.Tensor,
    *,
    scale_rule_id: int,
    scale_block_size: int = NVFP4_SCALE_BLOCK_SIZE,
    x_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale_rule_id not in {SCALE_RULE_ABS_MAX, SCALE_RULE_MAE, SCALE_RULE_MSE}:
        msg = f"unsupported adaptive IF3 scale rule id {scale_rule_id}"
        raise ValueError(msg)
    if scale_block_size not in {8, NVFP4_SCALE_BLOCK_SIZE}:
        msg = f"scale block size must be 8 or {NVFP4_SCALE_BLOCK_SIZE}, got {scale_block_size}"
        raise ValueError(msg)

    m, k = _validate_quantize_input(x, 6)
    if k % scale_block_size != 0:
        msg = f"columns must be divisible by {scale_block_size}, got {k}"
        raise ValueError(msg)
    x = x.contiguous()

    out = torch.empty_like(x)
    amax = _resolve_amax(x, x_amax)
    total_scale_blocks = m * (k // scale_block_size)
    num_blocks = _uncapped_launch_grid(
        total_scale_blocks,
        threads_per_block=THREADS_PER_BLOCK,
    )

    kernel = _compile_if3_adaptive_pseudo_quantize(
        k,
        scale_rule_id,
        scale_block_size,
    )
    kernel(
        x,
        out,
        total_scale_blocks,
        num_blocks,
        amax.reshape(1),
        cutlass_torch.current_stream(),
    )
    return out


def dequantize_fp6_values(
    values: torch.Tensor,
    scale_factors: torch.Tensor,
    *,
    use_e3m2: bool,
    is_if6: bool,
) -> torch.Tensor:
    if values.ndim != 2:
        msg = f"values must be 2D, got {values.ndim}D"
        raise ValueError(msg)
    if values.dtype != torch.uint8:
        msg = f"values must be uint8, got {values.dtype}"
        raise TypeError(msg)
    m, k = values.shape
    if k % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"columns must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {k}"
        raise ValueError(msg)
    values = values.contiguous()
    scale_factors = scale_factors.contiguous()

    out = torch.empty((m, k), dtype=torch.float32, device=values.device)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _launch_grid(total_scale_blocks, values.device)
    kernel = _compile_fp6_dequantize_values(k, use_e3m2, is_if6)
    kernel(
        values,
        scale_factors.reshape(-1),
        out,
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return out


def dequantize_int6_values(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        msg = f"values must be 2D, got {values.ndim}D"
        raise ValueError(msg)
    if values.dtype != torch.uint8:
        msg = f"values must be uint8, got {values.dtype}"
        raise TypeError(msg)
    m, k = values.shape
    if k % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"columns must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {k}"
        raise ValueError(msg)
    values = values.contiguous()

    out = torch.empty((m, k), dtype=torch.float32, device=values.device)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _launch_grid(total_scale_blocks, values.device)
    kernel = _compile_int6_dequantize_values(k)
    kernel(
        values,
        out,
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return out


def dequantize_int3_values(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        msg = f"values must be 2D, got {values.ndim}D"
        raise ValueError(msg)
    if values.dtype != torch.uint8:
        msg = f"values must be uint8, got {values.dtype}"
        raise TypeError(msg)
    m, k = values.shape
    if k % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"columns must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {k}"
        raise ValueError(msg)
    values = values.contiguous()

    out = torch.empty((m, k), dtype=torch.float32, device=values.device)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _launch_grid(total_scale_blocks, values.device)
    kernel = _compile_int3_dequantize_values(k)
    kernel(
        values,
        out,
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return out


def dequantize_fp3_values(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        msg = f"values must be 2D, got {values.ndim}D"
        raise ValueError(msg)
    if values.dtype != torch.uint8:
        msg = f"values must be uint8, got {values.dtype}"
        raise TypeError(msg)
    m, k = values.shape
    if k % NVFP4_SCALE_BLOCK_SIZE != 0:
        msg = f"columns must be divisible by {NVFP4_SCALE_BLOCK_SIZE}, got {k}"
        raise ValueError(msg)
    values = values.contiguous()

    out = torch.empty((m, k), dtype=torch.float32, device=values.device)
    total_scale_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _launch_grid(total_scale_blocks, values.device)
    kernel = _compile_fp3_dequantize_values(k)
    kernel(
        values,
        out,
        total_scale_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return out


def rht_transform(x: torch.Tensor) -> torch.Tensor:
    m, k = _validate_quantize_input(x, 4)
    x = x.contiguous()

    out = torch.empty((m, k), dtype=x.dtype, device=x.device)
    total_blocks = m * (k // NVFP4_SCALE_BLOCK_SIZE)
    num_blocks = _launch_grid(total_blocks, x.device)
    kernel = _compile_rht_transform(k)
    kernel(
        x,
        out,
        total_blocks,
        num_blocks,
        cutlass_torch.current_stream(),
    )
    return out
