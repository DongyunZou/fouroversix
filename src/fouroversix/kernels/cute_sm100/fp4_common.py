from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint16, Uint32, Uint64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def ld_global_v4_u32(
    base_ptr: Int64,
    *,
    loc=None,
    ip=None,
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def ld_global_u16(
    base_ptr: Int64,
    *,
    loc=None,
    ip=None,
) -> Uint16:
    return Uint16(
        llvm.inline_asm(
            T.i16(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.u16 $0, [$1];",
            "=h,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_u32(base_ptr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_u64(base_ptr: Int64, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_v4_f32(
    base_ptr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def nvfp4_compute_output_scale(
    fp8_val: Uint32,
    global_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(fp8_val).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_zero;
                .reg .b16 fp8_pair;
                .reg .b32 h2_32;
                .reg .b16 h_lo, h_hi;
                .reg .f32 scale_f32, decode_scale, product, result;

                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
                mov.b32 {h_lo, h_hi}, h2_32;
                cvt.f32.f16 scale_f32, h_lo;

                div.rn.f32 decode_scale, 0f3f800000, $2;
                mul.rn.f32 product, decode_scale, scale_f32;
                div.rn.f32 result, 0f3f800000, product;

                setp.eq.f32 p_zero, scale_f32, 0f00000000;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def nvfp4_compute_dequant_scale(
    fp8_val: Uint32,
    global_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(fp8_val).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 fp8_pair;
                .reg .b32 h2_32;
                .reg .b16 h_lo, h_hi;
                .reg .f32 scale_f32;

                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
                mov.b32 {h_lo, h_hi}, h2_32;
                cvt.f32.f16 scale_f32, h_lo;
                div.rn.f32 $0, scale_f32, $2;
            }
            """,
            "=f,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def nvfp4_compute_quant_scale_exact(
    fp8_val: Uint32,
    global_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(fp8_val).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_zero;
                .reg .b16 fp8_pair;
                .reg .b32 h2_32;
                .reg .b16 h_lo, h_hi;
                .reg .f32 scale_f32, decode_scale, product, result;

                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
                mov.b32 {h_lo, h_hi}, h2_32;
                cvt.f32.f16 scale_f32, h_lo;

                div.rn.f32 decode_scale, 0f3f800000, $2;
                mul.rn.f32 product, decode_scale, scale_f32;
                div.rn.f32 result, 0f3f800000, product;

                setp.eq.f32 p_zero, scale_f32, 0f00000000;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def float_to_ue8m0_ceil(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_has_mant, p_ovf;
                .reg .u32 bits, exp_biased, mantissa, bump, result;

                setp.le.f32 p_zero, $1, 0f00000000;
                mov.b32 bits, $1;
                shr.b32 exp_biased, bits, 23;
                and.b32 exp_biased, exp_biased, 255;
                and.b32 mantissa, bits, 0x7FFFFF;
                setp.ne.u32 p_has_mant, mantissa, 0;
                selp.u32 bump, 1, 0, p_has_mant;
                add.u32 result, exp_biased, bump;
                setp.gt.u32 p_ovf, result, 254;
                selp.u32 result, 254, result, p_ovf;
                selp.u32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0_to_inv_scale(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .s32 new_exp;
                .reg .b32 float_bits;
                .reg .pred p_zero;

                setp.eq.u32 p_zero, $1, 0;
                sub.s32 new_exp, 254, $1;
                max.s32 new_exp, new_exp, 0;
                shl.b32 float_bits, new_exp, 23;
                mov.b32 $0, float_bits;
                @p_zero mov.b32 $0, 0;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0_to_scale(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 float_bits;
                .reg .pred p_zero;

                setp.eq.u32 p_zero, $1, 0;
                shl.b32 float_bits, $1, 23;
                mov.b32 $0, float_bits;
                @p_zero mov.b32 $0, 0;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 a_lo, a_hi, b_lo, b_hi, max_lo, max_hi;
                .reg .f32 fa_lo, fa_hi, fb_lo, fb_hi;
                and.b32 a_lo, $1, 0xFFFF;
                shr.b32 a_hi, $1, 16;
                and.b32 b_lo, $2, 0xFFFF;
                shr.b32 b_hi, $2, 16;
                shl.b32 a_lo, a_lo, 16;
                shl.b32 a_hi, a_hi, 16;
                shl.b32 b_lo, b_lo, 16;
                shl.b32 b_hi, b_hi, 16;
                mov.b32 fa_lo, a_lo;
                mov.b32 fa_hi, a_hi;
                mov.b32 fb_lo, b_lo;
                mov.b32 fb_hi, b_hi;
                max.f32 fa_lo, fa_lo, fb_lo;
                max.f32 fa_hi, fa_hi, fb_hi;
                mov.b32 max_lo, fa_lo;
                mov.b32 max_hi, fa_hi;
                shr.b32 max_lo, max_lo, 16;
                and.b32 max_hi, max_hi, 0xFFFF0000;
                or.b32 $0, max_lo, max_hi;
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_hmax_reduce_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_to_float2_scaled(
    h2: Uint32,
    scale: Float32,
    *,
    loc=None,
    ip=None,
) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Uint32(h2).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


@dsl_user_op
def bfloat2_nvfp4_absmax_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi, qh2;
                .reg .b8 qbyte;
                .reg .b16 qlo, qhi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                cvt.rn.satfinite.e2m1x2.f32 qbyte, s1, s0;
                cvt.rn.f16x2.e2m1x2 qh2, qbyte;
                mov.b32 {qlo, qhi}, qh2;
                cvt.f32.f16 q0, qlo;
                cvt.f32.f16 q1, qhi;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                abs.f32 d0, d0;
                abs.f32 d1, d1;
                max.f32 $0, d0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_nvfp4_mae_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi, qh2;
                .reg .b8 qbyte;
                .reg .b16 qlo, qhi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                cvt.rn.satfinite.e2m1x2.f32 qbyte, s1, s0;
                cvt.rn.f16x2.e2m1x2 qh2, qbyte;
                mov.b32 {qlo, qhi}, qh2;
                cvt.f32.f16 q0, qlo;
                cvt.f32.f16 q1, qhi;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                abs.f32 d0, d0;
                abs.f32 d1, d1;
                add.rn.f32 $0, d0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_nvfp4_mse_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi, qh2;
                .reg .b8 qbyte;
                .reg .b16 qlo, qhi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                cvt.rn.satfinite.e2m1x2.f32 qbyte, s1, s0;
                cvt.rn.f16x2.e2m1x2 qh2, qbyte;
                mov.b32 {qlo, qhi}, qh2;
                cvt.f32.f16 q0, qlo;
                cvt.f32.f16 q1, qhi;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                fma.rn.f32 d0, d0, d0, 0f00000000;
                fma.rn.f32 d1, d1, d1, d0;
                mov.f32 $0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_int4_absmax_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;
                .reg .s32 i0, i1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                max.f32 s0, s0, 0fc0e00000;
                min.f32 s0, s0, 0f40e00000;
                max.f32 s1, s1, 0fc0e00000;
                min.f32 s1, s1, 0f40e00000;
                cvt.rni.s32.f32 i0, s0;
                cvt.rni.s32.f32 i1, s1;
                cvt.rn.f32.s32 q0, i0;
                cvt.rn.f32.s32 q1, i1;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                abs.f32 d0, d0;
                abs.f32 d1, d1;
                max.f32 $0, d0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_int4_mae_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;
                .reg .s32 i0, i1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                max.f32 s0, s0, 0fc0e00000;
                min.f32 s0, s0, 0f40e00000;
                max.f32 s1, s1, 0fc0e00000;
                min.f32 s1, s1, 0f40e00000;
                cvt.rni.s32.f32 i0, s0;
                cvt.rni.s32.f32 i1, s1;
                cvt.rn.f32.s32 q0, i0;
                cvt.rn.f32.s32 q1, i1;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                abs.f32 d0, d0;
                abs.f32 d1, d1;
                add.rn.f32 $0, d0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_int4_mse_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;
                .reg .s32 i0, i1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                max.f32 s0, s0, 0fc0e00000;
                min.f32 s0, s0, 0f40e00000;
                max.f32 s1, s1, 0fc0e00000;
                min.f32 s1, s1, 0f40e00000;
                cvt.rni.s32.f32 i0, s0;
                cvt.rni.s32.f32 i1, s1;
                cvt.rn.f32.s32 q0, i0;
                cvt.rn.f32.s32 q1, i1;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                fma.rn.f32 d0, d0, d0, 0f00000000;
                fma.rn.f32 d1, d1, d1, d0;
                mov.f32 $0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_fp6_absmax_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    use_e3m2: bool,
    *,
    loc=None,
    ip=None,
) -> Float32:
    cvt_inst = (
        "cvt.rn.satfinite.e3m2x2.f32 qbits, s1, s0;"
        if use_e3m2
        else "cvt.rn.satfinite.e2m3x2.f32 qbits, s1, s0;"
    )
    deq_inst = (
        "cvt.rn.f16x2.e3m2x2 qh2, qbits;"
        if use_e3m2
        else "cvt.rn.f16x2.e2m3x2 qh2, qbits;"
    )
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            f"""
            {{
                .reg .b32 lo, hi, qh2;
                .reg .b16 qbits, qlo, qhi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                {cvt_inst}
                {deq_inst}
                mov.b32 {{qlo, qhi}}, qh2;
                cvt.f32.f16 q0, qlo;
                cvt.f32.f16 q1, qhi;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                abs.f32 d0, d0;
                abs.f32 d1, d1;
                max.f32 $0, d0, d1;
            }}
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_fp6_mae_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    use_e3m2: bool,
    *,
    loc=None,
    ip=None,
) -> Float32:
    cvt_inst = (
        "cvt.rn.satfinite.e3m2x2.f32 qbits, s1, s0;"
        if use_e3m2
        else "cvt.rn.satfinite.e2m3x2.f32 qbits, s1, s0;"
    )
    deq_inst = (
        "cvt.rn.f16x2.e3m2x2 qh2, qbits;"
        if use_e3m2
        else "cvt.rn.f16x2.e2m3x2 qh2, qbits;"
    )
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            f"""
            {{
                .reg .b32 lo, hi, qh2;
                .reg .b16 qbits, qlo, qhi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                {cvt_inst}
                {deq_inst}
                mov.b32 {{qlo, qhi}}, qh2;
                cvt.f32.f16 q0, qlo;
                cvt.f32.f16 q1, qhi;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                abs.f32 d0, d0;
                abs.f32 d1, d1;
                add.rn.f32 $0, d0, d1;
            }}
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_fp6_mse_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    use_e3m2: bool,
    *,
    loc=None,
    ip=None,
) -> Float32:
    cvt_inst = (
        "cvt.rn.satfinite.e3m2x2.f32 qbits, s1, s0;"
        if use_e3m2
        else "cvt.rn.satfinite.e2m3x2.f32 qbits, s1, s0;"
    )
    deq_inst = (
        "cvt.rn.f16x2.e3m2x2 qh2, qbits;"
        if use_e3m2
        else "cvt.rn.f16x2.e2m3x2 qh2, qbits;"
    )
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            f"""
            {{
                .reg .b32 lo, hi, qh2;
                .reg .b16 qbits, qlo, qhi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                {cvt_inst}
                {deq_inst}
                mov.b32 {{qlo, qhi}}, qh2;
                cvt.f32.f16 q0, qlo;
                cvt.f32.f16 q1, qhi;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                fma.rn.f32 d0, d0, d0, 0f00000000;
                fma.rn.f32 d1, d1, d1, d0;
                mov.f32 $0, d1;
            }}
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_int6_absmax_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;
                .reg .s32 i0, i1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                max.f32 s0, s0, 0fc1f80000;
                min.f32 s0, s0, 0f41f80000;
                max.f32 s1, s1, 0fc1f80000;
                min.f32 s1, s1, 0f41f80000;
                cvt.rni.s32.f32 i0, s0;
                cvt.rni.s32.f32 i1, s1;
                cvt.rn.f32.s32 q0, i0;
                cvt.rn.f32.s32 q1, i1;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                abs.f32 d0, d0;
                abs.f32 d1, d1;
                max.f32 $0, d0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_int6_mae_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;
                .reg .s32 i0, i1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                max.f32 s0, s0, 0fc1f80000;
                min.f32 s0, s0, 0f41f80000;
                max.f32 s1, s1, 0fc1f80000;
                min.f32 s1, s1, 0f41f80000;
                cvt.rni.s32.f32 i0, s0;
                cvt.rni.s32.f32 i1, s1;
                cvt.rn.f32.s32 q0, i0;
                cvt.rn.f32.s32 q1, i1;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                abs.f32 d0, d0;
                abs.f32 d1, d1;
                add.rn.f32 $0, d0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_int6_mse_error(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 x0, x1, s0, s1, q0, q1, d0, d1;
                .reg .s32 i0, i1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                max.f32 s0, s0, 0fc1f80000;
                min.f32 s0, s0, 0f41f80000;
                max.f32 s1, s1, 0fc1f80000;
                min.f32 s1, s1, 0f41f80000;
                cvt.rni.s32.f32 i0, s0;
                cvt.rni.s32.f32 i1, s1;
                cvt.rn.f32.s32 q0, i0;
                cvt.rn.f32.s32 q1, i1;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                sub.f32 d0, q0, x0;
                sub.f32 d1, q1, x1;
                fma.rn.f32 d0, d0, d0, 0f00000000;
                fma.rn.f32 d1, d1, d1, d0;
                mov.f32 $0, d1;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def float2_to_bfloat2(
    lo: Float32,
    hi: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_nvfp4_dequant_bfloat2(
    h2: Uint32,
    output_scale: Float32,
    dequant_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(output_scale).ir_value(loc=loc, ip=ip),
                Float32(dequant_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi, qh2;
                .reg .b8 qbyte;
                .reg .b16 qlo, qhi;
                .reg .f32 x0, x1, s0, s1, q0, q1;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 x0, lo;
                mov.b32 x1, hi;
                mul.f32 s0, x0, $2;
                mul.f32 s1, x1, $2;
                cvt.rn.satfinite.e2m1x2.f32 qbyte, s1, s0;
                cvt.rn.f16x2.e2m1x2 qh2, qbyte;
                mov.b32 {qlo, qhi}, qh2;
                cvt.f32.f16 q0, qlo;
                cvt.f32.f16 q1, qhi;
                mul.rn.f32 q0, q0, $3;
                mul.rn.f32 q1, q1, $3;
                cvt.rn.bf16x2.f32 $0, q1, q0;
            }
            """,
            "=r,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e2m3x4_to_f32(
    packed: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 tmp, h2lo, h2hi;
            .reg .b16 lo16, hi16;
            .reg .b16 h0, h1, h2, h3;
            and.b32 tmp, $4, 0xFFFF;
            cvt.u16.u32 lo16, tmp;
            shr.b32 tmp, $4, 16;
            cvt.u16.u32 hi16, tmp;
            cvt.rn.f16x2.e2m3x2 h2lo, lo16;
            cvt.rn.f16x2.e2m3x2 h2hi, hi16;
            mov.b32 {h0, h1}, h2lo;
            mov.b32 {h2, h3}, h2hi;
            cvt.f32.f16 $0, h0;
            cvt.f32.f16 $1, h1;
            cvt.f32.f16 $2, h2;
            cvt.f32.f16 $3, h3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_e3m2x4_to_f32(
    packed: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 tmp, h2lo, h2hi;
            .reg .b16 lo16, hi16;
            .reg .b16 h0, h1, h2, h3;
            and.b32 tmp, $4, 0xFFFF;
            cvt.u16.u32 lo16, tmp;
            shr.b32 tmp, $4, 16;
            cvt.u16.u32 hi16, tmp;
            cvt.rn.f16x2.e3m2x2 h2lo, lo16;
            cvt.rn.f16x2.e3m2x2 h2hi, hi16;
            mov.b32 {h0, h1}, h2lo;
            mov.b32 {h2, h3}, h2hi;
            cvt.f32.f16 $0, h0;
            cvt.f32.f16 $1, h1;
            cvt.f32.f16 $2, h2;
            cvt.f32.f16 $3, h3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_int3x4_to_f32(
    packed: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 b0, b1, b2, b3;
            .reg .u32 u0, u1, u2, u3;
            .reg .u32 s0, s1, s2, s3;
            .reg .u32 m0, m1, m2, m3;
            .reg .s32 i0, i1, i2, i3;
            .reg .s32 n0, n1, n2, n3;
            .reg .pred p0, p1, p2, p3;

            mov.b32 {b0, b1, b2, b3}, $4;
            cvt.u32.u8 u0, b0;
            cvt.u32.u8 u1, b1;
            cvt.u32.u8 u2, b2;
            cvt.u32.u8 u3, b3;
            and.b32 s0, u0, 0x4;
            and.b32 s1, u1, 0x4;
            and.b32 s2, u2, 0x4;
            and.b32 s3, u3, 0x4;
            and.b32 m0, u0, 0x3;
            and.b32 m1, u1, 0x3;
            and.b32 m2, u2, 0x3;
            and.b32 m3, u3, 0x3;
            cvt.s32.u32 i0, m0;
            cvt.s32.u32 i1, m1;
            cvt.s32.u32 i2, m2;
            cvt.s32.u32 i3, m3;
            neg.s32 n0, i0;
            neg.s32 n1, i1;
            neg.s32 n2, i2;
            neg.s32 n3, i3;
            setp.ne.u32 p0, s0, 0;
            setp.ne.u32 p1, s1, 0;
            setp.ne.u32 p2, s2, 0;
            setp.ne.u32 p3, s3, 0;
            selp.s32 i0, n0, i0, p0;
            selp.s32 i1, n1, i1, p1;
            selp.s32 i2, n2, i2, p2;
            selp.s32 i3, n3, i3, p3;
            cvt.rn.f32.s32 $0, i0;
            cvt.rn.f32.s32 $1, i1;
            cvt.rn.f32.s32 $2, i2;
            cvt.rn.f32.s32 $3, i3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_int6x4_to_f32(
    packed: Uint32,
    int_expansion_factor: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(packed).ir_value(loc=loc, ip=ip),
            Float32(int_expansion_factor).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b32 b0, b1, b2, b3, m0, m1, m2, m3;
            .reg .pred p0, p1, p2, p3;
            .reg .f32 f0, f1, f2, f3, n0, n1, n2, n3;
            .reg .b16 h0, h1, h2, h3;
            and.b32 b0, $4, 0xFF;
            shr.b32 b1, $4, 8;
            and.b32 b1, b1, 0xFF;
            shr.b32 b2, $4, 16;
            and.b32 b2, b2, 0xFF;
            shr.b32 b3, $4, 24;
            and.b32 b3, b3, 0xFF;

            and.b32 m0, b0, 0x1F;
            and.b32 m1, b1, 0x1F;
            and.b32 m2, b2, 0x1F;
            and.b32 m3, b3, 0x1F;
            cvt.rn.f32.u32 f0, m0;
            cvt.rn.f32.u32 f1, m1;
            cvt.rn.f32.u32 f2, m2;
            cvt.rn.f32.u32 f3, m3;
            setp.ne.u32 p0, b0, m0;
            setp.ne.u32 p1, b1, m1;
            setp.ne.u32 p2, b2, m2;
            setp.ne.u32 p3, b3, m3;
            neg.f32 n0, f0;
            neg.f32 n1, f1;
            neg.f32 n2, f2;
            neg.f32 n3, f3;
            selp.f32 f0, n0, f0, p0;
            selp.f32 f1, n1, f1, p1;
            selp.f32 f2, n2, f2, p2;
            selp.f32 f3, n3, f3, p3;
            mul.rn.f32 f0, f0, $5;
            mul.rn.f32 f1, f1, $5;
            mul.rn.f32 f2, f2, $5;
            mul.rn.f32 f3, f3, $5;
            cvt.rn.f16.f32 h0, f0;
            cvt.rn.f16.f32 h1, f1;
            cvt.rn.f16.f32 h2, f2;
            cvt.rn.f16.f32 h3, f3;
            cvt.f32.f16 $0, h0;
            cvt.f32.f16 $1, h1;
            cvt.f32.f16 $2, h2;
            cvt.f32.f16 $3, h3;
        }
        """,
        "=f,=f,=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_e2m0x4_f32(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
                Float32(f2).ir_value(loc=loc, ip=ip),
                Float32(f3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f32 a0, a1, a2, a3, n0, n1, n2, n3;
                .reg .u32 c0, c1, c2, c3, s0, s1, s2, s3;
                .reg .b8 b0, b1, b2, b3;
                .reg .pred p0, p1, p2, p3;

                abs.f32 a0, $1;
                abs.f32 a1, $2;
                abs.f32 a2, $3;
                abs.f32 a3, $4;

                mov.u32 c0, 3;
                setp.le.f32 p0, a0, 0f40400000;
                selp.u32 c0, 2, c0, p0;
                setp.le.f32 p0, a0, 0f3FC00000;
                selp.u32 c0, 1, c0, p0;
                setp.le.f32 p0, a0, 0f3F000000;
                selp.u32 c0, 0, c0, p0;

                mov.u32 c1, 3;
                setp.le.f32 p1, a1, 0f40400000;
                selp.u32 c1, 2, c1, p1;
                setp.le.f32 p1, a1, 0f3FC00000;
                selp.u32 c1, 1, c1, p1;
                setp.le.f32 p1, a1, 0f3F000000;
                selp.u32 c1, 0, c1, p1;

                mov.u32 c2, 3;
                setp.le.f32 p2, a2, 0f40400000;
                selp.u32 c2, 2, c2, p2;
                setp.le.f32 p2, a2, 0f3FC00000;
                selp.u32 c2, 1, c2, p2;
                setp.le.f32 p2, a2, 0f3F000000;
                selp.u32 c2, 0, c2, p2;

                mov.u32 c3, 3;
                setp.le.f32 p3, a3, 0f40400000;
                selp.u32 c3, 2, c3, p3;
                setp.le.f32 p3, a3, 0f3FC00000;
                selp.u32 c3, 1, c3, p3;
                setp.le.f32 p3, a3, 0f3F000000;
                selp.u32 c3, 0, c3, p3;

                setp.lt.f32 p0, $1, 0f00000000;
                setp.lt.f32 p1, $2, 0f00000000;
                setp.lt.f32 p2, $3, 0f00000000;
                setp.lt.f32 p3, $4, 0f00000000;
                selp.u32 s0, 4, 0, p0;
                selp.u32 s1, 4, 0, p1;
                selp.u32 s2, 4, 0, p2;
                selp.u32 s3, 4, 0, p3;
                or.b32 c0, c0, s0;
                or.b32 c1, c1, s1;
                or.b32 c2, c2, s2;
                or.b32 c3, c3, s3;
                cvt.u8.u32 b0, c0;
                cvt.u8.u32 b1, c1;
                cvt.u8.u32 b2, c2;
                cvt.u8.u32 b3, c3;
                mov.b32 $0, {b0, b1, b2, b3};
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e2m0x4_to_f32(
    packed: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 b0, b1, b2, b3;
            .reg .u32 u0, u1, u2, u3, c0, c1, c2, c3, s0, s1, s2, s3;
            .reg .f32 f0, f1, f2, f3, n0, n1, n2, n3;
            .reg .pred p0, p1, p2, p3;

            mov.b32 {b0, b1, b2, b3}, $4;
            cvt.u32.u8 u0, b0;
            cvt.u32.u8 u1, b1;
            cvt.u32.u8 u2, b2;
            cvt.u32.u8 u3, b3;
            and.b32 c0, u0, 0x3;
            and.b32 c1, u1, 0x3;
            and.b32 c2, u2, 0x3;
            and.b32 c3, u3, 0x3;
            cvt.rn.f32.u32 f0, c0;
            cvt.rn.f32.u32 f1, c1;
            cvt.rn.f32.u32 f2, c2;
            cvt.rn.f32.u32 f3, c3;
            setp.eq.u32 p0, c0, 3;
            selp.f32 f0, 0f40800000, f0, p0;
            setp.eq.u32 p1, c1, 3;
            selp.f32 f1, 0f40800000, f1, p1;
            setp.eq.u32 p2, c2, 3;
            selp.f32 f2, 0f40800000, f2, p2;
            setp.eq.u32 p3, c3, 3;
            selp.f32 f3, 0f40800000, f3, p3;
            and.b32 s0, u0, 0x4;
            and.b32 s1, u1, 0x4;
            and.b32 s2, u2, 0x4;
            and.b32 s3, u3, 0x4;
            neg.f32 n0, f0;
            neg.f32 n1, f1;
            neg.f32 n2, f2;
            neg.f32 n3, f3;
            setp.ne.u32 p0, s0, 0;
            setp.ne.u32 p1, s1, 0;
            setp.ne.u32 p2, s2, 0;
            setp.ne.u32 p3, s3, 0;
            selp.f32 $0, n0, f0, p0;
            selp.f32 $1, n1, f1, p1;
            selp.f32 $2, n2, f2, p2;
            selp.f32 $3, n3, f3, p3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_e2m0x4_f32_values(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
            Float32(f2).ir_value(loc=loc, ip=ip),
            Float32(f3).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .f32 a0, a1, a2, a3, q0, q1, q2, q3, n0, n1, n2, n3;
            .reg .pred p0, p1, p2, p3;

            abs.f32 a0, $4;
            abs.f32 a1, $5;
            abs.f32 a2, $6;
            abs.f32 a3, $7;

            mov.f32 q0, 0f40800000;
            setp.le.f32 p0, a0, 0f40400000;
            selp.f32 q0, 0f40000000, q0, p0;
            setp.le.f32 p0, a0, 0f3FC00000;
            selp.f32 q0, 0f3F800000, q0, p0;
            setp.le.f32 p0, a0, 0f3F000000;
            selp.f32 q0, 0f00000000, q0, p0;

            mov.f32 q1, 0f40800000;
            setp.le.f32 p1, a1, 0f40400000;
            selp.f32 q1, 0f40000000, q1, p1;
            setp.le.f32 p1, a1, 0f3FC00000;
            selp.f32 q1, 0f3F800000, q1, p1;
            setp.le.f32 p1, a1, 0f3F000000;
            selp.f32 q1, 0f00000000, q1, p1;

            mov.f32 q2, 0f40800000;
            setp.le.f32 p2, a2, 0f40400000;
            selp.f32 q2, 0f40000000, q2, p2;
            setp.le.f32 p2, a2, 0f3FC00000;
            selp.f32 q2, 0f3F800000, q2, p2;
            setp.le.f32 p2, a2, 0f3F000000;
            selp.f32 q2, 0f00000000, q2, p2;

            mov.f32 q3, 0f40800000;
            setp.le.f32 p3, a3, 0f40400000;
            selp.f32 q3, 0f40000000, q3, p3;
            setp.le.f32 p3, a3, 0f3FC00000;
            selp.f32 q3, 0f3F800000, q3, p3;
            setp.le.f32 p3, a3, 0f3F000000;
            selp.f32 q3, 0f00000000, q3, p3;

            neg.f32 n0, q0;
            neg.f32 n1, q1;
            neg.f32 n2, q2;
            neg.f32 n3, q3;
            setp.lt.f32 p0, $4, 0f00000000;
            setp.lt.f32 p1, $5, 0f00000000;
            setp.lt.f32 p2, $6, 0f00000000;
            setp.lt.f32 p3, $7, 0f00000000;
            selp.f32 $0, n0, q0, p0;
            selp.f32 $1, n1, q1, p1;
            selp.f32 $2, n2, q2, p2;
            selp.f32 $3, n3, q3, p3;
        }
        """,
        "=f,=f,=f,=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_e2m1x8_f32(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    f4: Float32,
    f5: Float32,
    f6: Float32,
    f7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
                Float32(f2).ir_value(loc=loc, ip=ip),
                Float32(f3).ir_value(loc=loc, ip=ip),
                Float32(f4).ir_value(loc=loc, ip=ip),
                Float32(f5).ir_value(loc=loc, ip=ip),
                Float32(f6).ir_value(loc=loc, ip=ip),
                Float32(f7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def stochastic_round_e2m1_value(
    x: Float32,
    seed: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x).ir_value(loc=loc, ip=ip),
                Uint32(seed).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_neg, p_lt2, p_lt4;
                .reg .u32 h, t;
                .reg .s32 i;
                .reg .f32 ax, noise, work, rounded, result;

                mul.lo.u32 h, $2, 0x9E3779B9;
                shr.u32 t, h, 16;
                xor.b32 h, h, t;
                mul.lo.u32 h, h, 0x85EBCA6B;
                shr.u32 t, h, 13;
                xor.b32 h, h, t;
                and.b32 h, h, 0x00FFFFFF;
                cvt.rn.f32.u32 noise, h;
                fma.rn.f32 noise, noise, 0f33800000, 0fBF000000;

                abs.f32 ax, $1;
                setp.lt.f32 p_lt2, ax, 0f40000000;
                setp.lt.f32 p_lt4, ax, 0f40800000;
                @p_lt2 bra LT2;
                @p_lt4 bra LT4;

                fma.rn.f32 work, ax, 0f3F000000, noise;
                cvt.rni.s32.f32 i, work;
                cvt.rn.f32.s32 rounded, i;
                add.rn.f32 rounded, rounded, rounded;
                min.f32 rounded, rounded, 0f40C00000;
                bra SIGN;

            LT4:
                add.rn.f32 work, ax, noise;
                cvt.rni.s32.f32 i, work;
                cvt.rn.f32.s32 rounded, i;
                bra SIGN;

            LT2:
                fma.rn.f32 work, ax, 0f40000000, noise;
                cvt.rni.s32.f32 i, work;
                cvt.rn.f32.s32 rounded, i;
                mul.rn.f32 rounded, rounded, 0f3F000000;

            SIGN:
                setp.lt.f32 p_neg, $1, 0f00000000;
                neg.f32 result, rounded;
                selp.f32 $0, result, rounded, p_neg;
            }
            """,
            "=f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def stochastic_round_int4_value(
    x: Float32,
    seed: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x).ir_value(loc=loc, ip=ip),
                Uint32(seed).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_neg;
                .reg .u32 h, t;
                .reg .s32 i;
                .reg .f32 ax, noise, work, rounded, result;

                mul.lo.u32 h, $2, 0x9E3779B9;
                shr.u32 t, h, 16;
                xor.b32 h, h, t;
                mul.lo.u32 h, h, 0x85EBCA6B;
                shr.u32 t, h, 13;
                xor.b32 h, h, t;
                and.b32 h, h, 0x00FFFFFF;
                cvt.rn.f32.u32 noise, h;
                fma.rn.f32 noise, noise, 0f32FAE149, 0fBEFAE148;

                abs.f32 ax, $1;
                add.rn.f32 work, ax, noise;
                max.f32 work, work, 0f00000000;
                min.f32 work, work, 0f40E00000;
                cvt.rni.s32.f32 i, work;
                cvt.rn.f32.s32 rounded, i;
                setp.lt.f32 p_neg, $1, 0f00000000;
                neg.f32 result, rounded;
                selp.f32 $0, result, rounded, p_neg;
            }
            """,
            "=f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_int3x4_f32(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
                Float32(f2).ir_value(loc=loc, ip=ip),
                Float32(f3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f32 c0, c1, c2, c3;
                .reg .s32 i0, i1, i2, i3;
                .reg .u32 s0, s1, s2, s3;
                .reg .u32 m0, m1, m2, m3;
                .reg .b8 b0, b1, b2, b3;

                max.f32 c0, $1, 0fc0400000;
                min.f32 c0, c0, 0f40400000;
                max.f32 c1, $2, 0fc0400000;
                min.f32 c1, c1, 0f40400000;
                max.f32 c2, $3, 0fc0400000;
                min.f32 c2, c2, 0f40400000;
                max.f32 c3, $4, 0fc0400000;
                min.f32 c3, c3, 0f40400000;

                cvt.rni.s32.f32 i0, c0;
                cvt.rni.s32.f32 i1, c1;
                cvt.rni.s32.f32 i2, c2;
                cvt.rni.s32.f32 i3, c3;
                shr.u32 s0, i0, 31;
                shr.u32 s1, i1, 31;
                shr.u32 s2, i2, 31;
                shr.u32 s3, i3, 31;
                abs.s32 m0, i0;
                abs.s32 m1, i1;
                abs.s32 m2, i2;
                abs.s32 m3, i3;
                and.b32 m0, m0, 0x3;
                and.b32 m1, m1, 0x3;
                and.b32 m2, m2, 0x3;
                and.b32 m3, m3, 0x3;
                shl.b32 s0, s0, 2;
                shl.b32 s1, s1, 2;
                shl.b32 s2, s2, 2;
                shl.b32 s3, s3, 2;
                or.b32 m0, m0, s0;
                or.b32 m1, m1, s1;
                or.b32 m2, m2, s2;
                or.b32 m3, m3, s3;
                cvt.u8.u32 b0, m0;
                cvt.u8.u32 b1, m1;
                cvt.u8.u32 b2, m2;
                cvt.u8.u32 b3, m3;
                mov.b32 $0, {b0, b1, b2, b3};
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_int3x4_f32_values(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
            Float32(f2).ir_value(loc=loc, ip=ip),
            Float32(f3).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .f32 c0, c1, c2, c3;
            .reg .s32 i0, i1, i2, i3;

            max.f32 c0, $4, 0fc0400000;
            min.f32 c0, c0, 0f40400000;
            max.f32 c1, $5, 0fc0400000;
            min.f32 c1, c1, 0f40400000;
            max.f32 c2, $6, 0fc0400000;
            min.f32 c2, c2, 0f40400000;
            max.f32 c3, $7, 0fc0400000;
            min.f32 c3, c3, 0f40400000;

            cvt.rni.s32.f32 i0, c0;
            cvt.rni.s32.f32 i1, c1;
            cvt.rni.s32.f32 i2, c2;
            cvt.rni.s32.f32 i3, c3;
            cvt.rn.f32.s32 $0, i0;
            cvt.rn.f32.s32 $1, i1;
            cvt.rn.f32.s32 $2, i2;
            cvt.rn.f32.s32 $3, i3;
        }
        """,
        "=f,=f,=f,=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_int4x8_f32(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    f4: Float32,
    f5: Float32,
    f6: Float32,
    f7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
                Float32(f2).ir_value(loc=loc, ip=ip),
                Float32(f3).ir_value(loc=loc, ip=ip),
                Float32(f4).ir_value(loc=loc, ip=ip),
                Float32(f5).ir_value(loc=loc, ip=ip),
                Float32(f6).ir_value(loc=loc, ip=ip),
                Float32(f7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f32 c0, c1, c2, c3, c4, c5, c6, c7;
                .reg .s32 i0, i1, i2, i3, i4, i5, i6, i7;
                .reg .u32 u0, u1, u2, u3, u4, u5, u6, u7;
                .reg .b8 byte0, byte1, byte2, byte3;

                max.f32 c0, $1, 0fc0e00000;
                min.f32 c0, c0, 0f40e00000;
                max.f32 c1, $2, 0fc0e00000;
                min.f32 c1, c1, 0f40e00000;
                max.f32 c2, $3, 0fc0e00000;
                min.f32 c2, c2, 0f40e00000;
                max.f32 c3, $4, 0fc0e00000;
                min.f32 c3, c3, 0f40e00000;
                max.f32 c4, $5, 0fc0e00000;
                min.f32 c4, c4, 0f40e00000;
                max.f32 c5, $6, 0fc0e00000;
                min.f32 c5, c5, 0f40e00000;
                max.f32 c6, $7, 0fc0e00000;
                min.f32 c6, c6, 0f40e00000;
                max.f32 c7, $8, 0fc0e00000;
                min.f32 c7, c7, 0f40e00000;

                cvt.rni.s32.f32 i0, c0;
                cvt.rni.s32.f32 i1, c1;
                cvt.rni.s32.f32 i2, c2;
                cvt.rni.s32.f32 i3, c3;
                cvt.rni.s32.f32 i4, c4;
                cvt.rni.s32.f32 i5, c5;
                cvt.rni.s32.f32 i6, c6;
                cvt.rni.s32.f32 i7, c7;

                and.b32 u0, i0, 0xF;
                and.b32 u1, i1, 0xF;
                and.b32 u2, i2, 0xF;
                and.b32 u3, i3, 0xF;
                and.b32 u4, i4, 0xF;
                and.b32 u5, i5, 0xF;
                and.b32 u6, i6, 0xF;
                and.b32 u7, i7, 0xF;

                shl.b32 u1, u1, 4;
                shl.b32 u3, u3, 4;
                shl.b32 u5, u5, 4;
                shl.b32 u7, u7, 4;
                or.b32 u0, u0, u1;
                or.b32 u2, u2, u3;
                or.b32 u4, u4, u5;
                or.b32 u6, u6, u7;
                cvt.u8.u32 byte0, u0;
                cvt.u8.u32 byte1, u2;
                cvt.u8.u32 byte2, u4;
                cvt.u8.u32 byte3, u6;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_int4x4_to_f32(
    packed: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u32 u0, u1, u2, u3;
            .reg .s32 i0, i1, i2, i3;
            .reg .pred p0, p1, p2, p3;

            and.b32 u0, $4, 0xF;
            shr.u32 u1, $4, 4;
            and.b32 u1, u1, 0xF;
            shr.u32 u2, $4, 8;
            and.b32 u2, u2, 0xF;
            shr.u32 u3, $4, 12;
            and.b32 u3, u3, 0xF;

            cvt.s32.u32 i0, u0;
            cvt.s32.u32 i1, u1;
            cvt.s32.u32 i2, u2;
            cvt.s32.u32 i3, u3;

            setp.ge.s32 p0, i0, 8;
            setp.ge.s32 p1, i1, 8;
            setp.ge.s32 p2, i2, 8;
            setp.ge.s32 p3, i3, 8;
            @p0 add.s32 i0, i0, -16;
            @p1 add.s32 i1, i1, -16;
            @p2 add.s32 i2, i2, -16;
            @p3 add.s32 i3, i3, -16;

            cvt.rn.f32.s32 $0, i0;
            cvt.rn.f32.s32 $1, i1;
            cvt.rn.f32.s32 $2, i2;
            cvt.rn.f32.s32 $3, i3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_int6x8_f32(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    f4: Float32,
    f5: Float32,
    f6: Float32,
    f7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint64:
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
                Float32(f2).ir_value(loc=loc, ip=ip),
                Float32(f3).ir_value(loc=loc, ip=ip),
                Float32(f4).ir_value(loc=loc, ip=ip),
                Float32(f5).ir_value(loc=loc, ip=ip),
                Float32(f6).ir_value(loc=loc, ip=ip),
                Float32(f7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f32 c0, c1, c2, c3, c4, c5, c6, c7;
                .reg .s32 i0, i1, i2, i3, i4, i5, i6, i7;
                .reg .u32 u0, u1, u2, u3, u4, u5, u6, u7;
                .reg .u32 s0, s1, s2, s3, s4, s5, s6, s7;
                .reg .u32 m0, m1, m2, m3, m4, m5, m6, m7;
                .reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;
                .reg .b32 lo, hi;

                max.f32 c0, $1, 0fc1f80000;
                min.f32 c0, c0, 0f41f80000;
                max.f32 c1, $2, 0fc1f80000;
                min.f32 c1, c1, 0f41f80000;
                max.f32 c2, $3, 0fc1f80000;
                min.f32 c2, c2, 0f41f80000;
                max.f32 c3, $4, 0fc1f80000;
                min.f32 c3, c3, 0f41f80000;
                max.f32 c4, $5, 0fc1f80000;
                min.f32 c4, c4, 0f41f80000;
                max.f32 c5, $6, 0fc1f80000;
                min.f32 c5, c5, 0f41f80000;
                max.f32 c6, $7, 0fc1f80000;
                min.f32 c6, c6, 0f41f80000;
                max.f32 c7, $8, 0fc1f80000;
                min.f32 c7, c7, 0f41f80000;

                cvt.rni.s32.f32 i0, c0;
                cvt.rni.s32.f32 i1, c1;
                cvt.rni.s32.f32 i2, c2;
                cvt.rni.s32.f32 i3, c3;
                cvt.rni.s32.f32 i4, c4;
                cvt.rni.s32.f32 i5, c5;
                cvt.rni.s32.f32 i6, c6;
                cvt.rni.s32.f32 i7, c7;

                shr.u32 s0, i0, 31;
                shr.u32 s1, i1, 31;
                shr.u32 s2, i2, 31;
                shr.u32 s3, i3, 31;
                shr.u32 s4, i4, 31;
                shr.u32 s5, i5, 31;
                shr.u32 s6, i6, 31;
                shr.u32 s7, i7, 31;
                abs.s32 m0, i0;
                abs.s32 m1, i1;
                abs.s32 m2, i2;
                abs.s32 m3, i3;
                abs.s32 m4, i4;
                abs.s32 m5, i5;
                abs.s32 m6, i6;
                abs.s32 m7, i7;
                and.b32 m0, m0, 0x1F;
                and.b32 m1, m1, 0x1F;
                and.b32 m2, m2, 0x1F;
                and.b32 m3, m3, 0x1F;
                and.b32 m4, m4, 0x1F;
                and.b32 m5, m5, 0x1F;
                and.b32 m6, m6, 0x1F;
                and.b32 m7, m7, 0x1F;
                shl.b32 s0, s0, 5;
                shl.b32 s1, s1, 5;
                shl.b32 s2, s2, 5;
                shl.b32 s3, s3, 5;
                shl.b32 s4, s4, 5;
                shl.b32 s5, s5, 5;
                shl.b32 s6, s6, 5;
                shl.b32 s7, s7, 5;
                or.b32 u0, m0, s0;
                or.b32 u1, m1, s1;
                or.b32 u2, m2, s2;
                or.b32 u3, m3, s3;
                or.b32 u4, m4, s4;
                or.b32 u5, m5, s5;
                or.b32 u6, m6, s6;
                or.b32 u7, m7, s7;
                cvt.u8.u32 b0, u0;
                cvt.u8.u32 b1, u1;
                cvt.u8.u32 b2, u2;
                cvt.u8.u32 b3, u3;
                cvt.u8.u32 b4, u4;
                cvt.u8.u32 b5, u5;
                cvt.u8.u32 b6, u6;
                cvt.u8.u32 b7, u7;
                mov.b32 lo, {b0, b1, b2, b3};
                mov.b32 hi, {b4, b5, b6, b7};
                mov.b64 $0, {lo, hi};
            }
            """,
            "=l,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e2m3x4_f32(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
                Float32(f2).ir_value(loc=loc, ip=ip),
                Float32(f3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 tmp0, tmp1;
                cvt.rn.satfinite.e2m3x2.f32 tmp0, $2, $1;
                cvt.rn.satfinite.e2m3x2.f32 tmp1, $4, $3;
                mov.b32 $0, {tmp0, tmp1};
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e3m2x4_f32(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
                Float32(f2).ir_value(loc=loc, ip=ip),
                Float32(f3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 tmp0, tmp1;
                cvt.rn.satfinite.e3m2x2.f32 tmp0, $2, $1;
                cvt.rn.satfinite.e3m2x2.f32 tmp1, $4, $3;
                mov.b32 $0, {tmp0, tmp1};
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def bfloat2_max_abs_8(
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    v4: Uint32,
    v5: Uint32,
    v6: Uint32,
    v7: Uint32,
) -> Uint32:
    max01 = bfloat2_hmax2(bfloat2_habs2(v0), bfloat2_habs2(v1))
    max23 = bfloat2_hmax2(bfloat2_habs2(v2), bfloat2_habs2(v3))
    max45 = bfloat2_hmax2(bfloat2_habs2(v4), bfloat2_habs2(v5))
    max67 = bfloat2_hmax2(bfloat2_habs2(v6), bfloat2_habs2(v7))
    return bfloat2_hmax2(bfloat2_hmax2(max01, max23), bfloat2_hmax2(max45, max67))


@cute.jit
def bfloat2x4_to_e2m1x8_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    inv_scale: Float32,
) -> Uint32:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    return cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)


@cute.jit
def bfloat2x4_to_e2m0x8_values(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    inv_scale: Float32,
) -> Tuple[Uint32, Uint32]:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    return (
        cvt_e2m0x4_f32(s0, s1, s2, s3),
        cvt_e2m0x4_f32(s4, s5, s6, s7),
    )


@cute.jit
def bfloat2x8_to_e2m0x16_values(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)
    return (
        cvt_e2m0x4_f32(s0, s1, s2, s3),
        cvt_e2m0x4_f32(s4, s5, s6, s7),
        cvt_e2m0x4_f32(s8, s9, s10, s11),
        cvt_e2m0x4_f32(s12, s13, s14, s15),
    )


@cute.jit
def bfloat2x8_to_e2m1x16_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> Uint64:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    packed_lo = cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_e2m1x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def bfloat2x8_to_e2m1x16_packed_stochastic(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
    seed_base: Uint32,
) -> Uint64:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    s0 = stochastic_round_e2m1_value(s0, seed_base + Uint32(0))
    s1 = stochastic_round_e2m1_value(s1, seed_base + Uint32(1))
    s2 = stochastic_round_e2m1_value(s2, seed_base + Uint32(2))
    s3 = stochastic_round_e2m1_value(s3, seed_base + Uint32(3))
    s4 = stochastic_round_e2m1_value(s4, seed_base + Uint32(4))
    s5 = stochastic_round_e2m1_value(s5, seed_base + Uint32(5))
    s6 = stochastic_round_e2m1_value(s6, seed_base + Uint32(6))
    s7 = stochastic_round_e2m1_value(s7, seed_base + Uint32(7))
    s8 = stochastic_round_e2m1_value(s8, seed_base + Uint32(8))
    s9 = stochastic_round_e2m1_value(s9, seed_base + Uint32(9))
    s10 = stochastic_round_e2m1_value(s10, seed_base + Uint32(10))
    s11 = stochastic_round_e2m1_value(s11, seed_base + Uint32(11))
    s12 = stochastic_round_e2m1_value(s12, seed_base + Uint32(12))
    s13 = stochastic_round_e2m1_value(s13, seed_base + Uint32(13))
    s14 = stochastic_round_e2m1_value(s14, seed_base + Uint32(14))
    s15 = stochastic_round_e2m1_value(s15, seed_base + Uint32(15))

    packed_lo = cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_e2m1x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def bfloat2x4_to_int3x8_values(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    inv_scale: Float32,
) -> Tuple[Uint32, Uint32]:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)

    return (
        cvt_int3x4_f32(s0, s1, s2, s3),
        cvt_int3x4_f32(s4, s5, s6, s7),
    )


@cute.jit
def bfloat2x8_to_int3x16_values(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    return (
        cvt_int3x4_f32(s0, s1, s2, s3),
        cvt_int3x4_f32(s4, s5, s6, s7),
        cvt_int3x4_f32(s8, s9, s10, s11),
        cvt_int3x4_f32(s12, s13, s14, s15),
    )


@cute.jit
def bfloat2x4_to_int4x8_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    inv_scale: Float32,
) -> Uint32:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)

    return cvt_int4x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)


@cute.jit
def bfloat2x8_to_int4x16_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> Uint64:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    packed_lo = cvt_int4x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_int4x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def bfloat2x8_to_int4x16_packed_stochastic(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
    seed_base: Uint32,
) -> Uint64:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    s0 = stochastic_round_int4_value(s0, seed_base + Uint32(0))
    s1 = stochastic_round_int4_value(s1, seed_base + Uint32(1))
    s2 = stochastic_round_int4_value(s2, seed_base + Uint32(2))
    s3 = stochastic_round_int4_value(s3, seed_base + Uint32(3))
    s4 = stochastic_round_int4_value(s4, seed_base + Uint32(4))
    s5 = stochastic_round_int4_value(s5, seed_base + Uint32(5))
    s6 = stochastic_round_int4_value(s6, seed_base + Uint32(6))
    s7 = stochastic_round_int4_value(s7, seed_base + Uint32(7))
    s8 = stochastic_round_int4_value(s8, seed_base + Uint32(8))
    s9 = stochastic_round_int4_value(s9, seed_base + Uint32(9))
    s10 = stochastic_round_int4_value(s10, seed_base + Uint32(10))
    s11 = stochastic_round_int4_value(s11, seed_base + Uint32(11))
    s12 = stochastic_round_int4_value(s12, seed_base + Uint32(12))
    s13 = stochastic_round_int4_value(s13, seed_base + Uint32(13))
    s14 = stochastic_round_int4_value(s14, seed_base + Uint32(14))
    s15 = stochastic_round_int4_value(s15, seed_base + Uint32(15))

    packed_lo = cvt_int4x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_int4x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def bfloat2x8_to_int6x16_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> tuple[Uint64, Uint64]:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    packed_lo = cvt_int6x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_int6x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)
    return packed_lo, packed_hi


@cute.jit
def bfloat2x8_to_e2m3x16_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> tuple[Uint64, Uint64]:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    p0 = cvt_e2m3x4_f32(s0, s1, s2, s3)
    p1 = cvt_e2m3x4_f32(s4, s5, s6, s7)
    p2 = cvt_e2m3x4_f32(s8, s9, s10, s11)
    p3 = cvt_e2m3x4_f32(s12, s13, s14, s15)
    return (Uint64(p1) << Uint64(32)) | Uint64(p0), (
        Uint64(p3) << Uint64(32)
    ) | Uint64(p2)


@cute.jit
def bfloat2x8_to_e3m2x16_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> tuple[Uint64, Uint64]:
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    p0 = cvt_e3m2x4_f32(s0, s1, s2, s3)
    p1 = cvt_e3m2x4_f32(s4, s5, s6, s7)
    p2 = cvt_e3m2x4_f32(s8, s9, s10, s11)
    p3 = cvt_e3m2x4_f32(s12, s13, s14, s15)
    return (Uint64(p1) << Uint64(32)) | Uint64(p0), (
        Uint64(p3) << Uint64(32)
    ) | Uint64(p2)
