import functools

import torch
from fouroversix.matmul.backend import MatmulBackendBase
from fouroversix.quantize import QuantizedTensor
from fouroversix.utils import BLACKWELL_SM_IDS, SM_100, SM_120, DataType


def _to_blackwell_blocked(a: torch.Tensor) -> torch.Tensor:
    rows, cols = a.shape
    if rows % 128 != 0 or cols % 4 != 0:
        msg = (
            "Blackwell scale layout requires rows divisible by 128 and columns "
            f"divisible by 4, got {tuple(a.shape)}"
        )
        raise ValueError(msg)

    return (
        a.reshape(-1, 128, cols // 4, 4)
        .transpose(1, 2)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .contiguous()
        .reshape(-1)
    )


def _ensure_padded_values_clean(tensor: QuantizedTensor) -> None:
    padded_rows = tensor.padded_shape[0]
    if tensor.values.shape[0] != padded_rows:
        padded_values = torch.zeros(
            (padded_rows, tensor.values.shape[1]),
            device=tensor.values.device,
            dtype=tensor.values.dtype,
        )
        padded_values[: tensor.values.shape[0]].copy_(tensor.values)
        tensor.values = padded_values

    if tensor.scale_factors.ndim > 1 and tensor.scale_factors.shape[0] != padded_rows:
        padded_scales = torch.zeros(
            (padded_rows, tensor.scale_factors.shape[1]),
            device=tensor.scale_factors.device,
            dtype=tensor.scale_factors.dtype,
        )
        padded_scales[: tensor.scale_factors.shape[0]].copy_(tensor.scale_factors)
        tensor.scale_factors = padded_scales

    padded_rows = tensor.values.shape[0]
    original_rows = tensor.original_shape[0]

    if (
        not getattr(tensor, "_fouroversix_padded_values_clean", True)
        and padded_rows != original_rows
    ):
        tensor.values[original_rows:padded_rows].zero_()
    tensor._fouroversix_padded_values_clean = True

    if (
        not getattr(tensor, "_fouroversix_padded_scales_clean", True)
        and tensor.scale_factors.ndim > 1
        and tensor.scale_factors.shape[0] != original_rows
    ):
        tensor.scale_factors[original_rows : tensor.scale_factors.shape[0]].zero_()
    tensor._fouroversix_padded_scales_clean = True

    if not tensor.scale_factors_are_in_blackwell_layout:
        if tensor.scale_factors.ndim != 2:
            msg = "row-major scale factors must be a 2D tensor before conversion"
            raise ValueError(msg)
        tensor.scale_factors = _to_blackwell_blocked(tensor.scale_factors)
        tensor.scale_factors_are_in_blackwell_layout = True


class CUTLASSMatmulBackend(MatmulBackendBase):
    """
    The CUTLASS matrix multiplication backend. Uses CUTLASS kernels to perform fast
    FP4 matrix multiplication. Requires a Blackwell GPU.
    """

    @classmethod
    @functools.lru_cache
    def is_available(cls) -> bool:
        """Return True if the CUTLASS backend is available on the current machine."""

        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] not in BLACKWELL_SM_IDS
        ):
            return False

        try:
            import fouroversix._C  # noqa: F401
        except ModuleNotFoundError:
            return False

        return True

    @classmethod
    def is_supported(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> bool:
        """
        Return True if the CUTLASS backend supports the given inputs and output data
        type.
        """

        if not super().is_supported(input, other, out_dtype=out_dtype):
            return False

        return (
            input.dtype == other.dtype
            and input.dtype in {DataType.mxfp4, DataType.nvfp4}
            and input.device.type == "cuda"
        )

    @classmethod
    def quantized_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> torch.Tensor:
        """
        Perform a matrix multiplication (`a @ b.T`) between two quantized tensors using
        the CUTLASS backend.
        """

        from .ops import (
            gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt,
            gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120,
            gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt,
            gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120,
            gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt,
            gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120,
        )

        out_shape = (input.original_shape[0], other.original_shape[0])

        if input.dtype == DataType.mxfp4:
            alpha = torch.ones(
                1,
                device=input.values.device,
                dtype=torch.float32,
            )
        elif input.dtype == DataType.nvfp4:
            alpha = (
                (input.amax * other.amax)
                / (
                    input.dtype.quantized_value_type.get_maximum_value(input.scale_rule)
                    * input.dtype.scale_type.get_maximum_value(input.scale_rule)
                    * other.dtype.quantized_value_type.get_maximum_value(
                        other.scale_rule,
                    )
                    * other.dtype.scale_type.get_maximum_value(other.scale_rule)
                    * input.round_style.adjustment_factor
                    * other.round_style.adjustment_factor
                )
            ).to(torch.float32)

        gemm_fns = {
            (
                SM_100,
                DataType.mxfp4,
                DataType.bfloat16,
            ): gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt,
            (
                SM_120,
                DataType.mxfp4,
                DataType.bfloat16,
            ): gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120,
            (
                SM_100,
                DataType.nvfp4,
                DataType.bfloat16,
            ): gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt,
            (
                SM_120,
                DataType.nvfp4,
                DataType.bfloat16,
            ): gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120,
            (
                SM_100,
                DataType.nvfp4,
                DataType.float16,
            ): gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt,
            (
                SM_120,
                DataType.nvfp4,
                DataType.float16,
            ): gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120,
        }

        gemm_fn = gemm_fns.get(
            (torch.cuda.get_device_capability()[0], input.dtype, out_dtype),
        )

        if gemm_fn is None:
            msg = (
                "No gemm function found for the given device capability and "
                f"out_dtype: {torch.cuda.get_device_capability()[0]}, {out_dtype}"
            )
            raise ValueError(msg)

        _ensure_padded_values_clean(input)
        _ensure_padded_values_clean(other)

        out = gemm_fn(
            input.values,
            other.values,
            input.scale_factors,
            other.scale_factors,
            alpha,
        )

        if out_shape is not None and out.shape != out_shape:
            out = out[: out_shape[0], : out_shape[1]]

        return out
