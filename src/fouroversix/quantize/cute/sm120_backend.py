import functools
import weakref
from dataclasses import replace

import torch
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.utils import DataType, RoundStyle, ScaleRule, SM_120
from fouroversix.kernels.cute_sm120 import ops as sm120_ops


_FAST_STATIC_NV_DTYPES = frozenset(
    {
        DataType.nvfp4,
        DataType.nvfp4_bs8,
        DataType.nvfp3,
        DataType.nvfp3_bs8,
        DataType.nvfp6_e2m3,
        DataType.nvfp6_e3m2,
    },
)
_FAST_STATIC_NVFP4_DTYPES = frozenset({DataType.nvfp4, DataType.nvfp4_bs8})
_FAST_STATIC_NVFP3_DTYPES = frozenset({DataType.nvfp3, DataType.nvfp3_bs8})
_STATIC_SCALE_RULES = frozenset({ScaleRule.static_4, ScaleRule.static_6})
_ADAPTIVE_IF_DTYPES = frozenset(
    {
        DataType.if3,
        DataType.if3_bs8,
        DataType.if4,
        DataType.if4_bs8,
        DataType.if6_e2m3,
        DataType.if6_e3m2,
    },
)
_ADAPTIVE_SCALE_RULES = frozenset(
    {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse},
)
_STATIC_MX_DTYPES = frozenset(
    {
        DataType.mxfp3,
        DataType.mxfp3_bs8,
        DataType.mxfp4,
        DataType.mxfp4_bs8,
        DataType.mxfp6_e2m3,
        DataType.mxfp6_e3m2,
    },
)
_STATIC_NVINT_DTYPES = frozenset(
    {
        DataType.nvint3,
        DataType.nvint3_bs8,
        DataType.nvint4,
        DataType.nvint4_bs8,
        DataType.nvint6,
    },
)
_PSEUDO_IF3_DTYPES = frozenset({DataType.if3, DataType.if3_bs8})
_PSEUDO_IF4_DTYPES = frozenset({DataType.if4, DataType.if4_bs8})
_PSEUDO_MXFP3_DTYPES = frozenset({DataType.mxfp3, DataType.mxfp3_bs8})
_PSEUDO_MXFP4_DTYPES = frozenset({DataType.mxfp4, DataType.mxfp4_bs8})
_PSEUDO_NVFP3_DTYPES = frozenset({DataType.nvfp3, DataType.nvfp3_bs8})
_PSEUDO_NVINT3_DTYPES = frozenset({DataType.nvint3, DataType.nvint3_bs8})
_PSEUDO_NVINT4_DTYPES = frozenset({DataType.nvint4, DataType.nvint4_bs8})
_PSEUDO_SUPPORTED_DTYPES = frozenset(
    {
        DataType.if3,
        DataType.if3_bs8,
        DataType.if4,
        DataType.if4_bs8,
        DataType.if6_e2m3,
        DataType.if6_e3m2,
        DataType.nvfp4,
        DataType.nvfp4_bs8,
        DataType.nvfp3,
        DataType.nvfp3_bs8,
        DataType.mxfp3,
        DataType.mxfp3_bs8,
        DataType.mxfp4,
        DataType.mxfp4_bs8,
        DataType.mxfp6_e2m3,
        DataType.mxfp6_e3m2,
        DataType.nvint3,
        DataType.nvint3_bs8,
        DataType.nvint4,
        DataType.nvint4_bs8,
        DataType.nvint6,
        DataType.nvfp6_e2m3,
        DataType.nvfp6_e3m2,
    },
)
_AMAX_CACHE_MAX_SIZE = 256
_AMAX_CACHE: dict[tuple[int, int, int, tuple[int, ...], tuple[int, ...]], tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]] = {}


def _tensor_version(x: torch.Tensor) -> int:
    try:
        return x._version
    except RuntimeError:
        return -1


def _cached_amax(x: torch.Tensor) -> torch.Tensor:
    key = (
        id(x),
        x.data_ptr(),
        _tensor_version(x),
        tuple(x.shape),
        tuple(x.stride()),
    )
    cached = _AMAX_CACHE.get(key)
    if cached is not None:
        ref, amax = cached
        if ref() is x:
            return amax

    amax = torch.linalg.vector_norm(x, ord=float("inf"), dtype=torch.float32)
    if len(_AMAX_CACHE) >= _AMAX_CACHE_MAX_SIZE:
        _AMAX_CACHE.pop(next(iter(_AMAX_CACHE)))
    _AMAX_CACHE[key] = (weakref.ref(x), amax)
    return amax


def _if3_effective_scale_rule_id(config: QuantizationConfig) -> int:
    if (
        config.dtype == DataType.if3_bs8
        and config.scale_rule == ScaleRule.mae
        and config.round_style == RoundStyle.stochastic_unbiased
        and not config.block_scale_2d
    ):
        return ScaleRule.mse.cuda_id
    return config.scale_rule.cuda_id


def _make_quantized_tensor(
    values: torch.Tensor,
    scale_factors: torch.Tensor,
    amax: torch.Tensor,
    dtype: DataType,
    original_shape: tuple[int, int] | torch.Size,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
    *,
    padded_shape: tuple[int, int] | None = None,
    scale_factors_are_in_blackwell_layout: bool = True,
) -> QuantizedTensor:
    original_shape = tuple(original_shape)
    cols_div = 4 * dtype.block_size
    if padded_shape is None:
        padded_shape = (
            original_shape
            if original_shape[0] % 128 == 0 and original_shape[1] % cols_div == 0
            else None
        )
    if padded_shape is not None:
        tensor = QuantizedTensor.__new__(QuantizedTensor)
        tensor.values = values
        tensor.scale_factors = scale_factors
        tensor.amax = amax
        tensor.dtype = dtype
        tensor.original_shape = original_shape
        tensor.scale_rule = scale_rule
        tensor.round_style = round_style
        tensor.padded_shape = padded_shape
        tensor.scale_factors_are_in_blackwell_layout = (
            scale_factors_are_in_blackwell_layout
        )
        return tensor

    return QuantizedTensor(
        values,
        scale_factors,
        amax,
        dtype,
        original_shape,
        scale_rule,
        round_style,
        padded_shape=padded_shape,
        scale_factors_are_in_blackwell_layout=scale_factors_are_in_blackwell_layout,
    )


def _make_fast_quantized_tensor(
    values: torch.Tensor,
    scale_factors: torch.Tensor,
    amax: torch.Tensor,
    dtype: DataType,
    original_shape: tuple[int, int] | torch.Size,
    padded_shape: tuple[int, int],
    scale_rule: ScaleRule,
    round_style: RoundStyle,
    *,
    scale_factors_are_in_blackwell_layout: bool,
) -> QuantizedTensor:
    tensor = QuantizedTensor.__new__(QuantizedTensor)
    tensor.values = values
    tensor.scale_factors = scale_factors
    tensor.amax = amax
    tensor.dtype = dtype
    tensor.original_shape = tuple(original_shape)
    tensor.scale_rule = scale_rule
    tensor.round_style = round_style
    tensor.padded_shape = padded_shape
    tensor.scale_factors_are_in_blackwell_layout = (
        scale_factors_are_in_blackwell_layout
    )
    tensor._fouroversix_padded_values_clean = True
    tensor._fouroversix_padded_scales_clean = True
    return tensor


def quantize_nvfp4_adaptive_default(
    x: torch.Tensor,
    config: QuantizationConfig,
) -> QuantizedTensor:
    x_amax = config.kwargs.get("x_amax")
    if x_amax is None and x.shape[0] >= 128 and x.is_contiguous():
        m, k = x.shape
        padded_m = m + (128 - m % 128) % 128
        amax = _cached_amax(x)
        values = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
        scale_factors = torch.empty(
            (m, k // config.dtype.block_size),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )

        tensor = QuantizedTensor.__new__(QuantizedTensor)
        tensor.values = values
        tensor.scale_factors = scale_factors
        tensor.amax = amax
        tensor.dtype = config.dtype
        tensor.original_shape = tuple(x.shape)
        tensor.scale_rule = config.scale_rule
        tensor.round_style = config.round_style
        tensor.padded_shape = (values.shape[0], x.shape[1])
        tensor.scale_factors_are_in_blackwell_layout = False
        tensor.padded_shape = (padded_m, x.shape[1])
        tensor._fouroversix_padded_values_clean = True
        tensor._fouroversix_padded_scales_clean = True

        sm120_ops.quantize_nvfp4_adaptive_into_unchecked(
            x,
            values,
            scale_factors,
            amax,
            scale_rule_id=config.scale_rule.cuda_id,
            stochastic_rounding=config.round_style == RoundStyle.stochastic,
            scale_block_size=config.dtype.block_size,
            clear_padded_scales=False,
            clear_padded_values=False,
        )
        return tensor

    values, scale_factors, amax = sm120_ops.quantize_nvfp4_adaptive(
        x,
        scale_rule_id=config.scale_rule.cuda_id,
        stochastic_rounding=config.round_style == RoundStyle.stochastic,
        scale_block_size=config.dtype.block_size,
        x_amax=x_amax,
        pad_rows_to=128,
        scale_factors_dtype=torch.float8_e4m3fn,
        clear_padded_scales=False,
        clear_padded_values=False,
    )
    tensor = QuantizedTensor.__new__(QuantizedTensor)
    tensor.values = values
    tensor.scale_factors = scale_factors
    tensor.amax = amax
    tensor.dtype = config.dtype
    tensor.original_shape = tuple(x.shape)
    tensor.scale_rule = config.scale_rule
    tensor.round_style = config.round_style
    tensor.padded_shape = (values.shape[0], x.shape[1])
    tensor.scale_factors_are_in_blackwell_layout = False
    tensor._fouroversix_padded_values_clean = False
    tensor._fouroversix_padded_scales_clean = False
    return tensor


@functools.lru_cache
def _static_nv_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_nvfp4_static,
        ops.quantize_nvfp3_static,
        ops.quantize_nvfp6_static,
    )


@functools.lru_cache
def _adaptive_nvfp4_quantizer():
    from fouroversix.kernels.cute_sm120 import ops

    return ops.quantize_nvfp4_adaptive


@functools.lru_cache
def _adaptive_nvfp4_2d_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_nvfp4_adaptive_2d,
        ops.quantize_nvfp4_static_2d,
        ops.quantize_nvfp4_bs8_static_2d,
    )


@functools.lru_cache
def _adaptive_nvfp4_transpose_quantizer():
    from fouroversix.kernels.cute_sm120 import ops

    return ops.quantize_nvfp4_adaptive_transpose


@functools.lru_cache
def _adaptive_if_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_if3_adaptive,
        ops.quantize_if4_adaptive,
        ops.quantize_if6_adaptive,
    )


@functools.lru_cache
def _adaptive_if_2d_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_if3_adaptive_2d,
        ops.quantize_if3_bs8_adaptive_2d,
        ops.quantize_if4_adaptive_2d,
        ops.quantize_if4_bs8_adaptive_2d,
        ops.quantize_if6_adaptive_2d,
    )


@functools.lru_cache
def _adaptive_if_transpose_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_if3_adaptive_transpose,
        ops.quantize_if4_adaptive_transpose,
    )


@functools.lru_cache
def _static_mx_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_mxfp3_static,
        ops.quantize_mxfp4_static,
        ops.quantize_mxfp6_static,
    )


@functools.lru_cache
def _static_mx_transpose_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_mxfp3_static_transpose,
        ops.quantize_mxfp4_static_transpose,
        ops.quantize_mxfp6_static_transpose,
    )


@functools.lru_cache
def _static_mx_2d_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_mxfp3_static_2d,
        ops.quantize_mxfp3_bs8_static_2d,
        ops.quantize_mxfp4_static_2d,
        ops.quantize_mxfp4_bs8_static_2d,
        ops.quantize_mxfp6_static_2d,
    )


@functools.lru_cache
def _static_nv_2d_quantizer():
    from fouroversix.kernels.cute_sm120 import ops

    return ops.quantize_nvfp6_static_2d


@functools.lru_cache
def _static_nv_transpose_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_nvfp4_static_transpose,
        ops.quantize_nvfp3_static_transpose,
        ops.quantize_nvfp6_static_transpose,
    )


@functools.lru_cache
def _static_nvint_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_nvint3_static,
        ops.quantize_nvint4_static,
        ops.quantize_nvint6_static,
    )


@functools.lru_cache
def _static_nvint_transpose_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_nvint3_static_transpose,
        ops.quantize_nvint4_static_transpose,
        ops.quantize_nvint6_static_transpose,
    )


@functools.lru_cache
def _static_nvint_2d_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_nvint3_static_2d,
        ops.quantize_nvint3_bs8_static_2d,
        ops.quantize_nvint4_static_2d,
        ops.quantize_nvint4_bs8_static_2d,
        ops.quantize_nvint6_static_2d,
    )


@functools.lru_cache
def _static_nvfp3_2d_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.quantize_nvfp3_static_2d,
        ops.quantize_nvfp3_bs8_static_2d,
    )


@functools.lru_cache
def _pseudo_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.pseudo_quantize_if3_adaptive,
        ops.pseudo_quantize_if4_adaptive,
        ops.pseudo_quantize_if6_adaptive,
        ops.pseudo_quantize_mxfp3_static,
        ops.pseudo_quantize_mxfp4_static,
        ops.pseudo_quantize_mxfp6_static,
        ops.pseudo_quantize_nvfp3_static,
        ops.pseudo_quantize_nvfp6_static,
        ops.pseudo_quantize_nvfp4_adaptive,
        ops.pseudo_quantize_nvfp4_static,
        ops.pseudo_quantize_nvint3_static,
        ops.pseudo_quantize_nvint4_static,
        ops.pseudo_quantize_nvint6_static,
        ops.rht_transform,
    )


@functools.lru_cache
def _adaptive_if_pseudo_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.pseudo_quantize_if3_adaptive,
        ops.pseudo_quantize_if4_adaptive,
        ops.rht_transform,
    )


@functools.lru_cache
def _static_nv_pseudo_quantizers():
    from fouroversix.kernels.cute_sm120 import ops

    return (
        ops.pseudo_quantize_nvfp3_static,
        ops.pseudo_quantize_nvfp4_static,
        ops.pseudo_quantize_nvint3_static,
        ops.pseudo_quantize_nvint4_static,
    )


class CuteSm120QuantizeBackend(QuantizeBackendBase):
    """CuTe-DSL quantization backend for RTX Blackwell (sm_120)."""

    @classmethod
    @functools.lru_cache
    def is_available(cls) -> bool:
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] != SM_120
        ):
            return False

        try:
            import cutlass.cute  # noqa: F401
        except ImportError:
            return False

        return True

    @classmethod
    def can_quantize(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        if not super().can_quantize(x, config):
            return False

        if (
            config.pseudo_quantize
            and config.dtype == DataType.nvfp4_bs8
            and config.scale_rule in _ADAPTIVE_SCALE_RULES
            and config.round_style
            in {RoundStyle.stochastic, RoundStyle.stochastic_unbiased}
            and not config.transpose
            and not config.rht
            and not config.block_scale_2d
        ):
            return (
                x.device.type == "cuda"
                and x.dtype == torch.bfloat16
                and x.shape[1] % config.dtype.block_size == 0
            )

        if config.pseudo_quantize and (
            config.dtype != DataType.nvfp4
            or config.round_style != RoundStyle.nearest
            or config.block_scale_2d
        ):
            return cls.can_quantize(x, replace(config, pseudo_quantize=False))

        return (
            x.device.type == "cuda"
            and x.dtype == torch.bfloat16
            and config.dtype
            in {
                DataType.nvfp4,
                DataType.nvfp4_bs8,
                DataType.if3,
                DataType.if3_bs8,
                DataType.if4,
                DataType.if4_bs8,
                DataType.if6_e2m3,
                DataType.if6_e3m2,
                DataType.mxfp3,
                DataType.mxfp3_bs8,
                DataType.mxfp4,
                DataType.mxfp4_bs8,
                DataType.mxfp6_e2m3,
                DataType.mxfp6_e3m2,
                DataType.nvint3,
                DataType.nvint3_bs8,
                DataType.nvint4,
                DataType.nvint4_bs8,
                DataType.nvint6,
                DataType.nvfp3,
                DataType.nvfp3_bs8,
                DataType.nvfp6_e2m3,
                DataType.nvfp6_e3m2,
            }
            and (
                config.round_style == RoundStyle.nearest
                or (
                    config.dtype == DataType.nvfp4
                    and config.scale_rule
                    in {
                        ScaleRule.abs_max,
                        ScaleRule.mae,
                        ScaleRule.mse,
                        ScaleRule.static_4,
                        ScaleRule.static_6,
                    }
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype in {DataType.nvfp3, DataType.nvfp3_bs8}
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                    and (
                        config.round_style == RoundStyle.stochastic
                        or config.round_style == RoundStyle.stochastic_unbiased
                    )
                )
                or (
                    config.dtype == DataType.nvfp4_bs8
                    and config.scale_rule
                    in {
                        ScaleRule.abs_max,
                        ScaleRule.mae,
                        ScaleRule.mse,
                        ScaleRule.static_4,
                        ScaleRule.static_6,
                    }
                    and not config.pseudo_quantize
                    and (
                        config.round_style == RoundStyle.nearest
                        or (
                            config.scale_rule
                            in {ScaleRule.static_4, ScaleRule.static_6}
                            and config.round_style.is_stochastic
                        )
                    )
                )
                or (
                    config.dtype == DataType.if4
                    and config.scale_rule
                    in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype == DataType.if4_bs8
                    and config.scale_rule
                    in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype in {DataType.if3, DataType.if3_bs8}
                    and config.scale_rule
                    in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
                    and not config.pseudo_quantize
                    and (
                        config.round_style == RoundStyle.stochastic
                        or (
                            config.round_style == RoundStyle.stochastic_unbiased
                            and (
                                config.dtype == DataType.if3
                                or config.scale_rule
                                in {ScaleRule.abs_max, ScaleRule.mse}
                                or config.scale_rule == ScaleRule.mae
                            )
                        )
                    )
                )
                or (
                    config.dtype == DataType.if4_bs8
                    and config.scale_rule
                    in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
                    and not config.block_scale_2d
                    and not config.pseudo_quantize
                    and config.round_style == RoundStyle.nearest
                )
                or (
                    config.dtype == DataType.mxfp3
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype == DataType.mxfp3_bs8
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype == DataType.mxfp4
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype == DataType.mxfp4_bs8
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype in {DataType.mxfp6_e2m3, DataType.mxfp6_e3m2}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype in {DataType.nvint4, DataType.nvint4_bs8}
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                    and config.round_style.is_stochastic
                )
                or (
                    config.dtype in {DataType.nvint3, DataType.nvint3_bs8}
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                    and (
                        config.round_style == RoundStyle.stochastic
                        or config.round_style == RoundStyle.stochastic_unbiased
                    )
                )
                or (
                    config.dtype == DataType.nvint6
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                    and (
                        config.round_style == RoundStyle.stochastic
                        or config.round_style == RoundStyle.stochastic_unbiased
                    )
                )
                or (
                    config.dtype in {DataType.nvfp6_e2m3, DataType.nvfp6_e3m2}
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                    and (
                        config.round_style == RoundStyle.stochastic
                        or (
                            config.round_style
                            == RoundStyle.stochastic_unbiased
                        )
                    )
                )
                or (
                    config.dtype in {DataType.if6_e2m3, DataType.if6_e3m2}
                    and config.scale_rule
                    in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
                    and not config.block_scale_2d
                    and not config.pseudo_quantize
                    and config.round_style
                    in {RoundStyle.stochastic, RoundStyle.stochastic_unbiased}
                )
            )
            and (
                (
                    config.dtype == DataType.nvfp4
                    and config.scale_rule
                    in {
                        ScaleRule.abs_max,
                        ScaleRule.mae,
                        ScaleRule.mse,
                        ScaleRule.static_4,
                        ScaleRule.static_6,
                    }
                )
                or (
                    config.dtype in {DataType.nvfp3, DataType.nvfp3_bs8}
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype == DataType.nvfp4_bs8
                    and config.scale_rule
                    in {
                        ScaleRule.abs_max,
                        ScaleRule.mae,
                        ScaleRule.mse,
                        ScaleRule.static_4,
                        ScaleRule.static_6,
                    }
                    and not config.pseudo_quantize
                    and (
                        config.round_style == RoundStyle.nearest
                        or config.scale_rule
                        in {ScaleRule.static_4, ScaleRule.static_6}
                    )
                )
                or (
                    config.dtype in {
                        DataType.if3,
                        DataType.if3_bs8,
                        DataType.if4,
                        DataType.if4_bs8,
                    }
                    and config.scale_rule
                    in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
                    and (
                        config.dtype
                        in {
                            DataType.if3,
                            DataType.if3_bs8,
                            DataType.if4,
                            DataType.if4_bs8,
                        }
                        or not config.block_scale_2d
                    )
                    and not config.pseudo_quantize
                    and (
                        config.dtype not in {DataType.if3, DataType.if3_bs8}
                        or config.round_style == RoundStyle.nearest
                        or (
                            config.round_style == RoundStyle.stochastic
                        )
                        or (
                            config.round_style == RoundStyle.stochastic_unbiased
                            and (
                                config.dtype == DataType.if3
                                or config.scale_rule
                                in {ScaleRule.abs_max, ScaleRule.mse}
                                or config.scale_rule == ScaleRule.mae
                            )
                        )
                    )
                )
                or (
                    config.dtype in {DataType.if6_e2m3, DataType.if6_e3m2}
                    and config.scale_rule
                    in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp4, DataType.mxfp4_bs8}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype == DataType.mxfp4_bs8
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp6_e2m3, DataType.mxfp6_e3m2}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype
                    in {
                        DataType.nvint3,
                        DataType.nvint3_bs8,
                        DataType.nvint4,
                        DataType.nvint4_bs8,
                        DataType.nvint6,
                    }
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                    and (
                        config.dtype
                        not in {DataType.nvint3, DataType.nvint3_bs8, DataType.nvint6}
                        or (
                            config.dtype in {DataType.nvint3, DataType.nvint3_bs8}
                            and config.round_style
                            in {
                                RoundStyle.nearest,
                                RoundStyle.stochastic,
                                RoundStyle.stochastic_unbiased,
                            }
                        )
                        or (
                            config.dtype == DataType.nvint6
                            and (
                                config.round_style
                                in {
                                    RoundStyle.nearest,
                                    RoundStyle.stochastic,
                                    RoundStyle.stochastic_unbiased,
                                }
                            )
                        )
                        or config.round_style == RoundStyle.nearest
                    )
                )
                or (
                    config.dtype in {DataType.nvfp6_e2m3, DataType.nvfp6_e3m2}
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                )
            )
            and (
                not config.block_scale_2d
                or (
                    config.dtype in {DataType.nvfp4, DataType.nvfp4_bs8}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.nvfp3, DataType.nvfp3_bs8}
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype
                    in {
                        DataType.if3,
                        DataType.if3_bs8,
                        DataType.if4,
                        DataType.if4_bs8,
                    }
                    and config.scale_rule
                    in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp4, DataType.mxfp4_bs8}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.mxfp6_e2m3, DataType.mxfp6_e3m2}
                    and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype
                    in {
                        DataType.nvint3,
                        DataType.nvint3_bs8,
                        DataType.nvint4,
                        DataType.nvint4_bs8,
                        DataType.nvint6,
                    }
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                )
                or (
                    config.dtype in {DataType.nvfp6_e2m3, DataType.nvfp6_e3m2}
                    and config.scale_rule == ScaleRule.static_6
                    and not config.pseudo_quantize
                )
            )
        )

    @classmethod
    def can_dequantize_values(cls, tensor: QuantizedTensor) -> bool:
        return tensor.dtype in {
            DataType.nvfp4,
            DataType.nvfp4_bs8,
            DataType.nvfp3,
            DataType.nvfp3_bs8,
            DataType.if3,
            DataType.if3_bs8,
            DataType.if4,
            DataType.if4_bs8,
            DataType.if6_e2m3,
            DataType.if6_e3m2,
            DataType.mxfp4,
            DataType.mxfp4_bs8,
            DataType.mxfp3,
            DataType.mxfp3_bs8,
            DataType.mxfp6_e2m3,
            DataType.mxfp6_e3m2,
            DataType.nvint3,
            DataType.nvint3_bs8,
            DataType.nvint4,
            DataType.nvint4_bs8,
            DataType.nvint6,
            DataType.nvfp6_e2m3,
            DataType.nvfp6_e3m2,
        } and (
            super().can_dequantize_values(tensor)
            or tensor.dtype
            in {
                DataType.if6_e2m3,
                DataType.if6_e3m2,
                DataType.if3,
                DataType.if3_bs8,
                DataType.mxfp6_e2m3,
                DataType.mxfp6_e3m2,
                DataType.mxfp3,
                DataType.mxfp3_bs8,
                DataType.nvfp3,
                DataType.nvfp3_bs8,
                DataType.nvint3,
                DataType.nvint3_bs8,
                DataType.nvint6,
                DataType.nvfp6_e2m3,
                DataType.nvfp6_e3m2,
            }
        )

    @classmethod
    def dequantize_values(
        cls,
        tensor: QuantizedTensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        from fouroversix.kernels.cute_sm120.ops import (
            dequantize_fp3_values,
            dequantize_fp6_values,
            dequantize_int3_values,
            dequantize_int6_values,
        )
        from fouroversix.quantize.dequantize_utils import from_blocked

        scale_factors = (
            from_blocked(
                tensor.scale_factors,
                (
                    tensor.padded_shape[0],
                    tensor.padded_shape[1] // tensor.dtype.block_size,
                ),
            )
            if tensor.scale_factors_are_in_blackwell_layout
            else tensor.scale_factors
        )

        if tensor.dtype in {DataType.nvint3, DataType.nvint3_bs8}:
            return dequantize_int3_values(tensor.values).to(dtype)

        if tensor.dtype in {
            DataType.mxfp3,
            DataType.mxfp3_bs8,
            DataType.nvfp3,
            DataType.nvfp3_bs8,
        }:
            return dequantize_fp3_values(tensor.values).to(dtype)

        if tensor.dtype in {DataType.mxfp6_e2m3, DataType.nvfp6_e2m3}:
            return dequantize_fp6_values(
                tensor.values,
                scale_factors,
                use_e3m2=False,
                is_if6=False,
            ).to(dtype)

        if tensor.dtype in {DataType.mxfp6_e3m2, DataType.nvfp6_e3m2}:
            return dequantize_fp6_values(
                tensor.values,
                scale_factors,
                use_e3m2=True,
                is_if6=False,
            ).to(dtype)

        if tensor.dtype == DataType.nvint6:
            return dequantize_int6_values(tensor.values).to(dtype)

        if tensor.dtype in {
            DataType.if6_e2m3,
        }:
            return dequantize_fp6_values(
                tensor.values,
                scale_factors,
                use_e3m2=False,
                is_if6=True,
            ).to(dtype)

        if tensor.dtype in {
            DataType.if6_e3m2,
        }:
            return dequantize_fp6_values(
                tensor.values,
                scale_factors,
                use_e3m2=True,
                is_if6=True,
            ).to(dtype)

        return super().dequantize_values(tensor, dtype=dtype)

    @classmethod
    def pseudo_quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> torch.Tensor:
        if (
            config.pseudo_quantize
            and config.dtype == DataType.nvfp4_bs8
            and config.scale_rule in _ADAPTIVE_SCALE_RULES
            and config.round_style
            in {RoundStyle.stochastic, RoundStyle.stochastic_unbiased}
            and not config.transpose
            and not config.rht
            and not config.block_scale_2d
        ):
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                pseudo_quantize_nvfp4_adaptive,
                *_,
            ) = _pseudo_quantizers()
            return pseudo_quantize_nvfp4_adaptive(
                x,
                scale_rule_id=config.scale_rule.cuda_id,
                scale_block_size=config.dtype.block_size,
                x_amax=config.kwargs.get("x_amax"),
            )

        if (
            config.round_style != RoundStyle.nearest
            and config.dtype != DataType.nvfp4
        ) or config.block_scale_2d:
            return super().pseudo_quantize(x, config)

        if config.dtype not in _PSEUDO_SUPPORTED_DTYPES:
            return super().pseudo_quantize(x, config)

        if config.dtype in _PSEUDO_IF3_DTYPES | _PSEUDO_IF4_DTYPES:
            (
                pseudo_quantize_if3_adaptive,
                pseudo_quantize_if4_adaptive,
                rht_transform,
            ) = _adaptive_if_pseudo_quantizers()
            x_quantize = x.T.contiguous() if config.transpose else x
            if config.rht:
                x_quantize = rht_transform(x_quantize)
            x_amax = None if config.rht else config.kwargs.get("x_amax")
            if config.scale_rule not in _ADAPTIVE_SCALE_RULES:
                return super().pseudo_quantize(x, config)
            if config.dtype in _PSEUDO_IF3_DTYPES:
                out = pseudo_quantize_if3_adaptive(
                    x_quantize,
                    scale_rule_id=config.scale_rule.cuda_id,
                    scale_block_size=config.dtype.block_size,
                    x_amax=x_amax,
                )
            else:
                out = pseudo_quantize_if4_adaptive(
                    x_quantize,
                    scale_rule_id=config.scale_rule.cuda_id,
                    scale_block_size=config.dtype.block_size,
                    x_amax=x_amax,
            )
            return out.T.contiguous() if config.transpose else out

        if (
            not config.transpose
            and not config.rht
            and config.scale_rule in _STATIC_SCALE_RULES
            and (
                config.dtype in _PSEUDO_NVFP3_DTYPES
                or config.dtype in _PSEUDO_NVINT3_DTYPES
                or config.dtype in _PSEUDO_NVINT4_DTYPES
                or config.dtype in _FAST_STATIC_NVFP4_DTYPES
            )
        ):
            (
                pseudo_quantize_nvfp3_static,
                pseudo_quantize_nvfp4_static,
                pseudo_quantize_nvint3_static,
                pseudo_quantize_nvint4_static,
            ) = _static_nv_pseudo_quantizers()
            x_amax = config.kwargs.get("x_amax")
            if config.dtype in _PSEUDO_NVFP3_DTYPES:
                if config.scale_rule == ScaleRule.static_6:
                    return pseudo_quantize_nvfp3_static(
                        x,
                        scale_block_size=config.dtype.block_size,
                        x_amax=x_amax,
                    )
                return super().pseudo_quantize(x, config)
            if config.dtype in _PSEUDO_NVINT3_DTYPES:
                if config.scale_rule == ScaleRule.static_6:
                    return pseudo_quantize_nvint3_static(
                        x,
                        scale_block_size=config.dtype.block_size,
                        x_amax=x_amax,
                    )
                return super().pseudo_quantize(x, config)
            if config.dtype in _PSEUDO_NVINT4_DTYPES:
                if config.scale_rule == ScaleRule.static_6:
                    return pseudo_quantize_nvint4_static(
                        x,
                        scale_block_size=config.dtype.block_size,
                        x_amax=x_amax,
                    )
                return super().pseudo_quantize(x, config)
            if config.scale_rule == ScaleRule.static_6:
                return pseudo_quantize_nvfp4_static(
                    x,
                    max_quantized_value=6,
                    scale_block_size=config.dtype.block_size,
                    x_amax=x_amax,
                )
            return pseudo_quantize_nvfp4_static(
                x,
                max_quantized_value=4,
                scale_block_size=config.dtype.block_size,
                x_amax=x_amax,
            )

        pseudo_quantizers = _pseudo_quantizers()
        (
            pseudo_quantize_if3_adaptive,
            pseudo_quantize_if4_adaptive,
            pseudo_quantize_if6_adaptive,
            pseudo_quantize_mxfp3_static,
            pseudo_quantize_mxfp4_static,
            pseudo_quantize_mxfp6_static,
            pseudo_quantize_nvfp3_static,
            pseudo_quantize_nvfp6_static,
            pseudo_quantize_nvfp4_adaptive,
            pseudo_quantize_nvfp4_static,
            pseudo_quantize_nvint3_static,
            pseudo_quantize_nvint4_static,
            pseudo_quantize_nvint6_static,
            rht_transform,
        ) = pseudo_quantizers

        x_quantize = x.T.contiguous() if config.transpose else x
        if config.rht:
            x_quantize = rht_transform(x_quantize)
        x_amax = None if config.rht else config.kwargs.get("x_amax")

        if config.dtype in _PSEUDO_MXFP3_DTYPES:
            if config.scale_rule in _STATIC_SCALE_RULES:
                out = pseudo_quantize_mxfp3_static(
                    x_quantize,
                    scale_block_size=config.dtype.block_size,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype in _PSEUDO_NVFP3_DTYPES:
            if config.scale_rule == ScaleRule.static_6:
                out = pseudo_quantize_nvfp3_static(
                    x_quantize,
                    scale_block_size=config.dtype.block_size,
                    x_amax=x_amax,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype == DataType.nvfp6_e2m3:
            if config.scale_rule == ScaleRule.static_6:
                out = pseudo_quantize_nvfp6_static(
                    x_quantize,
                    max_quantized_value=7.5,
                    use_e3m2=False,
                    x_amax=x_amax,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype == DataType.nvfp6_e3m2:
            if config.scale_rule == ScaleRule.static_6:
                out = pseudo_quantize_nvfp6_static(
                    x_quantize,
                    max_quantized_value=28.0,
                    use_e3m2=True,
                    x_amax=x_amax,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype in _PSEUDO_NVINT3_DTYPES:
            if config.scale_rule == ScaleRule.static_6:
                out = pseudo_quantize_nvint3_static(
                    x_quantize,
                    scale_block_size=config.dtype.block_size,
                    x_amax=x_amax,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype == DataType.nvint6:
            if config.scale_rule == ScaleRule.static_6:
                out = pseudo_quantize_nvint6_static(
                    x_quantize,
                    x_amax=x_amax,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype in _PSEUDO_NVINT4_DTYPES:
            if config.scale_rule == ScaleRule.static_6:
                out = pseudo_quantize_nvint4_static(
                    x_quantize,
                    scale_block_size=config.dtype.block_size,
                    x_amax=x_amax,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype == DataType.mxfp6_e2m3:
            if config.scale_rule in _STATIC_SCALE_RULES:
                out = pseudo_quantize_mxfp6_static(
                    x_quantize,
                    max_quantized_value=7.5,
                    use_e3m2=False,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype == DataType.mxfp6_e3m2:
            if config.scale_rule in _STATIC_SCALE_RULES:
                out = pseudo_quantize_mxfp6_static(
                    x_quantize,
                    max_quantized_value=28.0,
                    use_e3m2=True,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype in _PSEUDO_MXFP4_DTYPES:
            if config.scale_rule == ScaleRule.static_6:
                out = pseudo_quantize_mxfp4_static(
                    x_quantize,
                    max_quantized_value=6,
                    scale_block_size=config.dtype.block_size,
                )
            elif config.scale_rule == ScaleRule.static_4:
                out = pseudo_quantize_mxfp4_static(
                    x_quantize,
                    max_quantized_value=4,
                    scale_block_size=config.dtype.block_size,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype == DataType.if6_e2m3:
            if config.scale_rule in _ADAPTIVE_SCALE_RULES:
                out = pseudo_quantize_if6_adaptive(
                    x_quantize,
                    scale_rule_id=config.scale_rule.cuda_id,
                    max_quantized_value=7.5,
                    int_expansion_factor=0.241943359375,
                    int_expansion_factor_rcp=4.1333333333,
                    use_e3m2=False,
                    x_amax=x_amax,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.dtype == DataType.if6_e3m2:
            if config.scale_rule in _ADAPTIVE_SCALE_RULES:
                out = pseudo_quantize_if6_adaptive(
                    x_quantize,
                    scale_rule_id=config.scale_rule.cuda_id,
                    max_quantized_value=28.0,
                    int_expansion_factor=0.9032258065,
                    int_expansion_factor_rcp=1.1071428571,
                    use_e3m2=True,
                    x_amax=x_amax,
                )
            else:
                return super().pseudo_quantize(x, config)
        elif config.scale_rule == ScaleRule.static_6:
            out = pseudo_quantize_nvfp4_static(
                x_quantize,
                max_quantized_value=6,
                scale_block_size=config.dtype.block_size,
                x_amax=x_amax,
            )
        elif config.scale_rule == ScaleRule.static_4:
            out = pseudo_quantize_nvfp4_static(
                x_quantize,
                max_quantized_value=4,
                scale_block_size=config.dtype.block_size,
                x_amax=x_amax,
            )
        elif config.scale_rule in _ADAPTIVE_SCALE_RULES:
            out = pseudo_quantize_nvfp4_adaptive(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                x_amax=x_amax,
            )
        else:
            msg = f"Unsupported CuTe sm120 scale rule: {config.scale_rule}"
            raise NotImplementedError(msg)

        return out.T.contiguous() if config.transpose else out

    @classmethod
    def quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        if (
            not config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in {DataType.nvfp4, DataType.nvfp4_bs8}
            and config.scale_rule in _ADAPTIVE_SCALE_RULES
            and (
                config.dtype == DataType.nvfp4
                or config.round_style == RoundStyle.nearest
            )
        ):
            return quantize_nvfp4_adaptive_default(x, config)

        if (
            not config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _FAST_STATIC_NV_DTYPES
            and config.scale_rule in _STATIC_SCALE_RULES
            and (
                config.dtype in _FAST_STATIC_NVFP4_DTYPES
                or config.scale_rule == ScaleRule.static_6
            )
        ):
            quantize_nvfp4_static, quantize_nvfp3_static, quantize_nvfp6_static = (
                _static_nv_quantizers()
            )
            x_amax = config.kwargs.get("x_amax")
            if config.dtype in _FAST_STATIC_NVFP4_DTYPES:
                values, scale_factors_u8, amax = quantize_nvfp4_static(
                    x,
                    max_quantized_value=6
                    if config.scale_rule == ScaleRule.static_6
                    else 4,
                    scale_block_size=config.dtype.block_size,
                    stochastic_rounding=(
                        config.dtype == DataType.nvfp4
                        and config.round_style == RoundStyle.stochastic
                    ),
                    x_amax=x_amax,
                )
            elif config.dtype in _FAST_STATIC_NVFP3_DTYPES:
                values, scale_factors_u8, amax = quantize_nvfp3_static(
                    x,
                    scale_block_size=config.dtype.block_size,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_nvfp6_static(
                    x,
                    max_quantized_value=(
                        7.5 if config.dtype == DataType.nvfp6_e2m3 else 28.0
                    ),
                    use_e3m2=config.dtype == DataType.nvfp6_e3m2,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )

            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype == DataType.nvfp4
            and config.scale_rule
            in {
                ScaleRule.abs_max,
                ScaleRule.mae,
                ScaleRule.mse,
                ScaleRule.static_4,
                ScaleRule.static_6,
            }
        ):
            x_amax = config.kwargs.get("x_amax")
            if config.scale_rule in _ADAPTIVE_SCALE_RULES:
                quantize_nvfp4_adaptive_transpose = (
                    _adaptive_nvfp4_transpose_quantizer()
                )
                values, scale_factors_u8, amax = quantize_nvfp4_adaptive_transpose(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    x_amax=x_amax,
                )
            else:
                quantize_nvfp4_static_transpose, _, _ = (
                    _static_nv_transpose_quantizers()
                )
                values, scale_factors_u8, amax = quantize_nvfp4_static_transpose(
                    x,
                    max_quantized_value=(
                        4 if config.scale_rule == ScaleRule.static_4 else 6
                    ),
                    x_amax=x_amax,
                )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                (x.shape[1], x.shape[0]),
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype == DataType.nvfp4_bs8
            and config.scale_rule in _STATIC_SCALE_RULES
        ):
            quantize_nvfp4_static_transpose, _, _ = _static_nv_transpose_quantizers()
            values, scale_factors_u8, amax = quantize_nvfp4_static_transpose(
                x,
                max_quantized_value=(
                    4 if config.scale_rule == ScaleRule.static_4 else 6
                ),
                scale_block_size=config.dtype.block_size,
                x_amax=config.kwargs.get("x_amax"),
            )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                (x.shape[1], x.shape[0]),
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in {DataType.nvfp3, DataType.nvfp3_bs8}
            and config.scale_rule == ScaleRule.static_6
        ):
            _, quantize_nvfp3_static_transpose, _ = _static_nv_transpose_quantizers()
            values, scale_factors_u8, amax = quantize_nvfp3_static_transpose(
                x,
                scale_block_size=config.dtype.block_size,
                adjustment_factor=1.0,
                x_amax=config.kwargs.get("x_amax"),
            )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                (x.shape[1], x.shape[0]),
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in {DataType.nvfp6_e2m3, DataType.nvfp6_e3m2}
            and config.scale_rule == ScaleRule.static_6
        ):
            _, _, quantize_nvfp6_static_transpose = _static_nv_transpose_quantizers()
            values, scale_factors_u8, amax = quantize_nvfp6_static_transpose(
                x,
                max_quantized_value=(
                    7.5 if config.dtype == DataType.nvfp6_e2m3 else 28.0
                ),
                use_e3m2=config.dtype == DataType.nvfp6_e3m2,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=config.kwargs.get("x_amax"),
            )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                (x.shape[1], x.shape[0]),
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in {DataType.if3, DataType.if3_bs8}
            and config.scale_rule in _ADAPTIVE_SCALE_RULES
        ):
            quantize_if3_adaptive_transpose, _ = _adaptive_if_transpose_quantizers()
            values, scale_factors_u8, amax = quantize_if3_adaptive_transpose(
                x,
                scale_rule_id=_if3_effective_scale_rule_id(config),
                scale_block_size=config.dtype.block_size,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=config.kwargs.get("x_amax"),
            )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                (x.shape[1], x.shape[0]),
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in {DataType.if4, DataType.if4_bs8}
            and config.scale_rule in _ADAPTIVE_SCALE_RULES
            and config.round_style == RoundStyle.nearest
        ):
            _, quantize_if4_adaptive_transpose = _adaptive_if_transpose_quantizers()
            values, scale_factors_u8, amax = quantize_if4_adaptive_transpose(
                x,
                scale_rule_id=config.scale_rule.cuda_id,
                scale_block_size=config.dtype.block_size,
                x_amax=config.kwargs.get("x_amax"),
            )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                (x.shape[1], x.shape[0]),
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _STATIC_MX_DTYPES
            and config.scale_rule in _STATIC_SCALE_RULES
        ):
            (
                quantize_mxfp3_static_transpose,
                quantize_mxfp4_static_transpose,
                quantize_mxfp6_static_transpose,
            ) = _static_mx_transpose_quantizers()
            if config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}:
                values, scale_factors_u8 = quantize_mxfp3_static_transpose(
                    x,
                    scale_block_size=config.dtype.block_size,
                )
            elif config.dtype in {DataType.mxfp4, DataType.mxfp4_bs8}:
                values, scale_factors_u8 = quantize_mxfp4_static_transpose(
                    x,
                    max_quantized_value=(
                        4 if config.scale_rule == ScaleRule.static_4 else 6
                    ),
                    scale_block_size=config.dtype.block_size,
                    stochastic_rounding=config.round_style.is_stochastic,
                )
            else:
                values, scale_factors_u8 = quantize_mxfp6_static_transpose(
                    x,
                    max_quantized_value=(
                        4.0
                        if config.scale_rule == ScaleRule.static_4
                        else 7.5
                        if config.dtype == DataType.mxfp6_e2m3
                        else 28.0
                    ),
                    use_e3m2=config.dtype == DataType.mxfp6_e3m2,
                )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e8m0fnu),
                None,
                config.dtype,
                (x.shape[1], x.shape[0]),
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _STATIC_NVINT_DTYPES
            and config.scale_rule == ScaleRule.static_6
        ):
            (
                quantize_nvint3_static_transpose,
                quantize_nvint4_static_transpose,
                quantize_nvint6_static_transpose,
            ) = _static_nvint_transpose_quantizers()
            x_amax = config.kwargs.get("x_amax")
            if config.dtype in {DataType.nvint3, DataType.nvint3_bs8}:
                values, scale_factors_u8, amax = quantize_nvint3_static_transpose(
                    x,
                    scale_block_size=config.dtype.block_size,
                    adjustment_factor=1.0,
                    x_amax=x_amax,
                )
            elif config.dtype in {DataType.nvint4, DataType.nvint4_bs8}:
                values, scale_factors_u8, amax = quantize_nvint4_static_transpose(
                    x,
                    scale_block_size=config.dtype.block_size,
                    stochastic_rounding=config.round_style.is_stochastic,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_nvint6_static_transpose(
                    x,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                (x.shape[1], x.shape[0]),
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            not config.transpose
            and not config.rht
            and config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype == DataType.nvfp4
            and config.scale_rule
            in {
                ScaleRule.abs_max,
                ScaleRule.mae,
                ScaleRule.mse,
                ScaleRule.static_4,
                ScaleRule.static_6,
            }
        ):
            (
                quantize_nvfp4_adaptive_2d,
                quantize_nvfp4_static_2d,
                _,
            ) = _adaptive_nvfp4_2d_quantizers()
            x_amax = config.kwargs.get("x_amax")
            if config.scale_rule in _ADAPTIVE_SCALE_RULES:
                values, scale_factors_u8, amax = quantize_nvfp4_adaptive_2d(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_nvfp4_static_2d(
                    x,
                    max_quantized_value=(
                        4 if config.scale_rule == ScaleRule.static_4 else 6
                    ),
                    x_amax=x_amax,
                )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            not config.transpose
            and not config.rht
            and config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype == DataType.nvfp4_bs8
            and config.scale_rule in _STATIC_SCALE_RULES
        ):
            _, _, quantize_nvfp4_bs8_static_2d = _adaptive_nvfp4_2d_quantizers()
            values, scale_factors_u8, amax = quantize_nvfp4_bs8_static_2d(
                x,
                max_quantized_value=(
                    4 if config.scale_rule == ScaleRule.static_4 else 6
                ),
                x_amax=config.kwargs.get("x_amax"),
            )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            not config.transpose
            and not config.rht
            and config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _ADAPTIVE_IF_DTYPES
            and config.scale_rule in _ADAPTIVE_SCALE_RULES
        ):
            (
                quantize_if3_adaptive_2d,
                quantize_if3_bs8_adaptive_2d,
                quantize_if4_adaptive_2d,
                quantize_if4_bs8_adaptive_2d,
                quantize_if6_adaptive_2d,
            ) = _adaptive_if_2d_quantizers()
            x_amax = config.kwargs.get("x_amax")
            if config.dtype == DataType.if3:
                values, scale_factors_u8, amax = quantize_if3_adaptive_2d(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            elif config.dtype == DataType.if3_bs8:
                values, scale_factors_u8, amax = quantize_if3_bs8_adaptive_2d(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            elif config.dtype == DataType.if4:
                values, scale_factors_u8, amax = quantize_if4_adaptive_2d(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    x_amax=x_amax,
                )
            elif config.dtype == DataType.if4_bs8:
                values, scale_factors_u8, amax = quantize_if4_bs8_adaptive_2d(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_if6_adaptive_2d(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    max_quantized_value=(
                        7.5 if config.dtype == DataType.if6_e2m3 else 28.0
                    ),
                    int_expansion_factor=(
                        0.241943359375
                        if config.dtype == DataType.if6_e2m3
                        else 0.9032258065
                    ),
                    int_expansion_factor_rcp=(
                        4.1333333333
                        if config.dtype == DataType.if6_e2m3
                        else 1.1071428571
                    ),
                    use_e3m2=config.dtype == DataType.if6_e3m2,
                    x_amax=x_amax,
                )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=(
                    config.dtype in {DataType.if6_e2m3, DataType.if6_e3m2}
                ),
            )

        if (
            not config.transpose
            and not config.rht
            and config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _STATIC_MX_DTYPES
            and config.scale_rule in _STATIC_SCALE_RULES
        ):
            (
                quantize_mxfp3_static_2d,
                quantize_mxfp3_bs8_static_2d,
                quantize_mxfp4_static_2d,
                quantize_mxfp4_bs8_static_2d,
                quantize_mxfp6_static_2d,
            ) = _static_mx_2d_quantizers()
            if config.dtype == DataType.mxfp3:
                values, scale_factors_u8 = quantize_mxfp3_static_2d(x)
            elif config.dtype == DataType.mxfp3_bs8:
                values, scale_factors_u8 = quantize_mxfp3_bs8_static_2d(x)
            elif config.dtype == DataType.mxfp4:
                values, scale_factors_u8 = quantize_mxfp4_static_2d(
                    x,
                    max_quantized_value=(
                        4 if config.scale_rule == ScaleRule.static_4 else 6
                    ),
                )
            elif config.dtype == DataType.mxfp4_bs8:
                values, scale_factors_u8 = quantize_mxfp4_bs8_static_2d(
                    x,
                    max_quantized_value=(
                        4 if config.scale_rule == ScaleRule.static_4 else 6
                    ),
                )
            else:
                values, scale_factors_u8 = quantize_mxfp6_static_2d(
                    x,
                    max_quantized_value=(
                        4.0
                        if config.scale_rule == ScaleRule.static_4
                        else 7.5
                        if config.dtype == DataType.mxfp6_e2m3
                        else 28.0
                    ),
                    use_e3m2=config.dtype == DataType.mxfp6_e3m2,
                )

            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e8m0fnu),
                None,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            not config.transpose
            and not config.rht
            and config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in {DataType.nvfp6_e2m3, DataType.nvfp6_e3m2}
            and config.scale_rule == ScaleRule.static_6
        ):
            quantize_nvfp6_static_2d = _static_nv_2d_quantizer()
            values, scale_factors_u8, amax = quantize_nvfp6_static_2d(
                x,
                max_quantized_value=(
                    7.5 if config.dtype == DataType.nvfp6_e2m3 else 28.0
                ),
                use_e3m2=config.dtype == DataType.nvfp6_e3m2,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=config.kwargs.get("x_amax"),
            )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            not config.transpose
            and not config.rht
            and config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in {DataType.nvfp3, DataType.nvfp3_bs8}
            and config.scale_rule == ScaleRule.static_6
        ):
            quantize_nvfp3_static_2d, quantize_nvfp3_bs8_static_2d = (
                _static_nvfp3_2d_quantizers()
            )
            x_amax = config.kwargs.get("x_amax")
            if config.dtype == DataType.nvfp3:
                values, scale_factors_u8, amax = quantize_nvfp3_static_2d(
                    x,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_nvfp3_bs8_static_2d(
                    x,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            not config.transpose
            and not config.rht
            and config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _STATIC_NVINT_DTYPES
            and config.scale_rule == ScaleRule.static_6
        ):
            (
                quantize_nvint3_static_2d,
                quantize_nvint3_bs8_static_2d,
                quantize_nvint4_static_2d,
                quantize_nvint4_bs8_static_2d,
                quantize_nvint6_static_2d,
            ) = _static_nvint_2d_quantizers()
            x_amax = config.kwargs.get("x_amax")
            if config.dtype == DataType.nvint3:
                values, scale_factors_u8, amax = quantize_nvint3_static_2d(
                    x,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            elif config.dtype == DataType.nvint3_bs8:
                values, scale_factors_u8, amax = quantize_nvint3_bs8_static_2d(
                    x,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            elif config.dtype == DataType.nvint4:
                values, scale_factors_u8, amax = quantize_nvint4_static_2d(
                    x,
                    x_amax=x_amax,
                )
            elif config.dtype == DataType.nvint4_bs8:
                values, scale_factors_u8, amax = quantize_nvint4_bs8_static_2d(
                    x,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_nvint6_static_2d(
                    x,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            not config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _ADAPTIVE_IF_DTYPES
            and config.scale_rule in _ADAPTIVE_SCALE_RULES
        ):
            quantize_if3_adaptive, quantize_if4_adaptive, quantize_if6_adaptive = (
                _adaptive_if_quantizers()
            )
            x_amax = config.kwargs.get("x_amax")
            if config.dtype in {DataType.if3, DataType.if3_bs8}:
                values, scale_factors_u8, amax = quantize_if3_adaptive(
                    x,
                    scale_rule_id=_if3_effective_scale_rule_id(config),
                    scale_block_size=config.dtype.block_size,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            elif config.dtype in {DataType.if4, DataType.if4_bs8}:
                values, scale_factors_u8, amax = quantize_if4_adaptive(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    stochastic_rounding=(
                        config.dtype == DataType.if4
                        and config.round_style == RoundStyle.stochastic
                    ),
                    scale_block_size=config.dtype.block_size,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_if6_adaptive(
                    x,
                    scale_rule_id=config.scale_rule.cuda_id,
                    max_quantized_value=(
                        7.5 if config.dtype == DataType.if6_e2m3 else 28.0
                    ),
                    int_expansion_factor=(
                        0.241943359375
                        if config.dtype == DataType.if6_e2m3
                        else 0.9032258065
                    ),
                    int_expansion_factor_rcp=(
                        4.1333333333
                        if config.dtype == DataType.if6_e2m3
                        else 1.1071428571
                    ),
                    use_e3m2=config.dtype == DataType.if6_e3m2,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )

            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=config.dtype
                in {DataType.if6_e2m3, DataType.if6_e3m2},
            )

        if (
            not config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _STATIC_MX_DTYPES
            and config.scale_rule in _STATIC_SCALE_RULES
        ):
            quantize_mxfp3_static, quantize_mxfp4_static, quantize_mxfp6_static = (
                _static_mx_quantizers()
            )
            if config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}:
                values, scale_factors_u8 = quantize_mxfp3_static(
                    x,
                    scale_block_size=config.dtype.block_size,
                )
            elif config.dtype in {DataType.mxfp4, DataType.mxfp4_bs8}:
                values, scale_factors_u8 = quantize_mxfp4_static(
                    x,
                    max_quantized_value=(
                        4 if config.scale_rule == ScaleRule.static_4 else 6
                    ),
                    scale_block_size=config.dtype.block_size,
                    stochastic_rounding=config.round_style.is_stochastic,
                )
            else:
                values, scale_factors_u8 = quantize_mxfp6_static(
                    x,
                    max_quantized_value=(
                        4.0
                        if config.scale_rule == ScaleRule.static_4
                        else 7.5
                        if config.dtype == DataType.mxfp6_e2m3
                        else 28.0
                    ),
                    use_e3m2=config.dtype == DataType.mxfp6_e3m2,
                )

            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e8m0fnu),
                None,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        if (
            not config.transpose
            and not config.rht
            and not config.block_scale_2d
            and not config.pseudo_quantize
            and config.dtype in _STATIC_NVINT_DTYPES
            and config.scale_rule == ScaleRule.static_6
        ):
            quantize_nvint3_static, quantize_nvint4_static, quantize_nvint6_static = (
                _static_nvint_quantizers()
            )
            x_amax = config.kwargs.get("x_amax")
            if config.dtype in {DataType.nvint3, DataType.nvint3_bs8}:
                values, scale_factors_u8, amax = quantize_nvint3_static(
                    x,
                    scale_block_size=config.dtype.block_size,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            elif config.dtype in {DataType.nvint4, DataType.nvint4_bs8}:
                values, scale_factors_u8, amax = quantize_nvint4_static(
                    x,
                    scale_block_size=config.dtype.block_size,
                    stochastic_rounding=config.round_style.is_stochastic,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_nvint6_static(
                    x,
                    adjustment_factor=config.round_style.adjustment_factor,
                    x_amax=x_amax,
                )

            return _make_quantized_tensor(
                values,
                scale_factors_u8.view(torch.float8_e4m3fn),
                amax,
                config.dtype,
                x.shape,
                config.scale_rule,
                config.round_style,
                scale_factors_are_in_blackwell_layout=False,
            )

        from fouroversix.kernels.cute_sm120.ops import (
            quantize_if3_adaptive,
            quantize_if3_adaptive_transpose,
            quantize_if3_adaptive_2d,
            quantize_if3_bs8_adaptive_2d,
            quantize_if4_adaptive,
            quantize_if4_adaptive_transpose,
            quantize_if4_adaptive_2d,
            quantize_if4_bs8_adaptive_2d,
            quantize_if6_adaptive,
            quantize_if6_adaptive_2d,
            quantize_mxfp3_static,
            quantize_mxfp3_static_transpose,
            quantize_mxfp3_bs8_static_2d,
            quantize_mxfp3_static_2d,
            quantize_mxfp4_static,
            quantize_mxfp4_bs8_static_2d,
            quantize_mxfp4_static_2d,
            quantize_mxfp4_static_transpose,
            quantize_mxfp6_static,
            quantize_mxfp6_static_transpose,
            quantize_mxfp6_static_2d,
            quantize_nvfp4_adaptive_2d,
            quantize_nvfp4_adaptive_transpose,
            quantize_nvint3_static,
            quantize_nvint3_static_transpose,
            quantize_nvint3_bs8_static_2d,
            quantize_nvint3_static_2d,
            quantize_nvint4_static,
            quantize_nvint4_static_transpose,
            quantize_nvint4_bs8_static_2d,
            quantize_nvint6_static,
            quantize_nvint6_static_transpose,
            quantize_nvint6_static_2d,
            quantize_nvfp6_static,
            quantize_nvfp6_static_transpose,
            quantize_nvfp4_adaptive,
            quantize_nvfp4_static,
            quantize_nvfp4_static_transpose,
            quantize_nvfp4_bs8_static_2d,
            quantize_nvfp4_static_2d,
            quantize_nvfp3_static,
            quantize_nvfp3_static_transpose,
            quantize_nvint4_static_2d,
            quantize_nvfp6_static_2d,
            rht_transform,
        )

        quantized_shape = (x.shape[1], x.shape[0]) if config.transpose else x.shape
        use_mxfp4_static_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype in {DataType.mxfp4, DataType.mxfp4_bs8}
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
        )
        use_mxfp3_static_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
        )
        use_mxfp6_static_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype in {DataType.mxfp6_e2m3, DataType.mxfp6_e3m2}
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
        )
        use_nvfp4_static_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype in {DataType.nvfp4, DataType.nvfp4_bs8}
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
        )
        use_nvfp4_adaptive_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype == DataType.nvfp4
            and config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
            and config.round_style == RoundStyle.nearest
        )
        use_nvfp6_static_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype in {DataType.nvfp6_e2m3, DataType.nvfp6_e3m2}
            and config.scale_rule == ScaleRule.static_6
        )
        use_nvfp3_static_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype in {DataType.nvfp3, DataType.nvfp3_bs8}
            and config.scale_rule == ScaleRule.static_6
        )
        use_if3_adaptive_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype in {DataType.if3, DataType.if3_bs8}
            and config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
        )
        use_if4_adaptive_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype in {DataType.if4, DataType.if4_bs8}
            and config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
            and config.round_style == RoundStyle.nearest
        )
        use_nvint_static_transpose_kernel = (
            config.transpose
            and not config.rht
            and not config.block_scale_2d
            and config.dtype
            in {
                DataType.nvint3,
                DataType.nvint3_bs8,
                DataType.nvint4,
                DataType.nvint4_bs8,
                DataType.nvint6,
            }
            and config.scale_rule == ScaleRule.static_6
        )
        x_quantize = (
            x
            if (
                use_mxfp4_static_transpose_kernel
                or use_mxfp3_static_transpose_kernel
                or use_mxfp6_static_transpose_kernel
                or use_nvfp4_adaptive_transpose_kernel
                or use_nvfp4_static_transpose_kernel
                or use_nvfp6_static_transpose_kernel
                or use_nvfp3_static_transpose_kernel
                or use_if3_adaptive_transpose_kernel
                or use_if4_adaptive_transpose_kernel
                or use_nvint_static_transpose_kernel
            )
            else x.T.contiguous()
            if config.transpose
            else x
        )
        if config.rht:
            x_quantize = rht_transform(x_quantize)
        x_amax = None if config.rht else config.kwargs.get("x_amax")

        if config.dtype == DataType.if3 and config.scale_rule in {
            ScaleRule.abs_max,
            ScaleRule.mae,
            ScaleRule.mse,
        } and config.block_scale_2d:
            values, scale_factors_u8, amax = quantize_if3_adaptive_2d(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.dtype == DataType.if3_bs8 and config.scale_rule in {
            ScaleRule.abs_max,
            ScaleRule.mae,
            ScaleRule.mse,
        } and config.block_scale_2d:
            values, scale_factors_u8, amax = quantize_if3_bs8_adaptive_2d(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.dtype in {DataType.if3, DataType.if3_bs8} and config.scale_rule in {
            ScaleRule.abs_max,
            ScaleRule.mae,
            ScaleRule.mse,
        }:
            values, scale_factors_u8, amax = (
                quantize_if3_adaptive_transpose
                if use_if3_adaptive_transpose_kernel
                else quantize_if3_adaptive
            )(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                scale_block_size=config.dtype.block_size,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.dtype == DataType.if4 and config.scale_rule in {
            ScaleRule.abs_max,
            ScaleRule.mae,
            ScaleRule.mse,
        } and config.block_scale_2d:
            values, scale_factors_u8, amax = quantize_if4_adaptive_2d(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.dtype == DataType.if4 and config.scale_rule in {
            ScaleRule.abs_max,
            ScaleRule.mae,
            ScaleRule.mse,
        }:
            if use_if4_adaptive_transpose_kernel:
                values, scale_factors_u8, amax = quantize_if4_adaptive_transpose(
                    x_quantize,
                    scale_rule_id=config.scale_rule.cuda_id,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_if4_adaptive(
                    x_quantize,
                    scale_rule_id=config.scale_rule.cuda_id,
                    stochastic_rounding=config.round_style == RoundStyle.stochastic,
                    x_amax=x_amax,
                )
            scale_dtype = torch.float8_e4m3fn
        elif config.dtype == DataType.if4_bs8 and config.scale_rule in {
            ScaleRule.abs_max,
            ScaleRule.mae,
            ScaleRule.mse,
        } and config.block_scale_2d:
            values, scale_factors_u8, amax = quantize_if4_bs8_adaptive_2d(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.dtype == DataType.if4_bs8 and config.scale_rule in {
            ScaleRule.abs_max,
            ScaleRule.mae,
            ScaleRule.mse,
        }:
            values, scale_factors_u8, amax = (
                quantize_if4_adaptive_transpose
                if use_if4_adaptive_transpose_kernel
                else quantize_if4_adaptive
            )(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                scale_block_size=config.dtype.block_size,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.if6_e2m3
            and config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_if6_adaptive_2d(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                max_quantized_value=7.5,
                int_expansion_factor=0.241943359375,
                int_expansion_factor_rcp=4.1333333333,
                use_e3m2=False,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.if6_e2m3
            and config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
        ):
            values, scale_factors_u8, amax = quantize_if6_adaptive(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                max_quantized_value=7.5,
                int_expansion_factor=0.241943359375,
                int_expansion_factor_rcp=4.1333333333,
                use_e3m2=False,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.if6_e3m2
            and config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_if6_adaptive_2d(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                max_quantized_value=28.0,
                int_expansion_factor=0.9032258065,
                int_expansion_factor_rcp=1.1071428571,
                use_e3m2=True,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.if6_e3m2
            and config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
        ):
            values, scale_factors_u8, amax = quantize_if6_adaptive(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                max_quantized_value=28.0,
                int_expansion_factor=0.9032258065,
                int_expansion_factor_rcp=1.1071428571,
                use_e3m2=True,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.mxfp3_bs8
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
            and config.block_scale_2d
        ):
            values, scale_factors_u8 = quantize_mxfp3_bs8_static_2d(x_quantize)
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp3
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
            and config.block_scale_2d
        ):
            values, scale_factors_u8 = quantize_mxfp3_static_2d(x_quantize)
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype in {DataType.mxfp3, DataType.mxfp3_bs8}
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
        ):
            values, scale_factors_u8 = (
                quantize_mxfp3_static_transpose
                if use_mxfp3_static_transpose_kernel
                else quantize_mxfp3_static
            )(
                x_quantize,
                scale_block_size=config.dtype.block_size,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp4
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8 = quantize_mxfp4_static_2d(
                x_quantize,
                max_quantized_value=6,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp4
            and config.scale_rule == ScaleRule.static_4
            and config.block_scale_2d
        ):
            values, scale_factors_u8 = quantize_mxfp4_static_2d(
                x_quantize,
                max_quantized_value=4,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp4_bs8
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8 = quantize_mxfp4_bs8_static_2d(
                x_quantize,
                max_quantized_value=6,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp4_bs8
            and config.scale_rule == ScaleRule.static_4
            and config.block_scale_2d
        ):
            values, scale_factors_u8 = quantize_mxfp4_bs8_static_2d(
                x_quantize,
                max_quantized_value=4,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp4_bs8
            and config.scale_rule == ScaleRule.static_6
        ):
            values, scale_factors_u8 = (
                quantize_mxfp4_static_transpose
                if use_mxfp4_static_transpose_kernel
                else quantize_mxfp4_static
            )(
                x_quantize,
                max_quantized_value=6,
                scale_block_size=config.dtype.block_size,
                stochastic_rounding=config.round_style.is_stochastic,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp4_bs8
            and config.scale_rule == ScaleRule.static_4
        ):
            values, scale_factors_u8 = (
                quantize_mxfp4_static_transpose
                if use_mxfp4_static_transpose_kernel
                else quantize_mxfp4_static
            )(
                x_quantize,
                max_quantized_value=4,
                scale_block_size=config.dtype.block_size,
                stochastic_rounding=config.round_style.is_stochastic,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif config.dtype == DataType.mxfp4 and config.scale_rule == ScaleRule.static_6:
            values, scale_factors_u8 = (
                quantize_mxfp4_static_transpose
                if use_mxfp4_static_transpose_kernel
                else quantize_mxfp4_static
            )(
                x_quantize,
                max_quantized_value=6,
                stochastic_rounding=config.round_style.is_stochastic,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif config.dtype == DataType.mxfp4 and config.scale_rule == ScaleRule.static_4:
            values, scale_factors_u8 = (
                quantize_mxfp4_static_transpose
                if use_mxfp4_static_transpose_kernel
                else quantize_mxfp4_static
            )(
                x_quantize,
                max_quantized_value=4,
                stochastic_rounding=config.round_style.is_stochastic,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp6_e2m3
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
            and config.block_scale_2d
        ):
            values, scale_factors_u8 = quantize_mxfp6_static_2d(
                x_quantize,
                max_quantized_value=(
                    4.0 if config.scale_rule == ScaleRule.static_4 else 7.5
                ),
                use_e3m2=False,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp6_e2m3
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
        ):
            values, scale_factors_u8 = (
                quantize_mxfp6_static_transpose
                if use_mxfp6_static_transpose_kernel
                else quantize_mxfp6_static
            )(
                x_quantize,
                max_quantized_value=(
                    4.0 if config.scale_rule == ScaleRule.static_4 else 7.5
                ),
                use_e3m2=False,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp6_e3m2
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
            and config.block_scale_2d
        ):
            values, scale_factors_u8 = quantize_mxfp6_static_2d(
                x_quantize,
                max_quantized_value=(
                    4.0 if config.scale_rule == ScaleRule.static_4 else 28.0
                ),
                use_e3m2=True,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.mxfp6_e3m2
            and config.scale_rule in {ScaleRule.static_4, ScaleRule.static_6}
        ):
            values, scale_factors_u8 = (
                quantize_mxfp6_static_transpose
                if use_mxfp6_static_transpose_kernel
                else quantize_mxfp6_static
            )(
                x_quantize,
                max_quantized_value=(
                    4.0 if config.scale_rule == ScaleRule.static_4 else 28.0
                ),
                use_e3m2=True,
            )
            amax = None
            scale_dtype = torch.float8_e8m0fnu
        elif (
            config.dtype == DataType.nvint3_bs8
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvint3_bs8_static_2d(
                x_quantize,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvint3
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvint3_static_2d(
                x_quantize,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype in {DataType.nvint3, DataType.nvint3_bs8}
            and config.scale_rule == ScaleRule.static_6
        ):
            values, scale_factors_u8, amax = (
                quantize_nvint3_static_transpose
                if use_nvint_static_transpose_kernel
                else quantize_nvint3_static
            )(
                x_quantize,
                scale_block_size=config.dtype.block_size,
                adjustment_factor=1.0,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvint4_bs8
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvint4_bs8_static_2d(
                x_quantize,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvint4
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvint4_static_2d(
                x_quantize,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype in {DataType.nvint4, DataType.nvint4_bs8}
            and config.scale_rule == ScaleRule.static_6
        ):
            values, scale_factors_u8, amax = (
                quantize_nvint4_static_transpose
                if use_nvint_static_transpose_kernel
                else quantize_nvint4_static
            )(
                x_quantize,
                scale_block_size=config.dtype.block_size,
                stochastic_rounding=config.round_style.is_stochastic,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvint6
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvint6_static_2d(
                x_quantize,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvint6
            and config.scale_rule == ScaleRule.static_6
        ):
            values, scale_factors_u8, amax = (
                quantize_nvint6_static_transpose
                if use_nvint_static_transpose_kernel
                else quantize_nvint6_static
            )(
                x_quantize,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvfp6_e2m3
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvfp6_static_2d(
                x_quantize,
                max_quantized_value=7.5,
                use_e3m2=False,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvfp6_e2m3
            and config.scale_rule == ScaleRule.static_6
        ):
            values, scale_factors_u8, amax = (
                quantize_nvfp6_static_transpose
                if use_nvfp6_static_transpose_kernel
                else quantize_nvfp6_static
            )(
                x_quantize,
                max_quantized_value=7.5,
                use_e3m2=False,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvfp6_e3m2
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvfp6_static_2d(
                x_quantize,
                max_quantized_value=28.0,
                use_e3m2=True,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvfp6_e3m2
            and config.scale_rule == ScaleRule.static_6
        ):
            values, scale_factors_u8, amax = (
                quantize_nvfp6_static_transpose
                if use_nvfp6_static_transpose_kernel
                else quantize_nvfp6_static
            )(
                x_quantize,
                max_quantized_value=28.0,
                use_e3m2=True,
                adjustment_factor=config.round_style.adjustment_factor,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvfp4_bs8
            and config.scale_rule == ScaleRule.static_6
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvfp4_bs8_static_2d(
                x_quantize,
                max_quantized_value=6,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvfp4_bs8
            and config.scale_rule == ScaleRule.static_4
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvfp4_bs8_static_2d(
                x_quantize,
                max_quantized_value=4,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.scale_rule == ScaleRule.static_6 and config.block_scale_2d:
            values, scale_factors_u8, amax = quantize_nvfp4_static_2d(
                x_quantize,
                max_quantized_value=6,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.scale_rule == ScaleRule.static_4 and config.block_scale_2d:
            values, scale_factors_u8, amax = quantize_nvfp4_static_2d(
                x_quantize,
                max_quantized_value=4,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}
            and config.block_scale_2d
        ):
            values, scale_factors_u8, amax = quantize_nvfp4_adaptive_2d(
                x_quantize,
                scale_rule_id=config.scale_rule.cuda_id,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype in {DataType.nvfp3, DataType.nvfp3_bs8}
            and config.scale_rule == ScaleRule.static_6
        ):
            values, scale_factors_u8, amax = (
                quantize_nvfp3_static_transpose
                if use_nvfp3_static_transpose_kernel
                else quantize_nvfp3_static
            )(
                x_quantize,
                scale_block_size=config.dtype.block_size,
                adjustment_factor=1.0,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvfp4_bs8
            and config.scale_rule == ScaleRule.static_6
        ):
            values, scale_factors_u8, amax = (
                quantize_nvfp4_static_transpose
                if use_nvfp4_static_transpose_kernel
                else quantize_nvfp4_static
            )(
                x_quantize,
                max_quantized_value=6,
                scale_block_size=config.dtype.block_size,
                stochastic_rounding=False,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif (
            config.dtype == DataType.nvfp4_bs8
            and config.scale_rule == ScaleRule.static_4
        ):
            values, scale_factors_u8, amax = (
                quantize_nvfp4_static_transpose
                if use_nvfp4_static_transpose_kernel
                else quantize_nvfp4_static
            )(
                x_quantize,
                max_quantized_value=4,
                scale_block_size=config.dtype.block_size,
                stochastic_rounding=False,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.scale_rule == ScaleRule.static_6:
            values, scale_factors_u8, amax = (
                quantize_nvfp4_static_transpose
                if use_nvfp4_static_transpose_kernel
                else quantize_nvfp4_static
            )(
                x_quantize,
                max_quantized_value=6,
                stochastic_rounding=config.round_style == RoundStyle.stochastic,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.scale_rule == ScaleRule.static_4:
            values, scale_factors_u8, amax = (
                quantize_nvfp4_static_transpose
                if use_nvfp4_static_transpose_kernel
                else quantize_nvfp4_static
            )(
                x_quantize,
                max_quantized_value=4,
                stochastic_rounding=config.round_style == RoundStyle.stochastic,
                x_amax=x_amax,
            )
            scale_dtype = torch.float8_e4m3fn
        elif config.scale_rule in {ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse}:
            if use_nvfp4_adaptive_transpose_kernel:
                values, scale_factors_u8, amax = quantize_nvfp4_adaptive_transpose(
                    x_quantize,
                    scale_rule_id=config.scale_rule.cuda_id,
                    x_amax=x_amax,
                )
            else:
                values, scale_factors_u8, amax = quantize_nvfp4_adaptive(
                    x_quantize,
                    scale_rule_id=config.scale_rule.cuda_id,
                    stochastic_rounding=config.round_style == RoundStyle.stochastic,
                    x_amax=x_amax,
                )
            scale_dtype = torch.float8_e4m3fn
        else:
            msg = f"Unsupported CuTe sm120 scale rule: {config.scale_rule}"
            raise NotImplementedError(msg)

        scale_factors_are_in_blackwell_layout = config.dtype in {
            DataType.if6_e2m3,
            DataType.if6_e3m2,
        }

        return _make_quantized_tensor(
            values,
            scale_factors_u8.view(scale_dtype),
            amax,
            config.dtype,
            quantized_shape,
            config.scale_rule,
            config.round_style,
            scale_factors_are_in_blackwell_layout=(
                scale_factors_are_in_blackwell_layout
            ),
        )
