import itertools
from typing import Any

import pytest
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
from fouroversix.quantize.utils import get_rht_matrix
from fouroversix.utils import SM_80, SM_100, SM_120

MAE_MSE_MISMATCH_TOLERANCE = 1e-3
NUM_RANDOM_SEEDS = 10
QUANTIZATION_METRIC_TOLERANCE = 1e-12
CUTE_VALUE_EQUAL_RATIO_FLOOR = 0.98
CUTE_SCALE_EQUAL_RATIO_FLOOR = 0.98
CUTE_DEQUANT_METRIC_TOLERANCE = 1e-4
CUTE_RHT_DEQUANT_MAE_TOLERANCE = 5e-4


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _require_cuda_for_cute_sm100_accuracy() -> None:
    if not torch.cuda.is_available():
        pytest.xfail("CUDA is required to validate the CuTe sm100 accuracy gate")

    if torch.cuda.get_device_capability()[0] != SM_100:
        pytest.xfail(
            "An sm100 GPU is required to validate the CuTe sm100 accuracy gate",
        )


def _scale_factors_as_matrix(
    tensor,
    shape: tuple[int, int],
    dtype: DataType,
) -> torch.Tensor:
    if tensor.scale_factors_are_in_blackwell_layout:
        return from_blocked(
            tensor.scale_factors.bfloat16(),
            (shape[0], shape[1] // dtype.block_size),
        )

    return tensor.scale_factors.bfloat16().reshape(
        shape[0],
        shape[1] // dtype.block_size,
    )


def _quantize_against_pytorch_reference_metrics(
    x: torch.Tensor,
    *,
    backend: QuantizeBackend,
    dtype: DataType,
    scale_rule: ScaleRule,
) -> dict[str, float]:
    config_backend = QuantizationConfig(
        backend=backend,
        dtype=dtype,
        scale_rule=scale_rule,
    )
    config_reference = QuantizationConfig(
        backend=QuantizeBackend.pytorch,
        dtype=dtype,
        scale_rule=scale_rule,
    )

    backend_cls = AVAILABLE_BACKENDS[backend]
    reference_cls = AVAILABLE_BACKENDS[QuantizeBackend.pytorch]

    if not backend_cls.is_available() or not reference_cls.is_available():
        pytest.skip("Required backend is not available")

    if not backend_cls.can_quantize(x, config_backend):
        pytest.skip(f"Backend {backend} does not support this configuration")

    quantized_backend = quantize(x.clone(), config_backend)
    quantized_reference = quantize(x.clone(), config_reference)

    values_equal_ratio = (
        quantized_backend.values == quantized_reference.values
    ).float().mean()

    backend_scales = _scale_factors_as_matrix(
        quantized_backend,
        x.shape,
        dtype,
    )
    reference_scales = _scale_factors_as_matrix(
        quantized_reference,
        x.shape,
        dtype,
    )
    scales_equal_ratio = (backend_scales == reference_scales).float().mean()

    backend_dequant = dequantize(
        quantized_backend,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    reference_dequant = dequantize(
        quantized_reference,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequant_diff = backend_dequant - reference_dequant
    backend_input_diff = backend_dequant - x.float()

    return {
        "values_equal_ratio": values_equal_ratio.item(),
        "scales_equal_ratio": scales_equal_ratio.item(),
        "mse": (dequant_diff * dequant_diff).mean().item(),
        "mae": dequant_diff.abs().mean().item(),
        "max_error": dequant_diff.abs().max().item(),
        "input_mse": (backend_input_diff * backend_input_diff).mean().item(),
        "input_mae": backend_input_diff.abs().mean().item(),
        "input_max_error": backend_input_diff.abs().max().item(),
    }


@pytest.mark.parametrize("input_shape", [(128, 256), (256, 256)])
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_triton_matches_pytorch_reference_quantize_metrics(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
) -> None:
    _require_cuda()

    triton_backend = AVAILABLE_BACKENDS[QuantizeBackend.triton]
    pytorch_backend = AVAILABLE_BACKENDS[QuantizeBackend.pytorch]
    if not triton_backend.is_available() or not pytorch_backend.is_available():
        pytest.skip("Required backend is not available")

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    metrics = _quantize_against_pytorch_reference_metrics(
        x,
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
    )

    print(
        f"{scale_rule=} "
        f"values_equal={metrics['values_equal_ratio']:.6f} "
        f"scales_equal={metrics['scales_equal_ratio']:.6f} "
        f"mse={metrics['mse']:.8e} mae={metrics['mae']:.8e} "
        f"max_error={metrics['max_error']:.8e}",
    )

    if scale_rule in {ScaleRule.static_4, ScaleRule.static_6, ScaleRule.abs_max}:
        assert metrics["values_equal_ratio"] == 1
        assert metrics["scales_equal_ratio"] == 1
        assert metrics["mse"] == 0
    else:
        assert metrics["values_equal_ratio"] >= 1 - MAE_MSE_MISMATCH_TOLERANCE
        assert metrics["scales_equal_ratio"] >= 1 - MAE_MSE_MISMATCH_TOLERANCE
        assert metrics["mse"] < 1e-6
        assert metrics["mae"] < 1e-4


@pytest.mark.parametrize("input_shape", [(128, 256), (256, 256), (1024, 1024)])
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_quantize_is_not_less_accurate_than_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    if QuantizeBackend.cute_sm100 not in AVAILABLE_BACKENDS:
        pytest.fail("QuantizeBackend.cute_sm100 is not registered")

    cute_backend = AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100]
    triton_backend = AVAILABLE_BACKENDS[QuantizeBackend.triton]
    pytorch_backend = AVAILABLE_BACKENDS[QuantizeBackend.pytorch]
    if (
        not cute_backend.is_available()
        or not triton_backend.is_available()
        or not pytorch_backend.is_available()
    ):
        pytest.xfail("Required backend is not available")

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")

    triton_metrics = _quantize_against_pytorch_reference_metrics(
        x,
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
    )
    cute_metrics = _quantize_against_pytorch_reference_metrics(
        x,
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
    )

    print(
        f"{scale_rule=} triton={triton_metrics} cute_sm100={cute_metrics}",
    )

    assert cute_metrics["values_equal_ratio"] >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert cute_metrics["scales_equal_ratio"] >= CUTE_SCALE_EQUAL_RATIO_FLOOR
    assert cute_metrics["input_mse"] <= (
        triton_metrics["input_mse"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_metrics["input_mae"] <= (
        triton_metrics["input_mae"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_metrics["input_max_error"] <= (
        triton_metrics["input_max_error"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_quantize_zero_input(scale_rule: ScaleRule) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
    )
    x = torch.zeros(128, 256, dtype=torch.bfloat16, device="cuda")

    quantized = quantize(x, config)
    dequantized = dequantize(
        quantized,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )

    assert quantized.values.count_nonzero().item() == 0
    assert quantized.scale_factors.view(torch.uint8).count_nonzero().item() == 0
    assert quantized.amax.item() == 0
    assert dequantized.abs().max().item() == 0


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_nvfp4_transpose_matches_triton(scale_rule: ScaleRule) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        transpose=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        transpose=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)

    assert quantized_cute.original_shape == (x.shape[1], x.shape[0])
    assert quantized_cute.values.shape == quantized_triton.values.shape

    values_equal_ratio = (
        quantized_cute.values == quantized_triton.values
    ).float().mean()
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert values_equal_ratio.item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert diff.abs().mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_nvfp4_pseudo_quantize_matches_triton(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().max().item() <= (
        triton_input_diff.abs().max().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.mse, ScaleRule.static_4],
)
def test_cute_sm100_nvfp4_uses_provided_x_amax(scale_rule: ScaleRule) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    x_amax = x.abs().max().float() * 2
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        kwargs={"x_amax": x_amax},
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        kwargs={"x_amax": x_amax},
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.amax, x_amax)
    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(
        _scale_factors_as_matrix(quantized_cute, x.shape, DataType.nvfp4),
        _scale_factors_as_matrix(quantized_triton, x.shape, DataType.nvfp4),
    )
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize(
    ("dtype", "scale_rule", "kwargs"),
    [
        (DataType.if4, ScaleRule.mse, {}),
        (DataType.if3_bs8, ScaleRule.mae, {"block_scale_2d": True}),
        (DataType.if6_e3m2, ScaleRule.abs_max, {}),
        (DataType.nvfp3, ScaleRule.static_6, {}),
        (DataType.nvfp6_e3m2, ScaleRule.static_6, {}),
        (DataType.nvint4_bs8, ScaleRule.static_6, {}),
        (DataType.nvint6, ScaleRule.static_6, {"block_scale_2d": True}),
    ],
)
def test_cute_sm100_nv_if_formats_use_provided_x_amax(
    dtype: DataType,
    scale_rule: ScaleRule,
    kwargs: dict[str, Any],
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    x_amax = x.abs().max().float() * 2
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        kwargs={"x_amax": x_amax},
        **kwargs,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        kwargs={"x_amax": x_amax},
        **kwargs,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    assert torch.equal(quantized_cute.amax, x_amax)
    assert torch.equal(
        _scale_factors_as_matrix(quantized_cute, x.shape, dtype),
        _scale_factors_as_matrix(quantized_triton, x.shape, dtype),
    )
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.mse, ScaleRule.static_4],
)
def test_cute_sm100_nvfp4_pseudo_quantize_uses_provided_x_amax(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    x_amax = x.abs().max().float() * 2
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        pseudo_quantize=True,
        kwargs={"x_amax": x_amax},
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        pseudo_quantize=True,
        kwargs={"x_amax": x_amax},
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_nvfp4_block_scale_2d_pseudo_quantize_matches_triton_error(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        block_scale_2d=True,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        block_scale_2d=True,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_nvfp4_stochastic_unbiased_pseudo_quantize_matches_triton_error(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        round_style=RoundStyle.stochastic_unbiased,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        round_style=RoundStyle.stochastic_unbiased,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().max().item() <= (
        triton_input_diff.abs().max().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


def test_cute_sm100_nvfp4_stochastic_pseudo_quantize_is_not_claimed() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=ScaleRule.mse,
        round_style=RoundStyle.stochastic,
        pseudo_quantize=True,
    )

    assert not AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].can_quantize(x, config)


@pytest.mark.parametrize(
    ("dtype", "scale_rule"),
    [
        (DataType.mxfp3, ScaleRule.static_4),
        (DataType.mxfp3_bs8, ScaleRule.static_6),
        (DataType.mxfp4, ScaleRule.static_4),
        (DataType.mxfp4_bs8, ScaleRule.static_6),
        (DataType.mxfp6_e2m3, ScaleRule.static_6),
        (DataType.mxfp6_e3m2, ScaleRule.static_4),
        (DataType.if4, ScaleRule.mse),
        (DataType.if4_bs8, ScaleRule.mae),
        (DataType.nvint4, ScaleRule.static_6),
        (DataType.nvint4_bs8, ScaleRule.static_6),
        (DataType.nvfp6_e3m2, ScaleRule.static_6),
    ],
)
def test_cute_sm100_generic_pseudo_quantize_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse],
)
def test_cute_sm100_nvfp4_rht_matches_triton(scale_rule: ScaleRule) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        rht=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        rht=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert diff.abs().mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE


def test_cute_sm100_rht_transform_matches_torch_reference() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    from fouroversix.kernels.cute_sm100.ops import rht_transform

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    had = get_rht_matrix(device=x.device)
    expected = (x.reshape(-1, had.shape[0]) @ had.to(x.dtype)).reshape_as(x)

    actual = rht_transform(x)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "scale_rule", "dequant_backend"),
    [
        (DataType.if4, ScaleRule.abs_max, QuantizeBackend.pytorch),
        (DataType.if4, ScaleRule.mae, QuantizeBackend.pytorch),
        (DataType.if4, ScaleRule.mse, QuantizeBackend.pytorch),
        (DataType.if4_bs8, ScaleRule.abs_max, QuantizeBackend.pytorch),
        (DataType.if4_bs8, ScaleRule.mae, QuantizeBackend.pytorch),
        (DataType.if4_bs8, ScaleRule.mse, QuantizeBackend.pytorch),
        (DataType.if3, ScaleRule.abs_max, QuantizeBackend.cute_sm100),
        (DataType.if3, ScaleRule.mae, QuantizeBackend.cute_sm100),
        (DataType.if3, ScaleRule.mse, QuantizeBackend.cute_sm100),
        (DataType.if3_bs8, ScaleRule.abs_max, QuantizeBackend.cute_sm100),
        (DataType.if3_bs8, ScaleRule.mae, QuantizeBackend.cute_sm100),
        (DataType.if3_bs8, ScaleRule.mse, QuantizeBackend.cute_sm100),
        (DataType.if6_e2m3, ScaleRule.abs_max, QuantizeBackend.triton),
        (DataType.if6_e2m3, ScaleRule.mae, QuantizeBackend.triton),
        (DataType.if6_e2m3, ScaleRule.mse, QuantizeBackend.triton),
        (DataType.if6_e3m2, ScaleRule.abs_max, QuantizeBackend.triton),
        (DataType.if6_e3m2, ScaleRule.mae, QuantizeBackend.triton),
        (DataType.if6_e3m2, ScaleRule.mse, QuantizeBackend.triton),
        (DataType.mxfp4, ScaleRule.static_4, QuantizeBackend.pytorch),
        (DataType.mxfp4, ScaleRule.static_6, QuantizeBackend.pytorch),
        (DataType.mxfp6_e2m3, ScaleRule.static_4, QuantizeBackend.triton),
        (DataType.mxfp6_e2m3, ScaleRule.static_6, QuantizeBackend.triton),
        (DataType.mxfp6_e3m2, ScaleRule.static_4, QuantizeBackend.triton),
        (DataType.mxfp6_e3m2, ScaleRule.static_6, QuantizeBackend.triton),
        (DataType.nvint4, ScaleRule.static_6, QuantizeBackend.pytorch),
        (DataType.nvfp6_e2m3, ScaleRule.static_6, QuantizeBackend.triton),
        (DataType.nvfp6_e3m2, ScaleRule.static_6, QuantizeBackend.triton),
    ],
)
def test_cute_sm100_rht_supported_dtype_rules_match_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
    dequant_backend: QuantizeBackend,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        rht=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        rht=True,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=dequant_backend,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    dequant_tolerance = CUTE_DEQUANT_METRIC_TOLERANCE
    if dtype in {DataType.if3, DataType.if3_bs8}:
        dequant_tolerance = 3 * CUTE_DEQUANT_METRIC_TOLERANCE

    assert (diff * diff).mean().item() <= dequant_tolerance
    assert diff.abs().mean().item() <= CUTE_RHT_DEQUANT_MAE_TOLERANCE


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_nvfp4_block_scale_2d_matches_triton(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, DataType.nvfp4)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, DataType.nvfp4)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert (cute_scales == triton_scales).float().mean().item() >= (
        CUTE_SCALE_EQUAL_RATIO_FLOOR
    )
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert diff.abs().mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_nvfp4_block_scale_2d_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        round_style=RoundStyle.stochastic,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
        round_style=RoundStyle.stochastic,
        block_scale_2d=True,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize("dtype", [DataType.mxfp4, DataType.mxfp4_bs8])
def test_cute_sm100_mxfp4_block_scale_2d_static_matches_triton(
    scale_rule: ScaleRule,
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(
        quantized_triton,
        x.shape,
        dtype,
    )
    cute_scales = _scale_factors_as_matrix(
        quantized_cute,
        x.shape,
        dtype,
    )
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert (cute_scales == triton_scales).float().mean().item() >= (
        CUTE_SCALE_EQUAL_RATIO_FLOOR
    )
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert diff.abs().mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize("dtype", [DataType.mxfp3, DataType.mxfp3_bs8])
@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.nearest, RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_mxfp3_block_scale_2d_static_matches_triton(
    scale_rule: ScaleRule,
    dtype: DataType,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize("dtype", [DataType.mxfp6_e2m3, DataType.mxfp6_e3m2])
@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.nearest, RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_mxfp6_block_scale_2d_static_matches_triton(
    dtype: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize(
    ("dtype", "scale_rule", "round_style", "dequant_backend"),
    [
        (
            DataType.mxfp4,
            ScaleRule.static_4,
            RoundStyle.stochastic,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.mxfp4_bs8,
            ScaleRule.static_4,
            RoundStyle.stochastic,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.mxfp4,
            ScaleRule.static_6,
            RoundStyle.stochastic,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.mxfp4_bs8,
            ScaleRule.static_6,
            RoundStyle.stochastic,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.nvint4,
            ScaleRule.static_6,
            RoundStyle.stochastic,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.nvfp6_e2m3,
            ScaleRule.static_6,
            RoundStyle.stochastic,
            QuantizeBackend.triton,
        ),
        (
            DataType.nvfp6_e3m2,
            ScaleRule.static_6,
            RoundStyle.stochastic,
            QuantizeBackend.triton,
        ),
        (
            DataType.nvfp6_e2m3,
            ScaleRule.static_6,
            RoundStyle.stochastic_unbiased,
            QuantizeBackend.triton,
        ),
        (
            DataType.nvfp6_e3m2,
            ScaleRule.static_6,
            RoundStyle.stochastic_unbiased,
            QuantizeBackend.triton,
        ),
        (
            DataType.mxfp4,
            ScaleRule.static_4,
            RoundStyle.stochastic_unbiased,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.mxfp4_bs8,
            ScaleRule.static_4,
            RoundStyle.stochastic_unbiased,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.mxfp4,
            ScaleRule.static_6,
            RoundStyle.stochastic_unbiased,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.mxfp4_bs8,
            ScaleRule.static_6,
            RoundStyle.stochastic_unbiased,
            QuantizeBackend.pytorch,
        ),
        (
            DataType.nvint4,
            ScaleRule.static_6,
            RoundStyle.stochastic_unbiased,
            QuantizeBackend.pytorch,
        ),
    ],
)
def test_cute_sm100_static_block_scale_2d_stochastic_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
    dequant_backend: QuantizeBackend,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=True,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=dequant_backend,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("dtype", [DataType.nvfp6_e2m3, DataType.nvfp6_e3m2])
def test_cute_sm100_nvfp6_block_scale_2d_stochastic_unbiased_is_claimed(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    x = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic_unbiased,
        block_scale_2d=True,
    )

    assert AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].can_quantize(
        x,
        config,
    )


@pytest.mark.parametrize("dtype", [DataType.nvint4, DataType.nvint4_bs8])
def test_cute_sm100_nvint4_block_scale_2d_static_matches_triton(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert (cute_scales == triton_scales).float().mean().item() >= (
        CUTE_SCALE_EQUAL_RATIO_FLOOR
    )
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert diff.abs().mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("dtype", [DataType.nvfp6_e2m3, DataType.nvfp6_e3m2])
def test_cute_sm100_nvfp6_block_scale_2d_static_matches_triton(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert (cute_scales == triton_scales).float().mean().item() >= (
        CUTE_SCALE_EQUAL_RATIO_FLOOR
    )
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert diff.abs().mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse],
)
@pytest.mark.parametrize(
    "dtype",
    [DataType.if4, DataType.if6_e2m3, DataType.if6_e3m2],
)
def test_cute_sm100_if_adaptive_is_not_less_accurate_than_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")

    if dtype in {DataType.if6_e2m3, DataType.if6_e3m2}:
        config_triton = QuantizationConfig(
            backend=QuantizeBackend.triton,
            dtype=dtype,
            scale_rule=scale_rule,
        )
        config_cute = QuantizationConfig(
            backend=QuantizeBackend.cute_sm100,
            dtype=dtype,
            scale_rule=scale_rule,
        )
        quantized_triton = quantize(x, config_triton)
        quantized_cute = quantize(x, config_cute)
        triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
        cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
        dequantized_triton = dequantize(
            quantized_triton,
            dtype=torch.float32,
            backend=QuantizeBackend.triton,
            intermediate_dtype=torch.float32,
        )
        dequantized_cute = dequantize(
            quantized_cute,
            dtype=torch.float32,
            backend=QuantizeBackend.triton,
            intermediate_dtype=torch.float32,
        )
        diff = dequantized_cute - dequantized_triton
        triton_input_diff = dequantized_triton - x.float()
        cute_input_diff = dequantized_cute - x.float()
        values_equal_ratio = (
            quantized_cute.values == quantized_triton.values
        ).float().mean().item()

        print(
            f"{dtype=} {scale_rule=} values_equal={values_equal_ratio:.6f} "
            f"mse={(diff * diff).mean().item():.8e} "
            f"mae={diff.abs().mean().item():.8e}",
        )

        is_e2m3_absmax = dtype == DataType.if6_e2m3 and (
            scale_rule == ScaleRule.abs_max
        )
        if not is_e2m3_absmax:
            assert values_equal_ratio >= CUTE_VALUE_EQUAL_RATIO_FLOOR
            assert (cute_scales == triton_scales).float().mean().item() >= (
                CUTE_SCALE_EQUAL_RATIO_FLOOR
            )
        assert (cute_input_diff * cute_input_diff).mean().item() <= (
            (triton_input_diff * triton_input_diff).mean().item()
            + CUTE_DEQUANT_METRIC_TOLERANCE
        )
        assert cute_input_diff.abs().mean().item() <= (
            triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
        )
        return

    triton_metrics = _quantize_against_pytorch_reference_metrics(
        x,
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
    )
    cute_metrics = _quantize_against_pytorch_reference_metrics(
        x,
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
    )

    assert cute_metrics["values_equal_ratio"] >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert cute_metrics["scales_equal_ratio"] >= CUTE_SCALE_EQUAL_RATIO_FLOOR
    assert cute_metrics["input_mse"] <= (
        triton_metrics["input_mse"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_metrics["input_mae"] <= (
        triton_metrics["input_mae"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_metrics["input_max_error"] <= (
        triton_metrics["input_max_error"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse],
)
@pytest.mark.parametrize(
    "dtype",
    [DataType.if3, DataType.if3_bs8, DataType.if4, DataType.if4_bs8],
)
def test_cute_sm100_if_block_scale_2d_matches_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        block_scale_2d=True,
        dtype=dtype,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        block_scale_2d=True,
        dtype=dtype,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequant_backend = (
        QuantizeBackend.cute_sm100
        if dtype in {DataType.if3, DataType.if3_bs8}
        else QuantizeBackend.pytorch
    )
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=dequant_backend,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=dequant_backend,
        intermediate_dtype=torch.float32,
    )

    triton_input_diff = dequantized_triton - x.float()
    cute_input_diff = dequantized_cute - x.float()
    values_equal_ratio = (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item()
    scales_equal_ratio = (cute_scales == triton_scales).float().mean().item()

    print(
        f"{dtype=} {scale_rule=} 2d values_equal={values_equal_ratio:.6f} "
        f"scales_equal={scales_equal_ratio:.6f}",
    )

    assert values_equal_ratio >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert scales_equal_ratio >= CUTE_SCALE_EQUAL_RATIO_FLOOR
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("scale_rule", [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse])
def test_cute_sm100_if4_bs8_adaptive_matches_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.if4_bs8,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.if4_bs8,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(
        quantized_triton,
        x.shape,
        DataType.if4_bs8,
    )
    cute_scales = _scale_factors_as_matrix(
        quantized_cute,
        x.shape,
        DataType.if4_bs8,
    )
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert (cute_scales == triton_scales).float().mean().item() >= (
        CUTE_SCALE_EQUAL_RATIO_FLOOR
    )
    triton_input_diff = dequantized_triton - x.float()
    cute_input_diff = dequantized_cute - x.float()
    assert (diff * diff).mean().item() <= 3 * CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_scale_2d": True},
    ],
)
def test_cute_sm100_if4_bs8_unsupported_modes_are_not_claimed(
    kwargs: dict[str, Any],
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    if kwargs == {"block_scale_2d": True}:
        pytest.skip("Covered by IF4_BS8 block_scale_2d accuracy test")

    x = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.if4_bs8,
        scale_rule=ScaleRule.mse,
        **kwargs,
    )

    assert not AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].can_quantize(
        x,
        config,
    )


@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
@pytest.mark.parametrize("scale_rule", [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse])
def test_cute_sm100_if4_bs8_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.if4_bs8,
        round_style=round_style,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.if4_bs8,
        round_style=round_style,
        scale_rule=scale_rule,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{scale_rule=} {round_style=} triton_dist={triton_dist} cute_dist={cute_dist}")
    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("dtype", [DataType.if4, DataType.if4_bs8])
@pytest.mark.parametrize("scale_rule", [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse])
def test_cute_sm100_if4_pseudo_quantize_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("scale_rule", [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse])
@pytest.mark.parametrize("dtype", [DataType.if3, DataType.if3_bs8])
def test_cute_sm100_if3_adaptive_matches_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton
    triton_input_diff = dequantized_triton - x.float()
    cute_input_diff = dequantized_cute - x.float()

    assert (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert (cute_scales == triton_scales).float().mean().item() >= (
        CUTE_SCALE_EQUAL_RATIO_FLOOR
    )
    assert (diff * diff).mean().item() <= 3 * CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("dtype", [DataType.if3, DataType.if3_bs8])
@pytest.mark.parametrize("scale_rule", [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse])
def test_cute_sm100_if3_pseudo_quantize_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= 3 * CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("dtype", [DataType.if6_e2m3, DataType.if6_e3m2])
@pytest.mark.parametrize("scale_rule", [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse])
def test_cute_sm100_if6_pseudo_quantize_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= 3 * CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    ("dtype", "scale_rule", "round_style", "block_scale_2d"),
    [
        (DataType.if3, ScaleRule.mse, RoundStyle.stochastic_unbiased, True),
        (DataType.if3_bs8, ScaleRule.mse, RoundStyle.stochastic_unbiased, True),
        (DataType.if3_bs8, ScaleRule.mae, RoundStyle.stochastic_unbiased, False),
    ],
)
def test_cute_sm100_if3_unsupported_modes_are_not_claimed(
    dtype: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
    block_scale_2d: bool,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    x = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=block_scale_2d,
    )

    assert not AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].can_quantize(
        x,
        config,
    )


@pytest.mark.parametrize("dtype", [DataType.if3, DataType.if3_bs8])
@pytest.mark.parametrize("scale_rule", [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse])
@pytest.mark.parametrize(
    ("round_style", "block_scale_2d"),
    [
        (RoundStyle.stochastic, False),
        (RoundStyle.stochastic, True),
        (RoundStyle.stochastic_unbiased, False),
    ],
)
def test_cute_sm100_if3_stochastic_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
    block_scale_2d: bool,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    if (
        dtype == DataType.if3_bs8
        and scale_rule == ScaleRule.mae
        and round_style == RoundStyle.stochastic_unbiased
    ):
        pytest.skip("IF3_BS8 mae stochastic-unbiased remains unsupported")

    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=block_scale_2d,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=block_scale_2d,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("dtype", [DataType.if4, DataType.if4_bs8])
@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
@pytest.mark.parametrize("scale_rule", [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse])
def test_cute_sm100_if4_block_scale_2d_stochastic_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
        block_scale_2d=True,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize(
    ("dtype", "scale_rule"),
    [
        (DataType.nvfp4, ScaleRule.abs_max),
        (DataType.nvfp4, ScaleRule.mae),
        (DataType.nvfp4, ScaleRule.mse),
        (DataType.nvfp4, ScaleRule.static_4),
        (DataType.nvfp4, ScaleRule.static_6),
        (DataType.if4, ScaleRule.abs_max),
        (DataType.if4, ScaleRule.mae),
        (DataType.if4, ScaleRule.mse),
        (DataType.if4_bs8, ScaleRule.abs_max),
        (DataType.if4_bs8, ScaleRule.mae),
        (DataType.if4_bs8, ScaleRule.mse),
        (DataType.if6_e2m3, ScaleRule.abs_max),
        (DataType.if6_e2m3, ScaleRule.mae),
        (DataType.if6_e2m3, ScaleRule.mse),
        (DataType.if6_e3m2, ScaleRule.abs_max),
        (DataType.if6_e3m2, ScaleRule.mae),
        (DataType.if6_e3m2, ScaleRule.mse),
        (DataType.mxfp4, ScaleRule.static_4),
        (DataType.mxfp4, ScaleRule.static_6),
        (DataType.nvint4, ScaleRule.static_6),
        (DataType.nvfp6_e2m3, ScaleRule.static_6),
        (DataType.nvfp6_e3m2, ScaleRule.static_6),
    ],
)
def test_cute_sm100_dequantize_backend_matches_pytorch(
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
    )
    quantized = quantize(x, config)

    dequantized_cute = dequantize(
        quantized,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    reference_backend = (
        QuantizeBackend.triton
        if dtype
        in {
            DataType.if6_e2m3,
            DataType.if6_e3m2,
            DataType.nvfp6_e2m3,
            DataType.nvfp6_e3m2,
        }
        else QuantizeBackend.pytorch
    )
    dequantized_reference = dequantize(
        quantized,
        dtype=torch.float32,
        backend=reference_backend,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_reference

    assert diff.abs().max().item() == 0
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_nvfp4_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        round_style=RoundStyle.stochastic,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        round_style=RoundStyle.stochastic,
        scale_rule=scale_rule,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{scale_rule=} triton_dist={triton_dist} cute_dist={cute_dist}")
    assert abs((cute_dist - triton_dist) / triton_dist) < 0.05


@pytest.mark.parametrize("block_scale_2d", [False, True])
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_cute_sm100_nvfp4_stochastic_unbiased_matches_triton_error(
    scale_rule: ScaleRule,
    block_scale_2d: bool,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        round_style=RoundStyle.stochastic_unbiased,
        scale_rule=scale_rule,
        block_scale_2d=block_scale_2d,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4,
        round_style=RoundStyle.stochastic_unbiased,
        scale_rule=scale_rule,
        block_scale_2d=block_scale_2d,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )

    assert torch.dist(dequantized_cute, x.float()) <= (
        torch.dist(dequantized_triton, x.float()) + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_mxfp4_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.mxfp4,
        round_style=round_style,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.mxfp4,
        round_style=round_style,
        scale_rule=scale_rule,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{scale_rule=} {round_style=} "
        f"triton_dist={triton_dist} cute_dist={cute_dist}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize(
    ("dtype", "round_style"),
    [
        (DataType.nvfp6_e2m3, RoundStyle.stochastic),
        (DataType.nvfp6_e2m3, RoundStyle.stochastic_unbiased),
        (DataType.nvfp6_e3m2, RoundStyle.stochastic),
        (DataType.nvfp6_e3m2, RoundStyle.stochastic_unbiased),
    ],
)
def test_cute_sm100_nvfp6_stochastic_matches_triton_error(
    dtype: DataType,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        round_style=round_style,
        scale_rule=ScaleRule.static_6,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        round_style=round_style,
        scale_rule=ScaleRule.static_6,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, tuple(x.shape), dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, tuple(x.shape), dtype)

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales.view(torch.uint8), triton_scales.view(torch.uint8))

    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{dtype=} {round_style=} triton_dist={triton_dist} cute_dist={cute_dist}")
    if round_style == RoundStyle.stochastic_unbiased:
        assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE
    else:
        assert abs((cute_dist - triton_dist) / triton_dist) < 0.05


@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_nvint4_stochastic_matches_triton_error(
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint4,
        round_style=round_style,
        scale_rule=ScaleRule.static_6,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint4,
        round_style=round_style,
        scale_rule=ScaleRule.static_6,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{round_style=} triton_dist={triton_dist} cute_dist={cute_dist}")
    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse],
)
def test_cute_sm100_if4_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.if4,
        round_style=RoundStyle.stochastic,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.if4,
        round_style=RoundStyle.stochastic,
        scale_rule=scale_rule,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{scale_rule=} triton_dist={triton_dist} cute_dist={cute_dist}")
    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("block_scale_2d", [False, True])
@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.abs_max, ScaleRule.mae, ScaleRule.mse],
)
def test_cute_sm100_if4_stochastic_unbiased_matches_triton_error(
    scale_rule: ScaleRule,
    block_scale_2d: bool,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.if4,
        round_style=RoundStyle.stochastic_unbiased,
        scale_rule=scale_rule,
        block_scale_2d=block_scale_2d,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.if4,
        round_style=RoundStyle.stochastic_unbiased,
        scale_rule=scale_rule,
        block_scale_2d=block_scale_2d,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )

    assert torch.dist(dequantized_cute, x.float()) <= (
        torch.dist(dequantized_triton, x.float()) + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    ("dtype", "scale_rule"),
    [
        (DataType.if6_e2m3, ScaleRule.abs_max),
        (DataType.if6_e2m3, ScaleRule.mae),
        (DataType.if6_e2m3, ScaleRule.mse),
        (DataType.if6_e3m2, ScaleRule.abs_max),
        (DataType.if6_e3m2, ScaleRule.mae),
        (DataType.if6_e3m2, ScaleRule.mse),
    ],
)
@pytest.mark.parametrize("round_style", [RoundStyle.stochastic, RoundStyle.stochastic_unbiased])
def test_cute_sm100_if6_stochastic_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        round_style=round_style,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        round_style=round_style,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())
    values_equal_ratio = (
        quantized_cute.values == quantized_triton.values
    ).float().mean().item()
    scales_equal_ratio = (cute_scales == triton_scales).float().mean().item()

    print(
        f"{dtype=} {scale_rule=} values_equal={values_equal_ratio:.6f} "
        f"scales_equal={scales_equal_ratio:.6f} "
        f"{round_style=} triton_dist={triton_dist} cute_dist={cute_dist}",
    )

    equal_ratio_floor = CUTE_VALUE_EQUAL_RATIO_FLOOR
    if dtype == DataType.if6_e2m3 and scale_rule == ScaleRule.abs_max:
        equal_ratio_floor = 0.975

    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE
    assert values_equal_ratio >= equal_ratio_floor
    assert scales_equal_ratio >= equal_ratio_floor


def test_cute_sm100_if6_e2m3_absmax_stochastic_is_claimed() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    x = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.if6_e2m3,
        round_style=RoundStyle.stochastic,
        scale_rule=ScaleRule.abs_max,
    )

    assert AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].can_quantize(
        x,
        config,
    )


def test_cute_sm100_if6_stochastic_unbiased_is_claimed() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    x = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.if6_e2m3,
        scale_rule=ScaleRule.abs_max,
        round_style=RoundStyle.stochastic_unbiased,
    )

    assert AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].can_quantize(
        x,
        config,
    )


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_mxfp4_static_matches_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.mxfp4,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.mxfp4,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)

    triton_scales = _scale_factors_as_matrix(
        quantized_triton,
        x.shape,
        DataType.mxfp4,
    )
    cute_scales = _scale_factors_as_matrix(
        quantized_cute,
        x.shape,
        DataType.mxfp4,
    )
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_mxfp4_bs8_static_matches_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.mxfp4_bs8,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.mxfp4_bs8,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(
        quantized_triton,
        x.shape,
        DataType.mxfp4_bs8,
    )
    cute_scales = _scale_factors_as_matrix(
        quantized_cute,
        x.shape,
        DataType.mxfp4_bs8,
    )
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("dtype", [DataType.mxfp4, DataType.mxfp4_bs8])
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_mxfp4_pseudo_quantize_matches_triton(
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("dtype", [DataType.mxfp3, DataType.mxfp3_bs8])
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_mxfp3_pseudo_quantize_matches_triton(
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("dtype", [DataType.mxfp6_e2m3, DataType.mxfp6_e3m2])
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_mxfp6_pseudo_quantize_matches_triton_error(
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_mxfp4_bs8_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.mxfp4_bs8,
        scale_rule=scale_rule,
        round_style=round_style,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.mxfp4_bs8,
        scale_rule=scale_rule,
        round_style=round_style,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{scale_rule=} {round_style=} "
        f"triton_dist={triton_dist} cute_dist={cute_dist}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize("dtype", [DataType.mxfp3, DataType.mxfp3_bs8])
def test_cute_sm100_mxfp3_static_matches_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize("dtype", [DataType.mxfp3, DataType.mxfp3_bs8])
@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_mxfp3_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
    dtype: DataType,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{dtype=} {scale_rule=} {round_style=} {triton_dist=} {cute_dist=}")
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize("dtype", [DataType.mxfp6_e2m3, DataType.mxfp6_e3m2])
def test_cute_sm100_mxfp6_static_matches_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize("dtype", [DataType.mxfp6_e2m3, DataType.mxfp6_e3m2])
@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_mxfp6_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
    dtype: DataType,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=scale_rule,
        round_style=round_style,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
def test_cute_sm100_nvint4_static_is_not_less_accurate_than_triton(
    input_shape: tuple[int, int],
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    triton_metrics = _quantize_against_pytorch_reference_metrics(
        x,
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint4,
        scale_rule=ScaleRule.static_6,
    )
    cute_metrics = _quantize_against_pytorch_reference_metrics(
        x,
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint4,
        scale_rule=ScaleRule.static_6,
    )

    assert cute_metrics["values_equal_ratio"] >= CUTE_VALUE_EQUAL_RATIO_FLOOR
    assert cute_metrics["scales_equal_ratio"] >= CUTE_SCALE_EQUAL_RATIO_FLOOR
    assert cute_metrics["input_mse"] <= (
        triton_metrics["input_mse"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_metrics["input_mae"] <= (
        triton_metrics["input_mae"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_metrics["input_max_error"] <= (
        triton_metrics["input_max_error"] + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
def test_cute_sm100_nvint4_bs8_static_matches_triton(
    input_shape: tuple[int, int],
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint4_bs8,
        scale_rule=ScaleRule.static_6,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint4_bs8,
        scale_rule=ScaleRule.static_6,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(
        quantized_triton,
        x.shape,
        DataType.nvint4_bs8,
    )
    cute_scales = _scale_factors_as_matrix(
        quantized_cute,
        x.shape,
        DataType.nvint4_bs8,
    )
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    values_equal_ratio = (
        quantized_cute.values == quantized_triton.values
    ).float().mean()
    diff = dequantized_cute - dequantized_triton

    assert values_equal_ratio.item() >= 0.999
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert torch.dist(dequantized_cute, x.float()) <= (
        torch.dist(dequantized_triton, x.float()) + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_nvint4_bs8_stochastic_matches_triton_error(
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint4_bs8,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint4_bs8,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{round_style=} triton_dist={triton_dist} cute_dist={cute_dist}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_nvfp4_bs8_static_matches_triton(
    input_shape: tuple[int, int],
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(
        quantized_triton,
        x.shape,
        DataType.nvfp4_bs8,
    )
    cute_scales = _scale_factors_as_matrix(
        quantized_cute,
        x.shape,
        DataType.nvfp4_bs8,
    )
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_nvfp4_bs8_pseudo_quantize_matches_triton(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_nvfp4_bs8_block_scale_2d_static_matches_triton(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(
        quantized_triton,
        x.shape,
        DataType.nvfp4_bs8,
    )
    cute_scales = _scale_factors_as_matrix(
        quantized_cute,
        x.shape,
        DataType.nvfp4_bs8,
    )
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_nvfp4_bs8_block_scale_2d_pseudo_quantize_matches_triton_error(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        block_scale_2d=True,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        block_scale_2d=True,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )
    assert cute_input_diff.abs().mean().item() <= (
        triton_input_diff.abs().mean().item() + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
@pytest.mark.parametrize("round_style", [RoundStyle.stochastic, RoundStyle.stochastic_unbiased])
def test_cute_sm100_nvfp4_bs8_stochastic_matches_triton_error(
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        round_style=round_style,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        round_style=round_style,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{scale_rule=} {round_style=} "
        f"triton_dist={triton_dist} cute_dist={cute_dist}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("scale_rule", [ScaleRule.static_4, ScaleRule.static_6])
def test_cute_sm100_nvfp4_bs8_block_scale_2d_stochastic_unbiased_matches_triton_error(
    scale_rule: ScaleRule,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        block_scale_2d=True,
        round_style=RoundStyle.stochastic_unbiased,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        block_scale_2d=True,
        round_style=RoundStyle.stochastic_unbiased,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.pytorch,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{scale_rule=} block_scale_2d=True "
        f"round_style={RoundStyle.stochastic_unbiased} "
        f"triton_dist={triton_dist} cute_dist={cute_dist}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.nearest, RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_nvfp4_bs8_nonstatic_or_stochastic_is_not_claimed(
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    if scale_rule in {ScaleRule.static_4, ScaleRule.static_6} and round_style in {
        RoundStyle.nearest,
        RoundStyle.stochastic,
        RoundStyle.stochastic_unbiased,
    }:
        pytest.skip("Covered by static NVFP4_BS8 accuracy tests")

    x = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvfp4_bs8,
        scale_rule=scale_rule,
        round_style=round_style,
    )

    assert not AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].can_quantize(
        x,
        config,
    )


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [DataType.nvfp3, DataType.nvfp3_bs8])
def test_cute_sm100_nvfp3_static_matches_triton(
    input_shape: tuple[int, int],
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() == 0


@pytest.mark.parametrize("dtype", [DataType.nvfp3, DataType.nvfp3_bs8])
def test_cute_sm100_nvfp3_pseudo_quantize_matches_triton(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("dtype", [DataType.nvfp3, DataType.nvfp3_bs8])
def test_cute_sm100_nvfp3_stochastic_matches_triton_error(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{dtype=} round_style={RoundStyle.stochastic} {triton_dist=} {cute_dist=}")
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("dtype", [DataType.nvfp3, DataType.nvfp3_bs8])
def test_cute_sm100_nvfp3_stochastic_unbiased_matches_triton_error(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic_unbiased,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic_unbiased,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{dtype=} round_style={RoundStyle.stochastic_unbiased} "
        f"{triton_dist=} {cute_dist=}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("dtype", [DataType.nvfp3, DataType.nvfp3_bs8])
@pytest.mark.parametrize("round_style", [RoundStyle.stochastic, RoundStyle.stochastic_unbiased])
def test_cute_sm100_nvfp3_block_scale_2d_stochastic_matches_triton_error(
    dtype: DataType,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
        block_scale_2d=True,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{dtype=} block_scale_2d=True {round_style=} "
        f"{triton_dist=} {cute_dist=}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("dtype", [DataType.nvint3, DataType.nvint3_bs8])
def test_cute_sm100_nvint3_block_scale_2d_static_matches_triton(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    values_equal_ratio = (
        quantized_cute.values == quantized_triton.values
    ).float().mean()
    diff = dequantized_cute - dequantized_triton

    assert values_equal_ratio.item() >= 0.999
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert torch.dist(dequantized_cute, x.float()) <= (
        torch.dist(dequantized_triton, x.float()) + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "dtype",
    [DataType.nvfp3, DataType.nvfp3_bs8],
)
def test_cute_sm100_nvfp3_block_scale_2d_static_matches_triton(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(quantized_cute.values, quantized_triton.values)
    assert torch.equal(cute_scales, triton_scales)
    assert diff.abs().max() == 0


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [DataType.nvint3, DataType.nvint3_bs8])
def test_cute_sm100_nvint3_static_matches_triton(
    input_shape: tuple[int, int],
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    values_equal_ratio = (
        quantized_cute.values == quantized_triton.values
    ).float().mean()
    diff = dequantized_cute - dequantized_triton

    assert values_equal_ratio.item() >= 0.999
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert torch.dist(dequantized_cute, x.float()) <= (
        torch.dist(dequantized_triton, x.float()) + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("dtype", [DataType.nvint3, DataType.nvint3_bs8])
def test_cute_sm100_nvint3_pseudo_quantize_matches_triton(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert torch.equal(pseudo_cute, pseudo_triton)


@pytest.mark.parametrize("dtype", [DataType.nvint4, DataType.nvint4_bs8])
def test_cute_sm100_nvint4_pseudo_quantize_matches_triton(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize("dtype", [DataType.nvint3, DataType.nvint3_bs8])
def test_cute_sm100_nvint3_stochastic_matches_triton_error(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{dtype=} round_style={RoundStyle.stochastic} {triton_dist=} {cute_dist=}")
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("dtype", [DataType.nvint3, DataType.nvint3_bs8])
def test_cute_sm100_nvint3_stochastic_unbiased_matches_triton_error(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic_unbiased,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic_unbiased,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{dtype=} round_style={RoundStyle.stochastic_unbiased} "
        f"{triton_dist=} {cute_dist=}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("dtype", [DataType.nvint3, DataType.nvint3_bs8])
@pytest.mark.parametrize("round_style", [RoundStyle.stochastic, RoundStyle.stochastic_unbiased])
def test_cute_sm100_nvint3_block_scale_2d_stochastic_matches_triton_error(
    dtype: DataType,
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
        block_scale_2d=True,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"{dtype=} block_scale_2d=True {round_style=} "
        f"{triton_dist=} {cute_dist=}",
    )
    assert cute_dist <= (triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE)


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
def test_cute_sm100_nvint6_static_matches_triton(
    input_shape: tuple[int, int],
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, DataType.nvint6)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, DataType.nvint6)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    values_equal_ratio = (
        quantized_cute.values == quantized_triton.values
    ).float().mean()
    diff = dequantized_cute - dequantized_triton

    assert values_equal_ratio.item() >= 0.999
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert torch.dist(dequantized_cute, x.float()) <= (
        torch.dist(dequantized_triton, x.float()) + CUTE_DEQUANT_METRIC_TOLERANCE
    )


def test_cute_sm100_nvint6_block_scale_2d_static_matches_triton() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        block_scale_2d=True,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, DataType.nvint6)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, DataType.nvint6)
    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    values_equal_ratio = (
        quantized_cute.values == quantized_triton.values
    ).float().mean()
    diff = dequantized_cute - dequantized_triton

    assert values_equal_ratio.item() >= 0.999
    assert torch.equal(cute_scales, triton_scales)
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert torch.dist(dequantized_cute, x.float()) <= (
        torch.dist(dequantized_triton, x.float()) + CUTE_DEQUANT_METRIC_TOLERANCE
    )


def test_cute_sm100_nvint6_pseudo_quantize_matches_triton_error() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
    assert (cute_input_diff * cute_input_diff).mean().item() <= (
        (triton_input_diff * triton_input_diff).mean().item()
        + CUTE_DEQUANT_METRIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "round_style",
    [RoundStyle.stochastic, RoundStyle.stochastic_unbiased],
)
def test_cute_sm100_nvint6_stochastic_matches_triton_error(
    round_style: RoundStyle,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"{round_style=} {triton_dist=} {cute_dist=}")
    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


def test_cute_sm100_nvint6_block_scale_2d_stochastic_matches_triton_error() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic,
        block_scale_2d=True,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(f"block_scale_2d=True round_style={RoundStyle.stochastic} {triton_dist=} {cute_dist=}")
    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


def test_cute_sm100_nvint6_block_scale_2d_stochastic_unbiased_matches_triton_error() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic_unbiased,
        block_scale_2d=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic_unbiased,
        block_scale_2d=True,
    )

    dequantized_triton = dequantize(
        quantize(x, config_triton),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantize(x, config_cute),
        dtype=torch.float32,
        backend=QuantizeBackend.cute_sm100,
        intermediate_dtype=torch.float32,
    )
    triton_dist = torch.dist(dequantized_triton, x.float())
    cute_dist = torch.dist(dequantized_cute, x.float())

    print(
        f"block_scale_2d=True round_style={RoundStyle.stochastic_unbiased} "
        f"{triton_dist=} {cute_dist=}",
    )
    assert cute_dist <= triton_dist + CUTE_DEQUANT_METRIC_TOLERANCE


def test_cute_sm100_nvint6_stochastic_unbiased_is_claimed() -> None:
    _require_cuda_for_cute_sm100_accuracy()

    x = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=DataType.nvint6,
        scale_rule=ScaleRule.static_6,
        round_style=RoundStyle.stochastic_unbiased,
    )

    assert AVAILABLE_BACKENDS[QuantizeBackend.cute_sm100].can_quantize(x, config)


@pytest.mark.parametrize("input_shape", [(128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [DataType.nvfp6_e2m3, DataType.nvfp6_e3m2])
def test_cute_sm100_nvfp6_static_matches_triton(
    input_shape: tuple[int, int],
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
    )

    quantized_triton = quantize(x, config_triton)
    quantized_cute = quantize(x, config_cute)
    triton_scales = _scale_factors_as_matrix(quantized_triton, x.shape, dtype)
    cute_scales = _scale_factors_as_matrix(quantized_cute, x.shape, dtype)

    dequantized_triton = dequantize(
        quantized_triton,
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    dequantized_cute = dequantize(
        quantized_cute,
        dtype=torch.float32,
        backend=QuantizeBackend.triton,
        intermediate_dtype=torch.float32,
    )
    diff = dequantized_cute - dequantized_triton

    assert torch.equal(cute_scales, triton_scales)
    if dtype == DataType.nvfp6_e2m3:
        assert torch.equal(quantized_cute.values, quantized_triton.values)
        assert (diff * diff).mean().item() == 0
    else:
        assert (
            quantized_cute.values == quantized_triton.values
        ).float().mean().item() >= CUTE_VALUE_EQUAL_RATIO_FLOOR
        assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
        assert diff.abs().mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE


@pytest.mark.parametrize("dtype", [DataType.nvfp6_e2m3, DataType.nvfp6_e3m2])
def test_cute_sm100_nvfp6_pseudo_quantize_matches_triton_error(
    dtype: DataType,
) -> None:
    _require_cuda_for_cute_sm100_accuracy()

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config_triton = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )
    config_cute = QuantizationConfig(
        backend=QuantizeBackend.cute_sm100,
        dtype=dtype,
        scale_rule=ScaleRule.static_6,
        pseudo_quantize=True,
    )

    pseudo_triton = quantize(x, config_triton)
    pseudo_cute = quantize(x, config_cute)
    diff = pseudo_cute.float() - pseudo_triton.float()
    triton_input_diff = pseudo_triton.float() - x.float()
    cute_input_diff = pseudo_cute.float() - x.float()

    assert isinstance(pseudo_cute, torch.Tensor)
    assert pseudo_cute.shape == x.shape
    assert pseudo_cute.dtype == x.dtype
    if dtype == DataType.nvfp6_e2m3:
        assert torch.equal(pseudo_cute, pseudo_triton)
    else:
        assert (diff * diff).mean().item() <= CUTE_DEQUANT_METRIC_TOLERANCE
        assert (cute_input_diff * cute_input_diff).mean().item() <= (
            (triton_input_diff * triton_input_diff).mean().item()
            + CUTE_DEQUANT_METRIC_TOLERANCE
        )


@pytest.mark.parametrize("input_type", ["zeros", "ones", "rand01", "randn"])
@pytest.mark.parametrize(
    "input_shape",
    [(1024, 1024), (1024, 512), (512, 1024)],
)
@pytest.mark.parametrize(
    ("backend_a", "kwargs_a", "backend_b", "kwargs_b"),
    [
        (a[0], a[1], b[0], b[1])
        for a, b in itertools.chain(
            itertools.combinations(
                [
                    (QuantizeBackend.cuda, {}),
                    (QuantizeBackend.triton, {}),
                    (QuantizeBackend.pytorch, {}),
                    (QuantizeBackend.transformer_engine, {}),
                ],
                r=2,
            ),
            [
                (
                    (QuantizeBackend.triton, {}),
                    (QuantizeBackend.triton, {"major_compute_capability": SM_120}),
                ),
                (
                    (QuantizeBackend.triton, {}),
                    (QuantizeBackend.triton, {"major_compute_capability": SM_100}),
                ),
                (
                    (QuantizeBackend.triton, {}),
                    (QuantizeBackend.triton, {"major_compute_capability": SM_80}),
                ),
            ],
        )
    ],
)
@pytest.mark.parametrize("block_scale_2d", ["block_scale_2d", "no_block_scale_2d"])
@pytest.mark.parametrize(
    "dtype",
    [
        DataType.if4,
        DataType.if6_e2m3,
        DataType.if6_e3m2,
        DataType.mxfp4,
        DataType.nvfp4,
        DataType.nvfp6_e2m3,
        DataType.nvfp6_e3m2,
        DataType.nvint4,
    ],
)
@pytest.mark.parametrize("rht", ["rht", "no_rht"])
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
@pytest.mark.parametrize("round_style", [RoundStyle.nearest])
@pytest.mark.parametrize("transpose", ["transpose", "no_transpose"])
def test_backend_outputs_are_consistent(  # noqa: C901, PLR0912, PLR0915
    input_type: str,
    input_shape: tuple[int, int],
    backend_a: QuantizeBackend,
    kwargs_a: dict[str, Any],
    backend_b: QuantizeBackend,
    kwargs_b: dict[str, Any],
    *,
    block_scale_2d: str,
    dtype: DataType,
    rht: str,
    round_style: RoundStyle,
    scale_rule: ScaleRule,
    transpose: str,
) -> None:
    _require_cuda()

    torch.set_printoptions(precision=10)

    if (
        kwargs_a.get("major_compute_capability") is not None
        or kwargs_b.get("major_compute_capability") is not None
    ) and torch.cuda.get_device_capability()[0] != SM_100:
        pytest.skip("Can only simulate different major compute capabilities on SM_100")

    if dtype in {
        DataType.nvfp6_e2m3,
        DataType.nvfp6_e3m2,
        DataType.if6_e2m3,
        DataType.if6_e3m2,
    } and (
        kwargs_a.get("major_compute_capability") is not None
        or kwargs_b.get("major_compute_capability") is not None
    ):
        pytest.skip("No simulation is allowed for NVFP6 right now")

    backend_a_cls = AVAILABLE_BACKENDS[backend_a]
    backend_b_cls = AVAILABLE_BACKENDS[backend_b]

    if not backend_a_cls.is_available() or not backend_b_cls.is_available():
        pytest.skip("Backend is not available")

    config_a = QuantizationConfig(
        backend=backend_a,
        block_scale_2d=block_scale_2d == "block_scale_2d",
        dtype=dtype,
        kwargs=kwargs_a,
        rht=rht == "rht",
        round_style=round_style,
        scale_rule=scale_rule,
        transpose=transpose == "transpose",
    )

    config_b = QuantizationConfig(
        backend=backend_b,
        block_scale_2d=block_scale_2d == "block_scale_2d",
        dtype=dtype,
        kwargs=kwargs_b,
        rht=rht == "rht",
        round_style=round_style,
        scale_rule=scale_rule,
        transpose=transpose == "transpose",
    )

    if round_style.is_stochastic:
        pytest.xfail("This test is not currently targeting stochastic rounding")

    for random_seed in range(NUM_RANDOM_SEEDS):
        print(f"Testing with random seed: {random_seed}")
        torch.manual_seed(random_seed)

        if input_type == "zeros":
            x = torch.zeros(*input_shape, dtype=torch.bfloat16, device="cuda")
        elif input_type == "ones":
            x = torch.ones(*input_shape, dtype=torch.bfloat16, device="cuda")
        elif input_type == "rand01":
            x = torch.randint(0, 2, input_shape, dtype=int, device="cuda").to(
                torch.bfloat16,
            )
        elif input_type == "randn":
            x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
        else:
            msg = f"Invalid input type: {input_type}"
            raise ValueError(msg)

        if not backend_a_cls.can_quantize(
            x,
            config_a,
        ) or not backend_b_cls.can_quantize(
            x,
            config_b,
        ):
            pytest.skip("Backend is not supported")

        quantized_a = quantize(x.clone(), config_a)
        quantized_b = quantize(x.clone(), config_b)

        if not torch.allclose(quantized_a.amax, quantized_b.amax):
            print("Backends A and B have different amax values!")
            print(f"{backend_a}: {quantized_a.amax}")
            print(f"{backend_b}: {quantized_b.amax}")
            pytest.fail("Backends A and B have different amax values!")

        if quantized_a.scale_factors_are_in_blackwell_layout:
            sf_a = from_blocked(
                quantized_a.scale_factors.bfloat16(),
                (input_shape[0], input_shape[1] // dtype.block_size),
            )
        else:
            sf_a = quantized_a.scale_factors.bfloat16().reshape(
                input_shape[0],
                input_shape[1] // dtype.block_size,
            )

        if quantized_b.scale_factors_are_in_blackwell_layout:
            sf_b = from_blocked(
                quantized_b.scale_factors.bfloat16(),
                (input_shape[0], input_shape[1] // dtype.block_size),
            )
        else:
            sf_b = quantized_b.scale_factors.bfloat16().reshape(
                input_shape[0],
                input_shape[1] // dtype.block_size,
            )

        quantized_parameters_per_group = (
            dtype.block_size // dtype.quantized_value_type.packing_factor
        )

        # When computing 4/6 with the MAE and MSE scale rules, computing the errors
        # requires summing the errors in each block of 16 values. This operation
        # differently (elements are summed in different orders, and floating-point
        # addition is not associative) in PyTorch and Triton, and can not be easily made
        # deterministic in a way that allows for good performance. As a result, we allow
        # a small number of mismatches between the scale factors and values for these
        # two rules. Fortunately, abs_max does not involve a summation, so we can use it
        # to test the correctness of the rest of the 4/6 implementation.
        scale_factors_mismatch_prop = (sf_a != sf_b).sum() / sf_a.numel()

        if (
            scale_rule in {ScaleRule.static_6, ScaleRule.static_4, ScaleRule.abs_max}
            and scale_factors_mismatch_prop > 0
        ) or scale_factors_mismatch_prop >= MAE_MSE_MISMATCH_TOLERANCE:
            print(
                "Backends A and B have different scale factors! "
                f"{scale_factors_mismatch_prop:.2%} mismatch",
            )

            [i, *_], [j, *_] = torch.where(sf_a != sf_b)
            print(i, j)
            print(backend_a)
            print("amax", quantized_a.amax)
            print("sf", sf_a[i, j])
            print(
                "e2m1",
                quantized_a.values[
                    i,
                    quantized_parameters_per_group
                    * j : quantized_parameters_per_group
                    * (j + 1),
                ],
            )
            print(backend_b)
            print("sf", sf_b[i, j])
            print(
                "e2m1",
                quantized_b.values[
                    i,
                    quantized_parameters_per_group
                    * j : quantized_parameters_per_group
                    * (j + 1),
                ],
            )
            print("original")
            print("x", x[i, dtype.block_size * j : dtype.block_size * (j + 1)])
            pytest.fail("Backends A and B have different scale factors!")

        values_mismatch_prop = (
            quantized_a.values != quantized_b.values
        ).sum() / quantized_a.values.numel()

        if (
            scale_rule in {ScaleRule.static_6, ScaleRule.static_4, ScaleRule.abs_max}
            and values_mismatch_prop > 0
        ) or values_mismatch_prop >= MAE_MSE_MISMATCH_TOLERANCE:
            print(
                "Backends A and B have different e2m1 values! "
                f"{values_mismatch_prop:.2%} mismatch",
            )

            [i, *_], [j, *_] = torch.where(
                quantized_a.values != quantized_b.values,
            )
            print(i, j)
            print("amax", quantized_a.amax)
            print("sf", sf_a[i, j // quantized_parameters_per_group])
            print(backend_a)
            print(
                "e2m1",
                quantized_a.values[
                    i,
                    quantized_parameters_per_group
                    * (
                        j // quantized_parameters_per_group
                    ) : quantized_parameters_per_group
                    * (j // quantized_parameters_per_group + 1),
                ],
            )
            print(backend_b)
            print(
                "e2m1",
                quantized_b.values[
                    i,
                    quantized_parameters_per_group
                    * (
                        j // quantized_parameters_per_group
                    ) : quantized_parameters_per_group
                    * (j // quantized_parameters_per_group + 1),
                ],
            )
            print("original")
            print(
                "x",
                x[
                    i,
                    dtype.block_size
                    * (j // quantized_parameters_per_group) : dtype.block_size
                    * (j // quantized_parameters_per_group + 1),
                ],
            )
            pytest.fail("Backends A and B have different e2m1 values!")


def test_stochastic_rounding() -> None:
    _require_cuda()

    test_cases = [
        ({"backend": "transformer_engine", "scale_rule": "static_6"}, 141, None),
        ({"backend": "triton", "scale_rule": "static_6"}, 137, 158),
        ({"backend": "triton", "scale_rule": "mse"}, 122, 139),
        ({"backend": "triton", "dtype": "nvint4", "scale_rule": "static_6"}, 125, 135),
        ({"backend": "triton", "dtype": "if4", "scale_rule": "mse"}, 110, 122),
        (
            {
                "backend": "triton",
                "scale_rule": "static_6",
                "kwargs": {"major_compute_capability": SM_120},
            },
            141,
            158,
        ),
        (
            {
                "backend": "triton",
                "scale_rule": "mse",
                "kwargs": {"major_compute_capability": SM_120},
            },
            126,
            139,
        ),
        (
            {
                "backend": "triton",
                "scale_rule": "static_6",
                "kwargs": {"major_compute_capability": SM_80},
            },
            141,
            158,
        ),
        (
            {
                "backend": "triton",
                "scale_rule": "mse",
                "kwargs": {"major_compute_capability": SM_80},
            },
            126,
            139,
        ),
        ({"backend": "pytorch", "scale_rule": "static_6"}, 137, 158),
        ({"backend": "pytorch", "scale_rule": "mse"}, 122, 139),
    ]

    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")

    for kwargs, expected_dist, expected_dist_unbiased in test_cases:
        config = QuantizationConfig(round_style=RoundStyle.stochastic, **kwargs)
        out = quantize(x, config)
        x_dequantized = dequantize(out)
        dist = torch.dist(x_dequantized, x)
        print(kwargs, expected_dist, dist)
        assert abs((dist - expected_dist) / expected_dist) < 0.05  # noqa: PLR2004

        if expected_dist_unbiased is None:
            continue

        config = QuantizationConfig(
            round_style=RoundStyle.stochastic_unbiased,
            **kwargs,
        )
        out = quantize(x, config)
        x_dequantized = dequantize(out)
        dist = torch.dist(x_dequantized, x)
        print(kwargs, expected_dist_unbiased, dist)
        assert (
            abs((dist - expected_dist_unbiased) / expected_dist_unbiased)
            < 0.05  # noqa: PLR2004
        )


@pytest.mark.parametrize(
    "input_shape",
    [(1024, 1024), (1024, 512), (512, 1024)],
)
@pytest.mark.parametrize(
    "dtype",
    [
        DataType.nvfp4,
        DataType.if4,
        DataType.mxfp4,
        DataType.nvint4,
    ],
)
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_pseudo_quantize(
    input_shape: tuple[int, int],
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    _require_cuda()

    if scale_rule not in dtype.supported_scale_rules:
        pytest.skip(f"Scale rule {scale_rule} not supported for dtype {dtype}")

    backend = QuantizeBackend.triton

    if not AVAILABLE_BACKENDS[backend].is_available():
        pytest.skip("Triton backend is not available")

    for _ in range(NUM_RANDOM_SEEDS):
        x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")

        config_roundtrip = QuantizationConfig(
            backend=backend,
            dtype=dtype,
            scale_rule=scale_rule,
        )

        if not AVAILABLE_BACKENDS[backend].can_quantize(x, config_roundtrip):
            pytest.skip("Backend does not support this configuration")

        expected = dequantize(
            quantize(x.clone(), config_roundtrip),
            dtype=x.dtype,
            intermediate_dtype=torch.float32,
        )

        config_pseudo = QuantizationConfig(
            backend=backend,
            dtype=dtype,
            scale_rule=scale_rule,
            pseudo_quantize=True,
        )

        result = quantize(x.clone(), config_pseudo)
        print(torch.dist(result, expected))

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.bfloat16
        assert result.shape == x.shape
        assert torch.equal(result, expected)
