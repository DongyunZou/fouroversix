import torch
from fouroversix.kernels.constants import (
    IF3_INT_EXPANSION_FACTOR,
    IF6_E2M3_INT_EXPANSION_FACTOR,
    IF6_E3M2_INT_EXPANSION_FACTOR,
)


def from_blocked(a: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = orig_shape
    return (
        a.view(-1, 32, 4, 4)
        .transpose(1, 2)
        .reshape(-1, cols // 4, 128, 4)
        .transpose(1, 2)
        .reshape(rows, cols)
    )


def convert_e2m1_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    sign = (x >> 3) & 0x1
    exponent = (x >> 1) & 0x3
    mantissa = x & 0x1

    # Make adjustments
    new_exponent = torch.where(
        (exponent == 0) & (mantissa == 0),
        0,
        (exponent + 6) & 0xF,
    )
    new_mantissa = torch.where(exponent == 0, 0, mantissa << 2)

    return ((sign << 7) | (new_exponent << 3) | new_mantissa).view(torch.float8_e4m3fn)


def unpack_packed_fp4(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if to_dtype == torch.float8_e4m3fn:
        convert_function = convert_e2m1_to_fp8_e4m3
    else:
        msg = f"Unsupported dtype: {to_dtype}"
        raise ValueError(msg)

    high = (x >> 4) & 0xF
    low = x & 0xF

    return torch.stack(
        [convert_function(low), convert_function(high)],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2)


def unpack_packed_if4(
    x: torch.Tensor,
    scale_factors: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
    block_size: int = 16,
) -> torch.Tensor:
    high = (x >> 4) & 0xF
    low = x & 0xF

    x_unpacked = torch.stack(
        [low, high],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2 // block_size, block_size)

    return torch.where(
        (scale_factors.view(torch.uint8) >= 128).unsqueeze(2),  # noqa: PLR2004
        ((x_unpacked.to(torch.int8) << 4) >> 4).to(to_dtype) * (6 / 7),
        convert_e2m1_to_fp8_e4m3(x_unpacked).to(to_dtype),
    ).reshape(x.shape[0], x.shape[1] * 2)


def unpack_packed_int4(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    high = (x >> 4) & 0xF
    low = x & 0xF

    x_unpacked = torch.stack(
        [low, high],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2 // 16, 16)

    return (
        ((x_unpacked.to(torch.int8) << 4) >> 4)
        .to(to_dtype)
        .reshape(x.shape[0], x.shape[1] * 2)
    )


def convert_e2m0_to_float(x: torch.Tensor) -> torch.Tensor:
    sign = torch.where(((x >> 2) & 0x1) == 0, 1.0, -1.0)
    code = x & 0x3
    value = torch.where(code == 3, 4, code).to(torch.float32)
    return value * sign


def unpack_if3(
    x: torch.Tensor,
    scale_factors: torch.Tensor,
    to_dtype: torch.dtype = torch.float16,
    block_size: int = 16,
) -> torch.Tensor:
    fp_values = convert_e2m0_to_float(x)
    magnitude = (x & 0x3).to(torch.int8)
    sign = ((x >> 2) & 0x1).to(torch.int8)
    int_values = (magnitude * (1 - 2 * sign)).to(torch.float32)
    int_values = int_values.reshape(x.shape[0], x.shape[1] // block_size, block_size)
    fp_values = fp_values.reshape(x.shape[0], x.shape[1] // block_size, block_size)

    return torch.where(
        (scale_factors.view(torch.uint8) >= 128).unsqueeze(2),  # noqa: PLR2004
        int_values * IF3_INT_EXPANSION_FACTOR,
        fp_values,
    ).reshape_as(x).to(to_dtype)


def convert_e2m3_to_float(x: torch.Tensor) -> torch.Tensor:
    sign = (x >> 5) & 0x1
    exponent = (x >> 3) & 0x3
    mantissa = x & 0x7

    x_float = x.to(torch.float32)
    sign_float = torch.where(sign == 0, 1.0, -1.0)
    exponent_float = exponent.to(torch.float32)
    mantissa_float = mantissa.to(torch.float32)

    normal = (1.0 + mantissa_float / 8.0) * torch.pow(2.0, exponent_float - 1.0)
    subnormal = mantissa_float / 8.0
    value = torch.where(exponent == 0, subnormal, normal)
    value = torch.where(x == 0, x_float, value)
    return value * sign_float


def convert_e3m2_to_float(x: torch.Tensor) -> torch.Tensor:
    sign = (x >> 5) & 0x1
    exponent = (x >> 2) & 0x7
    mantissa = x & 0x3

    x_float = x.to(torch.float32)
    sign_float = torch.where(sign == 0, 1.0, -1.0)
    exponent_float = exponent.to(torch.float32)
    mantissa_float = mantissa.to(torch.float32)

    normal = (1.0 + mantissa_float / 4.0) * torch.pow(2.0, exponent_float - 3.0)
    subnormal = mantissa_float / 16.0
    value = torch.where(exponent == 0, subnormal, normal)
    value = torch.where(x == 0, x_float, value)
    return value * sign_float


def unpack_fp6_e2m3(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return convert_e2m3_to_float(x & 0x3F).to(to_dtype)


def unpack_fp6_e3m2(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return convert_e3m2_to_float(x & 0x3F).to(to_dtype)


def unpack_if6_e2m3(
    x: torch.Tensor,
    scale_factors: torch.Tensor,
    to_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    fp_values = unpack_fp6_e2m3(x, torch.float32)
    magnitude = (x & 0x1F).to(torch.int8)
    sign = ((x >> 5) & 0x1).to(torch.int8)
    int_values = (magnitude * (1 - 2 * sign)).to(torch.float32)
    int_values = int_values.reshape(x.shape[0], x.shape[1] // 16, 16)
    fp_values = fp_values.reshape(x.shape[0], x.shape[1] // 16, 16)

    factor = torch.tensor(
        IF6_E2M3_INT_EXPANSION_FACTOR,
        device=x.device,
        dtype=torch.float16,
    )

    return torch.where(
        (scale_factors.view(torch.uint8) >= 128).unsqueeze(2),  # noqa: PLR2004
        int_values.to(torch.float16) * factor,
        fp_values,
    ).reshape_as(x).to(torch.float16).to(to_dtype)


def unpack_if6_e3m2(
    x: torch.Tensor,
    scale_factors: torch.Tensor,
    to_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    fp_values = unpack_fp6_e3m2(x, torch.float32)
    magnitude = (x & 0x1F).to(torch.int8)
    sign = ((x >> 5) & 0x1).to(torch.int8)
    int_values = (magnitude * (1 - 2 * sign)).to(torch.float32)
    int_values = int_values.reshape(x.shape[0], x.shape[1] // 16, 16)
    fp_values = fp_values.reshape(x.shape[0], x.shape[1] // 16, 16)

    factor = torch.tensor(
        IF6_E3M2_INT_EXPANSION_FACTOR,
        device=x.device,
        dtype=torch.float16,
    )

    return torch.where(
        (scale_factors.view(torch.uint8) >= 128).unsqueeze(2),  # noqa: PLR2004
        int_values.to(torch.float16) * factor,
        fp_values,
    ).reshape_as(x).to(torch.float16).to(to_dtype)
