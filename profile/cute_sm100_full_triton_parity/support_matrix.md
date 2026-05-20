# CuTe sm100 vs Triton support matrix

Generated from current `can_quantize` predicates. This is a gap report, not a
completion claim.

- torch.cuda.is_available: `True`
- device: `NVIDIA B200` capability `(10, 0)`
- cute_sm100 available: `True`
- triton available: `True`

## Test workload dtype/rule coverage

| dtype | supported rules in dtype | Triton nearest 1D | CuTe nearest 1D | missing |
|---|---|---:|---:|---|
| `if3` | abs_max, mae, mse | 3 | 3 | - |
| `if3_bs8` | abs_max, mae, mse | 3 | 3 | - |
| `if4` | abs_max, mae, mse | 3 | 3 | - |
| `if4_bs8` | abs_max, mae, mse | 3 | 3 | - |
| `if6_e2m3` | abs_max, mae, mse | 3 | 3 | - |
| `if6_e3m2` | abs_max, mae, mse | 3 | 3 | - |
| `mxfp3` | static_4, static_6 | 2 | 2 | - |
| `mxfp3_bs8` | static_4, static_6 | 2 | 2 | - |
| `mxfp4` | static_4, static_6 | 2 | 2 | - |
| `mxfp4_bs8` | static_4, static_6 | 2 | 2 | - |
| `mxfp6_e2m3` | static_4, static_6 | 2 | 2 | - |
| `mxfp6_e3m2` | static_4, static_6 | 2 | 2 | - |
| `nvfp4` | abs_max, mae, mse, static_4, static_6 | 5 | 5 | - |
| `nvfp4_bs8` | static_4, static_6 | 2 | 2 | - |
| `nvfp3` | static_6 | 1 | 1 | - |
| `nvfp3_bs8` | static_6 | 1 | 1 | - |
| `nvfp6_e2m3` | static_6 | 1 | 1 | - |
| `nvfp6_e3m2` | static_6 | 1 | 1 | - |
| `nvint3` | static_6 | 1 | 1 | - |
| `nvint3_bs8` | static_6 | 1 | 1 | - |
| `nvint4` | static_6 | 1 | 1 | - |
| `nvint4_bs8` | static_6 | 1 | 1 | - |
| `nvint6` | static_6 | 1 | 1 | - |

## Feature flags

| feature | Triton | CuTe sm100 current |
|---|---|---|
| `block_scale_2d` | `True` | `nvfp4`, static NVFP4_BS8, adaptive IF3/IF3_BS8/IF4/IF4_BS8, static MXFP3/MXFP3_BS8/MXFP4/MXFP4_BS8/MXFP6/NVINT3/NVINT3_BS8/NVINT4/NVINT4_BS8/NVINT6/NVFP6 |
| `transpose` | `True` | `True` |
| `rht` | `True` | `True` (CuTe RHT pre-transform + CuTe quantize) |
| `pseudo_quantize` | `True` | `True` for supported quantize configs; IF3/IF3_BS8/IF4/IF4_BS8/IF6 adaptive nearest 1D, NVFP3/NVFP3_BS8/NVFP4/NVFP4_BS8/NVFP6/NVINT3/NVINT3_BS8/NVINT4/NVINT4_BS8/NVINT6 nearest 1D, and static MXFP3/MXFP3_BS8/MXFP4/MXFP4_BS8/MXFP6 nearest 1D are fused; NVFP4 block-scale-2D/stochastic-unbiased and other dtypes use CuTe quantize plus CuTe/backend dequantize |
| `x_amax` kwarg | `True` | NV/IF formats with global amax honor a provided precomputed amax; MX formats do not use global amax |

## Rounding

| round_style | Triton nvfp4/mse | CuTe sm100 nvfp4/mse |
|---|---:|---:|
| `nearest` | `True` | `True` |
| `stochastic` | `True` | 1D NVFP4/IF3/IF3_BS8/IF4/IF4_BS8/IF6; static NVFP4_BS8/MXFP3/MXFP3_BS8/MXFP4/MXFP4_BS8/MXFP6/NVFP3/NVFP3_BS8/NVINT3/NVINT3_BS8/NVINT4/NVINT4_BS8/NVINT6/NVFP6; 2D NVFP4/IF3/IF3_BS8/IF4 and static MXFP3/MXFP3_BS8/MXFP4/MXFP4_BS8/NVINT4/NVFP6 |
| `stochastic_unbiased` | `True` | NVFP4/IF4/IF4_BS8; NVFP4 pseudo via quantize+dequantize; static `nvfp4_bs8`, `mxfp3`, `mxfp3_bs8`, `mxfp4`, `mxfp4_bs8`, `mxfp6_*`, `nvint4`, `nvint4_bs8`, and 1D `nvfp6_e3m2`; 2D NVFP4/IF4 and static `nvfp4_bs8`/`mxfp3`/`mxfp3_bs8`/`mxfp4`/`mxfp4_bs8`/`nvint4`/NVFP6 |

## Dequantize Values

| dtype family | CuTe sm100 current |
|---|---|
| `if3`, `if3_bs8`, `nvfp3`, `nvfp3_bs8`, `nvfp4`, `nvfp4_bs8`, `if4`, `if4_bs8`, `mxfp3`, `mxfp3_bs8`, `mxfp4`, `mxfp4_bs8`, `nvint3`, `nvint3_bs8`, `nvint4`, `nvint4_bs8`, `nvint6` | `True` through backend raw-value decode dispatch |
| FP6/IF6 formats, including MXFP6 | `True` through backend raw-value decode dispatch |
