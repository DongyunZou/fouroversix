# IF4 MAE 128x256 pseudo NCU profile

This profiles the supported benchmark row closest to the current 1.2x target:
`shape=(128, 256), dtype=if4, scale_rule=mae, pseudo_quantize=True`.

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 /usr/local/cuda-13.2/bin/ncu \
  --target-processes all --profile-from-start off --set full \
  --import-source no --force-overwrite \
  --export profile/cute_sm100_full_triton_parity/ncu/if4_mae128_pseudo_triton \
  .venv/bin/python -c "..."

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 /usr/local/cuda-13.2/bin/ncu \
  --target-processes all --profile-from-start off --set full \
  --import-source no --force-overwrite \
  --export profile/cute_sm100_full_triton_parity/ncu/if4_mae128_pseudo_cute \
  .venv/bin/python -c "..."
```

Artifacts:

- `if4_mae128_pseudo_triton.ncu-rep`
- `if4_mae128_pseudo_triton.csv`
- `if4_mae128_pseudo_cute.ncu-rep`
- `if4_mae128_pseudo_cute.csv`

The profiler captured three timed calls after warmup. Per call, Triton launches
four kernels:

| kernel | duration | grid | block | achieved occupancy |
| --- | ---: | ---: | ---: | ---: |
| vectorized helper | `3.87-4.03 us` | 32 | 128 | `5.75-7.03%` |
| reduce | `12.19-12.35 us` | 1 | 512 | `24.71-25.61%` |
| unrolled helper | `3.94-4.29 us` | 1 | 128 | `4.39-5.08%` |
| `pseudo_quantization_kernel` | `30.91-31.46 us` | 4 | 128 | `6.25-6.29%` |

CuTe launches two kernels per call:

| kernel | duration | grid | block | achieved occupancy |
| --- | ---: | ---: | ---: | ---: |
| reduce | `12.19-12.45 us` | 1 | 512 | `23.89-25.35%` |
| `Sm100IF4AdaptivePseudoQuantize` | `5.12-5.28 us` | 16 | 128 | `6.12-6.23%` |

The captured kernel-duration sum across the three timed calls is about
`154.46 us` for Triton and `52.67 us` for CuTe. The relevant main kernel is not
a Triton advantage: CuTe's fused pseudo kernel is roughly `6x` shorter than
Triton's `pseudo_quantization_kernel` on this small row. The remaining benchmark
headroom is fixed launch/reduction overhead and tiny-grid underutilization;
NCU flags both paths with very low waves per SM (`0.00-0.01`) and low achieved
occupancy for the main pseudo kernels.
