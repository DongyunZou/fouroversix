# NVFP4 abs_max 1024 profile

This is a profiling note for the remaining near-threshold ordinary workload
`1024x1024 nvfp4 + abs_max`.

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 /usr/local/cuda-13.2/bin/ncu \
  --target-processes all --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_absmax1024_cute \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype nvfp4 --scale-rule abs_max --shape 1024 \
  --iters 3 --no-transpose

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 /usr/local/cuda-13.2/bin/ncu \
  --target-processes all --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_absmax1024_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype nvfp4 --scale-rule abs_max --shape 1024 \
  --iters 3 --no-transpose
```

CuTe launches two GPU kernels per profiled iteration:

```text
torch AbsMax reduce:                    9.25-9.47 us
Sm100NVFP4AdaptiveQuantize:             5.95-6.02 us
```

Triton launches four GPU kernels per profiled iteration:

```text
torch abs elementwise:                  4.42-4.45 us
torch max reduce:                       8.99-9.28 us
scalar copy/cast:                       4.03-4.06 us
quantization_kernel:                   32.26-32.38 us
```

The summed profiled GPU kernel durations strongly favor CuTe
(`~15.2 us/iter`) over Triton (`~49.7-50.2 us/iter`). The retained benchmark
row is only near threshold (`1.125x`, `0.05095 ms` CuTe vs `0.05735 ms`
Triton), so this ordinary NVFP4 miss is not explained by a slower CuTe kernel
body. Like the NVFP4 2D profile, fixed frontend/enqueue spacing dominates the
short public `quantize()` measurement.
