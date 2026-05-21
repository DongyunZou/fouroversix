## NVFP4 adaptive 1024 base profile

Workload:

`1024x1024 nvfp4 abs_max`, no transpose, no pseudo, no block-scale-2d.

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_absmax1024_base_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype nvfp4 --scale-rule abs_max \
  --shape 1024 --iters 3 --no-transpose

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_absmax1024_base_cute \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype nvfp4 --scale-rule abs_max \
  --shape 1024 --iters 3 --no-transpose
```

Summary:

- Triton launches four kernels per timed call: abs helper (`~4.3 us`), reduce
  (`~9.8-10.1 us`), copy helper (`~4.3 us`), and `quantization_kernel`
  (`~32.0-32.3 us`, block `128`, grid `8x16`).
- CuTe launches reduce (`~9.6-9.9 us`) plus
  `Sm100NVFP4AdaptiveQuantize` (`~6.4-6.6 us`, block `128`, grid `256`).
- Across three profiled calls, summed GPU kernel duration is about `151.75 us`
  for Triton versus `48.62 us` for CuTe.
- The narrow end-to-end benchmark margin around `1.2x` for this class is not
  evidence that Triton's main kernel is stronger; NCU shows CuTe's kernel body
  is much faster and the remaining margin is dominated by fixed frontend,
  launch, and reduction overhead.
