## NVFP6 E3M2 2D 1024 profile

Workload:

`1024x1024 nvfp6_e3m2 static_6 block_scale_2d=True`.

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp6_2d_e3m2_1024_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype nvfp6_e3m2 --scale-rule static_6 \
  --shape 1024 --iters 3 --no-transpose --block-scale-2d

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp6_2d_e3m2_1024_cute_32cta \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype nvfp6_e3m2 --scale-rule static_6 \
  --shape 1024 --iters 3 --no-transpose --block-scale-2d
```

Summary:

- Triton launches four kernels per timed call: abs helper, reduce, copy helper,
  and `quantization_kernel`.
- Before the retune, CuTe launched reduce plus `Sm100NVFP6StaticQuantize2D`
  with block `256` and grid `16`; NCU reported only about `0.01` waves/SM for
  the CuTe quantize kernel.
- The retained CuTe retune uses block `32` and grid `128` for the 1024 row.
  The profiled CuTe quantize kernel moved from about `13.57-13.63 us` to
  `12.96-13.18 us`; total profiled GPU kernel duration moved from about
  `69.44 us` to `67.16 us` across three timed calls.
- The refreshed end-to-end benchmark has this row at `1.261x` and all NVFP6 2D
  rows at `1.242x-1.284x`.
