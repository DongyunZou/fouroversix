# NVFP4 static_4 2D 4096 profile

Workload:

`4096x4096 nvfp4 static_4 block_scale_2d=True`.

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_static4_2d_4096_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype nvfp4 --scale-rule static_4 \
  --shape 4096 --iters 3 --no-transpose --block-scale-2d

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_static4_2d_4096_cute \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype nvfp4 --scale-rule static_4 \
  --shape 4096 --iters 3 --no-transpose --block-scale-2d
```

Summary:

- Triton launches four kernels per timed call: abs helper, reduce, scalar copy,
  and `quantization_kernel`.
- CuTe launches two kernels per timed call: torch absmax reduce and
  `Sm100NVFP4StaticQuantize2D`.
- Triton's profiled GPU kernel-duration sum is about `178.1 us` across three
  timed calls, or about `59.4 us/call`.
- CuTe's profiled GPU kernel-duration sum is about `106.5 us` across three
  timed calls, or about `35.5 us/call`.
- Triton's main `quantization_kernel` is about `24.4-24.8 us`, grid
  `(32,64,1)`, block `128`, with about `45%` achieved occupancy.
- CuTe's main `Sm100NVFP4StaticQuantize2D` kernel is about `17.0-17.5 us`,
  grid `256`, block `256`, with about `20-21%` achieved occupancy.

Interpretation:

The CuTe 2D static kernel body is faster than Triton's main quantization
kernel on this row, and CuTe launches fewer kernels. The narrow end-to-end
benchmark margin is mostly fixed Python/frontend/launch overhead around very
short kernels, not a Triton kernel-body advantage. A simple CTA-size retune was
tested separately: 128-thread and 64-thread CTAs made the full benchmark's
lowest NVFP4 2D rows worse, so the retained path remains the 256-thread CTA
mapping.
