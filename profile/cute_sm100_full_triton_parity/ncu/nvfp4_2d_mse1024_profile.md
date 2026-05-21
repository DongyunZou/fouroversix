# NVFP4 2D MSE 1024 profile

This profiles a representative remaining 2D near-threshold row:
`1024x1024 nvfp4 mse block_scale_2d=True`.

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_2d_mse1024_cute \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype nvfp4 --scale-rule mse --shape 1024 \
  --iters 5 --no-transpose --block-scale-2d

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_2d_mse1024_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype nvfp4 --scale-rule mse --shape 1024 \
  --iters 5 --no-transpose --block-scale-2d
```

Reports:

- `nvfp4_2d_mse1024_cute.ncu-rep`
- `nvfp4_2d_mse1024_triton.ncu-rep`
- `nvfp4_2d_mse1024_cute.csv`
- `nvfp4_2d_mse1024_triton.csv`

NCU kernel-duration summary:

```text
CuTe, per profiled iteration:
  torch AbsMax reduce:                    about 9-10 us
  Sm100NVFP4AdaptiveQuantize2D:            about 36-37 us

Triton, per profiled iteration:
  torch abs elementwise:                   about 4-5 us
  torch max reduce:                        about 9 us
  scalar copy/cast:                        about 4 us
  quantization_kernel:                     about 36 us
```

The summed profiled GPU kernel durations are roughly `45-47 us/iter` for CuTe
and `53-54 us/iter` for Triton. The retained CUDA-event benchmark row is still
near threshold rather than comfortably above target, so this gap is not simply
the CuTe 2D kernel body being slower than Triton. Like several small/medium
pseudo rows, enqueue spacing and fixed framework overhead around short kernels
are significant. A retained improvement likely needs either fusing/reducing the
amax path or a larger structural 2D kernel change, not just CTA-size retuning.
