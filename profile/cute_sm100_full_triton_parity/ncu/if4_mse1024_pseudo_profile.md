# IF4 MSE pseudo 1024x1024 profile

This profiles a representative pseudo row below the old 1.2x target but above
Triton under the relaxed pseudo target:

`1024x1024 if4 mse pseudo_quantize=True`

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite --page raw \
  -o profile/cute_sm100_full_triton_parity/ncu/if4_mse1024_pseudo_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype if4 --scale-rule mse --shape 1024 --iters 5 \
  --no-transpose --pseudo-quantize

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite --page raw \
  -o profile/cute_sm100_full_triton_parity/ncu/if4_mse1024_pseudo_cute \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype if4 --scale-rule mse --shape 1024 --iters 5 \
  --no-transpose --pseudo-quantize
```

Reports:

- `if4_mse1024_pseudo_triton.ncu-rep`
- `if4_mse1024_pseudo_cute.ncu-rep`
- `if4_mse1024_pseudo_triton_raw.csv`
- `if4_mse1024_pseudo_cute_raw.csv`

NCU raw kernel-duration summary over five profiled iterations:

| Backend | Kernel group | Count | Sum us | Median us | Grid | Block | Regs/thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| Triton | `pseudo_quantization_kernel` | 5 | 157.60 | 31.52 | 128 | 128 | 40 |
| Triton | ATen reduce | 5 | 46.24 | 9.22 | 128 | 512 | 31 |
| Triton | ATen vectorized elementwise | 5 | 21.41 | 4.26 | 1024 | 128 | 32 |
| Triton | ATen unrolled elementwise | 5 | 20.70 | 4.10 | 1 | 128 | 30 |
| CuTe | ATen reduce | 5 | 45.50 | 8.99 | 128 | 512 | 28 |
| CuTe | `Sm100IF4AdaptivePseudoQuantize` | 5 | 30.24 | 6.05 | 512 | 128 | 64 |

Interpretation:

- Triton launches four GPU kernel groups per pseudo call; CuTe launches the
  same reduction plus one fused CuTe pseudo kernel.
- On the profiled GPU work, CuTe is much faster: about `15.15 us/iter` summed
  kernel time versus Triton's `49.19 us/iter`.
- The retained benchmark row is only moderately above Triton (`~1.17x`) because
  this pseudo workload is now dominated by fixed frontend/launch overhead rather
  than the CuTe kernel body.
- There is no evidence from this profile that Triton's IF4 MSE pseudo kernel
  body is doing something better than CuTe. The remaining old-1.2x gap is not a
  kernel-throughput gap.
