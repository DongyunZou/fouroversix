# IF3 pseudo abs_max 4096 profile

This is a profiling note for the remaining below-parity
`4096x4096 if3 + abs_max + pseudo_quantize=True` row.

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/if3_pseudo_absmax4096_cute \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype if3 --scale-rule abs_max --shape 4096 \
  --iters 3 --no-transpose --pseudo-quantize

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/if3_pseudo_absmax4096_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype if3 --scale-rule abs_max --shape 4096 \
  --iters 3 --no-transpose --pseudo-quantize
```

CuTe launches two kernels per profiled iteration:

```text
torch AbsMax reduce:                    17.98-18.40 us
Sm100IF3AdaptivePseudoQuantize:         48.77-49.22 us
```

Triton launches four kernels per profiled iteration:

```text
torch abs elementwise:                  12.58-12.86 us
torch max reduce:                       17.70-18.30 us
scalar copy/cast:                        4.19-4.32 us
pseudo_quantization_kernel:             33.57-34.30 us
```

The total frontend-visible time is nearly tied, matching the retained benchmark
row around `0.98x`. The CuTe path already has fewer kernels than Triton, so the
remaining gap is not a Python fallback or extra launch problem. The CuTe IF3
pseudo kernel body is roughly `15 us` slower than Triton's pseudo kernel and is
compute-heavy (`~80%` SM throughput, `~9%` memory throughput), so closing this
row needs reducing the FP3/INT3 candidate/error work inside
`Sm100IF3AdaptivePseudoQuantize`.
