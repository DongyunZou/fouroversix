# NVFP4 transpose profile notes

This is a profiling note for the 4096x4096 `nvfp4 + static_6 +
transpose=True` workload where Triton is still faster than the current CuTe
sm100 path.

Commands used:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --page raw \
  --metrics gpu__time_duration.sum,gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,launch__block_size,launch__grid_size,launch__registers_per_thread \
  --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_transpose4096_cute \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype nvfp4 --scale-rule static_6 --shape 4096 --iters 3

HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --page raw \
  --metrics gpu__time_duration.sum,gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,launch__block_size,launch__grid_size,launch__registers_per_thread \
  --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvfp4_transpose4096_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype nvfp4 --scale-rule static_6 --shape 4096 --iters 3
```

Summary:

| Backend stage | Time | SM throughput | DRAM throughput | L2 throughput | Registers/thread | Grid | Block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CuTe `x.T.contiguous()` copy | 72.1-72.3 us | 25.6-25.7% | 6.56-6.60% | 62.3-62.9% | 20 | 32768 | 128 |
| CuTe amax reduction | 17.2-18.4 us | 39.4-42.2% | 23.9-25.6% | 16.0-17.2% | 28 | 592 | 512 |
| CuTe NVFP4 quantize kernel | 11.3-11.7 us | 44.1-45.8% | 37.5-38.9% | 25.3-26.3% | 29 | 1184 | 256 |
| Triton abs helper | 12.6 us | 19.2% | 35.1-35.2% | 29.6% | 32 | 16384 | 128 |
| Triton amax reduction | 18.0-18.3 us | 33.8-34.4% | 24.1-24.4% | 16.2-16.4% | 31 | 592 | 512 |
| Triton scalar cast helper | 4.2 us | ~0% | ~0.10% | 0.38-0.48% | 30 | 1 | 128 |
| Triton quantization kernel | 43.4-43.7 us | 49.2-49.6% | 10.0-10.1% | 10.8-10.9% | 52 | 2048 | 128 |

Interpretation:

- The retained CuTe NVFP4 static quantize kernel is already much faster than
  Triton's main quantization kernel on this workload. The CuTe kernel takes
  about `11-12 us`; Triton's quantization kernel takes about `43-44 us`.
- End-to-end CuTe is still slower because the current non-MX transpose path
  materializes `x.T.contiguous()` before launching the CuTe quantizer. That copy
  alone costs about `72 us`, dominating the CuTe path.
- Moving the global amax reduction before transpose was tested separately. It
  improved 4096x4096 NV transpose rows by only about 1% and regressed a
  1024x1024 NVFP4 transpose row, so it was reverted.
- The next useful implementation step for NV transpose is a real fused
  transpose-plus-quantize kernel, not a launch-parameter retune. It should load
  source tiles cooperatively, rearrange in registers or shared memory, compute
  global/static-scale metadata without a separate materialized transpose, and
  pack the output in the transposed layout.
