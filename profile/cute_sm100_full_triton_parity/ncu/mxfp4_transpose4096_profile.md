# MXFP4 transpose profile notes

This is a profiling note for the 4096x4096 `mxfp4 + static_6 +
transpose=True` workload where Triton is still faster than the current CuTe
sm100 fused scalar-transpose path.

Commands used:

```bash
HOME=/tmp ncu --target-processes all --profile-from-start off --csv \
  --page raw \
  --metrics gpu__time_duration.sum,gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,launch__block_size,launch__grid_size,launch__registers_per_thread \
  --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/mxfp4_transpose4096_cute \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype mxfp4 --scale-rule static_6 --shape 4096 --iters 3

HOME=/tmp ncu --target-processes all --profile-from-start off --csv \
  --page raw \
  --metrics gpu__time_duration.sum,gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,launch__block_size,launch__grid_size,launch__registers_per_thread \
  --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/mxfp4_transpose4096_triton \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype mxfp4 --scale-rule static_6 --shape 4096 --iters 3
```

Summary:

| Backend | Main kernel time | SM throughput | DRAM throughput | L2 throughput | Registers/thread | Grid | Block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | 47.2-49.4 us | 43.6-45.9% | 8.87-9.29% | 5.94-6.22% | 96 | 1024 | 128 |
| CuTe | 66.0-66.4 us | 7.94-7.98% | 6.60-6.64% | 57.3-57.9% | 50 | 1184 | 256 |

Interpretation:

- The retained CuTe transpose path avoids `x.T.contiguous()`, but it is still a
  scalar strided-load kernel. It has much lower SM throughput than Triton's
  tiled quantization kernel and high L2 activity, so it is not exposing enough
  useful parallel work per memory transaction.
- This is not a small launch-parameter problem: the CuTe kernel already runs
  fewer registers per thread than Triton but still spends about 1.35x-1.40x
  longer in the main kernel.
- The likely fix is a real tiled transpose-plus-quantize kernel that
  cooperatively loads a transposed tile, stages or rearranges data with shared
  memory/register tiling, and then performs vectorized per-scale-block packing.
  The current one-thread-per-output-block scalar transpose path is mainly a
  semantic bridge, not the final performance shape.
