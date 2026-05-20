# IF3 2D profile notes

This is a profiling note for the 4096x4096 `if3 + abs_max +
block_scale_2d=True` workload where Triton is still faster than the current
CuTe sm100 path. The current benchmark snapshot reports this row at
`0.889x` versus Triton:

```text
Triton: 0.0878 ms
CuTe:   0.0988 ms
```

Commands used:

```bash
PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all --set full \
  --kernel-name 'regex:.*(if3|IF3|quant|Quant).*' \
  --launch-skip 10 --launch-count 3 \
  --export profile/cute_sm100_full_triton_parity/ncu/if3_2d_cute \
  --force-overwrite \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend cute_sm100 --dtype if3 --scale-rule abs_max --shape 4096 \
  --iters 3 --no-transpose --block-scale-2d

PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all --set full \
  --kernel-name 'regex:.*(quant|Quant).*' \
  --launch-skip 10 --launch-count 3 \
  --export profile/cute_sm100_full_triton_parity/ncu/if3_2d_triton \
  --force-overwrite \
  .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py \
  --backend triton --dtype if3 --scale-rule abs_max --shape 4096 \
  --iters 3 --no-transpose --block-scale-2d

ncu --import profile/cute_sm100_full_triton_parity/ncu/if3_2d_cute.ncu-rep \
  --csv --page raw > profile/cute_sm100_full_triton_parity/ncu/if3_2d_cute.csv
ncu --import profile/cute_sm100_full_triton_parity/ncu/if3_2d_triton.ncu-rep \
  --csv --page raw > profile/cute_sm100_full_triton_parity/ncu/if3_2d_triton.csv
```

Summary:

| Backend | Main kernel time | SM throughput | DRAM throughput | L2 throughput | Active warps | Registers/thread | Grid | Block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | 51.1-51.4 us | 62.3-62.6% | 8.53-8.58% | 5.67-5.90% | 40.3-40.6% | 64 | 2048 | 128 |
| CuTe | 71.5-72.0 us | 71.8-72.1% | 6.10-6.15% | 4.66-4.70% | 20.5-20.7% | 50 | 256 | 256 |

Interpretation:

- The CuTe kernel is not DRAM-bound on this workload. It uses less DRAM and L2
  throughput than Triton but still takes about 1.4x longer in the profiled main
  kernel.
- The retained CuTe IF3 2D mapping launches only 256 CTAs for the 4096x4096
  case. Triton launches 2048 CTAs and keeps roughly twice as many active warps.
- The current CuTe kernel assigns one thread to one 16x16 scale tile. That
  thread scans the tile once for max, scans it again for FP3-vs-INT3 error, and
  scans it a third time to write the chosen values. This gives simple
  semantics but serializes too much tile work inside one thread.
- This is not a transpose-materialization issue and is unlikely to be fixed by
  launch-parameter tuning alone. The likely fix is a warp- or CTA-cooperative
  IF3/IF3_BS8 2D kernel that computes the 16x16 tile max and candidate errors
  cooperatively, keeps the row data in registers/shared memory where practical,
  and exposes more CTAs/warps like Triton's tiled decomposition.
