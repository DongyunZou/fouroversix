# MXFP3 2D block-scale profile

Workload:

```text
4096x4096 mxfp3 static_6 block_scale_2d=True transpose=False pseudo_quantize=False
```

Commands:

```bash
env PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all --set full --import-source no --force-overwrite --export profile/cute_sm100_full_triton_parity/ncu/mxfp3_2d_triton .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py --backend triton --dtype mxfp3 --scale-rule static_6 --shape 4096 --iters 3 --no-transpose --block-scale-2d
env PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all --set full --import-source no --force-overwrite --export profile/cute_sm100_full_triton_parity/ncu/mxfp3_2d_cute .venv/bin/python profile/cute_sm100_full_triton_parity/profile_transpose_kernel.py --backend cute_sm100 --dtype mxfp3 --scale-rule static_6 --shape 4096 --iters 3 --no-transpose --block-scale-2d
ncu --csv --page details --import profile/cute_sm100_full_triton_parity/ncu/mxfp3_2d_triton.ncu-rep > profile/cute_sm100_full_triton_parity/ncu/mxfp3_2d_triton.csv
ncu --csv --page details --import profile/cute_sm100_full_triton_parity/ncu/mxfp3_2d_cute.ncu-rep > profile/cute_sm100_full_triton_parity/ncu/mxfp3_2d_cute.csv
```

Representative median profiled kernels:

| Backend | Kernel | Duration | Grid | Block | Waves/SM | Achieved occupancy | Active warps/SM | Registers/thread | SM throughput | Memory throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `quantization_kernel` | 34.40 us | 1024 | 128 | 1.38 | 24.01% | 15.37 | 90 | 54.87% | 977.16 GB/s |
| CuTe | `Sm100MXFP3StaticQuantize2D` | 77.82 us | 64 | 256 | 0.09 | 12.47% | 7.98 | 44 | 15.54% | 433.45 GB/s |

Observed CuTe bottlenecks:

- The current CuTe 2D block-scale kernel launches only 64 CTAs for the
  4096x4096 workload, so NCU reports only 0.09 full waves across all SMs.
- Theoretical occupancy is 62.50%, but achieved occupancy is only about 12.5%
  because the grid is too small to keep the device resident.
- NCU reports all compute pipelines under-utilized for the CuTe kernel.
- CuTe global accesses are still partly uncoalesced: the report shows
  2,621,440 excessive sectors, 50% of 5,259,264 total sectors, with 16.0/32
  byte average sector utilization for loads and about 16.2/32 for stores.
- Triton has some uncoalesced global-load warnings too, but its 1024 CTA grid
  and higher active warp count hide more of the cost and reach roughly 2.26x
  the CuTe kernel memory throughput in this profile.

Interpretation:

The gap is not a simple launch-parameter issue. The current CuTe mapping is too
coarse for large 2D block-scale static MXFP3 rows: one CTA covers too much work,
leaving the GPU with only 64 CTAs. The next kernel-design step should split the
32x32 scale tiles into a more parallel cooperative mapping, likely one or more
warps per scale tile with coalesced row-major vectorized loads/stores and a CTA
shape that yields hundreds to thousands of CTAs on 4096x4096 inputs.
