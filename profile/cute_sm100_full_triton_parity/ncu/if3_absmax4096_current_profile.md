# IF3 abs_max pseudo 4096x4096 current profile

This profile compares the only retained benchmark row where CuTe sm100 is
strictly slower than Triton under the relaxed pseudo target:
`4096x4096 if3 abs_max pseudo_quantize=True`.

Reports:

- `if3_absmax4096_triton_current.ncu-rep`
- `if3_absmax4096_triton_current_details.csv`
- `if3_absmax4096_cute_current.ncu-rep`
- `if3_absmax4096_cute_current_details.csv`

## Summary

| Backend | Kernel | Duration | Block | Grid | Threads | Waves/SM | Reg/thread | Theoretical occupancy | Achieved occupancy | SM throughput | DRAM throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `pseudo_quantization_kernel` | 34.43 us | 128 | 32x64 | 262,144 | 1.38 | 43 | 62.50% | 48.17% | 64.08% | 12.91% |
| CuTe | `Sm100IF3AdaptivePseudoQuantize` | 46.91 us | 256 | 4096 | 1,048,576 | 6.92 | 64 | 50.00% | 42.92% | 84.82% | 9.45% |

## Interpretation

The gap is not memory bandwidth. CuTe uses lower DRAM throughput
(`9.45%` vs Triton `12.91%`) and lower memory bandwidth (`724 GB/s` vs
`989 GB/s`), while taking longer. CuTe is compute-bound: SM throughput is
already `84.82%`, compared with Triton's `64.08%`.

Triton does less work per output element at the kernel-mapping level:

- It launches four times fewer threads (`262k` vs `1,048k`).
- It uses fewer registers per thread (`43` vs `64`).
- It has higher achieved occupancy (`48.17%` vs `42.92%`).
- Its grid is a 2D program layout (`32x64`) instead of CuTe's flat
  one-thread-per-scale-block layout (`4096` CTAs x `256` threads).

The CuTe kernel is spending cycles on compute pipeline pressure rather than
waiting on memory. NCU reports the dominant CuTe stalls as not-selected
(`34.57%`) and execution-pipe unavailable (`33.24%`), which matches an
oversubscribed compute/instruction mix. Triton instead reports a larger L1TEX
scoreboard component (`32.21%`) and lower SM saturation.

## Next Direction

Small launch-parameter changes are unlikely to close this row. The fix should
change the IF3 pseudo mapping/instruction shape, not just wrapper dispatch:

- reduce the one-thread-per-scale-block scalar work,
- reduce register pressure in the FP3-vs-INT3 selection path,
- or implement a tiled/cooperative IF3 pseudo kernel closer to Triton's 2D
  program decomposition.

