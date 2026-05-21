# IF6 base profile notes

Note: this is a historical NCU note from an earlier below-target snapshot. The
current default auto-amax benchmark no longer has IF6 base below Triton; see
`benchmark_current.json` and `progress.md` for current status.

This is a profiling note for the 4096x4096 `if6_e3m2 + abs_max` base
quantize workload. The full benchmark snapshot shows IF6 base below Triton,
but kernel-level profiling shows the main CuTe quantization kernel is not the
slow part.

Nsight Compute summary:

| Backend | Main kernel time | SM throughput | DRAM throughput | Active warps | Registers/thread | Grid | Block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | 44.5-45.2 us | 66.9-68.1% | 9.68-9.84% | 48.0-48.2% | 48 | 2048 | 128 |
| CuTe | 29.4-29.6 us | 62.7-63.1% | 14.8-14.9% | 51.2-51.3% | 46 | 1184 | 256 |

Torch profiler end-to-end CUDA breakdown over 20 calls:

| Backend | Main quantize kernel | Reduction | Elementwise helpers | Other |
| --- | ---: | ---: | ---: | ---: |
| Triton | 822.7 us | 289.3 us | 253.8 us | 16.9 us memset |
| CuTe | 538.8 us | 296.3 us | 438.1 us | 21.9 us memset |

Interpretation:

- CuTe's retained IF6 adaptive quantize kernel is faster than Triton's main
  quantization kernel on this workload.
- The remaining end-to-end gap/noise comes from helper work outside the main
  kernel, especially extra elementwise kernels in the CuTe path and the shared
  `x.abs().max()` reduction.
- The full benchmark's fixed one-shot Triton-then-CuTe timing order is too
  noisy for near-threshold workloads. Targeted comparisons should use repeated,
  alternating order with medians before deciding that a kernel needs rewriting.
