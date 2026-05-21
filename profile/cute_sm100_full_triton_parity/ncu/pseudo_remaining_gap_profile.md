# Pseudo remaining gap NCU profile

This profiles two representative rows from the current benchmark where CuTe is
strictly faster than Triton but still below the old `1.2x` target. These are
pseudo-quantize rows, which now only need to be faster than Triton.

Reports:

- `mxfp4_pseudo1024_triton.ncu-rep`
- `mxfp4_pseudo1024_triton.csv`
- `mxfp4_pseudo1024_cute.ncu-rep`
- `mxfp4_pseudo1024_cute.csv`
- `if4bs8_pseudo128_triton.ncu-rep`
- `if4bs8_pseudo128_triton.csv`
- `if4bs8_pseudo128_cute.ncu-rep`
- `if4bs8_pseudo128_cute.csv`

## 1024x1024 MXFP4 static_6 pseudo

Benchmark row: Triton `0.0312 ms`, CuTe `0.0271 ms`, speedup `1.150x`.

| Backend stage | Time | Grid | Block | Reg/thread | SM throughput | DRAM throughput | L2 throughput | Issue active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton fill helper | 3.0-3.4 us | 1 | 128 | 16 | ~0.005% | ~0.016% | 0.29-0.33% | 2.5-4.0% |
| Triton `pseudo_quantization_kernel` | 12.3-12.7 us | 64 | 128 | 56 | 4.57-4.83% | 2.17-2.24% | 2.07-2.14% | 14.3-14.8% |
| CuTe `Sm100MXFP4StaticPseudoQuantize` | 4.7-5.1 us | 256 | 128 | 32 | 7.56-8.07% | 5.43-5.87% | 7.94-8.64% | 21.9-22.3% |

The CuTe GPU kernel is already substantially faster than Triton's main pseudo
kernel and uses fewer registers. The remaining gap to `1.2x` is not because the
Triton kernel body is faster; the retained end-to-end event timing is dominated
by short-kernel fixed costs and wrapper/launch sequencing, where a 10 us GPU
kernel advantage is partially diluted.

## 128x128 IF4_BS8 abs_max pseudo

Benchmark row: Triton `0.0386 ms`, CuTe `0.0327 ms`, speedup `1.180x`.

| Backend stage | Time | Grid | Block | Reg/thread | SM throughput | DRAM throughput | L2 throughput | Issue active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton fill helper | 4.1-4.3 us | 16 | 128 | 32 | ~0.07% | ~0.12% | 0.30-0.31% | 2.1-2.5% |
| Triton reduce | 8.8-9.0 us | 1 | 512 | 31 | ~0.08% | ~0.145% | 0.32-0.35% | 17.0-17.5% |
| Triton scalar helper | 4.4-4.7 us | 1 | 128 | 30 | ~0.005% | ~0.09% | 0.35-0.43% | 2.1-2.5% |
| Triton `pseudo_quantization_kernel` | 26.5-27.3 us | 4 | 128 | 24 | ~0.29% | ~0.02% | 0.07-0.08% | 11.7-12.1% |
| CuTe reduce | 8.4-8.5 us | 1 | 512 | 28 | ~0.095% | ~0.152% | 0.34-0.36% | 17.3-18.1% |
| CuTe `Sm100IF4BS8AdaptivePseudoQuantize` | 4.6-4.8 us | 16 | 128 | 32 | ~0.60% | ~0.126% | 0.30-0.36% | 12.4-14.0% |

This row is mostly a tiny-shape fixed-cost case. Triton launches more GPU work
than CuTe, and its main pseudo kernel is much slower than the CuTe pseudo
kernel. The reason the event-level speedup is only `1.18x` is that both paths
spend most of the measured time in launch-scale helper/reduction work where the
GPU is heavily underfilled.

## Conclusion

The current retained pseudo rows below `1.2x` are not cases where Triton's main
kernel is better. NCU shows the CuTe pseudo kernels are faster on both a
representative medium shape and a small shape. The remaining gap to `1.2x` is
fixed overhead and short-kernel underutilization, not a Triton kernel advantage.
