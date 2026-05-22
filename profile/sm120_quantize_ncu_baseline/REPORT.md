# sm120 Quantize NCU Baseline

## Workloads

- Device: RTX 5090, `CUDA_VISIBLE_DEVICES=7`, compute capability 12.0.
- Harness: `harness/run_quantize_once.py`.
- Representative failing kernels:
  - `if3 abs_max transpose`, shape `4096x4096`.
  - `nvfp4 mse`, shape `4096x4096`.

## Launch Structure

`if3 abs_max transpose`:

- CuTe sm120: `reduce_kernel`, then CuTe main quantize kernel.
- Triton: `vectorized_elementwise_kernel`, `reduce_kernel`,
  `unrolled_elementwise_kernel`, `quantization_kernel`, then two elementwise
  kernels.

`nvfp4 mse`:

- CuTe sm120: `reduce_kernel`, then CuTe main quantize kernel.
- Triton: `vectorized_elementwise_kernel`, `reduce_kernel`,
  `unrolled_elementwise_kernel`, then `quantization_kernel`.

## Key Findings

`if3 abs_max transpose` CuTe main kernel is structurally bottlenecked on local
memory and scalar transposed loads:

- CuTe main kernel duration in NCU: `209.344 us`.
- Triton `quantization_kernel` duration in NCU: `81.120 us`.
- CuTe local spilling: `458752` register-spill instructions,
  `229376` local loads and `229376` local stores.
- CuTe global-load sector utilization: about `2.00` data bytes per 32-byte
  sector.
- CuTe local load/store sector utilization: `1.00` data byte per 32-byte
  sector.
- CuTe stalls are dominated by LG throttle (`38.77` cycles per issue-active
  instruction) and long scoreboard (`15.89`).

This path needs a sm120-specific transpose kernel design. The current per-thread
scalar transposed load path keeps too many values live while evaluating IF3
candidate errors and spills to local memory.

`nvfp4 mse` default non-transpose path has a different profile:

- CuTe main kernel duration in NCU: `28.960 us`.
- Triton `quantization_kernel` duration in NCU: `80.672 us`.
- CuTe/Triton `reduce_kernel` durations are similar: about `31 us` each.
- CuTe main kernel has no register spilling.

For default inference-like `nvfp4 mse`, the CuTe main kernel is already faster
than Triton. End-to-end event timing is limited by multi-launch overhead,
allocation/dispatch gaps, and the separate amax reduction. To make the observed
workflow consistently exceed `1.2x`, reduce launch overhead or fuse/avoid amax.

## Current sm120 Inference Shape Status

The temporary CUDA-delegating experiment was removed because
`CuteSm120QuantizeBackend` should remain a pure CuTe backend. The current sm120
route keeps the default inference path in `src/fouroversix/kernels/cute_sm120/`.

Latest pure-CuTe inference shape profile:
`profile/inference_kernel_profile_sm120_pure/latest.md`.

- Small activation shapes remain below target at about `0.77x-0.79x` versus
  Triton.
- Small/medium weight shapes remain below target at about `0.79x-0.81x`.
- The larger `1536x576` weight shape is close at `1.132x`.
- The large `49152x576` weight shape is above target at `1.552x`.

The NCU data above indicates the main CuTe kernel is fast for large
`nvfp4 mse`, but full `quantize()` timing is dominated by fixed launch/dispatch
overhead, the separate amax reduction, and for non-128-row-aligned inference
shapes the `QuantizedTensor` padding path. The next pure-CuTe optimization
should make the sm120 NVFP4 adaptive kernel write padded output directly so the
backend avoids Python-side `F.pad` launches.

## KernelWiki Notes

Relevant KernelWiki references:

- `sources/prs/flashinfer/PR-2838.md`: FlashInfer added CuTe-DSL NVFP4
  quantization with a default vectorized global-load kernel and a TMA variant.
- `sources/prs/flashinfer/PR-2904.md`: FlashInfer later moved FP4/FP8
  quantization toward a dual-path architecture: linear flat plus swizzled
  row-based kernels.

This suggests the sm120 path should not be a direct sm100 clone. Keep
`cute_sm120` separate and add architecture-specific kernels for the main
families.

## Next Optimization Targets

1. For inference-like default paths, attack launch overhead and the separate
   amax reduction first. Options: provided FP32 `x_amax`, CUDA graph capture,
   or a fused amax+quantize sm120 kernel.
2. For transpose paths, redesign the sm120 kernel around coalesced row/column
   tiling instead of one scalar transposed half load per thread per source row.
3. For IF3/IF4 adaptive paths, shorten register lifetimes by computing
   candidate error and packed output in smaller groups instead of keeping all
   candidate values live across the full block.

## Superseding Update

The inference-like `nvfp4 + mse` path has since been optimized in the pure
`cute_sm120` backend:

- Latest profile: `profile/inference_kernel_profile_sm120_pure/latest.md`
- generated_at_unix `1779380840`
- All listed inference weight and activation quantize shapes exceed the 1.2x
  Triton target.

The implementation keeps the SM120 backend pure CuTe. It uses:

- a separate `CuteSm120QuantizeBackend` and `src/fouroversix/kernels/cute_sm120/`
  kernels;
- lazy CUTLASS materialization for padded values/scales, so quantize-time work
  measures the quantize kernel rather than GEMM padding;
- an `amax` cache for repeated immutable weight tensors, invalidated by tensor
  `_version`.

Full pseudo/fake parity is recorded in
`profile/cute_sm120_triton_parity/benchmark_current.json`: all `138/138`
`pseudo_quantize=True` rows in the parity harness are faster than Triton across
`128x256`, `1024x1024`, and `4096x4096`.

The transpose rows in this report remain useful as future kernel-redesign work:
IF3/IF4 large transpose is no longer slower in event timings, but it is still
below a 1.2x non-pseudo target.
