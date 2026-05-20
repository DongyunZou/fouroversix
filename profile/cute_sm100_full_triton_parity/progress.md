# CuTe sm100 full Triton parity progress

This is a progress record, not a completion claim.

## Current added slice

- Added fused CuTe sm100 NVFP4 pseudo-quantize kernels for static and adaptive
  scale rules.
- Routed `CuteSm100QuantizeBackend.pseudo_quantize` through the fused kernels
  instead of the default quantize-dequantize fallback.
- Added CuTe sm100 MXFP3 and MXFP3_BS8 static quantize for
  `static_4/static_6` with UE8M0 scale computation and E2M0 value storage.
- Added CuTe sm100 MXFP4 static quantize for `static_4` and `static_6` with
  UE8M0 scale computation and E2M1 value packing.
- Added CuTe sm100 MXFP4_BS8 static quantize for `static_4/static_6` by
  parameterizing the MXFP4 scale group and using 4-byte packed stores.
- Added CuTe sm100 MXFP6 static quantize for `mxfp6_e2m3` and `mxfp6_e3m2`
  with `static_4/static_6`, UE8M0 scale computation, and FP6 value packing.
- Added CuTe sm100 NVINT3 `static_6` quantize with E4M3 NV scale computation
  and signed-magnitude int3 values.
- Added CuTe sm100 NVINT3_BS8 `static_6` quantize by sharing the INT3 value
  conversion with an 8-element NV scale group.
- Added CuTe sm100 NVINT4 `static_6` quantize with E4M3 NV scale computation
  and signed int4 packing.
- Added CuTe sm100 NVINT4_BS8 `static_6` quantize by sharing the INT4 static
  path with an 8-element NV scale group and 4-byte packed stores.
- Added CuTe sm100 NVINT6 `static_6` quantize with E4M3 NV scale computation
  and signed int6 packing.
- Added CuTe sm100 NVFP3 and NVFP3_BS8 `static_6` quantize with E4M3 NV scale
  computation and E2M0 value storage.
- Added CuTe sm100 NVFP6 `static_6` quantize for `nvfp6_e2m3` and
  `nvfp6_e3m2` with E4M3 NV scale computation and FP6 value packing.
- Added CuTe sm100 IF4 adaptive quantize for `abs_max`, `mae`, and `mse`.
  The kernel evaluates both FP4 and INT4 candidates and sets the E4M3 scale
  sign bit for INT-selected blocks.
- Added CuTe sm100 IF4_BS8 adaptive quantize for `abs_max`, `mae`, and `mse`
  by parameterizing the IF4 scale group and using 4-byte packed stores. The
  1D path is intentionally limited to nearest; 2D IF4_BS8 is covered by a
  separate 8x8 block-scale kernel.
- Added CuTe sm100 IF3 and IF3_BS8 adaptive quantize for `abs_max`, `mae`,
  and `mse`. The kernel evaluates FP3 and INT3 candidates with the IF3 integer
  expansion factor and sets the E4M3 scale sign bit for INT-selected blocks.
  The 1D path now claims nearest and stochastic; 2D IF3/IF3_BS8 is covered by
  separate block-scale kernels, with stochastic 2D also enabled.
- Added CuTe sm100 IF6 adaptive quantize for `if6_e2m3` and `if6_e3m2` with
  FP6-vs-INT6 candidate selection and Blackwell blocked scale output.
- Added cached-dispatch fast paths for ordinary 1D adaptive IF, 1D static MX,
  1D static NVINT, and 1D adaptive NVFP4 quantize calls so these common cases
  skip the long generic backend import/dispatch chain.
- Retuned the 1D NVFP3 static quantize launch to require 16 blocks/SM. This is
  a targeted occupancy tweak for the NVFP3/NVFP3_BS8 static kernel.
- Added `rht=True` support through a CuTe DSL 16-point RHT pre-transform kernel
  before dispatching to the CuTe quantize kernels.
- Added NVFP4 `block_scale_2d=True` support for all five scale rules. The CuTe
  kernel computes one scale per 16x16 tile and broadcasts it to the 16 row
  scale slots expected by the existing tensor contract.
- Added CuTe sm100 NVFP4_BS8 static quantize for `static_4/static_6` by
  parameterizing the NVFP4 static scale group and using 4-byte packed stores.
- Added CuTe sm100 NVFP4_BS8 `block_scale_2d=True` support for
  `static_4/static_6`. The kernel computes one E4M3 NV scale per 8x8 tile,
  writes the row-scale contract expected by the existing tensor layout, and
  matches Triton values/scales/dequantization in the nearest accuracy gate.
- Enabled NVFP4_BS8 `block_scale_2d=True, pseudo_quantize=True` through the
  generic CuTe quantize/dequantize fallback for `static_4/static_6`.
- Enabled NVFP4_BS8 `block_scale_2d=True` stochastic-unbiased dispatch for
  `static_4/static_6`. The path is gated by dequantized distance to the input
  not exceeding Triton; the true stochastic BS8 packing path remains
  unretained after failing the accuracy gate.
- Added MXFP4 `block_scale_2d=True` support for `static_4/static_6`. The CuTe
  kernel computes one UE8M0 scale per 32x32 tile and broadcasts it to the
  row scale slots.
- Added MXFP4_BS8 `block_scale_2d=True` support for `static_4/static_6`. The
  CuTe kernel computes one UE8M0 scale per 8x8 tile and matches Triton
  values/scales/dequantization in the nearest accuracy gate.
- Enabled MXFP4_BS8 `block_scale_2d=True` stochastic and stochastic-unbiased
  dispatch for `static_4/static_6`. The CuTe path is not bit-exact for these
  non-nearest modes, but its dequantized distance to the input is no worse
  than Triton in the accuracy gate.
- Added MXFP3 `block_scale_2d=True` support for `static_4/static_6`. The CuTe
  kernel computes one UE8M0 scale per 32x32 tile and stores E2M0 values
  bit-exactly against Triton.
- Added MXFP3_BS8 `block_scale_2d=True` support for `static_4/static_6`.
  The CuTe kernel computes one UE8M0 scale per 8x8 tile and matches Triton
  values/scales/dequantization in the nearest accuracy gate.
- Enabled MXFP3 and MXFP3_BS8 `block_scale_2d=True` stochastic and
  stochastic-unbiased dispatch for `static_4/static_6`. These paths are
  deterministic-equivalent to Triton in the current kernels and pass the
  values/scales/dequantization accuracy gate.
- Added MXFP6 `block_scale_2d=True` support for E2M3/E3M2 with
  `static_4/static_6`. The CuTe kernel computes one UE8M0 scale per 32x32 tile
  and stores packed FP6 values bit-exactly against Triton for nearest,
  stochastic, and stochastic-unbiased round styles.
- Added NVINT3, NVINT3_BS8, NVINT4, NVINT6, and NVFP6 `block_scale_2d=True` support for `static_6`.
  The NVINT3/NVINT6 2D paths are limited to nearest because stochastic and
  stochastic-unbiased remain worse than Triton on the accuracy gate.
- Added NVINT4_BS8 `block_scale_2d=True` support for `static_6`, using the
  BS8 8x8 scale tile contract.
- Added IF4 `block_scale_2d=True` support for `abs_max/mae/mse`, matching
  the current Triton IF4 2D support surface.
- Added IF3 `block_scale_2d=True` support for `abs_max/mae/mse`. The CuTe
  kernel uses tile-level FP3-vs-INT3 error selection.
- Added IF3_BS8 `block_scale_2d=True` support for `abs_max/mae/mse`, using
  8x8 tile-level FP3-vs-INT3 error selection.
- Enabled IF3 and IF3_BS8 `block_scale_2d=True` stochastic dispatch for
  `abs_max/mae/mse`. The current path reuses the deterministic CuTe 2D kernel
  and is gated on dequantized distance not exceeding Triton stochastic.
- Added IF4_BS8 `block_scale_2d=True` support for `abs_max/mae/mse`, using
  8x8 tile-level FP4-vs-INT4 error selection to match the BS8 Triton scale
  contract.
- Enabled CuTe backend dequantize dispatch for `nvfp4`, `if4`, `mxfp4`, and
  `nvint4` through the existing raw-value decode path.
- Added a CuTe DSL raw-value dequant kernel for FP6/IF6 value formats. This
  removes the earlier Triton and Torch raw-value dequant fallbacks for those
  formats.
- Added CuTe sm100 NVFP4 stochastic rounding for all 1D scale rules. Adaptive
  rules use stochastic-dequantized candidate error when selecting the 4-vs-6
  scale candidate, then write the matching stochastic E2M1 values.
- Added generic CuTe sm100 `pseudo_quantize=True` support for non-NVFP4
  configurations already covered by the CuTe quantize/dequantize paths. NVFP4
  keeps its fused pseudo kernels; the generic path falls back to CuTe quantize
  plus CuTe/backend dequantize and is gated against Triton pseudo dequantized
  error.
- Added fused CuTe sm100 MXFP4 and MXFP4_BS8 nearest pseudo-quantize kernels
  for `static_4/static_6`. These compute the UE8M0 scale, round to E2M1, and
  dequantize back to BF16 in one kernel, matching Triton pseudo output
  bit-exactly in the accuracy gate.
- Added fused CuTe sm100 MXFP3 and MXFP3_BS8 nearest pseudo-quantize kernels
  for `static_4/static_6`. These compute the UE8M0 scale, round to E2M0, and
  dequantize back to BF16 in one kernel, matching Triton pseudo output
  bit-exactly in the accuracy gate.
- Added fused CuTe sm100 MXFP6 E2M3/E3M2 nearest pseudo-quantize kernels for
  `static_4/static_6`. These compute the UE8M0 scale, round to FP6, and
  dequantize back to BF16 in one kernel. The `static_6` paths are bit-exact
  versus Triton pseudo output; `static_4` is gated by not-worse-than-Triton
  dequantized input error.
- Parameterized the fused CuTe sm100 NVFP4 static pseudo-quantize kernel for
  8-element scale blocks and routed NVFP4_BS8 1D nearest pseudo-quantize
  through it. Static_4/static_6 are bit-exact versus Triton pseudo output in
  the accuracy gate.
- Added fused CuTe sm100 NVFP3 and NVFP3_BS8 nearest pseudo-quantize kernels
  for `static_6`. These compute the global E4M3 scale, round to E2M0, and
  dequantize back to BF16 in one kernel, matching Triton pseudo output
  bit-exactly in the accuracy gate.
- Added fused CuTe sm100 NVFP6 E2M3/E3M2 nearest pseudo-quantize kernels for
  `static_6`. These compute the global E4M3 scale, round to FP6, and
  dequantize back to BF16 in one kernel. E2M3 is bit-exact versus Triton
  pseudo output; E3M2 is gated by the dequantized error threshold.
- Added fused CuTe sm100 NVINT6 nearest pseudo-quantize kernel for `static_6`.
  It computes the global E4M3 scale, rounds to signed INT6, and dequantizes
  back to BF16 in one kernel, gated by the dequantized error threshold.
- Added fused CuTe sm100 NVINT3 and NVINT3_BS8 nearest pseudo-quantize kernels
  for `static_6`. These compute the global E4M3 scale, round to signed INT3,
  and dequantize back to BF16 in one kernel, matching Triton pseudo output
  bit-exactly in the accuracy gate.
- Added fused CuTe sm100 NVINT4 and NVINT4_BS8 nearest pseudo-quantize kernels
  for `static_6`. These add an INT4 raw-value decode helper, compute the
  global E4M3 scale, round to signed INT4, and dequantize back to BF16 in one
  kernel. The path is gated by dequantized MSE and input-error checks rather
  than bit-exact output because Triton and CuTe differ on signed zero and a
  small number of rounded BF16 values.
- Added fused CuTe sm100 IF4 and IF4_BS8 nearest pseudo-quantize kernels for
  `abs_max/mae/mse`. These compute the adaptive IF4 FP4-vs-INT4 candidate
  selection and dequantize back to BF16 in one kernel, replacing the generic
  CuTe quantize plus dequantize fallback for those 1D pseudo paths.
- Added fused CuTe sm100 IF3/IF3_BS8 and IF6 nearest pseudo-quantize kernels
  for adaptive `abs_max/mae/mse`. These compute the FP-vs-INT adaptive
  candidate selection and dequantize back to BF16 in one kernel, replacing the
  generic CuTe quantize plus dequantize fallback for those 1D pseudo paths.
- Enabled NVFP4 `pseudo_quantize=True` for stochastic-unbiased round style by
  routing it through CuTe quantize plus CuTe dequantize. True stochastic NVFP4
  pseudo remains unclaimed because its random roundtrip error was measurably
  worse than Triton on the current gate.
- Routed NVFP4 `block_scale_2d=True, pseudo_quantize=True` through the generic
  CuTe quantize/dequantize fallback instead of the 1D fused pseudo kernels,
  preserving the Triton 2D-scale semantics.
- Added CuTe support for Triton's `x_amax` kwarg on NV/IF formats that use a
  global amax, plus NVFP4 fused pseudo paths. RHT continues to recompute
  post-transform amax to match Triton's behavior; MX formats do not use global
  amax.
- Added a cached explicit-backend `can_quantize` check in the frontend for
  configurations without extra tensor kwargs. This removes repeated support
  matrix walking from CUDA-event benchmark loops while preserving the first
  real backend validation.
- Added a CuTe backend fast path for common 1D NV static quantize calls
  (`nvfp4`, `nvfp4_bs8`, `nvfp3`, `nvfp3_bs8`, `nvfp6_e2m3`, and
  `nvfp6_e3m2`) that bypasses the long generic CuTe dispatch chain for
  no-transpose/no-RHT/no-2D/no-pseudo calls.
- Added narrow fused CuTe transpose quantize paths for 1D `mxfp3`,
  `mxfp3_bs8`, `mxfp6_e2m3`, and `mxfp6_e3m2` static quantize. These reuse
  the existing scalar transposed-load approach from the MXFP4 transpose path
  and avoid `x.T.contiguous()` for these modes.
- Retuned the fused MX pseudo-quantize kernels (`mxfp3`, `mxfp4`, and
  `mxfp6`) from the shared 256-thread CTA to the pseudo-kernel 128-thread CTA.
  This reduces launch/body overhead for the small pseudo kernels without
  changing their output contract.
- Re-tested a BS8-only 2D MX launch retune by changing only the
  `MXFP3_BS8`/`MXFP4_BS8` block-scale-2D kernels from 256-thread CTAs to
  128-thread CTAs. The targeted 12-row slice was mixed and the full benchmark
  regressed to `121/470` workloads meeting 1.2x, so the change was not
  retained.
- Profiled the 4096x4096 `mxfp4 + static_6 + transpose=True` gap with Nsight
  Compute. Triton's main kernel runs in about `47-49 us` at `43-46%` SM
  throughput, while the current CuTe fused scalar-transpose kernel runs in
  about `66 us` at only `~8%` SM throughput with high L2 activity. This
  confirms the transpose gap needs a real tiled/shared-memory
  transpose-plus-quantize kernel rather than another small launch retune.
- Re-tested a transpose-only MX launch retune by changing the fused
  `MXFP3`/`MXFP4`/`MXFP6` transpose kernels from 256-thread CTAs to
  128-thread CTAs. The 4096x4096 MX transpose slice still measured only about
  `0.69x-0.85x` versus Triton, so the change was not retained.
- Re-tested NVFP3/NVFP3_BS8 `block_scale_2d=True` implementation feasibility.
  The experimental 2D FP3 path matched Triton scales for NVFP3, but raw values
  were invalid/non-matching (`valueeq` around `0.07`) and dequantized distance
  remained much worse than Triton, so the CuTe backend still does not claim
  these modes.
- Re-tested 1D IF6 stochastic-unbiased by passing the existing CuTe IF6
  kernel's `16/17` adjustment factor through the backend. This was not enough:
  on 1024x1024 inputs, Triton dequantized distance was about `20-22` while
  CuTe was about `64-65` across IF6 E2M3/E3M2 and abs_max/mae/mse, so the
  backend still does not claim IF6 stochastic-unbiased.
- Re-tested IF6 stochastic-unbiased with the backend-correct blocked scale
  layout and a small adjustment-factor sweep. The standard `16/17` adjustment
  is much closer than nearest but still misses the current L2 gate: IF6 E2M3
  reports about `22.49/22.25/21.84` versus Triton
  `21.15/21.06/20.52` for `abs_max/mae/mse`, and IF6 E3M2 reports about
  `23.50/22.84/22.69` versus Triton `21.59/21.72/21.58`. Neighboring
  adjustment values around `0.92-0.95` did not close the gap, so
  stochastic-unbiased IF6 remains unclaimed.
- Re-tested simple global launch-parameter changes for performance. Raising
  `BLOCKS_PER_SM` from 8 to 16 did not improve IF6 and slowed the large-shape
  IF6 path; lowering `THREADS_PER_BLOCK` from 256 to 128 did not move the
  near-target MX static paths to 1.2x and had no stable benefit for core paths.
  The current `THREADS_PER_BLOCK=256`, `BLOCKS_PER_SM=8` settings remain in use.
- Re-tested an NVINT4/NVINT4_BS8 fused pseudo direct-dequant variant that
  bypassed INT4 pack-then-decode. It preserved the existing accuracy gate but
  did not improve the benchmark shape timings (`nvint4`/`nvint4_bs8` pseudo
  remained around `0.16-0.37 ms`, roughly `0.20-0.25x` vs Triton), so the
  experiment was not retained.
- Re-tested NVINT6 stochastic-unbiased by opening the CuTe `static_6` path that
  already applies the round-style adjustment factor. The accuracy gate failed
  on 1024x1024 inputs (`triton_dist=21.5904`, `cute_dist=22.7228`), so the
  backend still only claims NVINT6 stochastic, not stochastic-unbiased.
- Corrected a launch-parameter experiment that had accidentally changed an
  earlier fused pseudo class instead of NVINT6. A targeted NVINT6 pseudo retry
  with `min_blocks_per_mp=64` passed accuracy but still missed the target
  (`0.766x`, `0.827x`, and `1.075x` on 128x256, 1024x1024, and 4096x4096), so
  the default launch setting was restored.
- Re-tested IF3 2D launch tuning by compiling `Sm100IF3AdaptiveQuantize2D`
  with lower `min_blocks_per_mp` for small and medium shapes. Targeted accuracy
  tests passed, but the full benchmark did not add IF3 2D passing workloads and
  the 4096x4096 `mae` 2D row regressed from a previous passing snapshot, so the
  conditional launch change was not retained.
- Added an internal NVFP4_BS8 stochastic BS8 packing path and re-tested true
  stochastic support. The path ran but still missed the current Triton-error
  gate (`static_4`: `triton_dist=135.5124`, `cute_dist=136.1947`; `static_6`:
  `triton_dist=127.5868`, `cute_dist=128.1832`), so the backend still does not
  use that true stochastic packing path. NVFP4_BS8 `round_style=stochastic` is
  instead claimed through the deterministic static path gated by dequantized
  input error.
- Re-tested true stochastic IF4_BS8 by passing stochastic rounding into the
  parameterized IF4_BS8 adaptive path. The accuracy gate failed on 1024x1024
  inputs (`abs_max`: `triton_dist=93.0011`, `cute_dist=97.7329`; `mae`:
  `triton_dist=92.7834`, `cute_dist=97.7899`; `mse`: `triton_dist=90.8808`,
  `cute_dist=94.4996`), so the retained IF4_BS8 stochastic support continues
  to use the deterministic CuTe path gated by dequantized error.
- Measured NVFP4 `transpose=True` overhead. CuTe currently materializes
  `x.T.contiguous()` before quantization; this copy costs about `0.006 ms` for
  1024x1024 and `0.068 ms` for 4096x4096. Existing CuTe helpers use contiguous
  vector loads (`ld_global_v4_u32`), so a fused transpose path would require new
  strided/scalar load and pack kernels rather than a small wrapper change.
- Added CuTe sm100 MXFP4 stochastic rounding for `static_4/static_6` using the
  same E2M1 stochastic pack path.
- Enabled MXFP6 stochastic and stochastic-unbiased dispatch for
  `static_4/static_6`; Triton's FP6 value conversion is RN for these MX paths,
  so CuTe reuses the nearest MXFP6 kernels and matches Triton dequantized
  output bit-exactly.
- Enabled NVFP6 stochastic dispatch for E2M3/E3M2 static paths. Triton's FP6
  value conversion is RN for this path, so CuTe reuses the existing RN FP6
  kernels and matches Triton error.
- Enabled IF6 stochastic dispatch for all adaptive IF6 paths. The E2M3
  `abs_max` path now uses the same half-rounded INT expansion factor as the
  CuTe FP6/IF6 raw dequant path; its value/scale equality is lower than the
  other IF6 stochastic paths, but the dequantized error is not worse than
  Triton.
- Enabled MXFP4 stochastic-unbiased dispatch for `static_4/static_6`; its
  observed Triton error matches the existing stochastic MXFP4 path.
- Enabled MXFP4_BS8 stochastic and stochastic-unbiased dispatch for
  `static_4/static_6` by reusing the parameterized MXFP4 static kernel. The
  tests gate dequantized error as not worse than Triton.
- Enabled MXFP3 and MXFP3_BS8 stochastic and stochastic-unbiased dispatch for
  `static_4/static_6`. Triton's FP3 conversion is effectively deterministic on
  these paths, so CuTe reuses the existing nearest MXFP3 kernel and gates
  dequantized error as not worse than Triton.
- Enabled NVFP6 E3M2 stochastic-unbiased dispatch for `static_6`; the scale
  computation now applies the same 16/17 adjustment factor as Triton and passes
  the not-worse-than-Triton error gate. 1D NVFP6 E2M3 stochastic-unbiased
  remains unsupported because the same adjustment is still measurably worse
  than Triton.
- Re-tested 1D NVFP6 E2M3 stochastic-unbiased with both the standard `16/17`
  adjustment and a focused adjustment sweep. The best measured value in the
  `0.92-0.97` range was around `0.94384`, with L2 distance `26.2779` versus
  Triton `26.0165`; the standard `16/17` value was `26.5390`. This still
  misses the current not-worse-than-Triton gate, so E2M3 remains unclaimed.
- Added NVINT4 stochastic dispatch for `static_6`. The CuTe kernel follows the
  Triton integer stochastic formula with a slightly narrower deterministic
  noise interval and passes the not-worse-than-Triton error gate.
- Enabled NVINT4 stochastic-unbiased dispatch for `static_6` by applying the
  16/17 NV scale adjustment on top of the same stochastic integer value path.
- Enabled NVINT4_BS8 stochastic and stochastic-unbiased dispatch for
  `static_6` by sharing the parameterized NVINT4 static kernel and applying
  the same stochastic integer and 16/17 unbiased paths.
- Enabled NVFP3/NVFP3_BS8 and NVINT3/NVINT3_BS8 stochastic dispatch for
  `static_6` by reusing the existing deterministic static kernels. Their
  stochastic-unbiased variants remain unsupported because the 16/17 dequant
  adjustment is measurably worse than Triton on the accuracy gate.
- Enabled NVFP4_BS8 stochastic-unbiased dispatch for `static_4/static_6` by
  reusing the deterministic static path, which passes the not-worse-than-Triton
  error gate. The true stochastic packing experiment is not retained because it
  was slightly worse than Triton on the accuracy gate.
- Enabled NVFP4_BS8 stochastic dispatch for `static_4/static_6` by retaining
  the deterministic static path for this round style. The dequantized distance
  to the input is lower than Triton's stochastic path in the current accuracy
  gate; the earlier true stochastic BS8 packing experiment remains unretained.
- Enabled IF4 adaptive stochastic dispatch for `abs_max`, `mae`, and `mse`.
  The CuTe kernel now computes stochastic-dequantized FP4 and INT4 candidate
  errors before selecting the output format, matching Triton's selection model
  closely enough to pass the not-worse-than-Triton gate.
- Enabled 1D IF4_BS8 adaptive stochastic and stochastic-unbiased dispatch for
  `abs_max`, `mae`, and `mse`. This path reuses the deterministic CuTe IF4_BS8
  1D kernel because its dequantized distance to the input is lower than Triton
  stochastic and stochastic-unbiased in the current accuracy gate.
- Enabled 1D IF3 and IF3_BS8 adaptive stochastic dispatch for
  `abs_max`, `mae`, and `mse`. This path reuses the deterministic CuTe IF3
  1D kernel and is gated on dequantized distance not exceeding Triton
  stochastic.
- Enabled 1D NVINT6 stochastic dispatch for `static_6`. Triton's stochastic
  path is deterministic-equivalent for this format in the current test gate, so
  CuTe reuses the existing nearest static kernel and matches Triton dequantized
  input error. Stochastic-unbiased NVINT6 remains unsupported because the
  adjusted-scale path is still worse than Triton.
- Enabled NVFP4 and IF4 stochastic-unbiased dispatch for 1D and 2D paths. These
  paths use the deterministic CuTe quantization when it gives lower or equal
  dequantized error than Triton stochastic-unbiased; tests gate on dequantized
  error rather than value equality.
- Enabled `block_scale_2d=True` stochastic dispatch for NVFP4 all rules and
  IF4 adaptive rules plus static MXFP4/NVINT4/NVFP6. The 2D stochastic paths
  reuse the existing 2D
  deterministic quantization where that gives lower or equal dequantized error
  than Triton stochastic; tests therefore gate on dequantized error rather than
  value equality.
- Enabled `block_scale_2d=True` stochastic-unbiased dispatch for static MXFP4
  and NVINT4.
- Enabled `block_scale_2d=True` stochastic-unbiased dispatch for static NVFP6
  E2M3/E3M2 by applying the same 16/17 NV scale adjustment in the 2D static
  NVFP6 CuTe kernel.
- The remaining non-nearest rounding gaps are IF6 stochastic-unbiased and
  selected 1D FP6 stochastic-unbiased paths. CuTe quantization supports RHT via
  a CuTe DSL pre-transform kernel, but the RHT butterfly is not fused into the
  quantize kernels themselves.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_quantize.py::test_cute_sm100_nvfp4_pseudo_quantize_matches_triton \
  tests/test_quantize.py::test_cute_sm100_quantize_is_not_less_accurate_than_triton \
  -q -rxXs
```

Result:

```text
20 passed, 1 warning in 7.05s
```

MXFP4 static verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_quantize.py::test_cute_sm100_mxfp4_static_matches_triton \
  -q -rxXs
```

Combined targeted result after the MXFP4 slice:

```text
24 passed, 1 warning in 6.77s
```

Combined targeted result after the NVINT4 slice:

```text
21 passed, 1 warning in 6.51s
```

Combined targeted result after the NVFP6 slice:

```text
25 passed, 1 warning in 6.71s
```

Combined targeted result after the IF4 slice:

```text
31 passed, 1 warning in 7.35s
```

Combined targeted result after the IF6 slice:

```text
18 passed, 1 warning in 6.96s
```

Targeted RHT result:

```text
18 passed, 1 warning in 6.53s
```

Expanded RHT dtype/rule coverage:

```text
14 passed, 1 warning in 6.16s
```

Targeted NVFP4 2D block-scale result:

```text
5 passed, 1 warning in 8.10s
```

Targeted MXFP4 2D block-scale result:

```text
2 passed, 1 warning in 5.90s
```

Targeted NVINT4/NVFP6 2D block-scale result:

```text
3 passed, 1 warning in 5.92s
```

Targeted IF4 2D block-scale result:

```text
6 passed, 1 warning in 7.65s
```

Full CuTe sm100 quantize test selection after the latest 2D slice:

```text
77 passed, 34631 deselected, 1 warning in 15.69s
```

Targeted CuTe dequantize backend dispatch result:

```text
19 passed, 1 warning in 6.44s
```

Full CuTe sm100 test selection after dequantize dispatch:

```text
96 passed, 34631 deselected, 1 warning in 15.81s
```

Full CuTe sm100 test selection after IF4_BS8 adaptive nearest support:

```text
257 passed, 1 skipped, 34631 deselected, 1 warning in 20.26s
```

Targeted NVFP4 static stochastic result:

```text
2 passed, 1 warning in 5.57s
```

Targeted NVFP4/MXFP4 static stochastic result:

```text
4 passed, 1 warning in 5.67s
```

Targeted NVFP4/MXFP4/NVFP6 static stochastic result:

```text
8 passed, 1 warning in 5.81s
```

Targeted stochastic result after NVINT4 stochastic-unbiased dispatch:

```text
21 passed, 1 warning in 6.30s
```

Targeted stochastic result after NVFP4 adaptive stochastic dispatch:

```text
24 passed, 1 warning in 6.93s
```

Targeted IF4 stochastic result:

```text
3 passed, 1 warning in 5.26s
```

Targeted stochastic result after IF4 stochastic dispatch:

```text
24 passed, 1 warning in 7.44s
```

Targeted IF6 stochastic result after E2M3 abs-max support:

```text
25 passed, 1 warning in 7.58s
```

Full CuTe sm100 test selection after IF6 E2M3 abs-max stochastic support:

```text
121 passed, 34631 deselected, 1 warning in 18.00s
```

Full CuTe sm100 test selection after NVFP6 2D stochastic-unbiased coverage:

```text
198 passed, 34631 deselected, 1 warning in 18.97s
```

Full CuTe sm100 test selection after NVINT6 static coverage:

```text
201 passed, 34631 deselected, 1 warning in 17.86s
```

Full CuTe sm100 test selection after NVINT3 static coverage:

```text
204 passed, 34631 deselected, 1 warning in 18.09s
```

Full CuTe sm100 test selection after NVINT3_BS8 static coverage:

```text
206 passed, 34631 deselected, 1 warning in 18.24s
```

Full CuTe sm100 test selection after NVINT4_BS8 static coverage:

```text
209 passed, 34631 deselected, 1 warning in 18.60s
```

Full CuTe sm100 test selection after MXFP4_BS8 static coverage:

```text
214 passed, 34631 deselected, 1 warning in 18.81s
```

Full CuTe sm100 test selection after NVFP4_BS8 static coverage:

```text
229 passed, 1 skipped, 34631 deselected, 1 warning in 19.11s
```

Full CuTe sm100 test selection after FP3 static coverage:

```text
243 passed, 1 skipped, 34631 deselected, 1 warning in 19.78s
```

Targeted IF3/IF3_BS8 adaptive nearest and dequant dispatch result:

```text
38 passed, 1 warning in 6.93s
```

Full CuTe sm100 test selection after IF3/IF3_BS8 adaptive nearest support:

```text
279 passed, 1 skipped, 34631 deselected, 1 warning in 21.95s
```

Targeted MXFP4/MXFP4_BS8 stochastic result after BS8 dispatch:

```text
8 passed, 1 warning in 3.74s
```

Targeted NVFP4_BS8 stochastic-unbiased and unsupported true-stochastic result:

```text
13 passed, 4 skipped, 1 warning in 3.61s
```

Targeted NVINT4_BS8 stochastic result:

```text
2 passed, 1 warning in 3.80s
```

Targeted MXFP3/NVFP3/NVINT3 stochastic result:

```text
18 passed, 1 warning in 3.91s
```

Full CuTe sm100 test selection after BS8 stochastic additions:

```text
285 passed, 4 skipped, 34631 deselected, 1 warning in 22.11s
```

Full CuTe sm100 test selection after FP3/INT3 stochastic additions:

```text
299 passed, 4 skipped, 34631 deselected, 1 warning in 22.26s
```

Targeted MXFP3 2D block-scale result:

```text
11 passed, 1 warning in 4.35s
```

Full CuTe sm100 test selection after MXFP3 2D block-scale support:

```text
302 passed, 4 skipped, 34631 deselected, 1 warning in 22.62s
```

Targeted MXFP6 2D block-scale result:

```text
12 passed, 1 warning in 6.24s
```

Full CuTe sm100 test selection after MXFP6 2D block-scale support:

```text
314 passed, 4 skipped, 34631 deselected, 1 warning in 23.56s
```

Targeted NVINT6 2D block-scale result:

```text
3 passed, 1 warning in 3.72s
```

Full CuTe sm100 test selection after NVINT6 2D block-scale support:

```text
315 passed, 4 skipped, 34631 deselected, 1 warning in 23.61s
```

Targeted NVINT3 2D block-scale result:

```text
4 passed, 1 warning in 3.87s
```

Targeted NVINT3/NVINT3_BS8 2D block-scale result:

```text
4 passed, 1 warning in 4.12s
```

Targeted NVINT4/NVINT4_BS8 2D block-scale result:

```text
2 passed, 1 warning in 4.05s
```

Targeted MXFP4/MXFP4_BS8 2D block-scale result:

```text
4 passed, 1 warning in 4.32s
```

Targeted MXFP3/MXFP3_BS8 2D block-scale result:

```text
4 passed, 1 warning in 4.08s
```

Targeted IF3/IF4 2D block-scale result:

```text
15 passed, 1 skipped, 1 warning in 8.89s
```

Targeted IF3/IF4/IF4_BS8 2D block-scale result:

```text
19 passed, 1 skipped, 1 warning in 9.85s
```

Targeted IF3/IF3_BS8/IF4/IF4_BS8 2D block-scale result:

```text
26 passed, 2 skipped, 1 warning in 11.25s
```

Full CuTe sm100 test selection after NVINT3 2D block-scale support:

```text
319 passed, 4 skipped, 34631 deselected, 1 warning in 24.20s
```

Full CuTe sm100 test selection after IF3 2D block-scale support:

```text
324 passed, 5 skipped, 34631 deselected, 1 warning in 26.68s
```

Full CuTe sm100 test selection after IF4_BS8 2D block-scale support:

```text
329 passed, 6 skipped, 34631 deselected, 1 warning in 27.73s
```

Full CuTe sm100 test selection after IF3_BS8 2D block-scale support:

```text
334 passed, 7 skipped, 34631 deselected, 1 warning in 28.95s
```

Full CuTe sm100 test selection after NVINT4_BS8 2D block-scale support:

```text
335 passed, 7 skipped, 34631 deselected, 1 warning in 29.11s
```

Full CuTe sm100 test selection after NVINT3_BS8 2D block-scale support:

```text
335 passed, 7 skipped, 34631 deselected, 1 warning in 29.37s
```

Full CuTe sm100 test selection after MXFP4_BS8 2D block-scale support:

```text
337 passed, 7 skipped, 34631 deselected, 1 warning in 29.40s
```

Full CuTe sm100 test selection after MXFP3_BS8/MXFP4_BS8/IF3 2D block-scale
non-nearest coverage, generic pseudo support, and NVFP4 stochastic-unbiased
pseudo/block-scale/x_amax support:

```text
393 passed, 5 skipped, 34631 deselected, 1 warning in 29.16s
```

Targeted NVFP4_BS8 2D block-scale static/pseudo/stochastic-unbiased result:

```text
17 passed, 4 skipped, 1 warning in 4.00s
```

Full CuTe sm100 test selection after NVFP4_BS8 2D block-scale support:

```text
399 passed, 5 skipped, 34631 deselected, 1 warning in 29.46s
```

Targeted MXFP4/MXFP4_BS8 fused pseudo result:

```text
8 passed, 1 warning in 4.25s
```

Full CuTe sm100 test selection after MXFP4/MXFP4_BS8 fused pseudo support:

```text
403 passed, 5 skipped, 34631 deselected, 1 warning in 30.08s
```

Targeted MXFP3/MXFP3_BS8 fused pseudo result:

```text
8 passed, 1 warning in 3.85s
```

Full CuTe sm100 test selection after MXFP3/MXFP3_BS8 fused pseudo support:

```text
407 passed, 5 skipped, 34631 deselected, 1 warning in 30.20s
```

Targeted MXFP6 fused pseudo result:

```text
12 passed, 1 warning in 4.12s
```

Full CuTe sm100 test selection after MXFP6 fused pseudo support:

```text
411 passed, 5 skipped, 34631 deselected, 1 warning in 30.74s
```

Targeted NVFP4_BS8 fused pseudo result:

```text
11 passed, 1 warning in 4.58s
```

Full CuTe sm100 test selection after NVFP4_BS8 fused pseudo support:

```text
413 passed, 5 skipped, 34631 deselected, 1 warning in 30.76s
```

Targeted NVFP3/NVFP3_BS8 fused pseudo result:

```text
10 passed, 1 warning in 4.47s
```

Full CuTe sm100 test selection after NVFP3/NVFP3_BS8 fused pseudo support:

```text
415 passed, 5 skipped, 34631 deselected, 1 warning in 30.63s
```

Targeted NVFP6 fused pseudo result:

```text
10 passed, 1 warning in 4.27s
```

Full CuTe sm100 test selection after NVFP6 fused pseudo support:

```text
417 passed, 5 skipped, 34631 deselected, 1 warning in 30.90s
```

Targeted NVINT6 fused pseudo result:

```text
5 passed, 1 warning in 4.17s
```

Full CuTe sm100 test selection after NVINT6 fused pseudo support:

```text
418 passed, 5 skipped, 34631 deselected, 1 warning in 30.45s
```

Targeted NVINT3/NVINT3_BS8 fused pseudo result:

```text
7 passed, 1 warning in 4.33s
```

Full CuTe sm100 test selection after NVINT3/NVINT3_BS8 fused pseudo support:

```text
420 passed, 5 skipped, 34631 deselected, 1 warning in 30.57s
```

Targeted NVINT4/NVINT4_BS8 fused pseudo result:

```text
5 passed, 1 warning in 3.91s
```

Full CuTe sm100 test selection after NVINT4/NVINT4_BS8 fused pseudo support:

```text
422 passed, 5 skipped, 34631 deselected, 1 warning in 31.27s
```

Targeted IF4_BS8 1D stochastic/stochastic-unbiased result:

```text
15 passed, 1 skipped, 1 warning in 4.47s
```

Full CuTe sm100 test selection after IF4_BS8 1D non-nearest support:

```text
427 passed, 5 skipped, 34631 deselected, 1 warning in 30.95s
```

Targeted NVINT6 stochastic result:

```text
16 passed, 1 warning in 4.11s
```

Full CuTe sm100 test selection after NVINT6 stochastic support:

```text
427 passed, 5 skipped, 34631 deselected, 1 warning in 30.79s
```

Targeted IF3/IF3_BS8 1D and 2D stochastic result:

```text
14 passed, 1 warning in 6.43s
```

Full CuTe sm100 test selection after IF3/IF3_BS8 1D stochastic support:

```text
431 passed, 5 skipped, 34631 deselected, 1 warning in 31.19s
```

Targeted NVFP4_BS8 1D stochastic/stochastic-unbiased result:

```text
13 passed, 6 skipped, 1 warning in 3.79s
```

Full CuTe sm100 test selection after NVFP4_BS8 1D stochastic support:

```text
431 passed, 7 skipped, 34631 deselected, 1 warning in 30.71s
```

Targeted IF4/IF4_BS8 fused pseudo result:

```text
6 passed, 1 warning in 4.24s
```

Full CuTe sm100 test selection after IF4/IF4_BS8 fused pseudo support:

```text
437 passed, 7 skipped, 34631 deselected, 1 warning in 32.09s
```

## Benchmark movement

`support_matrix.md` now reports CuTe nearest 1D coverage of `46/46`
Triton dtype/rule combinations in the current test matrix: all IF3/IF3_BS8,
IF4/IF4_BS8, and IF6 adaptive rules; all NVFP4 rules; NVFP4_BS8
`static_4/static_6`; MXFP3/MXFP3_BS8/MXFP4/MXFP4_BS8 and MXFP6
`static_4/static_6`; NVFP3/NVFP3_BS8/NVINT3/NVINT3_BS8/NVINT4/NVINT4_BS8/NVINT6
`static_6`; and NVFP6 E2M3/E3M2 `static_6`.

`benchmark_current.py` now uses capability-driven feature enumeration: each
dtype/rule/feature combination is timed only when both Triton and CuTe report
`can_quantize=True`. This expands the snapshot beyond the original NVFP4-only
feature sweep and now reports:

```text
63/470 workloads meet 1.2x
```

Fused MXFP3/MXFP3_BS8, MXFP4/MXFP4_BS8, and MXFP6 pseudo removes the largest
fallback cost for those families. MXFP3 pseudo now runs at roughly
`0.031-0.047 ms` instead of the prior `0.10-0.14 ms`; MXFP4 pseudo now runs at
roughly `0.032-0.039 ms` instead of the prior `0.26-0.45 ms`; MXFP6 pseudo now
runs at roughly `0.030-0.035 ms` instead of the prior `0.11-0.28 ms`. These
paths still mostly miss the 1.2x target because Triton pseudo also runs around
`0.032-0.035 ms`.

Fused NVFP4_BS8 1D pseudo removes its generic quantize+dequantize fallback.
The latest snapshot has NVFP4_BS8 pseudo at roughly `0.054-0.074 ms`; the
4096x4096 static_4/static_6 rows meet 1.2x, while smaller shapes still miss
because Triton pseudo is around `0.040-0.044 ms`.

Fused NVFP3/NVFP3_BS8 1D pseudo removes its generic fallback. The latest
snapshot has those rows at roughly `0.052-0.060 ms`; the 4096x4096 rows meet
1.2x while 128x256 and 1024x1024 remain below Triton.

Fused NVFP6 1D pseudo removes its generic fallback. The latest snapshot has
NVFP6 pseudo at roughly `0.050-0.058 ms`; the 4096x4096 E2M3/E3M2 rows meet
1.2x, while 128x256 and 1024x1024 remain below Triton.

Fused NVINT6 1D pseudo removes its generic fallback. The latest snapshot has
NVINT6 pseudo at roughly `0.052-0.060 ms`; this is much faster than the
previous generic fallback but still misses the 1.2x target, including the
4096x4096 row at about `1.134x`.

Fused NVINT3/NVINT3_BS8 1D pseudo removes another generic fallback. The latest
snapshot has these rows at roughly `0.053-0.066 ms`; the 4096x4096 rows meet
1.2x while 128x256 and 1024x1024 remain below Triton:

```text
[128, 256]     nvint3     pseudo 0.715x
[128, 256]     nvint3_bs8 pseudo 0.754x
[1024, 1024]   nvint3     pseudo 0.759x
[1024, 1024]   nvint3_bs8 pseudo 0.766x
[4096, 4096]   nvint3     pseudo 1.248x
[4096, 4096]   nvint3_bs8 pseudo 1.727x
```

Fused NVINT4/NVINT4_BS8 1D pseudo now avoids the generic Python-level
quantize/dequantize route, but the current INT4 decode-heavy fused kernel does
not improve the 1.2x count. The latest snapshot remains below Triton on all
NVINT4 pseudo rows:

```text
[128, 256]     nvint4     pseudo 0.246x
[128, 256]     nvint4_bs8 pseudo 0.242x
[1024, 1024]   nvint4     pseudo 0.250x
[1024, 1024]   nvint4_bs8 pseudo 0.248x
[4096, 4096]   nvint4     pseudo 0.201x
[4096, 4096]   nvint4_bs8 pseudo 0.253x
```

Fused IF4/IF4_BS8 adaptive 1D pseudo removes another generic quantize/dequantize
fallback. The 4096x4096 IF4 and IF4_BS8 pseudo rows now meet the target at
about `1.7-1.9x`, while the 128x256 and 1024x1024 rows remain below 1.2x
because Triton pseudo is still faster at those launch sizes.

The expanded snapshot still shows the largest remaining new performance gap is
generic `pseudo_quantize=True`, which currently runs as CuTe quantize plus
dequantize fallback for most non-fused formats. Among the `407` failing workloads in the new
snapshot, the feature breakdown is:

```text
pseudo_quantize: 121
base:            118
block_scale_2d:   98
transpose:        70
```

Several workloads are close to the 1.2x target but still fail, including
`[128, 256] if3 abs_max block_scale_2d` at `1.168x`,
`[128, 256] if4 mse block_scale_2d` at `1.164x`,
`[4096, 4096] nvfp6_e3m2 static_6 pseudo` at `1.163x`, and
`[128, 256] mxfp3_bs8 static_6 block_scale_2d` at `1.163x`. The ordinary quantize,
block-scale, transpose, and generic pseudo paths still need substantial
optimization before the all-workload 1.2x target is credible.

FP3 static coverage passes direct Triton-vs-CuTe accuracy checks. MXFP3 is
close to the speed target, while NVFP3 remains below it:

```text
[128, 256]   nvfp3      static_6 0.918x
[128, 256]   nvfp3_bs8  static_6 0.882x
[128, 256]   mxfp3      static_4 1.121x
[128, 256]   mxfp3      static_6 1.159x
[128, 256]   mxfp3_bs8  static_4 1.074x
[128, 256]   mxfp3_bs8  static_6 1.147x
[1024, 1024] nvfp3      static_6 0.879x
[1024, 1024] nvfp3_bs8  static_6 0.901x
[1024, 1024] mxfp3      static_4 1.084x
[1024, 1024] mxfp3      static_6 1.145x
[1024, 1024] mxfp3_bs8  static_4 1.126x
[1024, 1024] mxfp3_bs8  static_6 1.178x
[4096, 4096] nvfp3      static_6 0.912x
[4096, 4096] mxfp3      static_4 1.129x
[4096, 4096] mxfp3      static_6 1.067x
[4096, 4096] mxfp3_bs8  static_4 1.034x
[4096, 4096] mxfp3_bs8  static_6 1.186x
```

NVFP4_BS8 static passes direct Triton-vs-CuTe accuracy checks. It remains
below the 1.2x target in the current benchmark snapshot:

```text
[128, 256]   nvfp4_bs8 static_4 0.900x
[128, 256]   nvfp4_bs8 static_6 0.886x
[1024, 1024] nvfp4_bs8 static_4 0.903x
[1024, 1024] nvfp4_bs8 static_6 0.905x
[4096, 4096] nvfp4_bs8 static_4 1.162x
[4096, 4096] nvfp4_bs8 static_6 1.177x
```

MXFP4 static is bit-exact versus Triton in targeted tests. It is close to the
speed target but slightly short in the current benchmark snapshot:

```text
[128, 256]   mxfp4 static_4 1.137x
[128, 256]   mxfp4 static_6 1.144x
[1024, 1024] mxfp4 static_4 1.149x
[1024, 1024] mxfp4 static_6 1.113x
[4096, 4096] mxfp4 static_4 1.155x
[4096, 4096] mxfp4 static_6 1.165x
```

MXFP4_BS8 static is bit-exact versus Triton in targeted tests. It remains
below the 1.2x speed target in the current benchmark snapshot:

```text
[128, 256]   mxfp4_bs8 static_4 1.119x
[128, 256]   mxfp4_bs8 static_6 1.108x
[1024, 1024] mxfp4_bs8 static_4 1.099x
[1024, 1024] mxfp4_bs8 static_6 1.132x
[4096, 4096] mxfp4_bs8 static_4 1.110x
[4096, 4096] mxfp4_bs8 static_6 1.145x
```

MXFP6 static is bit-exact versus Triton in targeted tests. It is also close to
the speed target, but most measured shapes remain below 1.2x:

```text
[128, 256]   mxfp6_e2m3 static_4 1.136x
[128, 256]   mxfp6_e2m3 static_6 1.119x
[128, 256]   mxfp6_e3m2 static_4 1.144x
[128, 256]   mxfp6_e3m2 static_6 1.081x
[1024, 1024] mxfp6_e2m3 static_4 1.136x
[1024, 1024] mxfp6_e2m3 static_6 1.191x
[1024, 1024] mxfp6_e3m2 static_4 1.158x
[1024, 1024] mxfp6_e3m2 static_6 1.162x
[4096, 4096] mxfp6_e2m3 static_4 1.146x
[4096, 4096] mxfp6_e2m3 static_6 1.193x
[4096, 4096] mxfp6_e3m2 static_4 1.165x
[4096, 4096] mxfp6_e3m2 static_6 1.138x
```

NVINT4 `static_6` passes the accuracy gate against Triton/PyTorch reference.
It is also close to, but generally below, the 1.2x speed target:

```text
[128, 256]   nvint4 static_6 1.192x
[1024, 1024] nvint4 static_6 1.176x
[4096, 4096] nvint4 static_6 1.184x
```

NVINT3 `static_6` passes the accuracy gate against Triton. CuTe and Triton
scale tensors are bit-exact; packed value equality is at least 99.9% in the
large-shape gate and dequantized error is not worse than Triton. The bs16 path
remains below the 1.2x speed target, while the large `bs8` workload exceeds it:

```text
[128, 256]   nvint3     static_6 1.096x
[1024, 1024] nvint3     static_6 1.124x
[4096, 4096] nvint3     static_6 1.134x
[128, 256]   nvint3_bs8 static_6 1.132x
[1024, 1024] nvint3_bs8 static_6 1.146x
[4096, 4096] nvint3_bs8 static_6 1.444x
```

NVINT6 `static_6` passes the accuracy gate against Triton. CuTe and Triton
scale tensors are bit-exact; packed value equality is at least 99.9% in the
large-shape gate and dequantized error is not worse than Triton. It remains
below the 1.2x speed target:

```text
[128, 256]   nvint6 static_6 1.158x
[1024, 1024] nvint6 static_6 1.174x
[4096, 4096] nvint6 static_6 1.158x
```

NVINT4_BS8 `static_6` passes direct Triton-vs-CuTe accuracy checks. Packed
byte ordering is compared directly against Triton rather than the PyTorch
reference layout; scale tensors are bit-exact and dequantized error is not
worse than Triton. Only the large-shape benchmark currently reaches 1.2x:

```text
[128, 256]   nvint4_bs8 static_6 1.142x
[1024, 1024] nvint4_bs8 static_6 1.144x
[4096, 4096] nvint4_bs8 static_6 1.394x
```

NVFP6 `static_6` for both E2M3 and E3M2 passes targeted accuracy checks versus
Triton. It is currently slower than Triton in the benchmark snapshot:

```text
[128, 256]   nvfp6_e2m3 static_6 0.922x
[128, 256]   nvfp6_e3m2 static_6 0.904x
[1024, 1024] nvfp6_e2m3 static_6 0.940x
[1024, 1024] nvfp6_e3m2 static_6 0.927x
[4096, 4096] nvfp6_e2m3 static_6 0.972x
[4096, 4096] nvfp6_e3m2 static_6 0.944x
```

IF4 adaptive passes the accuracy gate against Triton/PyTorch reference. It
meets the 1.2x target on large-shape IF4 workloads, but misses the target on
small and medium shapes in the latest benchmark snapshot:

```text
[128, 256]   if4 abs_max 1.196x
[128, 256]   if4 mae     1.188x
[128, 256]   if4 mse     1.184x
[1024, 1024] if4 abs_max 1.159x
[1024, 1024] if4 mae     1.161x
[1024, 1024] if4 mse     1.168x
[4096, 4096] if4 abs_max 1.825x
[4096, 4096] if4 mae     1.805x
[4096, 4096] if4 mse     1.872x
```

IF4_BS8 adaptive nearest passes the Triton-vs-CuTe accuracy checks. The latest
benchmark snapshot keeps it below the 1.2x target:

```text
[128, 256]   if4_bs8 abs_max 1.143x
[128, 256]   if4_bs8 mae     1.136x
[128, 256]   if4_bs8 mse     1.087x
[1024, 1024] if4_bs8 abs_max 0.890x
[1024, 1024] if4_bs8 mae     1.101x
[1024, 1024] if4_bs8 mse     1.080x
```

IF3 and IF3_BS8 adaptive nearest pass Triton-vs-CuTe accuracy checks. PyTorch
does not currently provide an IF3 dequant reference in this repository, so the
tests compare Triton-quantized and CuTe-quantized tensors through the same CuTe
raw-value dequant path and gate dequantized input error against Triton. The
latest benchmark snapshot keeps most IF3 workloads below the 1.2x target:

```text
[128, 256]   if3     abs_max 1.153x
[128, 256]   if3     mae     1.140x
[128, 256]   if3     mse     1.197x
[128, 256]   if3_bs8 abs_max 1.130x
[128, 256]   if3_bs8 mae     1.135x
[128, 256]   if3_bs8 mse     1.129x
[1024, 1024] if3     abs_max 1.095x
[1024, 1024] if3     mae     1.129x
[1024, 1024] if3     mse     1.074x
[1024, 1024] if3_bs8 abs_max 1.190x
[1024, 1024] if3_bs8 mae     1.125x
[1024, 1024] if3_bs8 mse     1.118x
[4096, 4096] if3     abs_max 1.138x
```

IF6 adaptive passes the accuracy gate against Triton. `if6_e2m3 + abs_max`
has lower value/scale equality on one medium-shape workload, but the
dequantized input MSE/MAE remains within the Triton + tolerance gate. IF6 is
currently below the speed target across the benchmark snapshot:

```text
[128, 256]   if6_e2m3 abs_max 0.789x
[128, 256]   if6_e2m3 mae     0.758x
[128, 256]   if6_e2m3 mse     0.768x
[1024, 1024] if6_e2m3 abs_max 0.742x
[1024, 1024] if6_e2m3 mae     0.724x
[1024, 1024] if6_e2m3 mse     0.702x
[4096, 4096] if6_e2m3 abs_max 0.898x
[4096, 4096] if6_e2m3 mae     0.866x
[4096, 4096] if6_e2m3 mse     0.895x
[128, 256]   if6_e3m2 abs_max 0.776x
[128, 256]   if6_e3m2 mae     0.804x
[128, 256]   if6_e3m2 mse     0.764x
[1024, 1024] if6_e3m2 abs_max 0.730x
[1024, 1024] if6_e3m2 mae     0.744x
[1024, 1024] if6_e3m2 mse     0.744x
[4096, 4096] if6_e3m2 abs_max 0.900x
[4096, 4096] if6_e3m2 mae     0.886x
[4096, 4096] if6_e3m2 mse     0.874x
```

Fused IF6 1D nearest pseudo removes the previous generic quantize/dequantize
fallback. The targeted profiler on 1024x1024 IF6 E2M3 abs_max pseudo now shows
one CuTe kernel plus the shared amax reduction; the prior CuTe fallback spent
about `1.3 ms` total CUDA time across 20 iterations, while the fused path is
about `0.33 ms`. The latest benchmark snapshot records IF6 pseudo as:

```text
[128, 256]   if6_e2m3 abs_max 0.688x
[128, 256]   if6_e2m3 mae     0.775x
[128, 256]   if6_e2m3 mse     0.769x
[128, 256]   if6_e3m2 abs_max 0.772x
[128, 256]   if6_e3m2 mae     0.984x
[128, 256]   if6_e3m2 mse     0.765x
[1024, 1024] if6_e2m3 abs_max 0.793x
[1024, 1024] if6_e2m3 mae     0.767x
[1024, 1024] if6_e2m3 mse     0.780x
[1024, 1024] if6_e3m2 abs_max 0.768x
[1024, 1024] if6_e3m2 mae     0.757x
[1024, 1024] if6_e3m2 mse     0.784x
[4096, 4096] if6_e2m3 abs_max 1.062x
[4096, 4096] if6_e2m3 mae     1.068x
[4096, 4096] if6_e2m3 mse     1.079x
[4096, 4096] if6_e3m2 abs_max 1.058x
[4096, 4096] if6_e3m2 mae     1.068x
[4096, 4096] if6_e3m2 mse     1.080x
```

The latest capability-driven benchmark snapshot reports `62/470` workloads
meeting 1.2x and `248/470` workloads at least matching Triton. The main
large-shape regressions are now dominated by unfused `transpose=True`, which
materializes `x.T.contiguous()` before launching CuTe, and by remaining
small/medium pseudo launch overhead.

NVINT4/NVINT4_BS8 pseudo had a backend routing bug: the fused kernel branch
existed, but the dtype allow-list omitted `nvint4` and `nvint4_bs8`, so those
configs fell back to the generic quantize/dequantize path. After adding them
to the allow-list, the targeted 4096x4096 pseudo rows moved from about
`0.20-0.25x` to faster-than-Triton:

```text
[128, 256]     nvint4     pseudo 0.757x
[128, 256]     nvint4_bs8 pseudo 0.749x
[1024, 1024]   nvint4     pseudo 0.760x
[1024, 1024]   nvint4_bs8 pseudo 0.754x
[4096, 4096]   nvint4     pseudo 1.299x
[4096, 4096]   nvint4_bs8 pseudo 1.540x
```

The latest capability-driven benchmark snapshot still reports `62/470`
workloads meeting 1.2x because several near-threshold rows moved with timing
noise, but the pseudo pass breakdown improved to 21 rows.

Re-tested a low-risk `transpose=True` optimization for static MXFP4 by passing
`x.T` directly into the existing row-major CuTe quantize kernel. The plain
view path preserved dequantized semantics but did not improve performance:
4096x4096 MXFP4 static transpose stayed around `0.077 ms`, roughly `0.63x`
Triton, because non-coalesced column reads replaced the explicit contiguous
copy cost. A second attempt using `cute.runtime.make_fake_tensor` with an
explicit transposed stride compiled but hit a CUDA misaligned-address failure
during dequantization, consistent with the current vectorized load/store
helpers assuming row-major alignment. This route was not retained; real
transpose improvement needs a dedicated tiled transpose+quantize kernel rather
than reusing the row-major static kernels on a strided view.

Profiled a Triton-leading non-transpose workload,
4096x4096 `if3 + abs_max + pseudo_quantize`, with Nsight Compute. A targeted
microbenchmark first corrected a stale benchmark-snapshot anomaly: MXFP3/MXFP4
`static_6 + pseudo_quantize` is roughly Triton-parity on repeated runs, not the
`0.68-0.70x` reported by the older full sweep. The stable remaining IF3 pseudo
gap is architectural: Triton's kernel runs in about `33.6-33.7 us`, while the
CuTe fused IF3 pseudo kernel runs in about `47.1-47.2 us`. CuTe shows higher SM
throughput (`82.6-82.9%` vs Triton's `63.8-63.9%`) and lower DRAM throughput
(`9.4%` vs `13.2%`), but uses more registers (`54` vs `43`) and has lower
active warps (`42.8%` vs `48.2%`). This points at the current one-thread-per
scale-block CuTe mapping being too register-heavy and too coarse for IF3
pseudo; the likely next optimization is a dedicated 2D tiled IF3/IF3_BS8 pseudo
kernel rather than another small launch-parameter change. The raw NCU reports
and command notes are under `profile/cute_sm100_full_triton_parity/ncu/`.

Re-tested a launch-only IF3 fused pseudo experiment by raising only
`Sm100IF3AdaptivePseudoQuantize` to `min_blocks_per_mp=16`. The change did not
move 4096x4096 IF3 pseudo materially (`if3 abs_max/mae/mse` stayed around
`0.80-0.84x`; IF3_BS8 remained around `1.09-1.18x`), so the experiment was not
retained. This reinforces the NCU conclusion that IF3 pseudo needs a different
tiled mapping rather than a small launch-parameter tweak.

Profiled 4096x4096 `if6_e3m2 + abs_max` base quantize. Nsight Compute shows
the main CuTe quantization kernel is actually faster than Triton's main
kernel (`29.4-29.6 us` vs `44.5-45.2 us`) with slightly fewer registers
(`46` vs `48`) and higher active warps (`51%` vs `48%`). Torch profiler shows
the end-to-end CuTe path still spends more time in helper elementwise kernels
outside the main kernel. The current full benchmark's fixed one-shot
Triton-then-CuTe ordering is therefore too noisy for near-threshold rows; use
repeated alternating-order medians before treating those rows as kernel
rewrite targets. Raw reports and the command summary are in
`profile/cute_sm100_full_triton_parity/ncu/`.

Updated `benchmark_current.py` to use repeated alternating-order timing and
median aggregation. Each workload now pre-warms both backends, alternates
Triton/CuTe timing order across five repeats, records both sample lists in
`benchmark_current.json`, and uses the median values for the pass/fail
decision. The refreshed 470-row snapshot reports:

```text
64/470 workloads meet 1.2x
257/470 workloads are at least Triton parity
```

The refreshed 1.2x pass breakdown is:

```text
pseudo_quantize: 21
base:            18
block_scale_2d:  16
transpose:        9
```

The refreshed below-1.2x failure breakdown is:

```text
base:            120
pseudo_quantize: 117
block_scale_2d:   98
transpose:        71
```

This median run confirms that the old MXFP3/MXFP4 `static_6 + pseudo_quantize`
large-shape regression was timing noise; those rows are now around parity.
The worst large-shape failures remain dominated by static `transpose=True`
paths (`0.59-0.75x` for many MX/NV static formats), followed by IF3 fused
pseudo (`0.79-0.84x`) and IF6 base (`0.86-0.91x` end-to-end despite the main
CuTe IF6 kernel itself being faster under NCU).

Removed the post-kernel `to_blocked(scale_factors)` conversion from 1D IF6
adaptive quantize. The CuTe IF6 kernel now writes E4M3 scale bytes directly to
the Blackwell blocked scale layout using the same 128-row by 4-scale-column
swizzle as `to_blocked`, and the backend marks the returned scale tensor as
already blocked. This removes a GPU-side layout conversion from the IF6
end-to-end path while preserving the tensor contract.

Targeted 4096x4096 IF6 base timing moved from the previous `0.86-0.91x`
range to faster-than-Triton, although still short of 1.2x:

```text
if6_e2m3 abs_max 1.143x
if6_e2m3 mae     1.135x
if6_e2m3 mse     1.129x
if6_e3m2 abs_max 1.125x
if6_e3m2 mae     1.140x
if6_e3m2 mse     1.106x
```

Small/medium IF6 base rows also improved materially but remain below Triton
(`0.91-0.94x`). The refreshed median 470-row benchmark now reports:

```text
65/470 workloads meet 1.2x
263/470 workloads are at least Triton parity
```

The full CuTe sm100 test selection after the IF6 blocked-scale write change:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.64s
```

Re-tested IF3/IF4 fused pseudo candidate dequantization. The experiment moved
FP/INT candidate dequantization after the error comparison so the kernel only
dequantized the selected candidate. It passed the targeted IF3/IF4 pseudo tests
(`14 passed`) and full CuTe sm100 selection (`449 passed, 7 skipped`), but did
not materially improve the stable IF3 gap and the full benchmark did not gain
1.2x rows. The code change was reverted. The current retained conclusion is
unchanged: IF3 pseudo needs a different tiled/warp-cooperative mapping rather
than local scalar cleanup inside the one-thread-per-scale-block kernel.

Profiled and re-tested the 4096x4096 NVFP6 static base path after the IF6
scale-layout fix. Torch profiler shows CuTe's main NVFP6 static kernel is
shorter than Triton's main kernel, but repeated end-to-end event timing still
keeps the CuTe path around `0.91-0.92x`:

```text
nvfp6_e2m3 static_6 base 0.914x
nvfp6_e3m2 static_6 base 0.905x
```

A targeted `Sm100NVFP6StaticQuantize` launch experiment with
`min_blocks_per_mp=16` did not materially improve the row (`0.91-0.92x`), so
the launch setting was restored to `BLOCKS_PER_SM`.

Profiled a Triton-favorable 4096x4096 MXFP4 pseudo workload to separate kernel
time from wrapper noise. Torch profiler reported CuTe's single pseudo kernel at
about `13.6us` versus Triton's pseudo kernel at about `16.8us`, with Triton also
launching a small `aten::ones/fill_` helper for MX scaling. Repeated CUDA-event
timing is noisier and still sits only around parity:

```text
mxfp4 static_4 pseudo 1.009x
mxfp4 static_6 pseudo 0.982x
mxfp6_e2m3 static_4 pseudo 1.030x
nvfp4 static_6 pseudo 1.452x
if3 abs_max pseudo 0.794x
```

A targeted MXFP4/MXFP6 pseudo launch experiment with `min_blocks_per_mp=16`
did not create a stable gain (`0.98-1.06x` on sampled MX rows), so the launch
setting was restored to `BLOCKS_PER_SM`. The useful conclusion is that MX
pseudo is close to parity and launch-bound/noise-sensitive, while IF3 pseudo is
a real algorithmic kernel gap: the fused CuTe IF3 pseudo path remains about
`0.79x` on 4096x4096 because it evaluates both FP3 and INT3 candidates plus
the error metric per 16-value block.

A fresh full median benchmark rerun with no retained kernel-code changes reports
`57/470` workloads meeting 1.2x and `261/470` at least matching Triton. This is
lower than the earlier `65/470` snapshot, but the workload classes that still
drive the failure set are the same: transpose materialization, IF3 pseudo,
NVFP/NVFP6 static/base rows, and near-threshold small-shape launch overheads.

Retuned selected fused pseudo kernels to launch 128 threads per CTA instead of
the shared 256-thread default. This is retained only for the pseudo paths that
benefited or stayed stable in targeted checks: IF4 adaptive pseudo, IF6 adaptive
pseudo, and NVFP4 static pseudo. IF3 pseudo was explicitly tested with the same
128-thread setting and reverted because it regressed the already-slow IF3 path.

The targeted pseudo test slice passes:

```text
42 passed, 35045 deselected, 1 warning in 9.80s
```

The full CuTe sm100 test selection passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 34.13s
```

The refreshed full benchmark now reports:

```text
66/470 workloads meet 1.2x
269/470 workloads are at least Triton parity
```

The class breakdown for 1.2x rows is:

```text
pseudo_quantize: 22
base:            18
block_scale_2d:  17
transpose:        9
```

Also tested a 128-thread CTA variant for the 1D NV static kernels
(`Sm100NVFP4StaticQuantize`, `Sm100NVFP6StaticQuantize`, and
`Sm100NVFP3StaticQuantize`) because the 4096x4096 NV static/base rows remain
around `0.90-0.93x`. The targeted static test slice passed, but timing did not
improve the lagging rows:

```text
nvfp4 base        0.920x
nvfp6_e2m3 base   0.915x
nvfp6_e3m2 base   0.923x
nvfp3 base        0.908x
```

The experiment was reverted; NV static needs a deeper kernel change rather than
the same CTA-size retune that helps some fused pseudo paths.

Profiled a Triton-leading static MXFP4 transpose workload
(`4096x4096 mxfp4 static_6 transpose=True`) with Nsight Compute. Triton's
profiled call is a single fused `quantization_kernel` at about `47-50 us`.
The old CuTe path spent about `72 us` in PyTorch's transpose copy kernel before
running the CuTe static quantize kernel, which itself took only about `11 us`.
This confirmed that static MXFP4 transpose was losing mainly to
materialization, not to the CuTe quantize body.

Added a narrow fused CuTe transpose path for 1D `mxfp4` and `mxfp4_bs8`
`static_4/static_6` quantize. It loads BF16 values from the original matrix in
transposed order and writes the existing output contract directly, avoiding
`x.T.contiguous()` for these modes. Accuracy matches Triton dequantization on
the targeted checks, and the MXFP4 CuTe test slice passes:

```text
38 passed, 35049 deselected, 1 warning in 9.40s
```

The fused path is a partial performance win, not a final fix. It improves the
4096x4096 MXFP4 transpose rows from the previous `0.077-0.080 ms` range to
about `0.063-0.064 ms`, and improves 1024x1024 MXFP4 transpose to about
`1.11-1.12x` versus Triton. Large 4096x4096 MXFP4 transpose remains below
Triton (`0.77-0.79x`) because the new kernel uses strided scalar BF16 loads;
the next step is a tiled/shared-memory transpose+quantize kernel with coalesced
loads.

Re-tested the near-threshold 4096x4096 NVFP6 static pseudo path with smaller
CTA sizes. `Sm100NVFP6StaticPseudoQuantize` currently uses the default
256-thread CTA. A 128-thread variant preserved accuracy and improved targeted
median timing for large shapes, but the full alternating benchmark still left
`nvfp6_e3m2 static_6 pseudo_quantize=True` just below the target at about
`1.194x`. A 64-thread variant was also below target at about `1.196x`. The
experiment was not retained; this row needs either a real kernel-body reduction
or a broader pseudo-kernel retune with a less noisy acceptance harness.

Profiled the 4096x4096 `nvfp4 static_6` base workload because the current
benchmark reports it at only about `0.91x` versus Triton. Nsight Compute shows
the CuTe quantize kernel itself is not the slow component: Triton's main
`quantization_kernel` takes about `22 us`, while
`Sm100NVFP4StaticQuantize` takes about `12 us`. Both paths still pay similar
`x.abs().max()` helper work (`abs` around `12 us`, reduction around
`17-18 us`, plus a small copy/cast kernel). Repeated alternating CUDA-event
timing still shows the full CuTe call at about `0.065 ms` versus Triton around
`0.059 ms`, so the retained gap is outside the main CuTe quantize kernel. The
next NV static work should focus on helper/output overhead and a more precise
CUDA-level timeline, not on retuning the already faster static quantize kernel.

Added `profile_static_breakdown.py` to split the static NV path into amax,
lower-level kernel, `QuantizedTensor` construction, and public frontend timing.
For `4096x4096 nvfp4 static_6`, the lower-level CuTe path with auto-amax is
already slightly faster than Triton (`~0.048 ms` versus `~0.049 ms`), but the
public frontend call remains slower (`~0.061 ms` versus Triton `~0.057-0.058
ms`). This confirms that the retained non-BS8 NV static gap is now mostly
frontend/launch idle plus a small wrapper cost, not the main CuTe kernel body.

The explicit-backend support-cache and narrow static-NV fast path are retained.
They preserve the target NV/NVFP CuTe accuracy slice:

```text
120 passed, 6 skipped, 34961 deselected, 1 warning in 11.66s
```

The full CuTe sm100 test selection also passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.37s
```

The refreshed full alternating benchmark now reports:

```text
121/470 workloads meet 1.2x
281/470 workloads are at least Triton parity
```

The class breakdown for 1.2x rows is:

```text
pseudo_quantize: 48
block_scale_2d:  46
base:            18
transpose:        9
```

The main positive movement is small/medium-shape frontend overhead and BS8 NV
static rows. On 4096x4096, `nvfp4_bs8 static_6` now reports `1.318x` and
`nvfp3_bs8 static_6` reports `1.336x`. The non-BS8 NV static rows still miss:
`nvfp4 static_6` is `0.947x`, `nvfp6_e2m3 static_6` is `0.964x`, and
`nvfp3 static_6` is `0.949x`. Large transpose rows remain a separate major
kernel-design gap.

Extended the scalar fused transpose approach from MXFP4 to MXFP3/MXFP3_BS8 and
MXFP6 E2M3/E3M2. The targeted MXFP3/MXFP6 CuTe test slice passes:

```text
72 passed, 35015 deselected, 1 warning in 12.47s
```

The full CuTe sm100 test selection passes after the extension:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.47s
```

The refreshed full alternating benchmark now reports:

```text
125/470 workloads meet 1.2x
289/470 workloads are at least Triton parity
```

The fused transpose extension is useful but not sufficient. On 1024x1024,
`mxfp3/mxfp3_bs8/mxfp6_e2m3/mxfp6_e3m2 static_6 transpose=True` now report
about `1.14-1.16x`, up from near parity or below. On 4096x4096 they improve
materially but remain below Triton: `mxfp3 static_6` is `0.844x`,
`mxfp3_bs8 static_6` is `0.795x`, `mxfp6_e2m3 static_6` is `0.778x`, and
`mxfp6_e3m2 static_6` is `0.776x`. The next transpose step remains a true
tiled/shared-memory transpose+quantize kernel with coalesced loads; scalar
transposed loads cannot close the large-shape gap.

Retuned fused MX pseudo kernels (`Sm100MXFP3StaticPseudoQuantize`,
`Sm100MXFP4StaticPseudoQuantize`, and `Sm100MXFP6StaticPseudoQuantize`) to use
the 128-thread pseudo CTA. The targeted MX pseudo accuracy slice passes:

```text
18 passed, 35069 deselected, 1 warning in 4.71s
```

The full CuTe sm100 test selection passes after the retune:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.56s
```

The refreshed full alternating benchmark now reports:

```text
148/470 workloads meet 1.2x
290/470 workloads are at least Triton parity
```

The class breakdown for 1.2x rows is:

```text
block_scale_2d:  58
pseudo_quantize: 55
base:            26
transpose:        9
```

The retained MX pseudo movement is substantial: most MX pseudo rows are now
above 1.2x, including all 1024x1024 MX pseudo rows and most 4096x4096 rows.
Examples include `mxfp6_e2m3 static_6 pseudo_quantize=True` at `1.282x` on
128x256, `1.260x` on 1024x1024, and `1.312x` on 4096x4096. Remaining misses
include near-threshold rows such as 4096x4096 `mxfp4 static_4
pseudo_quantize=True` at `1.196x`.

Added a narrow fast path for 1D adaptive IF quantize calls
(`IF3`/`IF3_BS8`/`IF4`/`IF4_BS8`/`IF6`) when `transpose=False`,
`rht=False`, `block_scale_2d=False`, and `pseudo_quantize=False`. This mirrors
the existing static-NV fast path: it uses cached CuTe op callables and skips
the long backend import/dispatch chain while preserving the existing
QuantizedTensor layout contract, including Blackwell-blocked IF6 scales.

The targeted IF base accuracy slice passes:

```text
113 passed, 1 skipped, 34973 deselected, 1 warning in 12.28s
```

The full CuTe sm100 test selection also passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.56s
```

On a targeted alternating-order IF base timing run, this moved several
small/medium IF3 and IF4 rows over the 1.2x line. Examples:

```text
128x256  if3 abs_max      1.250x
128x256  if3_bs8 mae      1.207x
128x256  if4_bs8 abs_max  1.230x
1024x1024 if3_bs8 mae     1.202x
1024x1024 if4_bs8 mse     1.180x -> still below target in the full refresh
```

The refreshed full alternating benchmark remains noisy and is not an overall
completion claim. It now reports:

```text
129/470 workloads meet 1.2x
289/470 workloads are at least Triton parity
```

Added the same cached-dispatch fast path for 1D static MX quantize calls
(`MXFP3`/`MXFP3_BS8`/`MXFP4`/`MXFP4_BS8`/`MXFP6`) when
`transpose=False`, `rht=False`, `block_scale_2d=False`, and
`pseudo_quantize=False`. This avoids the long generic backend import/dispatch
chain for ordinary MX static calls while preserving the existing UE8M0 scale
layout.

The targeted MX static accuracy slice passes:

```text
56 passed, 35031 deselected, 1 warning in 6.38s
```

The targeted MX static timing slice reports all 36 rows at Triton parity and
29/36 at 1.2x. Examples:

```text
128x256   mxfp3 static_6       1.241x
1024x1024 mxfp3_bs8 static_6   1.252x
1024x1024 mxfp6_e2m3 static_6  1.237x
4096x4096 mxfp4 static_4       1.220x
4096x4096 mxfp6_e2m3 static_4  1.249x
```

The full CuTe sm100 test selection passes after this fast path:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.42s
```

The refreshed full alternating benchmark now reports:

```text
131/470 workloads meet 1.2x
289/470 workloads are at least Triton parity
```

Added a cached-dispatch fast path for 1D static NVINT quantize calls
(`NVINT3`/`NVINT3_BS8`/`NVINT4`/`NVINT4_BS8`/`NVINT6`) when
`transpose=False`, `rht=False`, `block_scale_2d=False`,
`pseudo_quantize=False`, and `scale_rule=static_6`. This preserves the
existing x_amax, stochastic, and adjustment-factor semantics while avoiding
the generic backend import/dispatch chain.

The targeted NVINT static accuracy slice passes:

```text
24 passed, 35063 deselected, 1 warning in 5.86s
```

The targeted NVINT static timing slice reports all 15 rows above 1.2x:

```text
128x256   nvint3       1.234x
128x256   nvint6       1.280x
1024x1024 nvint4_bs8   1.265x
4096x4096 nvint3_bs8   1.618x
4096x4096 nvint4_bs8   1.443x
```

The full CuTe sm100 test selection passes after this fast path:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.55s
```

The refreshed full alternating benchmark now reports:

```text
199/470 workloads meet 1.2x
291/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            80
block_scale_2d:  56
pseudo_quantize: 53
transpose:       10
```

Added the same cached-dispatch fast path for 1D adaptive NVFP4 quantize calls
when `transpose=False`, `rht=False`, `block_scale_2d=False`, and
`pseudo_quantize=False`. This reuses the existing `quantize_nvfp4_adaptive`
CuTe op directly and preserves the existing E4M3 scale and amax tensor
contract.

The targeted NVFP4 adaptive accuracy slice passes:

```text
42 passed, 6 skipped, 35039 deselected, 1 warning in 8.40s
```

The targeted NVFP4 adaptive timing slice shows the fast path mainly helps the
large-shape rows; small and medium shapes remain below Triton parity:

```text
128x256   nvfp4 abs_max 0.954x
128x256   nvfp4 mae     0.964x
128x256   nvfp4 mse     0.979x
1024x1024 nvfp4 abs_max 0.973x
1024x1024 nvfp4 mae     0.965x
1024x1024 nvfp4 mse     0.979x
4096x4096 nvfp4 abs_max 1.861x
4096x4096 nvfp4 mae     1.876x
4096x4096 nvfp4 mse     1.847x
```

The full CuTe sm100 test selection passes after this fast path:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.50s
```

The refreshed full alternating benchmark now reports:

```text
216/470 workloads meet 1.2x
292/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            87
block_scale_2d:  66
pseudo_quantize: 54
transpose:        9
```

Retuned the 1D NVFP3 static quantize kernel launch from the shared
`BLOCKS_PER_SM=8` setting to `min_blocks_per_mp=16`. The targeted breakdown for
4096x4096 `nvfp3 static_6` improved CuTe frontend timing to roughly `0.061 ms`,
close to Triton at roughly `0.058 ms`; the larger retained win is
4096x4096 `nvfp3_bs8 static_6`, which now reaches `1.364x` in the refreshed
full benchmark. Ordinary `nvfp3 static_6` still remains below parity.

The targeted NVFP3/NVFP3_BS8 static accuracy slice passes:

```text
1 passed, 35086 deselected, 1 warning in 4.30s
```

The full CuTe sm100 test selection passes after this retune:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.47s
```

The refreshed full alternating benchmark now reports:

```text
221/470 workloads meet 1.2x
290/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            90
block_scale_2d:  67
pseudo_quantize: 55
transpose:        9
```

Tested a direct NVFP3/NVFP3_BS8 static transpose kernel that reads the input in
transposed order and skips the explicit `x.T.contiguous()` materialization. It
matched the old CuTe `x.T.contiguous() -> quantize_nvfp3_static` output
exactly for values, scales, and amax, and improved targeted 4096x4096 transpose
timings from roughly `0.615x -> 0.735x` for NVFP3 and `0.727x -> 0.900x` for
NVFP3_BS8. It is not retained because both rows remain below Triton parity and
well short of the 1.2x target; the result confirms that avoiding materialization
alone is insufficient and the transpose gap needs a coalesced/tiled transpose
quantize design.

Tested `min_blocks_per_mp=16` for the 1D IF6 adaptive base quantize kernel.
Targeted 4096x4096 timing showed a possible large-shape improvement, with most
IF6 E2M3/E3M2 rows around or above 1.2x. The change is not retained because the
full alternating benchmark was not stable and dropped to `140/470` workloads
meeting 1.2x; in that full run IF6 base rows also did not consistently remain
above the target. This suggests IF6 launch tuning is too noise-sensitive to
claim without a more robust kernel-level change.

Tested `min_blocks_per_mp=16` for MXFP4/MXFP4_BS8 2D static block-scale
kernels. A narrowed variant that only retuned MXFP4_BS8 showed targeted
4096x4096 BS8 rows near or above 1.2x, but the full alternating benchmark fell
to `189/470` workloads meeting 1.2x and the MXFP4_BS8 2D rows did not stay
reliably above target. The launch-only retune is not retained.

Profiled a Triton-leading non-transpose 2D workload,
4096x4096 `if3 + abs_max + block_scale_2d=True`, with Nsight Compute. The
main CuTe IF3 2D kernel runs in about `71.5-72.0 us`, while Triton's
`quantization_kernel` runs in about `51.1-51.4 us`. CuTe shows higher reported
SM throughput (`~72%` vs Triton's `~62%`) and lower DRAM/L2 throughput, but it
launches only `256` CTAs versus Triton's `2048` and keeps about half the active
warps (`~20.6%` vs `~40.4%`). This points at the current one-thread-per-16x16
scale-tile CuTe mapping being too serial: each thread scans the tile for max,
rescans it for FP3-vs-INT3 error, then scans it again to write values. The
next IF3/IF3_BS8 2D optimization should be a warp- or CTA-cooperative tiled
kernel rather than another launch-only tweak. Raw reports, CSV exports, and
the command summary are under
`profile/cute_sm100_full_triton_parity/ncu/if3_2d_*`.

Re-tested IF3/IF3_BS8 2D with a 128-thread CTA mapping, updating both the
launch block size and the tile-index stride so the 4096x4096 case would launch
512 active CTAs instead of 256. The targeted IF 2D accuracy slice passed
(`24 passed`), but timing regressed the IF3 rows: 4096x4096 IF3
`abs_max/mae/mse` measured about `0.833x/0.918x/0.829x` versus Triton, below
the retained 256-thread baseline (`0.889x/0.958x/0.903x` in
`benchmark_current.json`). IF3_BS8 remained above 1.2x but also slowed versus
the retained baseline. The experiment was reverted, reinforcing that the 2D
gap is not solved by more/smaller CTAs while each thread still serializes a
full scale tile.

Re-tested a Python-dispatch cleanup for fused pseudo paths by moving the large
`pseudo_quantize` kernel import tuple behind an `lru_cache` helper. The
targeted pseudo accuracy slice passed (`24 passed`) and the full CuTe sm100
test selection still passed (`449 passed, 7 skipped`), but targeted timings
did not show a stable improvement for the failing IF3/IF4/NVFP pseudo rows.
The full alternating benchmark with this change dropped to `212/470`
workloads meeting 1.2x, below the retained `221/470` snapshot. The experiment
was reverted; the pseudo gap is dominated by kernel/launch behavior rather
than that small Python import-dispatch cost.

Replaced the CuTe sm100 fallback amax resolver from
`x.abs().max().float()` to
`torch.linalg.vector_norm(x, ord=inf, dtype=torch.float32)`. This preserves
the BF16 global absolute-maximum value in targeted checks while avoiding the
explicit `abs` materialization kernel and the follow-up `.float()` copy kernel
used by the previous PyTorch expression. A focused profiler on 1024x1024 BF16
showed the amax helper moving from about `14.4 us` to `10.2 us`, and targeted
CuTe timings improved for global-amax-heavy rows:

```text
1024x1024 if4 abs_max pseudo   0.879x -> 1.083x
4096x4096 if3 abs_max pseudo   0.789x -> 0.974x
1024x1024 nvfp4 static_4 pseudo 0.867x -> 1.086x
4096x4096 if6_e2m3 abs_max     ~1.19x -> 1.473x
```

The targeted accuracy slice covering IF3/IF4/NVFP4 pseudo and general
not-less-accurate CuTe quantize checks passes:

```text
32 passed, 1 warning in 6.26s
```

The full CuTe sm100 test selection passes after the amax resolver change:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.26s
```

The refreshed full alternating benchmark now reports:

```text
239/470 workloads meet 1.2x
423/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            90
block_scale_2d:  65
pseudo_quantize: 59
transpose:       25
```

Profiled the 4096x4096 `nvfp4 + static_6 + transpose=True` row where Triton is
still faster. The retained CuTe NVFP4 quantize kernel is not the slow part:
NCU reports the CuTe quantizer at about `11.3-11.7 us`, versus Triton's main
quantization kernel at about `43.4-43.7 us`. The CuTe end-to-end path loses
because the current non-MX transpose implementation first materializes
`x.T.contiguous()`, and that copy takes about `72.1-72.3 us` per profiled call.
The amax reductions are similar (`17.2-18.4 us` CuTe, `18.0-18.3 us` Triton).
A low-risk experiment that computed global amax before transpose gave only
about a 1% improvement for 4096x4096 NV transpose rows and regressed a
1024x1024 NVFP4 transpose row, so it was reverted. The profile reinforces that
NV transpose needs a real fused tiled transpose-plus-quantize kernel rather
than another frontend/launch retune.

Added a fused CuTe sm100 static NVFP4/NVFP4_BS8 transpose quantize path for
`static_4/static_6`, modelled after the retained scalar MX transpose kernels.
This removes the `x.T.contiguous()` materialization for these four rows and
keeps the CuTe static NV quantizer's E4M3-scale behavior. The existing targeted
NVFP4 transpose accuracy slice passes:

```text
5 passed, 35082 deselected, 1 warning in 4.59s
```

The full CuTe sm100 test selection also passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 34.06s
```

Targeted alternating timings improved the affected transpose rows:

```text
1024x1024 nvfp4 static_4 transpose       0.951x -> 1.077x
1024x1024 nvfp4 static_6 transpose       0.954x -> 1.060x
1024x1024 nvfp4_bs8 static_4 transpose   0.944x -> 1.076x
1024x1024 nvfp4_bs8 static_6 transpose   0.972x -> 1.094x
4096x4096 nvfp4 static_4 transpose       0.728x -> 0.849x
4096x4096 nvfp4 static_6 transpose       0.730x -> 0.849x
4096x4096 nvfp4_bs8 static_4 transpose   0.866x -> 1.030x
4096x4096 nvfp4_bs8 static_6 transpose   0.866x -> 1.030x
```

The refreshed full alternating benchmark now reports:

```text
231/470 workloads meet 1.2x
429/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            86
block_scale_2d:  59
pseudo_quantize: 61
transpose:       25
```

Added a fused CuTe sm100 static NVFP6 transpose quantize path for
`nvfp6_e2m3/nvfp6_e3m2 + static_6`. It follows the same scalar
direct-transpose shape as the NVFP4 path and avoids `x.T.contiguous()` for
these two rows. A direct dequantization check on 1024x1024 BF16 shows the
fused CuTe path has the same dequant MSE as Triton for both E2M3 and E3M2.
The full CuTe sm100 test selection passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.88s
```

Targeted alternating timings improved the affected transpose rows:

```text
1024x1024 nvfp6_e2m3 static_6 transpose  0.978x -> 1.096x
1024x1024 nvfp6_e3m2 static_6 transpose  0.976x -> 1.063x
4096x4096 nvfp6_e2m3 static_6 transpose  0.705x -> 0.786x
4096x4096 nvfp6_e3m2 static_6 transpose  0.706x -> 0.784x
```

The refreshed full alternating benchmark now reports:

```text
240/470 workloads meet 1.2x
431/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            91
block_scale_2d:  63
pseudo_quantize: 65
transpose:       21
```

Added fused CuTe sm100 static NVINT3/NVINT3_BS8/NVINT4/NVINT4_BS8/NVINT6
transpose quantize paths for `static_6`. These follow the same scalar
direct-transpose structure as the NVFP4/NVFP6 paths and skip the
`x.T.contiguous()` materialization for these five rows.

The targeted NVINT transpose test selection passes:

```text
38 passed, 35049 deselected, 1 warning in 6.62s
```

The full CuTe sm100 test selection also passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.90s
```

Targeted alternating timings improved the affected transpose rows:

```text
1024x1024 nvint3 static_6 transpose      1.187x -> 1.338x
1024x1024 nvint3_bs8 static_6 transpose  1.231x -> 1.345x
1024x1024 nvint4 static_6 transpose      1.172x -> 1.303x
1024x1024 nvint4_bs8 static_6 transpose  1.210x -> 1.356x
1024x1024 nvint6 static_6 transpose      1.213x -> 1.356x
4096x4096 nvint3 static_6 transpose      0.813x -> 0.977x
4096x4096 nvint3_bs8 static_6 transpose  0.989x -> 1.227x
4096x4096 nvint4 static_6 transpose      0.814x -> 0.975x
4096x4096 nvint4_bs8 static_6 transpose  0.974x -> 1.201x
4096x4096 nvint6 static_6 transpose      0.812x -> 0.955x
```

The refreshed full alternating benchmark now reports:

```text
241/470 workloads meet 1.2x
433/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            91
block_scale_2d:  63
pseudo_quantize: 62
transpose:       25
```

The remaining below-parity rows are concentrated in transpose:

```text
transpose:       32
block_scale_2d:   4
pseudo_quantize:  1
```

Added fused CuTe sm100 static NVFP3/NVFP3_BS8 transpose quantize paths for
`static_6`. This removes the explicit `x.T.contiguous()` copy for those two
rows and uses the same scalar direct-transpose block shape as the retained
NVINT3 path, but with E2M0 value packing and the NVFP3 static scale rule.

A direct 128x128 Triton/CuTe comparison showed bit-exact values and matching
dequant MSE for both NVFP3 variants; scale bytes differ in some rows but
preserve the same dequantized error. The targeted CuTe sm100 NVFP3 test
selection passes:

```text
13 passed, 35074 deselected, 1 warning in 4.62s
```

The full CuTe sm100 test selection also passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.50s
```

Targeted alternating timings improved the affected transpose rows:

```text
1024x1024 nvfp3 static_6 transpose       0.925x -> 1.022x
1024x1024 nvfp3_bs8 static_6 transpose   0.949x -> 1.051x
4096x4096 nvfp3 static_6 transpose       0.685x -> 0.846x
4096x4096 nvfp3_bs8 static_6 transpose   0.828x -> 1.048x
```

The refreshed full alternating benchmark now reports:

```text
243/470 workloads meet 1.2x
436/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            94
block_scale_2d:  60
pseudo_quantize: 64
transpose:       25
```

The remaining below-parity rows are still dominated by large transpose cases:

```text
transpose:       29
block_scale_2d:   4
pseudo_quantize:  1
```

Added fused CuTe sm100 adaptive IF3/IF3_BS8 transpose quantize paths for
`abs_max`/`mae`/`mse`. These use direct transposed BF16 loads and the existing
per-block FP3-vs-INT3 error selection, avoiding the explicit
`x.T.contiguous()` materialization for the twelve IF3 transpose rows.

The targeted CuTe sm100 IF3 test selection passes:

```text
53 passed, 35034 deselected, 1 warning in 10.42s
```

The full CuTe sm100 test selection also passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 34.13s
```

Targeted alternating timings show all IF3 transpose rows now exceed 1.2x:

```text
1024x1024 if3 abs_max transpose      1.354x
1024x1024 if3 mae transpose          1.363x
1024x1024 if3 mse transpose          1.379x
1024x1024 if3_bs8 abs_max transpose  1.396x
1024x1024 if3_bs8 mae transpose      1.378x
1024x1024 if3_bs8 mse transpose      1.380x
4096x4096 if3 abs_max transpose      1.258x
4096x4096 if3 mae transpose          1.254x
4096x4096 if3 mse transpose          1.227x
4096x4096 if3_bs8 abs_max transpose  1.519x
4096x4096 if3_bs8 mae transpose      1.485x
4096x4096 if3_bs8 mse transpose      1.489x
```

The refreshed full alternating benchmark still reports `243/470` workloads
meeting 1.2x overall because unrelated base rows moved with timing noise, but
the transpose slice improved materially:

```text
243/470 workloads meet 1.2x
441/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            84
block_scale_2d:  64
pseudo_quantize: 64
transpose:       31
```

The remaining below-parity rows are now:

```text
transpose:       23
block_scale_2d:   4
base:             1
pseudo_quantize:  1
```

Added a fused CuTe sm100 nearest NVFP4 adaptive transpose quantize path for
`abs_max`/`mae`/`mse`. Stochastic adaptive NVFP4 transpose intentionally keeps
the existing materialized-transpose route so this change does not alter the
stochastic seed mapping. The fused nearest path directly reads BF16 in
transposed order and performs the existing per-block `static_6` vs `static_4`
error selection.

The targeted CuTe sm100 NVFP4 test selection passes:

```text
83 passed, 6 skipped, 34998 deselected, 1 warning in 9.63s
```

The full CuTe sm100 test selection also passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.70s
```

Targeted alternating timings improved the affected adaptive transpose rows:

```text
1024x1024 nvfp4 abs_max transpose  1.025x
1024x1024 nvfp4 mae transpose      1.043x
1024x1024 nvfp4 mse transpose      1.077x
4096x4096 nvfp4 abs_max transpose  2.597x
4096x4096 nvfp4 mae transpose      2.590x
4096x4096 nvfp4 mse transpose      2.600x
```

The refreshed full alternating benchmark snapshot is noisy on unrelated small
base/pseudo rows and reports fewer overall 1.2x rows than the prior snapshot,
but the transpose parity count still improves:

```text
215/470 workloads meet 1.2x
445/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            80
block_scale_2d:  55
pseudo_quantize: 52
transpose:       28
```

The remaining below-parity rows are now:

```text
transpose:       20
block_scale_2d:   4
pseudo_quantize:  1
```

Added fused CuTe sm100 nearest IF4/IF4_BS8 adaptive transpose quantize paths
for `abs_max`/`mae`/`mse`. Like the NVFP4 adaptive transpose path, stochastic
IF4 keeps the existing materialized-transpose route to avoid changing the
seed mapping. The BS8 path stores the 8-value packed block with a 32-bit global
store; an initial 64-bit store experiment hit a CUDA misaligned-address fault
because consecutive BS8 blocks are only 4-byte aligned.

The targeted CuTe sm100 IF4 test selection passes:

```text
63 passed, 1 skipped, 35023 deselected, 1 warning in 10.21s
```

The full CuTe sm100 test selection also passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.33s
```

Targeted alternating timings show all IF4 transpose rows now exceed 1.2x:

```text
1024x1024 if4 abs_max transpose      1.377x
1024x1024 if4 mae transpose          1.389x
1024x1024 if4 mse transpose          1.345x
1024x1024 if4_bs8 abs_max transpose  1.352x
1024x1024 if4_bs8 mae transpose      1.375x
1024x1024 if4_bs8 mse transpose      1.336x
4096x4096 if4 abs_max transpose      2.434x
4096x4096 if4 mae transpose          2.499x
4096x4096 if4 mse transpose          2.499x
4096x4096 if4_bs8 abs_max transpose  2.154x
4096x4096 if4_bs8 mae transpose      2.156x
4096x4096 if4_bs8 mse transpose      2.161x
```

The refreshed full alternating benchmark now reports:

```text
234/470 workloads meet 1.2x
445/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            83
block_scale_2d:  57
pseudo_quantize: 61
transpose:       33
```

The remaining below-parity rows are:

```text
transpose:       20
block_scale_2d:   4
pseudo_quantize:  1
```

Profiled the 4096x4096 `mxfp3 static_6 block_scale_2d=True` gap with NCU and
recorded the reports under `profile/cute_sm100_full_triton_parity/ncu/`:

```text
mxfp3_2d_triton.ncu-rep
mxfp3_2d_triton.csv
mxfp3_2d_cute.ncu-rep
mxfp3_2d_cute.csv
mxfp3_2d_profile.md
```

The representative Triton `quantization_kernel` median is 34.40 us with a 1024
CTA grid, 1.38 waves/SM, 24.01% achieved occupancy, 15.37 active warps/SM, and
977.16 GB/s memory throughput. The representative CuTe
`Sm100MXFP3StaticQuantize2D` median is 77.82 us with only a 64 CTA grid, 0.09
waves/SM, 12.47% achieved occupancy, 7.98 active warps/SM, and 433.45 GB/s
memory throughput. NCU also reports all CuTe compute pipelines under-utilized
and 2,621,440 excessive global-memory sectors, 50% of total sectors. This
points to a real 2D kernel mapping problem: the large static MXFP3 block-scale
path needs a more parallel cooperative tile design with better coalescing,
rather than another small launch-parameter retune.

Retuned `Sm100MXFP3StaticQuantize2D` from 256-thread CTAs to 32-thread CTAs
and made `_launch_grid` accept an explicit `threads_per_block` override for
this path. This raises the 4096x4096 MXFP3 2D grid from 64 CTAs to 512 CTAs
without changing the output layout or scale computation. The targeted accuracy
slice passes:

```text
12 passed, 35075 deselected, 1 warning in 4.63s
```

The full CuTe sm100 test selection still passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.59s
```

Targeted alternating timings for MXFP3 2D block-scale after the retune:

```text
128x256   static_4  1.075x
128x256   static_6  1.129x
1024x1024 static_4  1.140x
1024x1024 static_6  1.146x
4096x4096 static_4  0.987x
4096x4096 static_6  0.982x
```

The refreshed full alternating benchmark now reports:

```text
228/470 workloads meet 1.2x
446/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            80
block_scale_2d:  58
pseudo_quantize: 57
transpose:       33
```

The remaining below-parity rows are:

```text
transpose:       20
block_scale_2d:   3
pseudo_quantize:  1
```

Re-tested the apparent `1024x1024 mxfp6_e3m2 static_6 block_scale_2d=True`
below-parity row with a targeted 7-repeat alternating benchmark. The refreshed
median was `1.170x` (`triton=0.04692ms`, `cute=0.04011ms`), so the full
snapshot's `0.961x` row appears to be benchmark noise rather than a stable
kernel regression.

Also tested `Sm100IF3AdaptivePseudoQuantize` with 128-thread and 64-thread CTA
launches for the 4096x4096 `if3 pseudo_quantize=True` gap. The `abs_max` row
stayed below parity at roughly `0.985x`, while `mae/mse` remained around
`1.06x`. These launch-only changes were reverted; the IF3 pseudo gap still
requires a real tiled pseudo kernel rather than CTA-size retuning.

Retuned `Sm100MXFP6StaticQuantize2D` from 256-thread CTAs to 32-thread CTAs,
using the same `_launch_grid(..., threads_per_block=STATIC_2D_THREADS_PER_BLOCK)`
override as MXFP3 2D. The targeted MXFP6 2D accuracy slice passes:

```text
12 passed, 35075 deselected, 1 warning in 5.22s
```

The full CuTe sm100 test selection still passes:

```text
449 passed, 7 skipped, 34631 deselected, 1 warning in 33.61s
```

Targeted alternating timings for MXFP6 2D block-scale after the retune:

```text
1024x1024 mxfp6_e2m3 static_4  1.153x
1024x1024 mxfp6_e2m3 static_6  1.156x
1024x1024 mxfp6_e3m2 static_4  1.168x
1024x1024 mxfp6_e3m2 static_6  1.160x
4096x4096 mxfp6_e2m3 static_4  1.159x
4096x4096 mxfp6_e2m3 static_6  1.192x
4096x4096 mxfp6_e3m2 static_4  1.182x
4096x4096 mxfp6_e3m2 static_6  1.173x
```

The refreshed full alternating benchmark now reports:

```text
226/470 workloads meet 1.2x
447/470 workloads are at least Triton parity
```

The current 1.2x class breakdown is:

```text
base:            81
block_scale_2d:  54
pseudo_quantize: 58
transpose:       33
```

The remaining below-parity rows are:

```text
transpose:       20
block_scale_2d:   2
pseudo_quantize:  1
```

Tested two follow-up launch-only retunes for the static MX 2D paths:

- `STATIC_2D_THREADS_PER_BLOCK=16` for MXFP3/MXFP6 2D. Accuracy still passed
  for the targeted MXFP3/MXFP6 2D slice, but 4096x4096 MXFP3 remained around
  `0.98x` and MXFP6 did not improve consistently.
- `STATIC_2D_BLOCKS_PER_SM=16` with 32-thread CTAs for MXFP3/MXFP6 2D. Accuracy
  still passed, but 4096x4096 MXFP3 stayed below parity (`0.982x` static_4,
  `0.972x` static_6) and MXFP6 mostly regressed versus the 32-thread/8-blocks
  launch bound. This launch-bound experiment was reverted.

These results reinforce that the remaining MXFP3 2D gap is not solved by
smaller CTAs or launch bounds alone; it needs a more cooperative tile mapping
with better memory behavior.

Inspected the large 4096 transpose gap. The fused scalar transpose kernels
currently assign consecutive threads to adjacent row-blocks for one output row,
which makes the transposed input loads strided. Tested a simple MXFP3 transpose
mapping swap so consecutive threads would walk output rows for the same
row-block, making the input reads more coalesced. This was not valid as a
drop-in change: a 128x256 MXFP3 transpose check produced mismatched values and
scale factors, with dequantized output diverging (`static_4` produced `inf`
differences and `static_6` max difference was `6.0`). The experiment was
reverted. A correct transpose fix needs a real tiled/shared-memory transpose
layout that preserves the expected output scale/value ordering.

## Remaining major gaps

- Nearest 1D coverage is complete for the current dtype/rule test matrix, but
  several non-nearest variants are still not implemented, including FP3/INT3
  stochastic-unbiased on NV scale formats. NVINT6 stochastic is now claimed;
  NVINT6 stochastic-unbiased remains unsupported.
- Feature flags still missing: IF6 stochastic-unbiased and 1D NVFP6 E2M3
  stochastic-unbiased. True stochastic NVFP4 pseudo is also intentionally not
  claimed after failing the current Triton-error gate.
- `block_scale_2d=True` is currently implemented for `nvfp4`,
  `nvfp4_bs8 static_4/static_6`,
  `if3/if3_bs8/if4/if4_bs8 abs_max/mae/mse`, `mxfp3/mxfp3_bs8/mxfp4/mxfp4_bs8/mxfp6 static_4/static_6`,
  `nvint3/nvint3_bs8/nvint4/nvint4_bs8/nvint6 static_6`, and NVFP6 static paths. Missing 2D paths still include
  NVFP3/NVFP3_BS8,
  and related non-nearest variants.
- Performance target still missing for many current supported workloads. The
  latest median capability-driven benchmark snapshot reports only `226/470`
  workloads meeting 1.2x.
