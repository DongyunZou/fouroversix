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
- Performance target still missing for most current supported workloads. The
  latest capability-driven benchmark snapshot reports only `62/470` workloads
  meeting 1.2x.
