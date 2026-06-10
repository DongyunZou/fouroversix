# TASK.md

## Context

The CuTe SM120 quantization backend currently has deterministic numerical
mismatches against the Triton backend on larger FourOverSix NVFP4 shapes.
Earlier probes showed that repeated runs of the same backend are bitwise stable
and that passing the same `x_amax` to Triton and CuTe SM120 does not remove the
mismatch. This points to implementation differences in scale arithmetic,
rounding/conversion points, and fast-math behavior rather than nondeterminism.

The first phase is to make the CuTe SM120 quantization path use regular
round-to-nearest arithmetic wherever Triton does, then observe how much of the
current quantization mismatch disappears. This phase is an accuracy-alignment
task first; performance recovery is a follow-up task after the numerical
behavior is understood.

Relevant Triton reference behavior:

- `src/fouroversix/kernels/triton/quantize.py` uses explicit `tl.div_rn` for
  NVFP4 scale encode/decode and block scaling.
- Triton FourOverSix selection uses strict `<` for 4-vs-6 error comparison, so
  exact ties choose the six path.
- Triton FP4 conversion uses `cvt.rn.satfinite.e2m1x2.f32` on SM100/SM120.
- Triton E4M3 scale conversion uses round-to-nearest hardware conversion on
  supported architectures.

Known CuTe SM120 fast-math and parity risk areas:

- `src/fouroversix/kernels/cute_sm120/fp4_common.py` defines and uses
  `rcp.approx.ftz.f32`.
- `nvfp4_compute_output_scale` uses approximate reciprocal with FTZ for final
  quantization scaling.
- Several CuTe SM120 kernels compute scale values as
  `block_max * rcp_approx_ftz(constant)` instead of round-to-nearest division.
- Error helpers use `abs.ftz.f32` and `max.ftz.f32` in places where the Triton
  path does not explicitly request FTZ behavior.
- CuTe SM120 currently computes separate `output_scale`, `selection_scale`, and
  `dequant_scale` paths; the final quantized value path and the error-selection
  path may therefore disagree at rounding boundaries.

## Constraints

- MUST only modify CuTe SM120 code.
- MUST NOT modify Triton backend code.
- MUST NOT modify CUDA backend code.
- MUST NOT modify CuTe SM100 backend code.
- MUST NOT hide CUDA or Triton fallbacks behind the CuTe SM120 backend name.
- MUST NOT use fast math in the accepted CuTe SM120 quantization path.
- MUST NOT introduce compiler options that enable approximate or unsafe math.
- MUST replace explicit approximate operations such as `rcp.approx.ftz.f32`
  with regular round-to-nearest operations where they affect SM120
  quantization.
- MUST avoid FTZ-only arithmetic where the Triton reference path has no
  equivalent FTZ behavior.
- MUST preserve Triton-compatible tie-breaking for FourOverSix selection:
  strict `error_4 < error_6`, with ties selecting six.
- MUST keep changes focused on quantization/dequantization arithmetic and
  acceptance tooling for this phase.
- MUST record accuracy observations before and after the change on the same GPU
  and shape matrix.
- SHOULD bind profiling and accuracy probes to an idle RTX 5090 GPU.
- SHOULD keep performance observations separate from accuracy acceptance in
  this phase.

## Acceptance Criteria

- [x] CuTe SM120 code no longer uses `rcp.approx.ftz.f32` in the accepted
      FourOverSix NVFP4 quantization path.
- [x] CuTe SM120 scale computations that correspond to Triton `tl.div_rn`
      operations use regular round-to-nearest division or an equivalent exact
      instruction sequence.
- [x] CuTe SM120 error helpers avoid `abs.ftz.f32` and `max.ftz.f32` for paths
      being compared against Triton unless an explicit Triton-equivalent FTZ
      behavior is proven and documented.
- [x] CuTe SM120 final quantization scaling and FourOverSix error-selection
      scaling are aligned so that both paths make decisions from the same
      arithmetic model.
- [x] CuTe SM120 preserves Triton tie-breaking for 4-vs-6 selection.
- [x] A parity probe reports packed value equality ratio, E4M3 scale equality
      ratio, dequant MAE, dequant MSE, max absolute error, mismatch counts, and
      sample mismatching block indices.
- [x] The parity probe covers at least these shapes:
      `(1, 576)`, `(6, 1536)`, `(128, 256)`, `(1024, 1024)`,
      `(4096, 4096)`, and `(8192, 4096)`.
- [x] The parity probe covers at least NVFP4 `static_4`, `static_6`,
      `abs_max`, `mae`, and `mse` scale rules with nearest rounding.
- [x] Results compare CuTe SM120 directly against Triton after normalizing
      padding and scale-factor layout.
- [x] The before/after report clearly states which mismatches disappeared,
      which remain, and the likely remaining source.
- [x] Existing SM120 quantization tests still pass or any failures are
      explained with exact test names and failure modes.

## Completion Notes

- The default FourOverSix CuTe SM120 path keeps the simpler CuTe reduction
  tree. This is the production default.
- The optional proof path is enabled with
  `QuantizationConfig(kwargs={"match_triton_reduction": True})`; it uses the
  Triton PTX-aligned MAE/MSE reduction order for FourOverSix NVFP4 adaptive
  selection and is used by the parity and ablation reports.
- `profile/sm120_accuracy_acceptance/nvfp4_sm120_triton_parity_latest.md`
  was regenerated on 2026-06-09 on RTX 5090. It reports 30/30 cases bitwise
  equal, 0 value mismatches, 0 E4M3 scale mismatches, and zero dequant error.
- `profile/sm120_accuracy_acceptance/nvfp4_fouroversix_ablation_latest.md`
  was regenerated on 2026-06-09. It compares the default CuTe reduction and
  the Triton-matched reduction against the same Triton-generated 4/6 candidate
  payloads. In the default CuTe path, every residual final mismatch is
  explained by a 4-vs-6 candidate choice difference: `(1024, 1024)` has 2 MAE
  mismatches, `(4096, 4096)` has 10 MAE and 1 MSE mismatches, and
  `(8192, 4096)` has 16 MAE and 3 MSE mismatches, all with 0 unexplained
  mismatches. In the Triton-matched reduction path, all large adaptive cases
  report 0 final mismatches and 0 unexplained mismatches.
- The cause classification is not based on assumption alone:
  the ablation first proves both Triton and CuTe outputs are always one of the
  same Triton-generated 4/6 candidate payloads, then shows the default CuTe
  residual mismatches line up exactly with candidate choice differences, and
  finally shows the PTX-aligned reduction path removes those choice
  differences. This excludes amax, padding/layout normalization, final packing,
  E4M3 scale payload generation, and uncategorized dequant-only drift as the
  source of the remaining default-path differences.
- Fast-math audit command:
  `rg -n "rcp\.approx|rcp_approx|\.ftz|--use_fast_math|use_fast_math" src/fouroversix/kernels/cute_sm120 src/fouroversix/quantize/cute/sm120_backend.py`
  returned no matches.
- Regression command:
  `CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_quantize.py::test_cute_sm120_default_mse_quantize_is_not_less_accurate_than_triton tests/test_quantize.py::test_cute_sm120_default_mse_quantize_zero_input tests/test_matmul.py::test_cute_sm120_cutlass_matmul_matches_pytorch -q --junitxml=profile/sm120_accuracy_acceptance/latest.xml`
  passed with 15 tests and 2 warnings.

## (Optional) Steps

1. Establish a baseline.
   - Run the existing SM120 parity or accuracy scripts.
   - Add or reuse a targeted parity probe that compares Triton and CuTe SM120
     packed values, scale factors, and common-backend dequantized values.
   - Save baseline results for the required shape and scale-rule matrix.

2. Audit all CuTe SM120 fast-math sites.
   - Search only `src/fouroversix/kernels/cute_sm120/` and
     `src/fouroversix/quantize/cute/sm120_backend.py`.
   - Classify each `rcp.approx`, `.ftz`, approximate division, and helper
     wrapper by whether it affects FourOverSix NVFP4 quantization,
     dequantization, unrelated formats, or dead code.
   - Do not change SM100 while performing this audit.

3. Replace approximate reciprocal scale arithmetic.
   - Replace `rcp_approx_ftz` uses in the accepted SM120 NVFP4 quantization
     path with `div.rn.f32` or a CuTe DSL operation that lowers to regular
     round-to-nearest division.
   - Pay special attention to `block_max / 6`, `block_max / 4`, global-scale
     encode/decode, and final output scaling.

4. Align output scale and selection scale.
   - Ensure the path used to choose four vs six and the path used to write the
     final packed FP4 value use the same scale arithmetic model.
   - Keep dequant error computation consistent with Triton's sequence:
     quantize with RN conversion, decode the quantized value through the same
     effective scale, then compare errors.

5. Remove FTZ-only behavior from compared error paths.
   - Replace `abs.ftz.f32` and `max.ftz.f32` in relevant SM120 error helpers
     with non-FTZ equivalents.
   - Re-check any packed helper that computes absolute values or maxima before
     scale selection.

6. Re-run accuracy probes.
   - Run the same baseline matrix after the change.
   - Include runs with backend-provided `x_amax` and shared explicit `x_amax`
     to separate amax behavior from scale arithmetic behavior.
   - Confirm each backend remains bitwise stable across repeated runs.

7. Summarize accuracy deltas.
   - Record value mismatch counts, scale mismatch counts, and dequant error
     metrics before and after the change.
   - Identify remaining differences by category: scale mismatch, packed value
     mismatch with identical scale, dequant-only mismatch, or layout/padding
     artifact.

8. Run focused regression tests.
   - Run the SM120 quantization tests relevant to NVFP4 and FourOverSix.
   - Run any parity probe or profile script used for the before/after report.
   - Do not treat speed regressions as phase-one failures unless they are so
     large that the kernel becomes unusable; record them for the follow-up
     performance phase.
