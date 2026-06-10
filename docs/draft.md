# Draft: CuTe SM120 FourOverSix Quantization Work

## Scope

Current task focus:

- Backend: CuTe SM120 only, running on the RTX 5090 server.
- Algorithm: Four-over-six quantization and dequantization behavior.
- Goal: make the CuTe SM120 backend match the Triton backend as closely as
  possible, ideally bitwise-identical for quantized values, scale factors, and
  dequantized results after layout normalization.
- Performance requirement: for every benchmarked shape, CuTe SM120 must be
  faster than both CUDA and Triton. The comparison should measure only quantize
  and dequantize time unless a later task explicitly widens the scope.

## Hard Restrictions

- Do not modify Triton backend code.
- Do not modify CUDA backend code.
- Do not modify CuTe SM100 backend code.
- Only CuTe SM120 code may be changed for implementation work.
- Do not hide CUDA or Triton fallbacks behind the CuTe SM120 backend name.
- Do not use fast math in CuTe SM120.

Fast-math prohibition includes both:

- Compile options that enable approximate/unsafe math transformations.
- Explicit fast-math instructions or helpers, including approximate reciprocal,
  approximate division, approximate square root, flush-to-zero variants where
  Triton does not use them, or any helper wrapping those operations.

The current known violation class is explicit `rcp.approx.ftz.f32` usage in the
CuTe code path. This must be replaced or isolated out of the accepted SM120
quantization path before claiming Triton parity.

## Accuracy Acceptance Work

The existing acceptance logic is too weak for the new goal. It should be
updated so CuTe SM120 is compared directly against Triton, not only against a
PyTorch reference or broad error thresholds.

Required acceptance direction:

- Normalize metadata/layout before comparing raw data, so padding and Blackwell
  scale layout differences do not create false failures.
- Compare CuTe SM120 against Triton for quantized packed FP4 values.
- Compare CuTe SM120 against Triton for E4M3 scale factors.
- Compare CuTe SM120 against Triton for dequantized values where both backends
  expose the relevant path; otherwise dequantize both quantized tensors through
  the same trusted backend and compare the resulting tensors.
- Prefer bitwise equality as the target.
- Where bitwise equality is not yet possible, record exact mismatch counts,
  equal ratios, maximum absolute error, MAE, MSE, and the responsible shape,
  scale rule, rounding style, dtype, and backend path.
- Treat large-shape mismatches as implementation differences until proven
  otherwise. The algorithm is deterministic, so repeated stable mismatches
  should not be dismissed as nondeterminism.

Known current issue: the existing SM120 accuracy gate only requires CuTe to be
"not less accurate" than Triton for selected cases. That is insufficient for
the new task. The gate should move toward Triton bitwise parity.

## Current CuTe SM120 Audit Notes

Known suspicious points to address or prove harmless:

- CuTe SM120 uses explicit approximate reciprocal with FTZ in FP4 scale helpers.
- Some scale computations use multiplication by approximate reciprocal instead
  of round-to-nearest division.
- The final quantization scale and the error-selection scale are computed
  through different arithmetic paths.
- Error helpers use FTZ abs/max instructions in places where the Triton path
  has no explicit FTZ behavior.
- FP8 scale decode and FP4 conversion points are split across multiple CuTe
  helpers, while Triton keeps quantization and dequantized error values tied to
  the same conversion path.
- Shape-dependent frontend paths and amax caching can hide timing costs or
  make small-shape and large-shape behavior diverge.
- Padding and scale-factor layout differences must be normalized before
  asserting bitwise mismatch.

Observed probe summary:

- Repeating the same backend is bitwise stable.
- Passing the same `x_amax` into Triton and CuTe SM120 does not remove the large
  shape mismatch.
- Some static scale-rule probes have identical scale factors but still differ
  in packed values, which points to scaling/rounding/conversion path differences
  rather than amax reduction nondeterminism.

## Performance Acceptance Work

The new performance target is stricter than previous parity notes:

- For every tested shape, CuTe SM120 must be faster than CUDA.
- For every tested shape, CuTe SM120 must be faster than Triton.
- The benchmark should report quantize and dequantize time separately.
- Dequantize comparisons must clearly mark any backend fallback, especially if
  Triton does not expose a native NVFP4 dequant frontend path for the tested
  configuration.
- Benchmark runs should bind to an idle GPU on the RTX 5090 server and record
  the selected GPU id.

Current profiling context:

- Earlier quant/dequant profiling used an idle nonzero GPU because GPU 0 had an
  existing process.
- Triton NVFP4 dequant may route through a fallback path in the current
  frontend, so dequant benchmark tables must explicitly annotate fallback
  behavior.

## Suggested Order

1. Tighten SM120 accuracy tests and parity reports without changing Triton,
   CUDA, or SM100 code.
2. Remove fast-math and FTZ-only arithmetic from the accepted CuTe SM120
   quantization path.
3. Align CuTe SM120 scale arithmetic, rounding points, FP32/FP16 conversion
   points, and tie-breaking with Triton.
4. Re-run bitwise parity probes across small and large shapes.
5. Only after accuracy is understood, tune CuTe SM120 performance until every
   shape beats both CUDA and Triton for quantize/dequantize timing.
