# Quantize Kernel Report: RTX 5090 SM120 vs Triton, with SM100 Reference

## Executive Summary

On RTX 5090 / SM120, the current `cute_sm120` backend is useful for the target
single-batch inference quantization path:

- Target path: `nvfp4`, `mse`, default auto-amax, no transpose, no 2D block
  scaling, no pseudo quantization.
- Speed: all profiled SmolLM2 single-batch inference quantize shapes are faster
  than Triton by `1.365x-1.861x`.
- Precision: the default MSE inference accuracy gate passes on GPU7 with
  `15 passed, 0 failed, 0 skipped, 0 xfailed`.
- Limitation: this is not a full all-configuration acceptance. The wider
  all-scale-rule probe has one known `abs_max` failure at `(6, 1536)`.

SM100/B200 has stronger existing documentation:

- The authoritative SM100 parity report says the full default auto-amax matrix
  is complete: `476/476` rows meet target.
- SM100 full `cute_sm100` pytest selection: `488 passed, 10 skipped`.
- An older SmolLM2 inference snapshot exists for SM100 and shows mixed
  model-shape speedups (`1.054x-2.491x`), but the repo marks
  `profile/cute_sm100_full_triton_parity/CURRENT_STATUS.md` as the authoritative
  current status.

## SM120 / RTX 5090 Speed

Source: `profile/inference_kernel_profile_sm120_pure/latest.md`

Environment and scope:

- GPU: RTX 5090, SM120
- Model shape source: `HuggingFaceTB/SmolLM2-135M`
- Backend comparison: Triton baseline vs `cute_sm120`
- Dtype / scale rule: `nvfp4 + mse`
- Workload type: single-prompt/single-batch inference quantize shapes

Weights:

| shape | count | Triton mean ms | CuTe SM120 mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| `192x576` | 60 | 0.149589 | 0.102529 | 1.459x |
| `576x576` | 60 | 0.151241 | 0.104036 | 1.454x |
| `576x1536` | 30 | 0.152107 | 0.102654 | 1.482x |
| `1536x576` | 60 | 0.149482 | 0.103160 | 1.449x |
| `49152x576` | 1 | 0.204593 | 0.109917 | 1.861x |

Activations:

| shape | count | Triton mean ms | CuTe SM120 mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| `1x576` | 1268 | 0.142172 | 0.104135 | 1.365x |
| `1x1536` | 210 | 0.142733 | 0.101685 | 1.404x |
| `6x576` | 180 | 0.142193 | 0.100186 | 1.419x |
| `6x1536` | 30 | 0.141453 | 0.101796 | 1.390x |

Interpretation:

- The current target is single-batch inference, not large-batch serving.
- The activation rows are `1xhidden` decode-like and `6xhidden` short-prefill
  shapes. There is no current large-batch activation profile such as
  `128xhidden` or `1024xhidden`.
- Triton also has not been tuned for the large-batch setting in this repo, so
  large-batch data should be treated as future work rather than a current target.

## SM120 / RTX 5090 Precision

Source: `profile/sm120_accuracy_acceptance/README.md` and
`profile/sm120_accuracy_acceptance/latest.xml`

Accepted scope:

- Backend/config: `cute_sm120`, `nvfp4`, `mse`.
- Quantize accuracy: input error is no worse than Triton within the existing CuTe
  dequant metric tolerance across covered inference shapes.
- Zero input: output values, `amax`, and dequantized tensor remain zero.
- Matmul accuracy: CUTLASS output from `cute_sm120` quantized tensors matches the
  PyTorch dequantized matmul gate for covered inference/regression shapes.

Recorded command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_quantize.py::test_cute_sm120_default_mse_quantize_is_not_less_accurate_than_triton \
  tests/test_quantize.py::test_cute_sm120_default_mse_quantize_zero_input \
  tests/test_matmul.py::test_cute_sm120_cutlass_matmul_matches_pytorch \
  -q --junitxml=profile/sm120_accuracy_acceptance/latest.xml
```

Recorded result:

```text
15 passed, 0 failed, 0 skipped, 0 xfailed
```

Covered quantize shapes:

- `1x576`
- `6x576`
- `1x1536`
- `6x1536`
- `192x576`
- `576x576`
- `576x1536`
- `1536x576`

Covered matmul regression shapes:

- `(1, 192, 576)`
- `(1, 576, 1536)`
- `(6, 576, 576)`
- `(6, 576, 1536)`
- `(128, 256, 256)`
- `(256, 512, 512)`

Known non-scope:

- The all-scale-rule probe covers 50 `(shape, scale_rule)` rows and has one
  failure: `(6, 1536) + abs_max`.
- For that row, Triton input MAE is `0.07092130184173584`; CuTe SM120 input MAE
  is `0.07109253853559494`, which exceeds the current `1e-4` tolerance by about
  `7.12e-5`.

## SM120 Implementation Note

The current SM120 path is a pure CuTe backend path with separate files/classes
from SM100, not a hybrid CUDA fallback. A precision bug was found and fixed in
the CUTLASS matmul path:

- `cute_sm120` fast quantize can produce row-major scale factors for lazy
  materialization.
- CUTLASS GEMM expects Blackwell blocked scale layout.
- The matmul backend now pads/zeros values and scale factors, converts row-major
  scales to Blackwell blocked layout, then calls CUTLASS.

This fix is necessary for correctness. Before the fix, a case such as
`1x192x576` produced finite but numerically wrong CUTLASS output.

## SM100 / B200 Reference

Authoritative status source:
`profile/cute_sm100_full_triton_parity/CURRENT_STATUS.md`

SM100 completion scope:

- Default auto-amax CuTe SM100 vs Triton matrix.
- Non-pseudo rows require `>=1.2x` Triton speed.
- Pseudo rows require strict speedup over Triton (`>1.0x`).
- Provided-`x_amax` rows are measured separately and not included in completion.

Recorded SM100 results:

| Metric | Result |
| --- | ---: |
| Total benchmark rows | 476 |
| Rows meeting current target | 476/476 |
| Non-pseudo rows meeting `>=1.2x` | 338/338 |
| Pseudo rows faster than Triton | 138/138 |
| CuTe missing Triton-runnable rows | 0 |
| CuTe predicate failures | 0 |
| Full `cute_sm100` pytest selection | 488 passed, 10 skipped |

SM100 targeted precision source:
`profile/cute_sm100_accuracy_validation/verification.md`

Targeted precision result:

```text
25 passed, 1 warning in 6.03s
```

SM100 latency snapshot source:
`profile/cute_sm100_accuracy_validation/ptx_and_benchmark.md`

| shape | rule | Triton ms | CuTe SM100 ms | note |
| --- | --- | ---: | ---: | --- |
| `128x256` | mse | 0.0600 | 0.0577 | slightly faster |
| `128x256` | static_6 | 0.0574 | 0.0610 | slightly slower |
| `1024x1024` | mse | 0.0594 | 0.0610 | comparable |
| `1024x1024` | static_6 | 0.0589 | 0.0596 | comparable |
| `4096x4096` | mse | 0.1118 | 0.0639 | faster |
| `4096x4096` | static_6 | 0.0614 | 0.0657 | slightly slower |

There is also an older SmolLM2 inference-kernel profile at
`profile/inference_kernel_profile/latest.md`. It reports SM100 speedups from
`1.054x` to `2.491x` on the same model-shape family. Because
`CURRENT_STATUS.md` is explicitly marked authoritative, use the full parity
status as the main SM100 conclusion and treat the inference snapshot as
historical/model-specific context.

## Suggested Report Wording

For the current RTX 5090 work, the accurate statement is:

> On RTX 5090/SM120, our separate CuTe SM120 backend passes precision for the
> default single-batch inference quantization path (`nvfp4 + mse`) and is
> consistently faster than Triton on the profiled SmolLM2 inference quantize
> shapes, with `1.365x-1.861x` speedup. The current claim is intentionally
> scoped to single-batch inference; large-batch activation shapes and all
> scale-rule configurations are not yet accepted. A wider scale-rule probe has
> one known `abs_max` miss.

For SM100 comparison:

> On SM100/B200, the repo already has an authoritative full parity report:
> `476/476` default auto-amax benchmark rows meet the target, including
> `338/338` non-pseudo rows at `>=1.2x` and `138/138` pseudo rows faster than
> Triton. The full `cute_sm100` test selection records `488 passed, 10 skipped`.
