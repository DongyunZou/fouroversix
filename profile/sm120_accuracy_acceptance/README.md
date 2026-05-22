# SM120 Accuracy Acceptance

Last run: 2026-05-21 21:16:00 -04:00 on `yotta-5090-244`.

Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_quantize.py::test_cute_sm120_default_mse_quantize_is_not_less_accurate_than_triton \
  tests/test_quantize.py::test_cute_sm120_default_mse_quantize_zero_input \
  tests/test_matmul.py::test_cute_sm120_cutlass_matmul_matches_pytorch \
  -q --junitxml=profile/sm120_accuracy_acceptance/latest.xml
```

Result: `15 passed, 0 failed, 0 skipped, 0 xfailed`.

Scope:

- GPU: `CUDA_VISIBLE_DEVICES=7`, NVIDIA GeForce RTX 5090, capability `(12, 0)`.
- Backend/config: `cute_sm120`, `nvfp4`, default inference `mse` scale rule.
- Quantize accuracy: candidate input error is no worse than Triton within the existing CuTe dequant metric tolerance across the covered inference shapes.
- Zero input: output values, `amax`, and dequantized tensor remain zero.
- Matmul accuracy: CUTLASS output from `cute_sm120` quantized tensors matches the PyTorch dequantized matmul gate for covered inference and regression shapes.

Non-scope:

- This does not certify every `scale_rule`. `all_scale_rule_probe.json` covers 50 `(shape, scale_rule)` rows and currently has one failure: `(6, 1536) + abs_max`, where CuTe SM120 input MAE is `0.07109253853559494` versus Triton `0.07092130184173584`, exceeding the current `1e-4` tolerance by about `7.12e-5`.
