# FLUX.2-klein-4B Quantized End-to-End Metrics

Run command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/diffusers/eval_flux_klein_quant_metrics.py \
  --output-dir profile/flux_klein_quant_metrics \
  --height 1024 --width 1024 --steps 4 --seed 42
```

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Capability: SM120
- Model: `black-forest-labs/FLUX.2-klein-4B`
- Prompt: `A cozy bookshop in Tokyo at night, warm light spilling onto a rain-slicked street, cherry blossom petals drifting through the air, Studio Ghibli style`
- Resolution: `1024x1024`
- Steps: `4`
- Seed: `42`
- Quantized matmul backend: `cutlass`

Generated images:

- BF16: `profile/flux_klein_quant_metrics/bf16.png`
- Triton NVFP4: `profile/flux_klein_quant_metrics/nvfp4_triton.png`
- CuTe SM120 NVFP4: `profile/flux_klein_quant_metrics/nvfp4_cute_sm120.png`

Latency and memory:

| Case | Load s | Generate s | Peak alloc GB | Peak reserved GB |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 4.994 | 1.485 | 18.626 | 21.053 |
| NVFP4 Triton | 5.403 | 0.765 | 13.061 | 15.626 |
| NVFP4 CuTe SM120 | 9.069 | 2.033 | 13.314 | 15.544 |

Image metrics, computed on RGB pixels normalized to `[0, 1]`:

| Comparison | MSE | PSNR dB | Mean abs | Max abs |
| --- | ---: | ---: | ---: | ---: |
| BF16 vs NVFP4 Triton | 0.029601 | 15.287 | 0.118002 | 0.988235 |
| BF16 vs NVFP4 CuTe SM120 | 0.033403 | 14.762 | 0.127342 | 0.972549 |
| NVFP4 Triton vs NVFP4 CuTe SM120 | 0.022688 | 16.442 | 0.099031 | 0.945098 |

Notes:

- This is a real Diffusers pipeline run, not a quantize-kernel microbenchmark.
- The CuTe timing includes first-use CuTe compilation overhead for this run and should
  not be treated as steady-state serving latency.
- These metrics were produced during a diagnostic run that temporarily allowed forced
  Triton on the FLUX activation layout. The Triton backend workaround was reverted; the
  current branch intentionally does not modify CUDA or Triton backends.
- Current forced CUDA/Triton FLUX.2-klein activation quantization is not treated as
  validated. The official upstream examples use `FourOverSixConfig()` rather than
  explicit forced Triton/CUDA backend validation.
- The retained SM120 compatibility fix is that the CuTe SM120 amax cache tolerates
  PyTorch inference tensors that do not expose a version counter.
