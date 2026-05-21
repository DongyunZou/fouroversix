# Inference Kernel Profile

- model: `HuggingFaceTB/SmolLM2-135M`
- dtype: `nvfp4`
- scale_rule: `mse`
- generated_at_unix: `1779356002`

## Weights

| shape | count | Triton mean ms | CuTe sm100 mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 192x576 | 60 | 0.081028 | 0.074335 | 1.090x |
| 576x576 | 60 | 0.081932 | 0.074700 | 1.097x |
| 576x1536 | 30 | 0.082175 | 0.077937 | 1.054x |
| 1536x576 | 60 | 0.080969 | 0.049856 | 1.624x |
| 49152x576 | 1 | 0.187242 | 0.075169 | 2.491x |

## Activations

| shape | count | Triton mean ms | CuTe sm100 mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 1x576 | 1268 | 0.080612 | 0.070804 | 1.139x |
| 1x1536 | 210 | 0.080485 | 0.075441 | 1.067x |
| 6x576 | 180 | 0.080723 | 0.071022 | 1.137x |
| 6x1536 | 30 | 0.080198 | 0.071428 | 1.123x |
