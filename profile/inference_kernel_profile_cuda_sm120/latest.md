# Inference Kernel Profile

- model: `HuggingFaceTB/SmolLM2-135M`
- dtype: `nvfp4`
- scale_rule: `mse`
- baseline_backend: `triton`
- candidate_backend: `cuda`
- generated_at_unix: `1779374613`

## Weights

| shape | count | Baseline mean ms | Candidate mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 192x576 | 60 | 0.157528 | 0.055346 | 2.846x |
| 576x576 | 60 | 0.154160 | 0.053816 | 2.865x |
| 576x1536 | 30 | 0.156123 | 0.054211 | 2.880x |
| 1536x576 | 60 | 0.152494 | 0.054350 | 2.806x |
| 49152x576 | 1 | 0.206499 | 0.176442 | 1.170x |

## Activations

| shape | count | Baseline mean ms | Candidate mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 1x576 | 1268 | 0.145760 | 0.056075 | 2.599x |
| 1x1536 | 210 | 0.144646 | 0.053747 | 2.691x |
| 6x576 | 180 | 0.148645 | 0.053365 | 2.785x |
| 6x1536 | 30 | 0.143597 | 0.053562 | 2.681x |
