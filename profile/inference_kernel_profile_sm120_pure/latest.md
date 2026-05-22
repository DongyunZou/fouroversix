# Inference Kernel Profile

- model: `HuggingFaceTB/SmolLM2-135M`
- dtype: `nvfp4`
- scale_rule: `mse`
- baseline_backend: `triton`
- candidate_backend: `cute_sm120`
- generated_at_unix: `1779380840`

## Weights

| shape | count | Baseline mean ms | Candidate mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 192x576 | 60 | 0.149589 | 0.102529 | 1.459x |
| 576x576 | 60 | 0.151241 | 0.104036 | 1.454x |
| 576x1536 | 30 | 0.152107 | 0.102654 | 1.482x |
| 1536x576 | 60 | 0.149482 | 0.103160 | 1.449x |
| 49152x576 | 1 | 0.204593 | 0.109917 | 1.861x |

## Activations

| shape | count | Baseline mean ms | Candidate mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 1x576 | 1268 | 0.142172 | 0.104135 | 1.365x |
| 1x1536 | 210 | 0.142733 | 0.101685 | 1.404x |
| 6x576 | 180 | 0.142193 | 0.100186 | 1.419x |
| 6x1536 | 30 | 0.141453 | 0.101796 | 1.390x |
