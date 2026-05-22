# Inference Kernel Profile

- model: `HuggingFaceTB/SmolLM2-135M`
- dtype: `nvfp4`
- scale_rule: `mse`
- baseline_backend: `triton`
- candidate_backend: `cute_sm120`
- generated_at_unix: `1779374709`

## Weights

| shape | count | Baseline mean ms | Candidate mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 192x576 | 60 | 0.152225 | 0.055868 | 2.725x |
| 576x576 | 60 | 0.150502 | 0.055811 | 2.697x |
| 576x1536 | 30 | 0.147234 | 0.055450 | 2.655x |
| 1536x576 | 60 | 0.150435 | 0.055428 | 2.714x |
| 49152x576 | 1 | 0.204451 | 0.137740 | 1.484x |

## Activations

| shape | count | Baseline mean ms | Candidate mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| 1x576 | 1268 | 0.146756 | 0.056097 | 2.616x |
| 1x1536 | 210 | 0.150224 | 0.055666 | 2.699x |
| 6x576 | 180 | 0.143743 | 0.055584 | 2.586x |
| 6x1536 | 30 | 0.143818 | 0.055575 | 2.588x |
