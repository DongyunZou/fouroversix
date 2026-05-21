# Inference End-to-End Profile

- model: `HuggingFaceTB/SmolLM2-135M`
- max_new_tokens: `32`
- warmup: `2`
- repeats: `6`
- dtype: `nvfp4`
- scale_rule: `mse`
- cutlass_available: `True`
- generated_at_unix: `1779356930`

This profiles a local end-to-end generate wrapper. It keeps `lm_head` in BF16,
pre-quantizes Linear weights, and uses CUTLASS FP4 matmul for replaced Linear
modules. The result measures complete `generate(...)` latency, not just the
standalone quantize kernel.

| config | replacements | mean ms | median ms | tokens/s | speedup vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 0 | 210.263 | 210.203 | 152.190 | 1.000x |
| w_triton__a_triton | 210 | 953.715 | 951.709 | 33.553 | 0.220x |
| w_triton__a_cute_sm100 | 210 | 1081.936 | 1080.899 | 29.577 | 0.194x |
| w_cute_sm100__a_triton | 210 | 950.761 | 950.400 | 33.657 | 0.221x |
| w_cute_sm100__a_cute_sm100 | 210 | 1083.522 | 1083.152 | 29.533 | 0.194x |
