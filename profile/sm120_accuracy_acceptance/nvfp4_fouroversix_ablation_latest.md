# SM120 NVFP4 FourOverSix Ablation

Generated: `2026-06-09T16:13:23.019367+00:00`
Device: `NVIDIA GeForce RTX 5090`

## Default CuTe Reduction

| shape | scale_rule | triton_unmatched | cute_unmatched | triton_choice_diff | cute_choice_diff | final_mismatch | explained_by_choice | unexplained |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (1024, 1024) | abs_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (1024, 1024) | mae | 0 | 0 | 0 | 2 | 2 | 2 | 0 |
| (1024, 1024) | mse | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (4096, 4096) | abs_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (4096, 4096) | mae | 0 | 0 | 0 | 10 | 10 | 10 | 0 |
| (4096, 4096) | mse | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| (8192, 4096) | abs_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (8192, 4096) | mae | 0 | 0 | 0 | 16 | 16 | 16 | 0 |
| (8192, 4096) | mse | 0 | 0 | 0 | 3 | 3 | 3 | 0 |

## Triton-Matched CuTe Reduction

| shape | scale_rule | triton_unmatched | cute_unmatched | triton_choice_diff | cute_choice_diff | final_mismatch | explained_by_choice | unexplained |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (1024, 1024) | abs_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (1024, 1024) | mae | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (1024, 1024) | mse | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (4096, 4096) | abs_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (4096, 4096) | mae | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (4096, 4096) | mse | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (8192, 4096) | abs_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (8192, 4096) | mae | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (8192, 4096) | mse | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
