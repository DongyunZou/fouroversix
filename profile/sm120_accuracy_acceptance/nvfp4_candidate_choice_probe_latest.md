# SM120 NVFP4 Candidate Choice Probe

Generated: `2026-06-09T15:35:34.335561+00:00`
Device: `NVIDIA GeForce RTX 5090`

| shape | scale_rule | value_mismatch_blocks | scale_mismatch_blocks | choice_mismatch_blocks | py_vs_triton | py_vs_cute | margin_abs_max | same_choice_mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (1024, 1024) | abs_max | 0 | 0 | 0 | 0 | 0 | 0.00000000e+00 | 0 |
| (1024, 1024) | mae | 2 | 2 | 2 | 4 | 4 | 0.00000000e+00 | 0 |
| (1024, 1024) | mse | 0 | 0 | 0 | 1 | 1 | 0.00000000e+00 | 0 |
| (4096, 4096) | abs_max | 0 | 0 | 0 | 0 | 0 | 0.00000000e+00 | 0 |
| (4096, 4096) | mae | 10 | 10 | 10 | 13 | 15 | 1.19209290e-07 | 0 |
| (4096, 4096) | mse | 1 | 1 | 1 | 1 | 2 | 0.00000000e+00 | 0 |
| (8192, 4096) | abs_max | 0 | 0 | 0 | 0 | 0 | 0.00000000e+00 | 0 |
| (8192, 4096) | mae | 16 | 16 | 16 | 29 | 27 | 1.19209290e-07 | 0 |
| (8192, 4096) | mse | 3 | 3 | 3 | 4 | 5 | 1.49011612e-08 | 0 |
