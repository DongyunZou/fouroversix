# Four Over Six NVFP4 Quant/Dequant Backend Benchmark

- timestamp: 2026-06-09 05:53:16 EDT
- device: NVIDIA GeForce RTX 5090
- capability: (12, 0)
- CUDA_VISIBLE_DEVICES: 1
- torch: 2.12.0+cu130
- cuda: 13.0
- repeats: 7
- warmups: 10
- include_amax: True
- dtype: nvfp4
- scale_rule: mse
- round_style: nearest

| shape | backend | status | dequant backend | quant median ms | dequant median ms | quant mean ms | dequant mean ms |
|---:|---|---|---|---:|---:|---:|---:|
| 128x256 | cuda | ok | cuda | 0.038557 | 0.378137 | 0.038544 | 0.378035 |
| 128x256 | triton | ok | cuda fallback | 0.120843 | 0.375041 | 0.120912 | 0.375152 |
| 128x256 | cute_sm120 | ok | cute_sm120 | 0.112753 | 0.352761 | 0.145042 | 0.353076 |
| 512x2048 | cuda | ok | cuda | 0.038629 | 0.379177 | 0.038638 | 0.379501 |
| 512x2048 | triton | ok | cuda fallback | 0.130051 | 0.380565 | 0.130035 | 0.381067 |
| 512x2048 | cute_sm120 | ok | cute_sm120 | 0.117957 | 0.357842 | 0.118085 | 0.358779 |
| 1024x1024 | cuda | ok | cuda | 0.038688 | 0.378753 | 0.038698 | 0.379291 |
| 1024x1024 | triton | ok | cuda fallback | 0.128426 | 0.378519 | 0.128425 | 0.378546 |
| 1024x1024 | cute_sm120 | ok | cute_sm120 | 0.117212 | 0.357435 | 0.117515 | 0.358120 |
| 4096x4096 | cuda | ok | cuda | 0.108575 | 0.559553 | 0.108565 | 0.561293 |
| 4096x4096 | triton | ok | cuda fallback | 0.131473 | 0.559731 | 0.132791 | 0.559716 |
| 4096x4096 | cute_sm120 | ok | cute_sm120 | 0.120287 | 0.552205 | 0.121364 | 0.552327 |
| 8192x4096 | cuda | ok | cuda | 0.200019 | 1.363678 | 0.200091 | 1.363557 |
| 8192x4096 | triton | ok | cuda fallback | 0.167738 | 1.363526 | 0.167762 | 1.363715 |
| 8192x4096 | cute_sm120 | ok | cute_sm120 | 0.119769 | 1.354342 | 0.119952 | 1.354434 |
