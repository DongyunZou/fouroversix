# NVFP4 quant backend + CUTLASS GEMM benchmark

- device: NVIDIA GeForce RTX 5090
- capability: (12, 0)
- scale_rule: mse
- repeats: 50
- warmups: 20

| shape M,N,K | quant backend | status | quant x ms | quant y ms | quant pair ms | cutlass gemm ms |
|---:|---|---|---:|---:|---:|---:|
| 16384,1024,1024 | cuda | ok | 0.1271 | 0.0529 | 0.1564 | 0.0858 |
| 16384,1024,1024 | cute_sm120 | ok | 0.1057 | 0.1033 | 0.1933 | 0.0873 |
| 16384,1024,1024 | triton | ok | 0.1728 | 0.1504 | 0.2898 | 0.0867 |
| 16384,1024,1024 | pytorch | ok | 6.0977 | 1.1799 | 7.2332 | 0.0856 |
| 16384,1024,1024 | transformer_engine | not_available |  |  |  |  |
