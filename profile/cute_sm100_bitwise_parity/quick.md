# SM100 CuTe Four-Over-Six Bitwise Parity Probe

- Compared cases: 51
- Compared against unmodified Triton: 42
- Compared against PyTorch reference: 9
- Bitwise equal: 51
- Bitwise mismatches: 0
- Unsupported by one backend: 0
- Skipped: 9
- Exceptions: 0

## Skipped

- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=abs_max, round_style=nearest, block_scale_2d=False, transpose=False, rht=True, shape=(128, 256), input=randn
- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=abs_max, round_style=nearest, block_scale_2d=False, transpose=True, rht=True, shape=(128, 256), input=randn
- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=abs_max, round_style=nearest, block_scale_2d=True, transpose=False, rht=True, shape=(128, 256), input=randn
- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=mae, round_style=nearest, block_scale_2d=False, transpose=False, rht=True, shape=(128, 256), input=randn
- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=mae, round_style=nearest, block_scale_2d=False, transpose=True, rht=True, shape=(128, 256), input=randn
- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=mae, round_style=nearest, block_scale_2d=True, transpose=False, rht=True, shape=(128, 256), input=randn
- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=mse, round_style=nearest, block_scale_2d=False, transpose=False, rht=True, shape=(128, 256), input=randn
- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=mse, round_style=nearest, block_scale_2d=False, transpose=True, rht=True, shape=(128, 256), input=randn
- reason=rht_preprocess_is_outside_four_over_six_quant_dequant, dtype=nvfp4_bs8, scale_rule=mse, round_style=nearest, block_scale_2d=True, transpose=False, rht=True, shape=(128, 256), input=randn
