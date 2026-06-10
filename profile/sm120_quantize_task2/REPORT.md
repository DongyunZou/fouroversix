# SM120 CuTe Quantize Task2 Report

## Status

Task2 is not complete under the corrected fair benchmark. The prior CUDA Graph replay result is invalid for task2 acceptance because only CuTe SM120 used graph replay while CUDA and Triton did not. The graph path has been removed from the code and is excluded from this report.

The current fair measurement uses ordinary CUDA events around public `quantize()` for all three backends. CuTe SM120 beats Triton on every required shape, but it beats CUDA on only 2 of 14 shapes.

## Environment

- timestamp: `2026-06-09 22:08:48 EDT`
- device: `NVIDIA GeForce RTX 5090`
- compute capability: `[12, 0]`
- CUDA_VISIBLE_DEVICES: `7`
- PyTorch: `2.12.0+cu130`
- CUDA: `13.0`
- repeats: `7`
- warmups: `10`
- include_amax: `True`
- timing method: ordinary CUDA events around public `quantize()`; no CUDA Graph replay for any backend
- iterations per sample: listed per shape below
- acceptance config: `dtype=nvfp4`, `scale_rule=mse`, `round_style=nearest`, `pseudo_quantize=False`, `transpose=False`, `block_scale_2d=False`, `rht=False`

## Pass/Fail Summary

| shape | iters/sample | CUDA median ms | Triton median ms | CuTe SM120 median ms | CuTe-vs-CUDA speedup | CuTe-vs-Triton speedup | pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1x576 | 500 | 0.039347 | 0.121310 | 0.080970 | 0.486x | 1.498x | FAIL |
| 1x1536 | 500 | 0.039384 | 0.123169 | 0.081657 | 0.482x | 1.508x | FAIL |
| 6x576 | 500 | 0.039163 | 0.122593 | 0.081251 | 0.482x | 1.509x | FAIL |
| 6x1536 | 500 | 0.039252 | 0.123290 | 0.081211 | 0.483x | 1.518x | FAIL |
| 128x256 | 500 | 0.039355 | 0.121991 | 0.081289 | 0.484x | 1.501x | FAIL |
| 512x2048 | 200 | 0.039408 | 0.129372 | 0.114122 | 0.345x | 1.134x | FAIL |
| 1024x1024 | 200 | 0.039630 | 0.129436 | 0.113630 | 0.349x | 1.139x | FAIL |
| 4096x4096 | 60 | 0.109419 | 0.133250 | 0.117879 | 0.928x | 1.130x | FAIL |
| 8192x4096 | 30 | 0.199037 | 0.166382 | 0.119194 | 1.670x | 1.396x | PASS |
| 192x576 | 200 | 0.039542 | 0.124118 | 0.109291 | 0.362x | 1.136x | FAIL |
| 576x576 | 200 | 0.039897 | 0.129845 | 0.114802 | 0.348x | 1.131x | FAIL |
| 576x1536 | 200 | 0.039639 | 0.129539 | 0.114268 | 0.347x | 1.134x | FAIL |
| 1536x576 | 200 | 0.039750 | 0.130606 | 0.113897 | 0.349x | 1.147x | FAIL |
| 49152x576 | 30 | 0.162695 | 0.138032 | 0.118489 | 1.373x | 1.165x | PASS |

- passing shapes: `2/14`
- failing shapes: `12/14`

## Quantization Timing Details

This is quantization-only and excludes dequant/fake quant from pass/fail.

| shape | backend | status | median ms | mean ms | min ms | max ms | samples ms |
|---:|---|---|---:|---:|---:|---:|---|
| 1x576 | cuda | ok | 0.039347 | 0.039510 | 0.039097 | 0.040684 | `0.040684, 0.039394, 0.039347, 0.039330, 0.039375, 0.039097, 0.039345` |
| 1x576 | triton | ok | 0.121310 | 0.121756 | 0.121088 | 0.123708 | `0.123708, 0.122309, 0.121088, 0.121176, 0.121466, 0.121236, 0.121310` |
| 1x576 | cute_sm120 | ok | 0.080970 | 0.081110 | 0.080197 | 0.082366 | `0.081377, 0.082366, 0.080868, 0.081048, 0.080970, 0.080197, 0.080942` |
| 1x1536 | cuda | ok | 0.039384 | 0.039354 | 0.039156 | 0.039498 | `0.039156, 0.039355, 0.039384, 0.039270, 0.039393, 0.039421, 0.039498` |
| 1x1536 | triton | ok | 0.123169 | 0.123135 | 0.122709 | 0.123554 | `0.123554, 0.123398, 0.123015, 0.122809, 0.122709, 0.123291, 0.123169` |
| 1x1536 | cute_sm120 | ok | 0.081657 | 0.081710 | 0.081066 | 0.082492 | `0.082492, 0.081707, 0.081657, 0.081066, 0.081523, 0.081900, 0.081627` |
| 6x576 | cuda | ok | 0.039163 | 0.039191 | 0.038986 | 0.039359 | `0.039256, 0.039359, 0.039100, 0.039155, 0.038986, 0.039163, 0.039319` |
| 6x576 | triton | ok | 0.122593 | 0.122829 | 0.122249 | 0.123630 | `0.123566, 0.122593, 0.123630, 0.122249, 0.122920, 0.122336, 0.122514` |
| 6x576 | cute_sm120 | ok | 0.081251 | 0.081282 | 0.081059 | 0.081617 | `0.081186, 0.081617, 0.081059, 0.081231, 0.081251, 0.081363, 0.081268` |
| 6x1536 | cuda | ok | 0.039252 | 0.039253 | 0.039133 | 0.039346 | `0.039252, 0.039346, 0.039214, 0.039333, 0.039167, 0.039328, 0.039133` |
| 6x1536 | triton | ok | 0.123290 | 0.123163 | 0.122476 | 0.123601 | `0.123161, 0.123366, 0.123601, 0.123368, 0.123290, 0.122476, 0.122880` |
| 6x1536 | cute_sm120 | ok | 0.081211 | 0.081333 | 0.080951 | 0.081807 | `0.081807, 0.081137, 0.081423, 0.081211, 0.081123, 0.080951, 0.081676` |
| 128x256 | cuda | ok | 0.039355 | 0.039392 | 0.039258 | 0.039630 | `0.039326, 0.039529, 0.039258, 0.039284, 0.039355, 0.039364, 0.039630` |
| 128x256 | triton | ok | 0.121991 | 0.122061 | 0.121426 | 0.122528 | `0.121980, 0.122368, 0.121991, 0.121426, 0.122528, 0.121837, 0.122299` |
| 128x256 | cute_sm120 | ok | 0.081289 | 0.081258 | 0.080793 | 0.081611 | `0.081535, 0.081000, 0.081611, 0.081371, 0.081210, 0.080793, 0.081289` |
| 512x2048 | cuda | ok | 0.039408 | 0.039339 | 0.039113 | 0.039466 | `0.039241, 0.039442, 0.039256, 0.039408, 0.039449, 0.039466, 0.039113` |
| 512x2048 | triton | ok | 0.129372 | 0.129071 | 0.127696 | 0.130319 | `0.130319, 0.129507, 0.129372, 0.127696, 0.128460, 0.129719, 0.128424` |
| 512x2048 | cute_sm120 | ok | 0.114122 | 0.114172 | 0.113356 | 0.115176 | `0.115176, 0.114348, 0.114019, 0.113356, 0.114122, 0.113592, 0.114593` |
| 1024x1024 | cuda | ok | 0.039630 | 0.039741 | 0.039479 | 0.040067 | `0.040067, 0.039618, 0.039629, 0.039887, 0.039630, 0.039875, 0.039479` |
| 1024x1024 | triton | ok | 0.129436 | 0.129559 | 0.129103 | 0.130728 | `0.129436, 0.129104, 0.130728, 0.129103, 0.129123, 0.129529, 0.129888` |
| 1024x1024 | cute_sm120 | ok | 0.113630 | 0.113805 | 0.113219 | 0.114762 | `0.114762, 0.113623, 0.113520, 0.113674, 0.113219, 0.113630, 0.114208` |
| 4096x4096 | cuda | ok | 0.109419 | 0.109422 | 0.109033 | 0.109842 | `0.109842, 0.109332, 0.109244, 0.109610, 0.109033, 0.109419, 0.109476` |
| 4096x4096 | triton | ok | 0.133250 | 0.133351 | 0.132476 | 0.134471 | `0.133704, 0.134471, 0.133576, 0.132964, 0.133013, 0.132476, 0.133250` |
| 4096x4096 | cute_sm120 | ok | 0.117879 | 0.117776 | 0.114833 | 0.120970 | `0.120970, 0.117526, 0.115384, 0.114833, 0.119373, 0.117879, 0.118463` |
| 8192x4096 | cuda | ok | 0.199037 | 0.199025 | 0.198388 | 0.199584 | `0.199584, 0.199409, 0.198519, 0.199293, 0.198943, 0.199037, 0.198388` |
| 8192x4096 | triton | ok | 0.166382 | 0.166295 | 0.166059 | 0.166428 | `0.166386, 0.166382, 0.166388, 0.166278, 0.166141, 0.166428, 0.166059` |
| 8192x4096 | cute_sm120 | ok | 0.119194 | 0.120841 | 0.118391 | 0.124036 | `0.119194, 0.124036, 0.118734, 0.123366, 0.118391, 0.123083, 0.119086` |
| 192x576 | cuda | ok | 0.039542 | 0.039568 | 0.039413 | 0.039737 | `0.039665, 0.039542, 0.039737, 0.039661, 0.039423, 0.039413, 0.039535` |
| 192x576 | triton | ok | 0.124118 | 0.124126 | 0.123565 | 0.124731 | `0.124362, 0.124709, 0.124731, 0.123565, 0.124118, 0.123752, 0.123643` |
| 192x576 | cute_sm120 | ok | 0.109291 | 0.109333 | 0.108469 | 0.110665 | `0.109708, 0.109559, 0.110665, 0.109291, 0.108469, 0.108950, 0.108686` |
| 576x576 | cuda | ok | 0.039897 | 0.039873 | 0.039631 | 0.040064 | `0.040012, 0.040064, 0.040033, 0.039897, 0.039736, 0.039736, 0.039631` |
| 576x576 | triton | ok | 0.129845 | 0.129772 | 0.129200 | 0.130175 | `0.129745, 0.129200, 0.129335, 0.130158, 0.130175, 0.129949, 0.129845` |
| 576x576 | cute_sm120 | ok | 0.114802 | 0.114800 | 0.113907 | 0.115332 | `0.115300, 0.115332, 0.114802, 0.114811, 0.114748, 0.114701, 0.113907` |
| 576x1536 | cuda | ok | 0.039639 | 0.039716 | 0.039415 | 0.040109 | `0.040109, 0.039769, 0.039415, 0.039603, 0.039867, 0.039639, 0.039613` |
| 576x1536 | triton | ok | 0.129539 | 0.129576 | 0.129368 | 0.129924 | `0.129441, 0.129607, 0.129368, 0.129539, 0.129924, 0.129721, 0.129434` |
| 576x1536 | cute_sm120 | ok | 0.114268 | 0.113958 | 0.113032 | 0.115106 | `0.114351, 0.114427, 0.115106, 0.114268, 0.113259, 0.113260, 0.113032` |
| 1536x576 | cuda | ok | 0.039750 | 0.039761 | 0.039665 | 0.039897 | `0.039665, 0.039808, 0.039699, 0.039794, 0.039710, 0.039750, 0.039897` |
| 1536x576 | triton | ok | 0.130606 | 0.130485 | 0.129819 | 0.131189 | `0.131189, 0.130069, 0.130674, 0.129819, 0.130191, 0.130606, 0.130848` |
| 1536x576 | cute_sm120 | ok | 0.113897 | 0.113892 | 0.113429 | 0.114841 | `0.114841, 0.113959, 0.113429, 0.113642, 0.113553, 0.113922, 0.113897` |
| 49152x576 | cuda | ok | 0.162695 | 0.162669 | 0.162335 | 0.163005 | `0.162572, 0.162860, 0.163005, 0.162695, 0.162483, 0.162734, 0.162335` |
| 49152x576 | triton | ok | 0.138032 | 0.138066 | 0.137574 | 0.138777 | `0.138777, 0.137915, 0.138041, 0.138032, 0.137939, 0.138186, 0.137574` |
| 49152x576 | cute_sm120 | ok | 0.118489 | 0.120514 | 0.117806 | 0.124654 | `0.124654, 0.118247, 0.122752, 0.117806, 0.123574, 0.118489, 0.118076` |

## Correction Note

The previous report claimed 14/14 pass using a CuTe-only CUDA Graph replay optimization. That was not a fair backend comparison. The CUDA Graph code has been removed, and this report supersedes that result.

The only retained task2-adjacent performance code change is the ordinary CuTe SM120 backend threshold that lets `m == 128` use the existing fused single-CTA path. It improves `128x256` versus the old separate-amax branch but still does not beat CUDA.

## NCU Evidence

- `reports/after_cute_nvfp4_mse_128x256_full.ncu-rep`: `128x256` ordinary CuTe path, representative fused kernel duration about `91.65 us`, public median `0.081289 ms`, still slower than CUDA `0.039355 ms`.
- `reports/after_cute_nvfp4_mse_512x2048_full.ncu-rep` and `reports/after_cute_nvfp4_mse_512x2048_main_full.ncu-rep`: `512x2048` ordinary CuTe path; reduction about `7.49 us`, main kernel about `10.46 us`, but public median `0.114122 ms`, slower than CUDA `0.039408 ms`.
- This NCU evidence still points to public dispatch/launch/allocation overhead as the main remaining gap versus CUDA for most shapes.

## Verification

- Fast-math audit: `rg -n "rcp\.approx|rcp_approx|\.ftz|--use_fast_math|use_fast_math" src/fouroversix/kernels/cute_sm120 src/fouroversix/quantize/cute/sm120_backend.py` returned no matches.
- Focused regression: `CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_quantize.py::test_cute_sm120_default_mse_quantize_is_not_less_accurate_than_triton tests/test_quantize.py::test_cute_sm120_default_mse_quantize_zero_input tests/test_matmul.py::test_cute_sm120_cutlass_matmul_matches_pytorch -q --junitxml=profile/sm120_quantize_task2/latest.xml` passed: `15 passed, 2 warnings`.

## Acceptance Audit

| criterion | status | evidence |
|---|---|---|
| CuTe SM120 median lower than CUDA for every required shape | FAIL | Only 2/14 shapes pass CUDA comparison. |
| CuTe SM120 median lower than Triton for every required shape | PASS | All 14 shapes pass Triton comparison. |
| Report includes required timing statistics and environment metadata | PASS | This report and `latest.json` include medians/means/min/max, samples, repeats, warmups, iterations/sample, GPU, device, compute capability, PyTorch, CUDA, and `CUDA_VISIBLE_DEVICES`. |
| Report includes pass/fail speedup table | PASS | See Pass/Fail Summary. |
| Acceptance measures quantization only | PASS | Pass/fail uses quantization timings only. |
| Cold-cache/include_amax mode does not hide SM120 amax cache | PASS | Benchmark ran with `include_amax=True`; `_AMAX_CACHE` is cleared before each measured CuTe call. |
| Cached-amax result separated from acceptance | PASS | No cached-amax result is used for pass/fail. |
| Existing task1 parity/accuracy tests pass or failures documented | PASS | Focused tests passed: `15 passed`. |
| Final fast-math audit has no matches | PASS | Audit command returned no matches. |
| NCU before/after evidence for one failing small and one failing medium/large shape | INCOMPLETE | Current NCU evidence documents the remaining gap, but there is no accepted optimization that closes the CUDA failures. |
