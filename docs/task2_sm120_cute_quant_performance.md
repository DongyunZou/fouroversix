# TASK2: SM120 CuTe Quantization Performance

## Context

Task1 completed the SM120 CuTe accuracy and fast-math cleanup phase. The
accepted FourOverSix NVFP4 quantization path no longer uses explicit
`rcp.approx.ftz.f32`, and the Triton-matched reduction proof path can reach
bitwise parity against Triton for the task1 shape/rule matrix.

Task2 is a performance-only follow-up for the SM120 CuTe backend. The goal is
to make CuTe SM120 quantization faster than both CUDA and Triton on every
benchmarked shape. Fake quantization and dequantization are out of scope for
this task; they may be measured for context, but they must not decide task2
completion.

Current relevant baseline:

- `profile/quant_backend_fouroversix_current/latest.md` compares CUDA, Triton,
  and CuTe SM120 for NVFP4 `mse`, nearest rounding, `include_amax=True`.
- On that snapshot, CuTe SM120 beats Triton quantization on all listed shapes
  but does not beat CUDA on `128x256`, `512x2048`, `1024x1024`, and
  `4096x4096`.
- `profile/sm120_quantize_ncu_baseline/REPORT.md` shows the large
  non-transpose NVFP4 `mse` CuTe main kernel is already faster than Triton's
  main quantization kernel; the remaining public `quantize()` gap is dominated
  by fixed launch/dispatch cost, the separate amax reduction, allocation, and
  padding/materialization behavior.
- `profile/inference_kernel_profile_sm120_pure/latest.md` shows inference-like
  shapes beat Triton, but that profile does not include CUDA as a baseline.

## Hard Restrictions

- MUST only modify SM120 CuTe backend code:
  - `src/fouroversix/kernels/cute_sm120/`
  - `src/fouroversix/quantize/cute/sm120_backend.py`
- MUST NOT modify Triton backend code.
- MUST NOT modify CUDA backend code.
- MUST NOT modify CuTe SM100 backend code.
- MUST NOT hide CUDA or Triton fallbacks behind the CuTe SM120 backend name.
- MUST NOT use fast math in the accepted SM120 CuTe quantization path.
- MUST NOT add compiler flags or inline PTX that enable approximate or unsafe
  math.
- MUST preserve task1 numerical behavior for NVFP4 FourOverSix quantization.
- MUST keep `pseudo_quantize`, fake quant, and dequant optimization out of the
  task2 completion criteria.

Fast-math prohibition remains active. The final audit must show no matches for:

```bash
rg -n "rcp\.approx|rcp_approx|\.ftz|--use_fast_math|use_fast_math" \
  src/fouroversix/kernels/cute_sm120 \
  src/fouroversix/quantize/cute/sm120_backend.py
```

## Scope

Primary target:

- Backend: `QuantizeBackend.cute_sm120`
- Device class: RTX 5090 / compute capability 12.0
- Operation: public `quantize()` time only
- Config:
  - `dtype=DataType.nvfp4`
  - `scale_rule=ScaleRule.mse`
  - `round_style=RoundStyle.nearest`
  - `pseudo_quantize=False`
  - `transpose=False`
  - `block_scale_2d=False`
  - `rht=False`
  - `include_amax=True`

Required task2 benchmark shapes:

- `1x576`
- `1x1536`
- `6x576`
- `6x1536`
- `128x256`
- `512x2048`
- `1024x1024`
- `4096x4096`
- `8192x4096`
- `192x576`
- `576x576`
- `576x1536`
- `1536x576`
- `49152x576`

The first four activation-like shapes and five weight-like inference shapes
come from `profile/inference_kernel_profile_sm120_pure/latest.md`. The five
matrix benchmark shapes come from
`profile/quant_backend_fouroversix_current/latest.md`. If additional shapes are
added to either benchmark harness before task2 is completed, they become part
of the task2 required matrix unless explicitly marked out of scope in the
report.

## Acceptance Criteria

- [ ] For every required shape, CuTe SM120 public `quantize()` median time is
      strictly lower than CUDA public `quantize()` median time.
- [ ] For every required shape, CuTe SM120 public `quantize()` median time is
      strictly lower than Triton public `quantize()` median time.
- [ ] The benchmark report includes per-shape CUDA, Triton, and CuTe SM120
      median/mean/min/max quantization timings, repeat count, warmup count,
      iterations per sample, GPU id, device name, compute capability, PyTorch
      version, CUDA version, and `CUDA_VISIBLE_DEVICES`.
- [ ] The report includes a pass/fail table with CuTe-vs-CUDA speedup and
      CuTe-vs-Triton speedup for every shape.
- [ ] The report measures quantization only for acceptance. Dequant/fake quant
      timings may be present only as non-gating context.
- [ ] The benchmark includes a cold-cache or `include_amax=True` mode that does
      not let the SM120 CuTe amax cache hide repeated-call amax cost.
- [ ] If an optimized repeated-weight path uses the existing SM120 amax cache,
      the report must separate cached-amax results from task2 acceptance
      results.
- [ ] Existing task1 parity/accuracy tests still pass, or every failure is
      documented with exact test name, shape, and mismatch mode.
- [ ] The final fast-math audit command above returns no matches.
- [ ] NCU before/after evidence is recorded for at least one failing small
      shape and one failing medium/large shape from the current baseline.

## Required Profiling Workflow

The task should use the KernelWiki findings already recorded in
`profile/sm120_quantize_ncu_baseline/REPORT.md` and `docs/cute_backend_notes.md`:
SM120 is not SM100, RTX 5090 has compute capability 12.0 constraints, and
FlashInfer/vLLM style prior work uses SM120-specific NVFP4 paths rather than
blindly reusing SM100 kernels.

Use NCU for every optimization claim. If the external `ncu-profile-skill`
is available in a later environment, use it for capture/export/reporting.
In this environment it was not discoverable, so the fallback is explicit NCU
commands and checked-in reports.

Minimum NCU capture set:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:".*(Sm120|quant|reduce).*" \
  --export profile/sm120_quantize_task2/reports/<case>.ncu-rep \
  --force-overwrite \
  .venv/bin/python profile/sm120_quantize_ncu_baseline/harness/run_quantize_once.py \
  --backend cute_sm120 --dtype nvfp4 --scale-rule mse --shape <MxK>
```

For each profiled case, record:

- CuTe SM120 main quantize kernel time.
- CuTe SM120 amax/reduction kernel time.
- CUDA and Triton public quantize event time for the same shape.
- Launch count and kernel sequence.
- Register count, spills, occupancy, memory throughput, global-load/store
  efficiency, and top stall reasons.
- Whether the optimization changed main-kernel time, reduction time, Python
  dispatch/allocation cost, or padding/materialization cost.

## Optimization Direction

Prioritize fixes that can move public `quantize()` below CUDA on small and
medium shapes:

1. Reduce fixed dispatch and allocation cost in `CuteSm120QuantizeBackend`
   narrow NVFP4 adaptive paths.
2. Avoid quantize-time padding/materialization for non-128-row-aligned shapes
   while preserving downstream CUTLASS matmul correctness.
3. Keep using lazy materialization where the quantized tensor can safely defer
   padded-value and padded-scale cleanup outside the quantize timing window.
4. Consider a SM120-specific fused amax-plus-quantize path only if NCU and event
   timelines show the separate amax launch is the remaining dominant cost and
   the fused path can still beat CUDA on all required shapes.
5. For tiny activation shapes, benchmark direct backend calls and public
   `quantize()` separately to decide whether the gap is kernel work or frontend
   overhead.
6. Do not spend task2 time on `pseudo_quantize=True`, dequantization, transpose,
   2D block-scale, RHT, or non-NVFP4 formats unless a change is required to
   keep the NVFP4 public quantize path correct.

## Suggested Execution Plan

1. Establish a fresh baseline on an idle RTX 5090.
   - Run `profile/quant_backend_fouroversix_current/benchmark.py` with all
     required shapes and `--include-amax`.
   - Extend the report if needed so it prints CuTe-vs-CUDA and
     CuTe-vs-Triton quant speedups.

2. Capture NCU for representative failing shapes.
   - At minimum: one activation-like tiny shape, `128x256`, and one of
     `512x2048` or `1024x1024`.
   - Include `4096x4096` if it still fails CUDA after the fresh baseline.

3. Classify the gap per shape.
   - Main kernel slower than CUDA.
   - Amax/reduction launch dominates.
   - Public frontend/dispatch overhead dominates.
   - Allocation or lazy materialization bookkeeping dominates.
   - Padding/scale layout cleanup dominates.

4. Implement only SM120 CuTe changes.
   - Keep the fast path narrow and predicate-heavy enough that unrelated
     backends and formats are not affected.
   - Preserve task1 arithmetic and tie-breaking behavior.

5. Re-run the benchmark matrix.
   - Compare against both CUDA and Triton in the same process, same GPU, and
     same shape order.
   - Use medians for pass/fail and include raw samples in JSON.

6. Re-run accuracy and fast-math gates.
   - Use the task1 parity/accuracy tests for NVFP4 SM120.
   - Run the fast-math `rg` audit.

7. Write the completion report.
   - Save JSON and Markdown under `profile/sm120_quantize_task2/`.
   - Include baseline, final results, speedups, NCU conclusions, commands, and
     any residual risk.

## Non-Goals

- No fake quant performance target.
- No dequant performance target.
- No CUDA/Triton/SM100 changes.
- No backend fallback masking.
- No broad refactor of quantization config/frontend behavior unless it is
  strictly necessary for the SM120 CuTe NVFP4 quantize fast path and does not
  change other backends.
