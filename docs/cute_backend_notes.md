# CuTe Backend Development Notes

This note records the practical context from the sm100 CuTe backend work and
the references most useful for continuing with sm120 support on RTX 5090-class
hardware.

## Current sm100 Status

The current completed scope is the default auto-amax CuTe sm100 versus Triton
benchmark matrix under this policy:

- Non-pseudo rows require `>=1.2x` Triton speed.
- Pseudo rows require strict speedup over Triton (`>1.0x`).
- Provided-`x_amax` rows are measured separately and are not part of the
  completion scope.

Authoritative status artifacts:

- `profile/cute_sm100_full_triton_parity/CURRENT_STATUS.md`
- `profile/cute_sm100_full_triton_parity/completion_status.json`
- `profile/cute_sm100_full_triton_parity/benchmark_current.json`
- `profile/cute_sm100_full_triton_parity/executable_support_current.json`

Fresh recorded validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python profile/cute_sm100_full_triton_parity/benchmark_current.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python profile/cute_sm100_full_triton_parity/audit_executable_support.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_quantize.py -q -rxXs -k cute_sm100
```

The latest recorded results are:

- `476/476` default auto-amax workloads meet the required target.
- `338/338` non-pseudo rows meet `>=1.2x`.
- `138/138` pseudo rows are strictly faster than Triton.
- CuTe has `0` missing Triton-runnable rows in the executable support audit.
- `488 passed, 10 skipped` for the full `cute_sm100` test selection.

## Source Layout

Primary CuTe sm100 implementation:

- `src/fouroversix/kernels/cute_sm100/fp4_common.py`
- `src/fouroversix/kernels/cute_sm100/ops.py`
- `src/fouroversix/quantize/cute/backend.py`

Backend dispatch and benchmark plumbing:

- `src/fouroversix/quantize/frontend.py`
- `src/fouroversix/quantize/config.py`
- `profile/cute_sm100_full_triton_parity/benchmark_current.py`
- `profile/cute_sm100_full_triton_parity/audit_executable_support.py`
- `tests/test_quantize.py`

Reference implementations and parity targets:

- `src/fouroversix/quantize/triton.py`
- `src/fouroversix/kernels/triton/ops.py`
- `src/fouroversix/kernels/triton/quantize.py`
- `src/fouroversix/quantize/pytorch/reference.py`

Compiled CUTLASS FP4 GEMM path:

- `src/fouroversix/matmul/cutlass/backend.py`
- `src/fouroversix/csrc/fp4_gemm.cu`
- `src/fouroversix/csrc/fp4_gemm_sm120.cu`
- `src/fouroversix/csrc/include/fp4_quant_kernel.h`
- `src/fouroversix/csrc/include/fp4_quant_launch_template.h`

## sm120 Build Notes

For RTX 5090 / sm120 work, build only the target architecture while iterating:

```bash
git submodule update --init third_party/cutlass
CUDA_ARCHS=120 MAX_JOBS=4 FORCE_BUILD=1 .venv/bin/python setup.py build_ext --inplace
```

If editable install is needed, prefer:

```bash
CUDA_ARCHS=120 MAX_JOBS=4 FORCE_BUILD=1 uv pip install -e . --no-build-isolation --python .venv/bin/python
```

If this fails on old packaging metadata validation, check the local setuptools
version. The project expects `setuptools>=77.0.3`; an older version can reject
the `pyproject.toml` license field.

The generated extension `src/fouroversix/_C*.so` and `build/` are local build
artifacts and should remain uncommitted.

After build, verify availability:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from fouroversix.matmul.cutlass.backend import CUTLASSMatmulBackend
print(CUTLASSMatmulBackend.is_available())
PY
```

## sm120 Porting Checklist

Start by mirroring sm100 coverage before tuning:

- Add a dedicated backend enum or dispatch path only if sm120 behavior cannot be
  represented cleanly by the current `cute_sm100` backend name. Otherwise keep
  the public backend stable and branch internally on compute capability.
- Audit all compute-capability checks. Existing utility constants live in
  `src/fouroversix/utils.py` (`SM_100`, `SM_120`, `BLACKWELL_SM_IDS`).
- Confirm CUTLASS Python/CuTe DSL APIs used by `cute_sm100` accept sm120 target
  lowering. Do not assume sm100 launch parameters transfer to sm120.
- Port support predicates first, then executable support, then performance.
- Keep support predicate checks and actual executable audit separate. Triton has
  rows where predicates claim support but actual kernels fail to run.
- Reuse `benchmark_current.py` policy logic so the sm120 comparison keeps the
  same non-pseudo and pseudo target semantics.
- Keep provided-`x_amax` rows separate from the default auto-amax completion
  scope unless the target policy is explicitly changed.

Suggested early validation loop:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python profile/cute_sm100_full_triton_parity/audit_executable_support.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_quantize.py -q -rxXs -k cute
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python profile/cute_sm100_full_triton_parity/benchmark_current.py
```

If the backend name changes for sm120, clone the profile harness first rather
than overloading historical sm100 output paths.

## Performance Lessons

The most useful profiling artifacts are under
`profile/cute_sm100_full_triton_parity/ncu/`. The negative experiments are worth
keeping in mind:

- CTA/thread-count retunes helped only specific pseudo paths and often hurt
  static/base paths.
- Transpose-view shortcuts did not replace the need for real tiled
  transpose-plus-quantize kernels.
- Small decode-like activation rows are dominated by fixed launch overhead; a
  faster single quantize kernel may not move end-to-end latency.
- Layout compatibility with CUTLASS FP4 GEMM matters as much as raw quantize
  kernel speed.

The kernel-level inference profile is recorded in:

- `profile/inference_kernel_profile/latest.md`
- `profile/inference_kernel_profile/latest.json`

On `HuggingFaceTB/SmolLM2-135M`, CuTe sm100 was faster than Triton for the
measured Linear weight and activation quantize shapes. This is a quantize-kernel
profile only.

The local end-to-end wrapper profile is recorded in:

- `profile/inference_end2end_profile/latest.md`
- `profile/inference_end2end_profile/latest.json`

That wrapper showed no end-to-end LLM speedup. Treat it as a bottleneck-finding
artifact, not an official deployment benchmark.

## Official Example Reality Check

There are three different example classes in this repository:

- `scripts/speedtest/quantize.py` and `scripts/speedtest/matmul.py` are kernel
  speedtests.
- `scripts/ptq` is the newer vLLM-based PTQ/eval path. The current vLLM
  integration pseudo-quantizes and dequantizes back to high precision for the
  linear operation, so it should not be treated as a real FP4 GEMM serving
  speedup example.
- `scripts/diffusers/bench_flux_klein.py` is the clearest real model-level
  speed example. It compares BF16 FLUX.2-klein-4B against a FourOverSix
  `FourOverSixLinear` Diffusers transformer path that uses quantized weights and
  `quantized_matmul` for Linear layers.

Relevant files:

- `scripts/diffusers/bench_flux_klein.py`
- `src/fouroversix/diffusers/quantizer.py`
- `src/fouroversix/model/modules/linear.py`
- `src/fouroversix/integrations/vllm_quantization.py`

For sm120 deployment work, use the Diffusers benchmark and the matmul speedtest
as the first real acceleration targets. The vLLM path needs a true low-bit GEMM
integration before it can be used as evidence of serving acceleration.

## Housekeeping

Use `PYTHONDONTWRITEBYTECODE=1` for profile/test commands when possible. Before
committing, verify:

```bash
git status --short --branch
find profile src tests docs -type d -name __pycache__ -prune -print
git ls-files --others --exclude-standard
```

Do not commit generated local build outputs such as `build/`, `.venv/`, or
`src/fouroversix/_C*.so`.
