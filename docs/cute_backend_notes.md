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
- `src/fouroversix/quantize/cute/sm100_backend.py`

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

Local server note: GPU 0 currently has a persistent task attached. Use GPU 7 for
development runs, for example with `CUDA_VISIBLE_DEVICES=7`, unless this server
state changes.

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

- Keep sm100 and sm120 separated. The sm120 path uses
  `QuantizeBackend.cute_sm120`, `CuteSm120QuantizeBackend`, and
  `src/fouroversix/kernels/cute_sm120/` rather than widening the sm100 backend.
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

Current sm120 artifacts:

- `src/fouroversix/quantize/cute/sm120_backend.py`
- `src/fouroversix/kernels/cute_sm120/`
- `profile/cute_sm120_triton_parity/`
- `profile/sm120_quantize_ncu_baseline/REPORT.md`
- `profile/inference_kernel_profile_sm120/latest.md`

`CuteSm120QuantizeBackend` should stay a pure CuTe backend. A temporary
CUDA-delegating experiment was removed because it made the backend name
misleading. As of `profile/inference_kernel_profile_sm120_pure/latest.md`,
default inference quantize shapes (`nvfp4 + mse`, no transpose/no 2D/no pseudo)
still need sm120 CuTe optimization to consistently reach the Triton `1.2x`
target. The large `49152x576` weight shape is above target, but smaller
activation/weight shapes are dominated by fixed dispatch, amax, and
`QuantizedTensor` padding overhead. The next pure-CuTe target is to write padded
NVFP4 adaptive output directly from the sm120 kernel. The full non-pseudo parity
matrix is also not complete; transpose / 2D block-scale formats still need
sm120-specific kernel redesign.

Latest incremental update: the sm120 path now has pure-CuTe fused paths for
small activation-like shapes and `192x576` weight-like shapes. For larger
medium weight shapes, the fast path enqueues the true amax reduction first,
allocates the padded output tensors while that reduction is running, then
launches the CuTe quantize kernel into the preallocated tensors. Direct float8
scale allocation avoids the extra view while preserving CUTLASS matmul
correctness. The latest `profile/inference_kernel_profile_sm120_pure/latest.md`
still has `576x576` and `1536x576` just below target, so the inference target is
not complete yet. A uint8 scale-factor fast path was tested and rejected because
CUTLASS matmul requires `torch.float8_e4m3fn` scale tensors and aborts with
uint8 scales.

## RTX 5090 / SM120 Reality Check

Do not treat RTX 5090 / SM120 as a small variant of B200 / SM100. The current
upstream evidence points the other way:

- NVIDIA's RTX Blackwell architecture brief lists GeForce RTX 5090 as GB202
  with 170 SMs, 21,760 CUDA cores, 680 fifth-generation Tensor Cores, 32 GB
  GDDR7, a 512-bit memory interface, and 1792 GB/s memory bandwidth:
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- CUTLASS lists GeForce RTX 50x0 as compute capability 12.0 and B200 as
  compute capability 10.0. It explicitly warns that SM100 datacenter kernels
  compiled with architecture-conditional `sm100a` features are not compatible
  with RTX 50 series GPUs:
  https://docs.nvidia.com/cutlass/4.4.2/overview.html
- CUDA's Blackwell tuning guide separates compute capability 10.0 and 12.0:
  CC 12.0 has 48 max concurrent warps/SM, 128 KB shared memory/SM, and 99 KB
  max shared memory/thread block. CC 10.0 has 64 max concurrent warps/SM,
  228 KB shared memory/SM, and 227 KB max shared memory/thread block:
  https://docs.nvidia.com/cuda/archive/13.0.2/pdf/Blackwell_Tuning_Guide.pdf
- CUTLASS SM120 GEMM documentation says GeForce has no multicast, so cluster
  shape is fixed to `1x1x1`; only TN layout is supported; SM120 uses
  `mma.sync.aligned.kind::*` narrow-precision instructions rather than the
  SM100 `tcgen05.mma` path; and valid FP4/NVFP4 tile shapes and schedules are
  listed separately under "Blackwell SM120 GEMMs":
  https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms

FlashAttention-4 is also a useful style guide here. The FA4 paper and
KernelWiki page are about SM100/B200: the main optimizations exploit
asymmetric Blackwell scaling on B200, fully asynchronous MMA, software exp,
TMEM, and 2-CTA MMA mode. Current upstream FlashAttention keeps separate
classes for SM100 and SM120:

- `FlashAttentionForwardSm100` is a large CuTe DSL implementation using
  `tcgen05`, TMA, TMEM, warp-specialized roles, and optional 2-CTA execution.
- `FlashAttentionForwardSm120` is a separate class. It subclasses the SM80-era
  path, keeps `arch = 80` for cp.async/MMA code paths, and only adjusts the
  SM120 shared-memory capacity check.
- Dispatch in `flash_attn/cute/interface.py` handles `arch // 10 == 12`
  separately: SM120 uses 128 threads, smaller tile choices such as `128x64`
  for `head_dim > 64`, disables block sparsity / paged KV / split-KV in that
  path, and uses separate backward tile/stage choices.

The design implication for FourOverSix is that `CuteSm120QuantizeBackend`
should remain a distinct pure-CuTe backend and should not widen
`CuteSm100QuantizeBackend` or hide CUDA fallbacks behind a CuTe name. However,
we should also not blindly port SM100 kernel structure. SM120 work should bias
toward SM120 constraints first: 99 KB per-block shared-memory limit,
lower warp occupancy ceiling, no GeForce multicast / fixed `1x1x1` clusters
for GEMM-style kernels, TN-oriented low-bit layouts, and separate launch/tile
choices for small inference shapes.

KernelWiki PR references reinforce the same separation. vLLM and FlashInfer
landed SM120-specific NVFP4 files rather than relying only on SM100 code:

- `pr-vllm-21309`: adds `nvfp4_scaled_mm_sm120_kernels.cu` and SM120 NVFP4
  quant / scaled-mm entry changes, tested on RTX 5090.
- `pr-flashinfer-1609`: adds `fp4_gemm_cutlass_sm120.cu` and separate
  `fp4_gemm_template_sm120.h` / `fp4_gemm_cutlass_template_sm120.h`.
- `pr-flashinfer-2725`: adds SM120 desktop Blackwell support for NVFP4 MoE
  by removing SM100-only checks and adding major version 12 support.

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

## SM120 Current Stable Baseline

The current `cute_sm120` NVFP4 adaptive inference path is now split between
quantize-time work and CUTLASS-matmul materialization:

- Clearing padded `values` inside the same CuTe kernel was tested with both
  2D indexing and flat pointer stores. Both variants can intermittently corrupt
  real scale factor bytes on RTX 5090/SM120, for example producing `0xff` in
  real `576x1536` weight scales.
- The stable fast path therefore skips row-padding work during quantize and
  marks the `QuantizedTensor` for lazy materialization. `CUTLASSMatmulBackend`
  pads/zeros values and scale factors just before GEMM. This keeps quantize
  kernels fast while preserving finite CUTLASS output.
- Repeated `m >= 128` weight quantization caches `amax` by tensor identity,
  data pointer, shape/stride, and tensor `_version`. In-place tensor mutation
  invalidates the cache.

Latest stable profile:

- `profile/inference_kernel_profile_sm120_pure/latest.md`
- generated_at_unix `1779380840`
- All listed inference quantize shapes exceed the 1.2x Triton target:
  weights are `1.449x-1.861x` and activations are `1.365x-1.419x`.

Known next optimization targets:

- Full fake/pseudo benchmark output is recorded in
  `profile/cute_sm120_triton_parity/benchmark_current.json`. It covers all
  `pseudo_quantize=True` rows in the parity harness for shapes `128x256`,
  `1024x1024`, and `4096x4096`; `138/138` rows are faster than Triton
  (`>1.0x`).
- Support audit passes for CuTe sm120: no missing Triton-runnable rows and no
  CuTe predicate/runtime failures in
  `profile/cute_sm120_triton_parity/executable_support_current.json`.
- Transpose kernels are improved versus the old NCU report but are not all
  `1.2x`: IF3/IF4 `4096x4096 transpose=True` are around `1.03x-1.09x`, and
  NVFP4 static transpose is around `1.17x`. A real tiled transpose redesign
  remains the next non-inference performance target.

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
