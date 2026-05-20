# CuTe sm100 PTX comparison and benchmark

## Triton PTX features observed

The Triton `quantization_kernel` cache contains Blackwell-specific FP8/FP4
conversion and tensor memory movement instructions. Representative cache paths:

- `/home/dongyun/.triton/cache/6EWANFPXHQMLIH3ITP5DDDOG434PAYBZ7XJIN466LX2FKDBSSDHA/quantization_kernel.ptx`
- `/home/dongyun/.triton/cache/C6RT7FZT2SEMADNRWLDJLL7S5WU6MENUQGQW7I7ZVAGOW5UV44EA/quantization_kernel.ptx`

Relevant instruction families found by `rg`:

- `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`
- `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group`
- `cp.async.bulk.tensor.1d.global.shared::cta.bulk_group`
- `cp.async.bulk.wait_group.read`
- `cvt.rn.satfinite.e4m3x2.f32`
- `cvt.rn.f16x2.e4m3x2`
- `cvt.rn.satfinite.e2m1x2.f32`
- `cvt.rn.f16x2.e2m1x2`

The CuTe sm100 kernel mirrors the hardware FP8/FP4 conversion instructions via
inline PTX in `src/fouroversix/kernels/cute_sm100/fp4_common.py`. It currently
uses vector global loads/stores instead of Triton's TMA tensor-descriptor
load/store path, and returns linear scale-factor layout with
`scale_factors_are_in_blackwell_layout=False`.

## Host-sync and extra-launch fixes

The first CuTe wrapper computed `global_scale` with `amax.item()` and used
`amax.item()` for the zero-input fast path. That forced a CPU/GPU sync every
quantize call. The current implementation computes global scale as a CUDA tensor
inside the CuTe kernel from the device `amax` tensor, so the amax reduction and
CuTe kernel remain queued on-device without a host scalar read or an extra
PyTorch elementwise launch.

## Targeted correctness

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_quantize.py::test_triton_matches_pytorch_reference_quantize_metrics tests/test_quantize.py::test_cute_sm100_quantize_is_not_less_accurate_than_triton tests/test_quantize.py::test_cute_sm100_quantize_zero_input -q -rxXs
```

Result:

```text
30 passed, 1 warning in 6.12s
```

Zero-input smoke test after the host-sync fix:

```text
mse [0] [0] 0.0 0.0
static_6 [0] [0] 0.0 0.0
static_4 [0] [0] 0.0 0.0
abs_max [0] [0] 0.0 0.0
mae [0] [0] 0.0 0.0
```

## Latency snapshot

Environment:

```text
torch 2.10.0+cu130, CUDA 13.0, NVIDIA B200, capability (10, 0)
```

Per-call event timing, including amax and quantization:

```text
shape (128, 256)
  mse       triton 0.0600 ms, cute_sm100 0.0577 ms, value_equal 0.996155
  static_6  triton 0.0574 ms, cute_sm100 0.0610 ms, value_equal 0.997498

shape (1024, 1024)
  mse       triton 0.0594 ms, cute_sm100 0.0610 ms, value_equal 0.999874
  static_6  triton 0.0589 ms, cute_sm100 0.0596 ms, value_equal 1.000000

shape (4096, 4096)
  mse       triton 0.1118 ms, cute_sm100 0.0639 ms, value_equal 0.999968
  static_6  triton 0.0614 ms, cute_sm100 0.0657 ms, value_equal 1.000000
```

The CuTe adaptive MSE path is faster than Triton at the large shape in this
snapshot and comparable at small/medium shapes. Static quantization is within
single-digit microseconds of Triton in this measurement. The remaining structural
difference is scale layout/TMA: Triton uses tensor-descriptor/TMA movement and
Blackwell scale-factor megablock layout, while the CuTe backend currently emits
linear scale factors and relies on direct vector global memory operations.
