# NVINT4 pseudo 128x256 profile

This profiles a representative fixed-overhead-limited row:
`128x256 nvint4 static_6 pseudo_quantize=True`.

Commands:

```bash
HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ncu --target-processes all \
  --profile-from-start off --csv --force-overwrite \
  -o profile/cute_sm100_full_triton_parity/ncu/nvint4_pseudo128_cute \
  .venv/bin/python - <<'PY'
import torch
from fouroversix import QuantizationConfig, QuantizeBackend, quantize
from fouroversix.utils import DataType, ScaleRule

torch.manual_seed(0)
x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
config = QuantizationConfig(
    backend=QuantizeBackend.cute_sm100,
    dtype=DataType.nvint4,
    scale_rule=ScaleRule.static_6,
    pseudo_quantize=True,
)
for _ in range(10):
    quantize(x, config)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(5):
    quantize(x, config)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
PY
```

The Triton command is identical except `backend=QuantizeBackend.triton` and
`-o .../nvint4_pseudo128_triton`.

NCU kernel-duration summary over five profiled iterations:

```text
CuTe:
  torch AbsMax reduce:                12.16-12.54 us
  Sm100NVINT4StaticPseudoQuantize:     4.61-5.28 us

Triton:
  torch abs elementwise:               3.71-3.97 us
  torch max reduce:                   12.48-12.83 us
  scalar copy/cast:                    4.16-4.38 us
  pseudo_quantization_kernel:         20.70-21.70 us
```

The summed profiled GPU kernel durations are about `17.3 us/iter` for CuTe and
`41.7 us/iter` for Triton, but the retained CUDA-event benchmark row is only
around `1.1x` faster for CuTe. That means this small-shape row is not limited by
the CuTe pseudo kernel body; the missing speedup is mostly frontend/enqueue
idle time and fixed framework overhead around the very short GPU work.
