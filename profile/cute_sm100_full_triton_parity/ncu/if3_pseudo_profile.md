# IF3 pseudo profile notes

This is a profiling note for the 4096x4096 `if3 + abs_max + pseudo_quantize`
workload where Triton is still faster than the current CuTe sm100 path.

Commands used:

```bash
/usr/local/cuda-13.2/bin/ncu --target-processes all \
  --kernel-name 'regex:.*pseudo_quantization_kernel.*' \
  --launch-skip 3 --launch-count 2 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active \
  -o profile/cute_sm100_full_triton_parity/ncu/if3_pseudo_triton \
  --force-overwrite .venv/bin/python - <<'PY'
import torch
from fouroversix import QuantizationConfig, QuantizeBackend, quantize
from fouroversix.utils import DataType, ScaleRule
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
cfg = QuantizationConfig(
    backend=QuantizeBackend.triton,
    dtype=DataType.if3,
    scale_rule=ScaleRule.abs_max,
    pseudo_quantize=True,
)
for _ in range(8):
    quantize(x, cfg)
torch.cuda.synchronize()
PY

/usr/local/cuda-13.2/bin/ncu --target-processes all \
  --kernel-name 'regex:.*Sm100IF3AdaptivePseudoQuantize.*' \
  --launch-skip 3 --launch-count 2 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active \
  -o profile/cute_sm100_full_triton_parity/ncu/if3_pseudo_cute \
  --force-overwrite .venv/bin/python - <<'PY'
import torch
from fouroversix import QuantizationConfig, QuantizeBackend, quantize
from fouroversix.utils import DataType, ScaleRule
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
cfg = QuantizationConfig(
    backend=QuantizeBackend.cute_sm100,
    dtype=DataType.if3,
    scale_rule=ScaleRule.abs_max,
    pseudo_quantize=True,
)
for _ in range(8):
    quantize(x, cfg)
torch.cuda.synchronize()
PY
```

Summary:

| Backend | Kernel time | SM throughput | DRAM throughput | Active warps | Registers/thread | Grid | Block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | 33.6-33.7 us | 63.8-63.9% | 13.16-13.19% | 48.1-48.3% | 43 | 2048 | 128 |
| CuTe | 47.1-47.2 us | 82.6-82.9% | 9.39-9.42% | 42.8% | 54 | 1184 | 256 |

Interpretation:

- The gap is not DRAM bandwidth-bound. CuTe uses less DRAM percentage and
  higher SM throughput, but takes longer.
- The retained CuTe IF3 pseudo kernel maps one thread to one scale block. That
  keeps the implementation simple but uses more registers and exposes less
  useful parallelism than Triton's 2D tiled pseudo kernel.
- A small launch-parameter change is unlikely to close this gap. The likely
  fix is a dedicated tiled IF3/IF3_BS8 pseudo kernel, where a CTA cooperatively
  covers multiple rows and scale blocks, closer to Triton's
  `BLOCK_SIZE_M=128, BLOCK_SIZE_N=4 * block_size` decomposition.
