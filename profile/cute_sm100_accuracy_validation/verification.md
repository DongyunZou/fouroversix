# CuTe sm100 accuracy validation

## Environment
```text
torch 2.10.0+cu130
torch.version.cuda 13.0
torch.cuda.is_available() True
torch.cuda.device_count() 1
torch.cuda.get_device_name(0) NVIDIA B200
torch.cuda.get_device_capability(0) (10, 0)
CuteSm100QuantizeBackend.is_available() True
```

## Targeted pytest
Command:
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_quantize.py::test_triton_matches_pytorch_reference_quantize_metrics tests/test_quantize.py::test_cute_sm100_quantize_is_not_less_accurate_than_triton -q -rxXs
```

Output:
```text
.........................                                                [100%]
=============================== warnings summary ===============================
src/fouroversix/__init__.py:16
  /home/dongyun/workspace/projects/fouroversix/src/fouroversix/__init__.py:16: UserWarning: Install diffusers>=0.32.0 to use the diffusers integration: pip install 'fouroversix[diffusers]'
    from .diffusers import FourOverSixConfig

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
25 passed, 1 warning in 6.03s
```
