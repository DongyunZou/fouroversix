import torch
from fouroversix import DataType, QuantizationConfig, QuantizeBackend, ScaleRule, quantize


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=QuantizeBackend.triton,
        dtype=DataType.nvfp4,
        scale_rule=ScaleRule.mse,
    )

    for _ in range(3):
        quantize(x, config)
    torch.cuda.synchronize()

    quantize(x, config)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
