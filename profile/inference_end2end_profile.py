from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from fouroversix import DataType, QuantizationConfig, QuantizeBackend, ScaleRule, quantize
from fouroversix.matmul import quantized_matmul
from fouroversix.matmul.cutlass.backend import CUTLASSMatmulBackend
from fouroversix.utils import MatmulBackend
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class FourOverSixBenchLinear(nn.Module):
    def __init__(
        self,
        module: nn.Linear,
        *,
        activation_backend: QuantizeBackend,
        dtype: DataType,
        scale_rule: ScaleRule,
        weight_backend: QuantizeBackend,
    ) -> None:
        super().__init__()
        self.out_features = module.out_features
        self.bias = module.bias
        self.activation_config = QuantizationConfig(
            backend=activation_backend,
            dtype=dtype,
            scale_rule=scale_rule,
        )
        weight_config = QuantizationConfig(
            backend=weight_backend,
            dtype=dtype,
            scale_rule=scale_rule,
        )
        with torch.inference_mode():
            self.quantized_weight = quantize(module.weight.detach(), weight_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = quantized_matmul(
            x.reshape(-1, x.shape[-1]),
            self.quantized_weight,
            backend=MatmulBackend.cutlass,
            input_config=self.activation_config,
            out_dtype=DataType.bfloat16,
        )
        out = out.reshape(*x.shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_linears(
    root: nn.Module,
    *,
    activation_backend: QuantizeBackend,
    dtype: DataType,
    scale_rule: ScaleRule,
    weight_backend: QuantizeBackend,
) -> int:
    replaced = 0
    for name, child in list(root.named_children()):
        if name == "lm_head":
            continue
        if isinstance(child, nn.Linear):
            setattr(
                root,
                name,
                FourOverSixBenchLinear(
                    child,
                    activation_backend=activation_backend,
                    dtype=dtype,
                    scale_rule=scale_rule,
                    weight_backend=weight_backend,
                ),
            )
            replaced += 1
        else:
            replaced += replace_linears(
                child,
                activation_backend=activation_backend,
                dtype=dtype,
                scale_rule=scale_rule,
                weight_backend=weight_backend,
            )
    return replaced


def summarize_times(times: list[float], max_new_tokens: int) -> dict[str, float]:
    mean_ms = statistics.mean(times)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "tokens_per_second": max_new_tokens / (mean_ms / 1000),
    }


def benchmark_generate(
    model: nn.Module,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    prompt: str,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return summarize_times(times, max_new_tokens)


def load_model(model_name: str) -> nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to("cuda")


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    results: dict[str, Any] = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dtype": args.dtype.value,
        "scale_rule": args.scale_rule.value,
        "generated_at_unix": int(time.time()),
        "cutlass_available": CUTLASSMatmulBackend.is_available(),
        "rows": [],
    }

    model = load_model(args.model_name)
    bf16 = benchmark_generate(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    results["rows"].append({"name": "bf16", "linear_replacements": 0, **bf16})
    del model
    gc.collect()
    torch.cuda.empty_cache()

    configs = [
        (QuantizeBackend.triton, QuantizeBackend.triton),
        (QuantizeBackend.triton, QuantizeBackend.cute_sm100),
        (QuantizeBackend.cute_sm100, QuantizeBackend.triton),
        (QuantizeBackend.cute_sm100, QuantizeBackend.cute_sm100),
    ]
    for weight_backend, activation_backend in configs:
        model = load_model(args.model_name)
        replacements = replace_linears(
            model,
            activation_backend=activation_backend,
            dtype=args.dtype,
            scale_rule=args.scale_rule,
            weight_backend=weight_backend,
        )
        row = benchmark_generate(
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            prompt=args.prompt,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        name = f"w_{weight_backend.value}__a_{activation_backend.value}"
        row["speedup_vs_bf16"] = bf16["mean_ms"] / row["mean_ms"]
        results["rows"].append(
            {
                "name": name,
                "weight_backend": weight_backend.value,
                "activation_backend": activation_backend.value,
                "linear_replacements": replacements,
                **row,
            },
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def write_markdown(output: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Inference End-to-End Profile",
        "",
        f"- model: `{results['model_name']}`",
        f"- max_new_tokens: `{results['max_new_tokens']}`",
        f"- warmup: `{results['warmup']}`",
        f"- repeats: `{results['repeats']}`",
        f"- dtype: `{results['dtype']}`",
        f"- scale_rule: `{results['scale_rule']}`",
        f"- cutlass_available: `{results['cutlass_available']}`",
        f"- generated_at_unix: `{results['generated_at_unix']}`",
        "",
        "This profiles a local end-to-end generate wrapper. It keeps `lm_head` in BF16,",
        "pre-quantizes Linear weights, and uses CUTLASS FP4 matmul for replaced Linear",
        "modules. The result measures complete `generate(...)` latency, not just the",
        "standalone quantize kernel.",
        "",
        "| config | replacements | mean ms | median ms | tokens/s | speedup vs BF16 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["rows"]:
        speedup = row.get("speedup_vs_bf16", 1.0)
        lines.append(
            "| "
            f"{row['name']} | {row['linear_replacements']} | {row['mean_ms']:.3f} | "
            f"{row['median_ms']:.3f} | {row['tokens_per_second']:.3f} | "
            f"{speedup:.3f}x |",
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--prompt", default="Four over six quantization accelerates")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--dtype", type=DataType, default=DataType.nvfp4)
    parser.add_argument("--scale-rule", type=ScaleRule, default=ScaleRule.mse)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profile/inference_end2end_profile"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for end-to-end inference profiling")
    if not CUTLASSMatmulBackend.is_available():
        raise RuntimeError("CUTLASS FP4 matmul backend is not available")

    results = run_profile(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "latest.json"
    md_path = args.output_dir / "latest.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, results)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
