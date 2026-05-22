from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from fouroversix import DataType, QuantizationConfig, QuantizeBackend, ScaleRule, quantize
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def cuda_event_ms(fn, *, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def bench_quantize(
    x: torch.Tensor,
    *,
    backend: QuantizeBackend,
    dtype: DataType,
    scale_rule: ScaleRule,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    config = QuantizationConfig(
        backend=backend,
        dtype=dtype,
        scale_rule=scale_rule,
    )

    def run() -> None:
        quantize(x, config)

    times = cuda_event_ms(run, repeats=repeats, warmup=warmup)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def collect_model_shapes(
    model_name: str,
    *,
    prompt: str,
    max_new_tokens: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()

    weight_shapes: dict[tuple[int, int], int] = {}
    activation_shapes: dict[tuple[int, int], int] = {}
    hooks = []

    for module in model.modules():
        if isinstance(module, nn.Linear):
            shape = tuple(int(v) for v in module.weight.shape)
            weight_shapes[shape] = weight_shapes.get(shape, 0) + 1

            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.ndim == 0:
                    return
                flat_shape = (int(x.numel() // x.shape[-1]), int(x.shape[-1]))
                activation_shapes[flat_shape] = activation_shapes.get(flat_shape, 0) + 1

            hooks.append(module.register_forward_pre_hook(hook))

    encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)

    for hook in hooks:
        hook.remove()

    del model
    torch.cuda.empty_cache()
    return weight_shapes, activation_shapes


def profile_shapes(
    shapes: dict[tuple[int, int], int],
    *,
    baseline_backend: QuantizeBackend,
    candidate_backend: QuantizeBackend,
    dtype: DataType,
    scale_rule: ScaleRule,
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    rows = []
    for shape, count in sorted(shapes.items(), key=lambda item: (item[0][0] * item[0][1], item[0])):
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        baseline = bench_quantize(
            x,
            backend=baseline_backend,
            dtype=dtype,
            scale_rule=scale_rule,
            repeats=repeats,
            warmup=warmup,
        )
        candidate = bench_quantize(
            x,
            backend=candidate_backend,
            dtype=dtype,
            scale_rule=scale_rule,
            repeats=repeats,
            warmup=warmup,
        )
        speedup = baseline["mean_ms"] / candidate["mean_ms"]
        rows.append(
            {
                "shape": list(shape),
                "count": count,
                "baseline_mean_ms": baseline["mean_ms"],
                "candidate_mean_ms": candidate["mean_ms"],
                "speedup": speedup,
            },
        )
        del x
    return rows


def write_markdown(output: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Inference Kernel Profile",
        "",
        f"- model: `{results['model_name']}`",
        f"- dtype: `{results['dtype']}`",
        f"- scale_rule: `{results['scale_rule']}`",
        f"- baseline_backend: `{results['baseline_backend']}`",
        f"- candidate_backend: `{results['candidate_backend']}`",
        f"- generated_at_unix: `{results['generated_at_unix']}`",
        "",
    ]

    for section in ("weights", "activations"):
        rows = results[section]
        lines += [
            f"## {section.title()}",
            "",
            "| shape | count | Baseline mean ms | Candidate mean ms | speedup |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            shape = "x".join(str(v) for v in row["shape"])
            lines.append(
                "| "
                f"{shape} | {row['count']} | {row['baseline_mean_ms']:.6f} | "
                f"{row['candidate_mean_ms']:.6f} | {row['speedup']:.3f}x |",
            )
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--prompt", default="Four over six quantization accelerates")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", type=DataType, default=DataType.nvfp4)
    parser.add_argument("--scale-rule", type=ScaleRule, default=ScaleRule.mse)
    parser.add_argument(
        "--baseline-backend",
        type=QuantizeBackend,
        default=QuantizeBackend.triton,
    )
    parser.add_argument(
        "--candidate-backend",
        type=QuantizeBackend,
        default=QuantizeBackend.cute_sm120,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profile/inference_kernel_profile"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for quantize backend profiling")

    weight_shapes, activation_shapes = collect_model_shapes(
        args.model_name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    results = {
        "model_name": args.model_name,
        "dtype": args.dtype.value,
        "scale_rule": args.scale_rule.value,
        "baseline_backend": args.baseline_backend.value,
        "candidate_backend": args.candidate_backend.value,
        "generated_at_unix": int(time.time()),
        "weights": profile_shapes(
            weight_shapes,
            baseline_backend=args.baseline_backend,
            candidate_backend=args.candidate_backend,
            dtype=args.dtype,
            scale_rule=args.scale_rule,
            repeats=args.repeats,
            warmup=args.warmup,
        ),
        "activations": profile_shapes(
            activation_shapes,
            baseline_backend=args.baseline_backend,
            candidate_backend=args.candidate_backend,
            dtype=args.dtype,
            scale_rule=args.scale_rule,
            repeats=args.repeats,
            warmup=args.warmup,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "latest.json"
    md_path = args.output_dir / "latest.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, results)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
