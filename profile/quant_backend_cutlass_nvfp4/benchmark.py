from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from fouroversix import (
    DataType,
    MatmulBackend,
    QuantizationConfig,
    QuantizeBackend,
    ScaleRule,
    quantize,
    quantized_matmul,
)
from fouroversix.matmul.frontend import AVAILABLE_BACKENDS as MATMUL_BACKENDS
from fouroversix.quantize.frontend import AVAILABLE_BACKENDS as QUANT_BACKENDS
from fouroversix.utils import SM_100, SM_120


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 3:
        msg = f"Expected M,N,K shape, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parts


def _mean_cuda_ms(fn, *, repeats: int, warmups: int) -> tuple[float, float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    std_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return statistics.mean(samples), std_ms


def _quantize_pair(x: torch.Tensor, y: torch.Tensor, config: QuantizationConfig):
    xq = quantize(x, config)
    yq = quantize(y, config)
    return xq, yq


def _default_quant_backends() -> list[QuantizeBackend]:
    sm = torch.cuda.get_device_capability()[0]
    cute_backend = {
        SM_100: QuantizeBackend.cute_sm100,
        SM_120: QuantizeBackend.cute_sm120,
    }.get(sm)

    backends = [QuantizeBackend.cuda]
    if cute_backend is not None:
        backends.append(cute_backend)
    backends.extend(
        [
            QuantizeBackend.triton,
            QuantizeBackend.pytorch,
            QuantizeBackend.transformer_engine,
        ],
    )
    return backends


def _benchmark_case(
    *,
    backend: QuantizeBackend,
    m: int,
    n: int,
    k: int,
    repeats: int,
    warmups: int,
    scale_rule: ScaleRule,
) -> dict[str, object]:
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    y = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(
        backend=backend,
        dtype=DataType.nvfp4,
        scale_rule=scale_rule,
    )

    backend_cls = QUANT_BACKENDS[backend]
    if not backend_cls.is_available():
        return {"backend": backend.value, "status": "not_available"}
    if not backend_cls.can_quantize(x, config) or not backend_cls.can_quantize(
        y,
        config,
    ):
        return {"backend": backend.value, "status": "not_supported"}

    # Materialize one pair for CUTLASS support checks and GEMM timing.
    xq, yq = _quantize_pair(x, y, config)
    if not MATMUL_BACKENDS[MatmulBackend.cutlass].is_supported(
        xq,
        yq,
        out_dtype=DataType.bfloat16,
    ):
        return {"backend": backend.value, "status": "cutlass_not_supported"}

    qx_ms, qx_std_ms = _mean_cuda_ms(
        lambda: quantize(x, config),
        repeats=repeats,
        warmups=warmups,
    )
    qy_ms, qy_std_ms = _mean_cuda_ms(
        lambda: quantize(y, config),
        repeats=repeats,
        warmups=warmups,
    )
    qpair_ms, qpair_std_ms = _mean_cuda_ms(
        lambda: _quantize_pair(x, y, config),
        repeats=repeats,
        warmups=warmups,
    )

    # Warmup mutates scale-factor layout for CUTLASS; keep it outside the measured loop.
    gemm_ms, gemm_std_ms = _mean_cuda_ms(
        lambda: quantized_matmul(
            xq,
            yq,
            backend=MatmulBackend.cutlass,
            out_dtype=DataType.bfloat16,
        ),
        repeats=repeats,
        warmups=max(warmups, 3),
    )

    return {
        "backend": backend.value,
        "status": "ok",
        "quant_x_ms": qx_ms,
        "quant_x_std_ms": qx_std_ms,
        "quant_y_ms": qy_ms,
        "quant_y_std_ms": qy_std_ms,
        "quant_pair_ms": qpair_ms,
        "quant_pair_std_ms": qpair_std_ms,
        "cutlass_gemm_ms": gemm_ms,
        "cutlass_gemm_std_ms": gemm_std_ms,
    }


def _write_markdown(results: dict[str, object], path: Path) -> None:
    lines = [
        "# NVFP4 quant backend + CUTLASS GEMM benchmark",
        "",
        f"- device: {results['device']}",
        f"- capability: {results['capability']}",
        f"- scale_rule: {results['scale_rule']}",
        f"- repeats: {results['repeats']}",
        f"- warmups: {results['warmups']}",
        "",
        "| shape M,N,K | quant backend | status | quant x ms | quant y ms | quant pair ms | cutlass gemm ms |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for case in results["cases"]:
        shape = f"{case['m']},{case['n']},{case['k']}"
        for item in case["results"]:
            if item["status"] != "ok":
                lines.append(
                    f"| {shape} | {item['backend']} | {item['status']} |  |  |  |  |",
                )
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        shape,
                        item["backend"],
                        item["status"],
                        f"{item['quant_x_ms']:.4f}",
                        f"{item['quant_y_ms']:.4f}",
                        f"{item['quant_pair_ms']:.4f}",
                        f"{item['cutlass_gemm_ms']:.4f}",
                    ],
                )
                + " |",
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        action="append",
        choices=[backend.value for backend in QuantizeBackend],
        dest="backends",
    )
    parser.add_argument("--shape", action="append", type=_parse_shape, dest="shapes")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument(
        "--scale-rule",
        choices=[rule.value for rule in ScaleRule],
        default="mse",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    selected_backends = [
        QuantizeBackend(value)
        for value in (
            args.backends
            if args.backends is not None
            else [backend.value for backend in _default_quant_backends()]
        )
    ]
    shapes = args.shapes or [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    results: dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "scale_rule": args.scale_rule,
        "cases": [],
    }

    for m, n, k in shapes:
        case = {"m": m, "n": n, "k": k, "results": []}
        print(f"\nM,N,K={m},{n},{k}")
        for backend in selected_backends:
            print(f"  {backend.value}...", flush=True)
            item = _benchmark_case(
                backend=backend,
                m=m,
                n=n,
                k=k,
                repeats=args.repeats,
                warmups=args.warmups,
                scale_rule=ScaleRule(args.scale_rule),
            )
            case["results"].append(item)
            if item["status"] == "ok":
                print(
                    "    "
                    f"quant_pair={item['quant_pair_ms']:.4f} ms, "
                    f"cutlass_gemm={item['cutlass_gemm_ms']:.4f} ms",
                    flush=True,
                )
            else:
                print(f"    {item['status']}", flush=True)
        results["cases"].append(case)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "latest.json"
    md_path = args.output_dir / "latest.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    _write_markdown(results, md_path)
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
