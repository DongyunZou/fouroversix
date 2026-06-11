from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from fouroversix import (
    DataType,
    QuantizationConfig,
    QuantizeBackend,
    RoundStyle,
    ScaleRule,
    dequantize,
    quantize,
)
from fouroversix.quantize.frontend import AVAILABLE_BACKENDS
from fouroversix.utils import SM_100, SM_120


DEFAULT_SHAPES = [
    (128, 256),
    (512, 2048),
    (1024, 1024),
    (4096, 4096),
    (8192, 4096),
]
DEFAULT_BACKENDS = [
    QuantizeBackend.cuda,
    QuantizeBackend.triton,
    "cute",
]


def _parse_shape(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    try:
        shape = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid shape {value!r}") from exc
    if len(shape) != 2:
        raise argparse.ArgumentTypeError(f"Expected M,N shape, got {value!r}")
    if shape[0] <= 0 or shape[1] <= 0:
        raise argparse.ArgumentTypeError(f"Shape dimensions must be positive: {value!r}")
    return shape


def _cute_backend_for_device() -> QuantizeBackend:
    sm = torch.cuda.get_device_capability()[0]
    if sm == SM_100:
        return QuantizeBackend.cute_sm100
    if sm == SM_120:
        return QuantizeBackend.cute_sm120
    return QuantizeBackend.cute_sm120


def _parse_backend(value: str) -> QuantizeBackend | str:
    return "cute" if value == "cute" else QuantizeBackend(value)


def _resolve_backends(values: list[QuantizeBackend | str]) -> list[QuantizeBackend]:
    resolved = []
    for backend in values:
        resolved.append(_cute_backend_for_device() if backend == "cute" else backend)
    return resolved


def _iters_for_shape(shape: tuple[int, int]) -> int:
    numel = shape[0] * shape[1]
    if numel <= 128 * 256:
        return 500
    if numel <= 1024 * 1024:
        return 200
    if numel <= 4096 * 4096:
        return 60
    return 30


def _clear_cute_amax_cache(backend: QuantizeBackend) -> None:
    if backend == QuantizeBackend.cute_sm100:
        from fouroversix.quantize.cute import sm100_backend

        cache = getattr(sm100_backend, "_AMAX_CACHE", None)
    else:
        cache = None
    if cache is not None:
        cache.clear()


def _time_cuda_ms(
    fn,
    *,
    iters: int,
    warmups: int,
    before_each=None,
) -> float:
    for _ in range(warmups):
        if before_each is not None:
            before_each()
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if before_each is not None:
            before_each()
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _can_quantize(
    backend: QuantizeBackend,
    x: torch.Tensor,
    config: QuantizationConfig,
) -> tuple[bool, str | None]:
    backend_cls = AVAILABLE_BACKENDS[backend]
    try:
        if not backend_cls.is_available():
            return False, "not_available"
        if not backend_cls.can_quantize(x, config):
            return False, "not_supported"
    except Exception as exc:  # noqa: BLE001
        return False, f"probe_failed: {type(exc).__name__}: {exc}"
    return True, None


def _can_dequantize(backend: QuantizeBackend, tensor) -> bool:
    try:
        return AVAILABLE_BACKENDS[backend].can_dequantize(tensor)
    except Exception:  # noqa: BLE001
        return False


def _pick_dequant_backend(
    quant_backend: QuantizeBackend,
    tensor,
) -> tuple[QuantizeBackend | None, bool]:
    if _can_dequantize(quant_backend, tensor):
        return quant_backend, True

    candidates = [
        QuantizeBackend.cuda,
        _cute_backend_for_device(),
        QuantizeBackend.triton,
        QuantizeBackend.pytorch,
    ]
    for backend in candidates:
        if backend != quant_backend and _can_dequantize(backend, tensor):
            return backend, False
    return None, False


def _summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _benchmark_backend(
    *,
    backend: QuantizeBackend,
    x: torch.Tensor,
    repeats: int,
    warmups: int,
    iters: int,
    include_amax: bool,
) -> dict[str, object]:
    config = QuantizationConfig(
        backend=backend,
        dtype=DataType.nvfp4,
        round_style=RoundStyle.nearest,
        scale_rule=ScaleRule.mse,
    )

    can_quantize, reason = _can_quantize(backend, x, config)
    if not can_quantize:
        return {
            "backend": backend.value,
            "status": reason,
        }

    before_quant = (
        lambda: _clear_cute_amax_cache(backend)
        if include_amax and backend == QuantizeBackend.cute_sm100
        else None
    )
    if include_amax and backend == QuantizeBackend.cute_sm100:
        _clear_cute_amax_cache(backend)

    q = quantize(x, config)
    dequant_backend, same_dequant_backend = _pick_dequant_backend(backend, q)
    if dequant_backend is None:
        return {
            "backend": backend.value,
            "status": "dequant_not_supported",
        }

    # Trigger JIT/extension compilation outside the measured samples.
    dequantize(q, torch.bfloat16, backend=dequant_backend)
    torch.cuda.synchronize()

    quant_samples: list[float] = []
    dequant_samples: list[float] = []
    for _ in range(repeats):
        quant_samples.append(
            _time_cuda_ms(
                lambda: quantize(x, config),
                iters=iters,
                warmups=warmups,
                before_each=before_quant,
            ),
        )
        dequant_samples.append(
            _time_cuda_ms(
                lambda: dequantize(q, torch.bfloat16, backend=dequant_backend),
                iters=iters,
                warmups=warmups,
            ),
        )

    return {
        "backend": backend.value,
        "status": "ok",
        "dequant_backend": dequant_backend.value,
        "same_dequant_backend": same_dequant_backend,
        "iters_per_sample": iters,
        "quant": _summarize(quant_samples),
        "dequant": _summarize(dequant_samples),
    }


def _write_markdown(results: dict[str, object], path: Path) -> None:
    lines = [
        "# Four Over Six NVFP4 Quant/Dequant Backend Benchmark",
        "",
        f"- timestamp: {results['timestamp']}",
        f"- device: {results['device']}",
        f"- capability: {results['capability']}",
        f"- CUDA_VISIBLE_DEVICES: {results['cuda_visible_devices']}",
        f"- torch: {results['torch']}",
        f"- cuda: {results['cuda']}",
        f"- repeats: {results['repeats']}",
        f"- warmups: {results['warmups']}",
        f"- include_amax: {results['include_amax']}",
        f"- dtype: {results['config']['dtype']}",
        f"- scale_rule: {results['config']['scale_rule']}",
        f"- round_style: {results['config']['round_style']}",
        "",
        "| shape | backend | status | dequant backend | quant median ms | dequant median ms | quant mean ms | dequant mean ms |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for case in results["cases"]:
        shape = f"{case['shape'][0]}x{case['shape'][1]}"
        for item in case["results"]:
            if item["status"] != "ok":
                lines.append(f"| {shape} | {item['backend']} | {item['status']} |  |  |  |  |  |")
                continue
            dequant_backend = item["dequant_backend"]
            if not item["same_dequant_backend"]:
                dequant_backend = f"{dequant_backend} fallback"
            lines.append(
                "| "
                + " | ".join(
                    [
                        shape,
                        item["backend"],
                        item["status"],
                        dequant_backend,
                        f"{item['quant']['median_ms']:.6f}",
                        f"{item['dequant']['median_ms']:.6f}",
                        f"{item['quant']['mean_ms']:.6f}",
                        f"{item['dequant']['mean_ms']:.6f}",
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
        choices=["cuda", "triton", "cute", "cute_sm100", "cute_sm120"],
        dest="backends",
    )
    parser.add_argument("--shape", action="append", type=_parse_shape, dest="shapes")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument(
        "--include-amax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include amax reduction in measured quant time where the backend caches it.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for quant/dequant backend benchmark")

    selected_backends = _resolve_backends(
        [
            _parse_backend(value)
            for value in (
                args.backends
                if args.backends is not None
                else [
                    backend.value if isinstance(backend, QuantizeBackend) else backend
                    for backend in DEFAULT_BACKENDS
                ]
            )
        ],
    )
    shapes = args.shapes or DEFAULT_SHAPES

    torch.manual_seed(0)
    results: dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "include_amax": args.include_amax,
        "config": {
            "dtype": DataType.nvfp4.value,
            "scale_rule": ScaleRule.mse.value,
            "round_style": RoundStyle.nearest.value,
            "block_scale_2d": False,
            "transpose": False,
            "rht": False,
            "pseudo_quantize": False,
        },
        "timing": {
            "method": "CUDA events; median of per-backend repeated samples",
            "note": "Triton NVFP4 dequant is not exposed as a supported frontend backend in this revision; fallback backend is recorded per row.",
        },
        "cases": [],
    }

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        iters = _iters_for_shape(shape)
        case = {"shape": shape, "iters_per_sample": iters, "results": []}
        print(f"\nshape={shape[0]}x{shape[1]} iters={iters}")
        for backend in selected_backends:
            print(f"  {backend.value}...", flush=True)
            item = _benchmark_backend(
                backend=backend,
                x=x,
                repeats=args.repeats,
                warmups=args.warmups,
                iters=iters,
                include_amax=args.include_amax,
            )
            case["results"].append(item)
            if item["status"] == "ok":
                print(
                    "    "
                    f"quant={item['quant']['median_ms']:.6f} ms, "
                    f"dequant={item['dequant']['median_ms']:.6f} ms "
                    f"via {item['dequant_backend']}",
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
