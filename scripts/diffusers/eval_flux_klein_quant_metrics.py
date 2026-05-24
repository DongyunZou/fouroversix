"""Run FLUX.2-klein-4B BF16/NVFP4 generation and image-difference metrics."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image

from fouroversix.diffusers import FourOverSixConfig


DEFAULT_PROMPT = (
    "A cozy bookshop in Tokyo at night, warm light spilling onto a rain-slicked "
    "street, cherry blossom petals drifting through the air, Studio Ghibli style"
)


def flush_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0}
    return {
        "allocated": torch.cuda.max_memory_allocated() / 1e9,
        "reserved": torch.cuda.max_memory_reserved() / 1e9,
    }


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def compare_images(reference: Image.Image, candidate: Image.Image) -> dict[str, float]:
    ref = image_to_array(reference)
    cand = image_to_array(candidate)
    diff = ref - cand
    mse = float(np.mean(diff * diff))
    psnr = math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    return {
        "mse": mse,
        "psnr_db": psnr,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
    }


def load_pipeline(
    *,
    model_id: str,
    label: str,
    quantize_backend: str | None,
    matmul_backend: str,
) -> tuple[Flux2KleinPipeline, float]:
    flush_cuda()
    start = time.perf_counter()
    if quantize_backend is None:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
    else:
        quant_config = FourOverSixConfig(
            dtype="nvfp4",
            scale_rule="mse",
            quantize_backend=quantize_backend,
            matmul_backend=matmul_backend,
        )
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    elapsed = time.perf_counter() - start
    print(f"[{label}] loaded in {elapsed:.2f}s")
    return pipe, elapsed


def generate_image(
    pipe: Flux2KleinPipeline,
    *,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    seed: int,
) -> tuple[Image.Image, float, dict[str, float]]:
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return image, elapsed, peak_vram_gb()


def run_case(
    *,
    label: str,
    model_id: str,
    quantize_backend: str | None,
    matmul_backend: str,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    seed: int,
    output_dir: Path,
) -> tuple[Image.Image, dict[str, Any]]:
    pipe, load_time = load_pipeline(
        model_id=model_id,
        label=label,
        quantize_backend=quantize_backend,
        matmul_backend=matmul_backend,
    )
    image, generate_time, vram = generate_image(
        pipe,
        prompt=prompt,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
    )
    image_path = output_dir / f"{label}.png"
    image.save(image_path)
    print(f"[{label}] generated in {generate_time:.2f}s, saved {image_path}")

    del pipe
    flush_cuda()
    return image, {
        "label": label,
        "quantize_backend": quantize_backend,
        "matmul_backend": None if quantize_backend is None else matmul_backend,
        "load_time_s": load_time,
        "generate_time_s": generate_time,
        "peak_vram_gb": vram,
        "image_path": str(image_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matmul-backend", default="cutlass")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profile/flux_klein_quant_metrics"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLUX quantized end-to-end metrics")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("bf16", None),
        ("nvfp4_triton", "triton"),
        ("nvfp4_cute_sm120", "cute_sm120"),
    ]

    images: dict[str, Image.Image] = {}
    results: dict[str, Any] = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cases": {},
        "comparisons": {},
    }

    for label, backend in cases:
        image, case_result = run_case(
            label=label,
            model_id=args.model_id,
            quantize_backend=backend,
            matmul_backend=args.matmul_backend,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        images[label] = image
        results["cases"][label] = case_result

    for reference, candidate in [
        ("bf16", "nvfp4_triton"),
        ("bf16", "nvfp4_cute_sm120"),
        ("nvfp4_triton", "nvfp4_cute_sm120"),
    ]:
        key = f"{reference}_vs_{candidate}"
        results["comparisons"][key] = compare_images(
            images[reference],
            images[candidate],
        )

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"metrics saved {metrics_path}")
    print(json.dumps(results["comparisons"], indent=2))


if __name__ == "__main__":
    main()
