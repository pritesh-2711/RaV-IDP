"""Standalone tiled DocEnTR inference entry point for its isolated environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from vit_pytorch import ViT


def _build_model(repo_dir: Path, model_size: str, split_size: int, patch_size: int):
    sys.path.insert(0, str(repo_dir))
    from models.binae import BinModel

    hyperparameters = {
        "small": (3, 4, 512),
        "base": (6, 8, 768),
        "large": (12, 16, 1024),
    }
    layers, heads, dimension = hyperparameters[model_size]
    encoder = ViT(
        image_size=(split_size, split_size),
        patch_size=patch_size,
        num_classes=1000,
        dim=dimension,
        depth=layers,
        heads=heads,
        mlp_dim=2048,
    )
    return BinModel(
        encoder=encoder,
        decoder_dim=dimension,
        decoder_depth=layers,
        decoder_heads=heads,
    )


def _state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model_state_dict", "model_state"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in checkpoint.items()
    }


def _tiles(image: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    padded_height = ((height + size - 1) // size) * size
    padded_width = ((width + size - 1) // size) * size
    padded = cv2.copyMakeBorder(
        image,
        0,
        padded_height - height,
        0,
        padded_width - width,
        cv2.BORDER_REFLECT_101,
    )
    tiles = [
        padded[y : y + size, x : x + size]
        for y in range(0, padded_height, size)
        for x in range(0, padded_width, size)
    ]
    return np.stack(tiles), (padded_height, padded_width)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(Path(args.repo_dir), args.model_size, args.split_size, args.patch_size)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(_state_dict(checkpoint))
    model.to(device).eval()

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read input image: {args.input}")
    original_height, original_width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tiles, (padded_height, padded_width) = _tiles(rgb, args.split_size)
    reconstructed: list[np.ndarray] = []
    for start in range(0, len(tiles), args.batch_size):
        batch = tiles[start : start + args.batch_size].astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad():
            _, _, predicted = model(tensor, torch.zeros_like(tensor))
        images = rearrange(
            predicted,
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            p1=args.patch_size,
            p2=args.patch_size,
            h=args.split_size // args.patch_size,
        )
        reconstructed.extend(images.detach().cpu().clamp(0, 1).numpy())

    canvas = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    index = 0
    for y in range(0, padded_height, args.split_size):
        for x in range(0, padded_width, args.split_size):
            canvas[y : y + args.split_size, x : x + args.split_size] = reconstructed[index]
            index += 1
    canvas = canvas[:original_height, :original_width]
    gray = cv2.cvtColor((canvas * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    binary = np.where(gray > round(args.threshold * 255), 255, 0).astype(np.uint8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), binary):
        raise RuntimeError(f"Unable to write DocEnTR output: {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="base")
    parser.add_argument("--split-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
