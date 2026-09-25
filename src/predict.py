"""Inference CLI for a single image or a directory of images.

Examples::

    python -m src.predict --input samples/malignant_01.jpg
    python -m src.predict --input samples/ --csv reports/predictions.csv
    python -m src.predict --input samples/ --threshold 0.35
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from .config import POSITIVE_CLASS_INDEX, Config, load_config, resolve_path
from .dataset import IMAGE_EXTENSIONS
from .model import build_model
from .transforms import build_eval_transform
from .utils import load_checkpoint, select_device, setup_logging

LOGGER = logging.getLogger("src.predict")


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line interface for prediction."""
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Classify one image or a folder of images (research demo only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load.")
    parser.add_argument("--device", type=str, default=None, help="auto, cuda, mps or cpu.")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per forward pass.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability of the malignant class above which the image is flagged.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    return parser


def collect_images(target: Path, recursive: bool = False) -> list[Path]:
    """Return the list of image files under ``target``.

    Args:
        target: A single image file or a directory.
        recursive: Search subdirectories as well.

    Raises:
        FileNotFoundError: If ``target`` does not exist.
        ValueError: If a directory contains no supported image files.
    """
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    if target.is_file():
        if target.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {target.suffix}")
        return [target]

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in target.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No images with extensions {IMAGE_EXTENSIONS} found in {target}")
    return files


def load_predictor(
    checkpoint_path: Path, config: Config, device: torch.device
):
    """Load a checkpoint and return ``(model, class_names, image_size)``.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        RuntimeError: If the weights cannot be loaded.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    metadata = checkpoint.get("extra", {}) or {}
    class_names = list(metadata.get("class_names") or ["Benign", "Malignant"])
    image_size = int(metadata.get("image_size", config.data.image_size))

    model = build_model(
        name=metadata.get("model_name", config.model.name),
        num_classes=len(class_names),
        pretrained=False,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, image_size


def predict_paths(
    model: torch.nn.Module,
    paths: list[Path],
    class_names: list[str],
    image_size: int,
    device: torch.device,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> list[dict[str, object]]:
    """Classify every path, skipping (and logging) files that fail to open."""
    transform = build_eval_transform(image_size)
    results: list[dict[str, object]] = []

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        kept: list[Path] = []
        for path in chunk:
            try:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
                kept.append(path)
            except (OSError, UnidentifiedImageError) as exc:
                LOGGER.warning("Skipping %s: %s", path, exc)

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(batch).float(), dim=1).cpu()

        for path, probability_row in zip(kept, probabilities, strict=False):
            malignant_probability = float(probability_row[POSITIVE_CLASS_INDEX])
            predicted_index = (
                POSITIVE_CLASS_INDEX
                if malignant_probability >= threshold
                else 1 - POSITIVE_CLASS_INDEX
            )
            results.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "predicted_class": class_names[predicted_index],
                    "confidence": float(probability_row[predicted_index]),
                    "malignant_probability": malignant_probability,
                    "threshold": threshold,
                }
            )
    return results


def write_csv(rows: list[dict[str, object]], destination: Path) -> None:
    """Write prediction rows to ``destination`` as CSV."""
    if not rows:
        LOGGER.warning("No predictions to write to %s", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise OSError(f"Could not write CSV to {destination}: {exc}") from exc
    LOGGER.info("Wrote %d rows to %s", len(rows), destination)


def main(argv: list | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    setup_logging()

    try:
        config = load_config(args.config)
    except ValueError as exc:
        LOGGER.error("Invalid configuration: %s", exc)
        return 2
    if args.device:
        config.device = args.device

    device = select_device(config.device)
    checkpoint_path = resolve_path(
        args.checkpoint or os.getenv("CHECKPOINT_PATH") or "checkpoints/best.pt"
    )

    try:
        model, class_names, image_size = load_predictor(checkpoint_path, config, device)
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        LOGGER.error("Could not load checkpoint: %s", exc)
        return 1

    try:
        paths = collect_images(resolve_path(args.input), recursive=args.recursive)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Classifying %d image(s) at threshold %.2f", len(paths), args.threshold)
    rows = predict_paths(
        model, paths, class_names, image_size, device,
        batch_size=args.batch_size, threshold=args.threshold,
    )

    for row in rows:
        LOGGER.info(
            "%-28s -> %-10s (confidence %.3f, P(malignant) %.3f)",
            row["filename"], row["predicted_class"], row["confidence"],
            row["malignant_probability"],
        )

    if args.csv:
        try:
            write_csv(rows, resolve_path(args.csv))
        except OSError as exc:
            LOGGER.error("%s", exc)
            return 1

    LOGGER.warning(
        "Research demo output only. These predictions are not a diagnosis and must not "
        "be used for clinical decisions."
    )
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
