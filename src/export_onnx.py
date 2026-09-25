"""Export a trained checkpoint to ONNX with a dynamic batch axis.

The exported graph takes a normalised float tensor of shape
``(N, 3, image_size, image_size)`` and returns raw logits of shape
``(N, num_classes)``; softmax is left to the consumer.

If ``onnxruntime`` is installed the export is verified numerically against the
PyTorch model. If it is not, the export still succeeds and a note is logged.

Example::

    python -m src.export_onnx --checkpoint checkpoints/best.pt --output melanoma.onnx
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch

from .config import load_config, resolve_path
from .predict import load_predictor
from .utils import select_device, setup_logging

LOGGER = logging.getLogger("src.export_onnx")


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line interface for ONNX export."""
    parser = argparse.ArgumentParser(
        prog="python -m src.export_onnx",
        description="Export the trained classifier to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to export.")
    parser.add_argument(
        "--output", type=str, default="checkpoints/model.onnx", help="Destination .onnx file."
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size of the tracing sample."
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip the onnxruntime numerical check."
    )
    return parser


def verify_with_onnxruntime(
    onnx_path, sample: torch.Tensor, reference: np.ndarray, tolerance: float = 1e-3
) -> bool:
    """Compare ONNX Runtime output against the PyTorch reference.

    Args:
        onnx_path: Path to the exported model.
        sample: The input tensor used for tracing, on CPU.
        reference: PyTorch logits for ``sample``.
        tolerance: Maximum tolerated absolute difference.

    Returns:
        ``True`` when the outputs agree, ``False`` when they differ or when
        ``onnxruntime`` is unavailable or fails to load the model.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        LOGGER.warning(
            "onnxruntime is not installed; skipping verification. "
            "Install it with: pip install onnxruntime"
        )
        return False

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: sample.numpy()})
    except Exception as exc:  # onnxruntime raises its own exception hierarchy
        LOGGER.error("ONNX Runtime could not run the exported model: %s", exc)
        return False

    difference = float(np.max(np.abs(outputs[0] - reference)))
    if difference <= tolerance:
        LOGGER.info("Verification passed: max absolute difference %.2e", difference)
        return True
    LOGGER.error(
        "Verification failed: max absolute difference %.2e exceeds tolerance %.2e",
        difference, tolerance,
    )
    return False


def main(argv: list | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    setup_logging()

    try:
        config = load_config(args.config)
    except ValueError as exc:
        LOGGER.error("Invalid configuration: %s", exc)
        return 2

    # Export on CPU: the traced graph is device independent and this avoids
    # CUDA-specific kernels leaking into the graph.
    device = select_device("cpu")
    checkpoint_path = resolve_path(
        args.checkpoint or os.getenv("CHECKPOINT_PATH") or "checkpoints/best.pt"
    )

    try:
        model, class_names, image_size = load_predictor(checkpoint_path, config, device)
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        LOGGER.error("Could not load checkpoint: %s", exc)
        return 1

    output_path = resolve_path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.error("Could not create output directory: %s", exc)
        return 1

    sample = torch.randn(max(1, args.batch_size), 3, image_size, image_size)
    with torch.no_grad():
        reference = model(sample).cpu().numpy()

    try:
        torch.onnx.export(
            model,
            sample,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            external_data=False,  # single self-contained .onnx file
        )
    except Exception as exc:  # torch.onnx raises several unrelated types
        LOGGER.error("ONNX export failed: %s", exc)
        return 1

    size_mb = output_path.stat().st_size / (1024 * 1024)
    LOGGER.info(
        "Exported %s (%.1f MB, opset %d, classes: %s)",
        output_path, size_mb, args.opset, ", ".join(class_names),
    )
    LOGGER.info(
        "Input 'input': (batch_size, 3, %d, %d), normalised with ImageNet statistics. "
        "Output 'logits': (batch_size, %d) - apply softmax downstream.",
        image_size, image_size, len(class_names),
    )

    if not args.no_verify:
        verify_with_onnxruntime(output_path, sample, reference)
    return 0


if __name__ == "__main__":
    sys.exit(main())
