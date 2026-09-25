"""Grad-CAM saliency maps implemented directly with forward/backward hooks.

The class-activation map is the ReLU of a weighted sum of the target layer's
feature maps, where each channel weight is the spatial average of the gradient
of the target logit with respect to that channel (Selvaraju et al., 2017).

No third-party Grad-CAM package is used; the hook bookkeeping lives in
:class:`GradCAM`.

Example::

    python -m src.gradcam --input samples/malignant_01.jpg --output docs/images/cam.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch import nn

from .config import POSITIVE_CLASS_INDEX, load_config, resolve_path
from .model import find_gradcam_target_layer
from .predict import collect_images, load_predictor
from .transforms import build_eval_transform
from .utils import select_device, setup_logging

LOGGER = logging.getLogger("src.gradcam")


class GradCAM:
    """Compute Grad-CAM heatmaps for a single target layer.

    The hooks are registered on construction and removed by :meth:`close`, or
    automatically when the instance is used as a context manager.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Register the forward and full-backward hooks on ``target_layer``."""
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        # A tensor hook on the layer's output is used for the gradient instead of
        # register_full_backward_hook: several torchvision models (DenseNet,
        # EfficientNet) apply an in-place activation right after the hooked
        # block, which full backward hooks do not allow.
        self._handles = [target_layer.register_forward_hook(self._save_activation)]

    def _save_activation(self, _module: nn.Module, _inputs, output: torch.Tensor) -> None:
        """Forward hook: keep the feature maps and hook their gradient."""
        self._activations = output.detach().clone()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        """Tensor hook: keep the gradient of the target logit w.r.t. the feature maps."""
        self._gradients = grad.detach()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Remove the registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(
        self, inputs: torch.Tensor, class_index: int | None = None
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Produce a heatmap for one image.

        Args:
            inputs: Input tensor of shape ``(1, C, H, W)``.
            class_index: Logit to explain. Defaults to the predicted class.

        Returns:
            ``(cam, class_index, probabilities)`` where ``cam`` has shape
            ``(H, W)`` with values normalised to ``[0, 1]``.

        Raises:
            ValueError: If ``inputs`` is not a single-image batch.
            RuntimeError: If the hooks captured no activations or gradients.
        """
        if inputs.ndim != 4 or inputs.size(0) != 1:
            raise ValueError(f"Expected input of shape (1, C, H, W), got {tuple(inputs.shape)}")

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None

        # Grad-CAM needs gradients, so no torch.no_grad() here.
        logits = self.model(inputs)
        probabilities = torch.softmax(logits.float(), dim=1).detach()
        target = int(logits.argmax(dim=1).item()) if class_index is None else int(class_index)
        logits[0, target].backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks captured nothing. The chosen target layer may not "
                "participate in the forward pass."
            )

        # Channel weights = global-average-pooled gradients.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        if was_training:
            self.model.train()
        return cam.detach().cpu(), target, probabilities.cpu()


def apply_colormap(cam: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Map a ``[0, 1]`` heatmap to an RGB uint8 array using matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    mapper = colormaps[colormap]
    coloured = mapper(np.clip(cam, 0.0, 1.0))[..., :3]
    return (coloured * 255).astype(np.uint8)


def overlay_heatmap(
    original: Image.Image, cam: np.ndarray, alpha: float = 0.45, colormap: str = "jet"
) -> Image.Image:
    """Blend a heatmap over the original image.

    Args:
        original: The source image (any size; it is used as the output size).
        cam: Heatmap with values in ``[0, 1]``.
        alpha: Heatmap opacity in ``[0, 1]``.
        colormap: Matplotlib colormap name.

    Returns:
        The blended RGB image.
    """
    base = original.convert("RGB")
    heat = Image.fromarray(apply_colormap(cam, colormap)).resize(base.size, Image.BILINEAR)
    return Image.blend(base, heat, alpha=float(np.clip(alpha, 0.0, 1.0)))


def generate_overlay(
    model: nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
    class_index: int | None = None,
    alpha: float = 0.45,
) -> tuple[Image.Image, int, np.ndarray]:
    """Run Grad-CAM on a PIL image and return ``(overlay, class_index, probabilities)``."""
    transform = build_eval_transform(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    target_layer = find_gradcam_target_layer(model)
    with GradCAM(model, target_layer) as cam_engine:
        cam, target, probabilities = cam_engine(tensor, class_index)
    overlay = overlay_heatmap(image, cam.numpy(), alpha=alpha)
    return overlay, target, probabilities.numpy()[0]


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line interface for Grad-CAM."""
    parser = argparse.ArgumentParser(
        prog="python -m src.gradcam",
        description="Save Grad-CAM overlays for one image or a folder of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Image file or directory.")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/figures",
        help="Output file (single image) or directory (folder input).",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load.")
    parser.add_argument("--device", type=str, default=None, help="auto, cuda, mps or cpu.")
    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        help="Explain this class index instead of the predicted one (1 = Malignant).",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity.")
    return parser


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

    input_path = resolve_path(args.input)
    try:
        paths: list[Path] = collect_images(input_path)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    output_arg = resolve_path(args.output)
    single_output = len(paths) == 1 and output_arg.suffix.lower() in {".png", ".jpg", ".jpeg"}
    output_dir = output_arg.parent if single_output else output_arg
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.error("Could not create output directory %s: %s", output_dir, exc)
        return 1

    written = 0
    for path in paths:
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            LOGGER.warning("Skipping %s: %s", path, exc)
            continue

        try:
            overlay, target, probabilities = generate_overlay(
                model, image, image_size, device,
                class_index=args.class_index, alpha=args.alpha,
            )
        except (ValueError, RuntimeError) as exc:
            LOGGER.error("Grad-CAM failed for %s: %s", path, exc)
            continue

        destination = output_arg if single_output else output_dir / f"{path.stem}_gradcam.png"
        try:
            overlay.save(destination)
        except OSError as exc:
            LOGGER.error("Could not save %s: %s", destination, exc)
            continue

        written += 1
        LOGGER.info(
            "%s -> %s | explained class '%s' | P(malignant) %.3f",
            path.name, destination, class_names[target],
            float(probabilities[POSITIVE_CLASS_INDEX]),
        )

    if written == 0:
        LOGGER.error("No overlays were produced")
        return 1
    LOGGER.info("Wrote %d overlay(s)", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
