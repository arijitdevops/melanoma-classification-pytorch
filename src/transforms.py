"""Image transform pipelines built on ``torchvision.transforms.v2``.

Training uses light geometric and photometric augmentation appropriate for
dermoscopic images (lesions have no canonical orientation, so both horizontal
and vertical flips are valid). Evaluation uses a deterministic resize plus
centre crop so that validation, test, Grad-CAM and the web demo all see exactly
the same preprocessing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torchvision.transforms import v2

LOGGER = logging.getLogger(__name__)

#: ImageNet channel statistics, matching the pretrained torchvision weights.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> v2.Compose:
    """Build the augmentation pipeline used for the training split.

    Args:
        image_size: Side length of the square network input.
        mean: Per-channel normalisation means.
        std: Per-channel normalisation standard deviations.

    Returns:
        A composed transform mapping a PIL image to a normalised float tensor
        of shape ``(3, image_size, image_size)``.
    """
    resize_to = int(round(image_size * 1.14))
    return v2.Compose(
        [
            v2.Resize(resize_to, antialias=True),
            v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=20),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_eval_transform(
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> v2.Compose:
    """Build the deterministic pipeline used for validation, test and inference.

    Args:
        image_size: Side length of the square network input.
        mean: Per-channel normalisation means.
        std: Per-channel normalisation standard deviations.

    Returns:
        A composed transform with no random components.
    """
    resize_to = int(round(image_size * 1.14))
    return v2.Compose(
        [
            v2.Resize(resize_to, antialias=True),
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def denormalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Invert :class:`~torchvision.transforms.v2.Normalize` for visualisation.

    Args:
        tensor: Normalised image tensor of shape ``(C, H, W)`` or ``(N, C, H, W)``.
        mean: The means used during normalisation.
        std: The standard deviations used during normalisation.

    Returns:
        A tensor of the same shape with values clamped to ``[0, 1]``.
    """
    mean_t = torch.tensor(list(mean), dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(list(std), dtype=tensor.dtype, device=tensor.device)
    shape = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)
    return (tensor * std_t.view(*shape) + mean_t.view(*shape)).clamp(0.0, 1.0)
