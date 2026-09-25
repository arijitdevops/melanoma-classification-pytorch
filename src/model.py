"""Backbone construction with torchvision's modern ``weights=`` API.

Four architectures are supported. Each one exposes its classifier head in a
different place, so the head replacement is handled per family rather than by
guessing attribute names.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import nn
from torchvision import models

LOGGER = logging.getLogger(__name__)

#: Supported backbone name -> (constructor, default weights enum).
SUPPORTED_MODELS: dict[str, tuple] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
}


def available_models() -> list[str]:
    """Return the sorted list of supported backbone names."""
    return sorted(SUPPORTED_MODELS)


def _replace_head(model: nn.Module, name: str, num_classes: int, dropout: float) -> nn.Module:
    """Swap the pretrained ImageNet classifier for a ``num_classes`` head."""
    if name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif name.startswith("efficientnet"):
        # torchvision's EfficientNet classifier is Sequential(Dropout, Linear).
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes)
        )
    elif name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_features, num_classes)
        )
    else:  # pragma: no cover - guarded by build_model's validation
        raise ValueError(f"No head-replacement rule for architecture '{name}'")
    return model


def get_classifier_head(model: nn.Module) -> nn.Module:
    """Return the classifier submodule of a model built by :func:`build_model`."""
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    raise AttributeError("Model exposes neither 'fc' nor 'classifier'")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze every parameter except the classifier head.

    Args:
        model: Model produced by :func:`build_model`.
        trainable: ``False`` freezes the feature extractor (linear probing),
            ``True`` restores full fine-tuning.
    """
    head = get_classifier_head(model)
    head_params = {id(p) for p in head.parameters()}
    for param in model.parameters():
        if id(param) not in head_params:
            param.requires_grad = trainable
    LOGGER.info("Backbone parameters %s", "unfrozen" if trainable else "frozen")


def build_model(
    name: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a torchvision backbone with a fresh classification head.

    Args:
        name: One of :func:`available_models`.
        num_classes: Output dimension of the new head.
        pretrained: Load ImageNet weights. Requires network access the first
            time; set ``False`` for fully offline runs.
        freeze_backbone: Train only the head (linear probing).
        dropout: Dropout probability inserted before the final linear layer.

    Returns:
        The assembled :class:`torch.nn.Module`.

    Raises:
        ValueError: If ``name`` is not supported or ``num_classes`` < 2.
    """
    key = name.strip().lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{name}'. Choose one of: {', '.join(available_models())}"
        )
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    constructor: Callable[..., nn.Module]
    constructor, default_weights = SUPPORTED_MODELS[key]
    weights: object | None = default_weights if pretrained else None

    try:
        model = constructor(weights=weights)
    except Exception as exc:
        # Most commonly a download failure for the pretrained weights.
        raise RuntimeError(
            f"Could not instantiate '{key}' (pretrained={pretrained}): {exc}. "
            "If you are offline, pass --no-pretrained or set model.pretrained: false."
        ) from exc

    model = _replace_head(model, key, num_classes, dropout)
    if freeze_backbone:
        set_backbone_trainable(model, trainable=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info(
        "Built %s: %d classes, %s weights, %s/%s trainable parameters",
        key,
        num_classes,
        "ImageNet" if pretrained else "random",
        f"{trainable:,}",
        f"{total:,}",
    )
    return model


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the deepest ``nn.Conv2d``-producing block, for Grad-CAM hooks.

    For the supported architectures this is the last convolutional module in
    ``model.modules()`` order, which coincides with the final feature block of
    ResNet, EfficientNet and DenseNet.

    Raises:
        ValueError: If the model contains no convolutional layer.
    """
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Model contains no nn.Conv2d layer; Grad-CAM is not applicable")
    return last_conv


def find_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the layer Grad-CAM should hook for a model built by :func:`build_model`.

    The standard choice is the output of the last convolutional *stage*, after
    normalisation and the residual/dense connections, not a single raw conv:

    * ResNet: ``model.layer4``
    * EfficientNet / DenseNet: ``model.features`` (ends with the final conv block
      or ``norm5``)

    Any other model falls back to :func:`find_last_conv_layer`.
    """
    if isinstance(getattr(model, "layer4", None), nn.Module):
        return model.layer4
    if isinstance(getattr(model, "features", None), nn.Module):
        return model.features
    return find_last_conv_layer(model)


@torch.no_grad()
def infer_head_out_features(model: nn.Module) -> int:
    """Return the ``out_features`` of the final linear layer of the head."""
    head = get_classifier_head(model)
    linear_layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("Classifier head contains no nn.Linear layer")
    return int(linear_layers[-1].out_features)
