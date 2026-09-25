"""Tests for backbone construction and head replacement.

These tests always pass ``pretrained=False`` so no weights are downloaded.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model import (
    available_models,
    build_model,
    find_last_conv_layer,
    infer_head_out_features,
    set_backbone_trainable,
)


@pytest.mark.parametrize("name", ["resnet18", "resnet50", "efficientnet_b0", "densenet121"])
def test_build_model_head_out_features(name):
    """Every architecture gets a head with exactly ``num_classes`` outputs."""
    model = build_model(name=name, num_classes=2, pretrained=False)
    assert infer_head_out_features(model) == 2


@pytest.mark.parametrize("num_classes", [2, 5])
def test_build_model_respects_num_classes(num_classes):
    """The head width follows ``num_classes`` rather than a hard-coded value."""
    model = build_model(name="resnet18", num_classes=num_classes, pretrained=False)
    assert infer_head_out_features(model) == num_classes


def test_forward_pass_shape():
    """A forward pass returns logits of shape (batch, num_classes)."""
    model = build_model(name="resnet18", num_classes=2, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 2)


def test_available_models_is_sorted_and_complete():
    """The advertised model list matches what ``build_model`` accepts."""
    names = available_models()
    assert names == sorted(names)
    assert {"resnet18", "resnet50", "efficientnet_b0", "densenet121"}.issubset(names)


def test_unknown_model_raises():
    """An unsupported backbone name fails loudly."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model(name="not_a_real_backbone", pretrained=False)


def test_too_few_classes_raises():
    """A single-class head is rejected: this is a binary classifier."""
    with pytest.raises(ValueError, match="num_classes"):
        build_model(name="resnet18", num_classes=1, pretrained=False)


def test_freeze_backbone_leaves_only_the_head_trainable():
    """Freezing keeps head parameters trainable and freezes everything else."""
    model = build_model(
        name="resnet18", num_classes=2, pretrained=False, freeze_backbone=True
    )
    head_params = list(model.fc.parameters())
    assert all(p.requires_grad for p in head_params)

    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    assert backbone and not any(p.requires_grad for p in backbone)

    set_backbone_trainable(model, trainable=True)
    assert all(p.requires_grad for p in model.parameters())


def test_find_last_conv_layer_returns_a_conv():
    """Grad-CAM's target-layer lookup finds a Conv2d module."""
    model = build_model(name="resnet18", num_classes=2, pretrained=False)
    layer = find_last_conv_layer(model)
    assert isinstance(layer, nn.Conv2d)


def test_find_last_conv_layer_rejects_conv_free_models():
    """A model with no convolution cannot be explained with Grad-CAM."""
    with pytest.raises(ValueError):
        find_last_conv_layer(nn.Sequential(nn.Flatten(), nn.Linear(4, 2)))


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0", "densenet121"])
def test_gradcam_produces_a_heatmap_for_every_family(name):
    from src.gradcam import GradCAM
    from src.model import find_gradcam_target_layer

    torch.manual_seed(0)
    model = build_model(name, num_classes=2, pretrained=False)
    with GradCAM(model, find_gradcam_target_layer(model)) as engine:
        cam, target, probabilities = engine(torch.randn(1, 3, 64, 64), class_index=1)
    assert cam.shape == (64, 64)
    assert target == 1
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0
    assert probabilities.shape == (1, 2)
