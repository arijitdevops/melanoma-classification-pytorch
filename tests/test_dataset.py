"""Tests for the transform pipelines and the stratified split."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dataset import (
    build_weighted_sampler,
    class_distribution,
    compute_class_weights,
    stratified_split,
)
from src.transforms import build_eval_transform, build_train_transform, denormalize


def test_train_transform_output_shape_and_dtype(sample_image):
    """The training pipeline yields a normalised float tensor of the requested size."""
    tensor = build_train_transform(image_size=224)(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    # ImageNet normalisation pushes values outside [0, 1] but keeps them bounded.
    assert torch.isfinite(tensor).all()
    assert tensor.abs().max() < 10.0


def test_eval_transform_is_deterministic(sample_image):
    """Two passes of the eval pipeline over the same image are identical."""
    transform = build_eval_transform(image_size=160)
    first = transform(sample_image)
    second = transform(sample_image)
    assert first.shape == (3, 160, 160)
    assert torch.allclose(first, second)


def test_denormalize_roundtrip(sample_image):
    """Denormalising the eval output returns values inside [0, 1]."""
    tensor = build_eval_transform(image_size=64)(sample_image)
    restored = denormalize(tensor)
    assert restored.shape == tensor.shape
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_stratified_split_preserves_class_ratio(imbalanced_targets):
    """Both subsets keep the 70/30 class ratio within one sample of the ideal."""
    train_idx, val_idx = stratified_split(imbalanced_targets, val_split=0.2, seed=42)

    assert len(train_idx) + len(val_idx) == len(imbalanced_targets)
    assert set(train_idx).isdisjoint(val_idx)

    val_labels = [imbalanced_targets[i] for i in val_idx]
    train_labels = [imbalanced_targets[i] for i in train_idx]
    assert val_labels.count(0) == 14  # round(70 * 0.2)
    assert val_labels.count(1) == 6  # round(30 * 0.2)
    assert train_labels.count(0) == 56
    assert train_labels.count(1) == 24


def test_stratified_split_is_deterministic(imbalanced_targets):
    """The same seed reproduces the split; a different seed changes it."""
    first = stratified_split(imbalanced_targets, val_split=0.15, seed=7)
    second = stratified_split(imbalanced_targets, val_split=0.15, seed=7)
    different = stratified_split(imbalanced_targets, val_split=0.15, seed=8)

    assert first == second
    assert first[1] != different[1]


@pytest.mark.parametrize("val_split", [0.0, 1.0, -0.1, 1.5])
def test_stratified_split_rejects_invalid_fraction(imbalanced_targets, val_split):
    """A split fraction outside (0, 1) is an error, not a silent clamp."""
    with pytest.raises(ValueError):
        stratified_split(imbalanced_targets, val_split=val_split, seed=0)


def test_class_distribution_counts(imbalanced_targets):
    """The distribution helper maps class names to counts."""
    assert class_distribution(imbalanced_targets, ["Benign", "Malignant"]) == {
        "Benign": 70,
        "Malignant": 30,
    }


def test_class_weights_favour_the_minority_class(imbalanced_targets):
    """Inverse-frequency weights give the rarer class the larger weight."""
    weights = compute_class_weights(imbalanced_targets, num_classes=2)
    assert weights.shape == (2,)
    assert float(weights[1]) > float(weights[0])
    assert np.isclose(float(weights.mean()), 1.0, atol=1e-6)


def test_weighted_sampler_length_matches_dataset(imbalanced_targets):
    """The sampler draws as many samples as the dataset contains."""
    sampler = build_weighted_sampler(imbalanced_targets, num_classes=2)
    assert sampler.num_samples == len(imbalanced_targets)
    assert len(list(iter(sampler))) == len(imbalanced_targets)


def test_limit_indices_is_stratified_and_reproducible():
    from src.dataset import limit_indices

    targets = [0] * 80 + [1] * 20
    kept = limit_indices(range(100), targets, 10, seed=1)
    assert len(kept) == 10
    assert sum(targets[i] for i in kept) == 2
    assert kept == limit_indices(range(100), targets, 10, seed=1)
    assert limit_indices(range(100), targets, None) == list(range(100))
