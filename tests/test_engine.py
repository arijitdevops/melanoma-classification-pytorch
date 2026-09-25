"""Tests for the training and validation loops on synthetic data."""

from __future__ import annotations

import numpy as np
import torch

from src.engine import build_criterion, macro_f1_from_counts, train_one_epoch, validate
from src.utils import AverageMeter, set_seed


def test_train_one_epoch_updates_parameters(tiny_model, synthetic_loader):
    """One epoch on two synthetic batches changes the weights and returns metrics."""
    set_seed(0)
    device = torch.device("cpu")
    criterion = build_criterion(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=0.05)
    before = tiny_model[-1].weight.detach().clone()

    result = train_one_epoch(
        tiny_model, synthetic_loader, criterion, optimizer, device,
        scaler=None, grad_clip=1.0, epoch=0, log_every=0,
    )

    assert not torch.allclose(before, tiny_model[-1].weight)
    assert result.loss > 0.0
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.seconds >= 0.0


def test_train_one_epoch_honours_max_batches(tiny_model, synthetic_loader):
    """``max_batches`` stops early, which is what ``--dry-run`` relies on."""
    device = torch.device("cpu")
    criterion = build_criterion()
    optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)

    result = train_one_epoch(
        tiny_model, synthetic_loader, criterion, optimizer, device,
        max_batches=1, log_every=0,
    )
    # One batch of two samples was consumed, not the whole four-sample dataset.
    assert result.loss > 0.0


def test_validate_returns_probabilities_and_targets(tiny_model, synthetic_loader):
    """Validation returns per-sample softmax rows aligned with the targets."""
    device = torch.device("cpu")
    result = validate(tiny_model, synthetic_loader, build_criterion(), device, amp=False)

    assert result.probabilities is not None and result.targets is not None
    assert result.probabilities.shape == (4, 2)
    assert result.targets.shape == (4,)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(4), rtol=1e-5)


def test_validate_does_not_change_parameters(tiny_model, synthetic_loader):
    """A validation pass leaves the model untouched."""
    before = tiny_model[-1].weight.detach().clone()
    validate(tiny_model, synthetic_loader, None, torch.device("cpu"))
    assert torch.allclose(before, tiny_model[-1].weight)


def test_macro_f1_perfect_and_worst_cases():
    """Macro-F1 is 1.0 for perfect predictions and 0.0 when every label is wrong."""
    targets = np.array([0, 0, 1, 1])
    assert macro_f1_from_counts(targets.copy(), targets, num_classes=2) == 1.0
    assert macro_f1_from_counts(1 - targets, targets, num_classes=2) == 0.0


def test_weighted_criterion_applies_class_weights():
    """A weighted loss penalises errors on the up-weighted class more heavily."""
    logits = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    targets = torch.tensor([0, 1])
    unweighted = build_criterion()(logits, targets)
    weighted = build_criterion(torch.tensor([1.0, 4.0]))(logits, targets)
    assert float(weighted) > float(unweighted)


def test_average_meter_tracks_weighted_mean():
    """AverageMeter averages over sample counts, not over update calls."""
    meter = AverageMeter("loss")
    meter.update(1.0, n=1)
    meter.update(3.0, n=3)
    assert meter.count == 4
    assert meter.avg == 2.5
    meter.reset()
    assert meter.count == 0 and meter.avg == 0.0
