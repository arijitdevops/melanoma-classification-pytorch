"""Training and validation loops with AMP, gradient clipping and running metrics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import AverageMeter

LOGGER = logging.getLogger(__name__)


@dataclass
class EpochResult:
    """Metrics produced by one pass over a dataloader."""

    loss: float
    accuracy: float
    macro_f1: float
    seconds: float
    probabilities: np.ndarray | None = field(default=None, repr=False)
    targets: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, float]:
        """Return the scalar metrics only, ready for logging."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "seconds": self.seconds,
        }


def macro_f1_from_counts(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Compute macro-averaged F1 without a scikit-learn dependency.

    Args:
        preds: Predicted class indices, shape ``(N,)``.
        targets: Ground-truth class indices, shape ``(N,)``.
        num_classes: Number of classes.

    Returns:
        The unweighted mean of the per-class F1 scores.
    """
    if preds.size == 0:
        return 0.0
    scores: list[float] = []
    for cls in range(num_classes):
        tp = float(np.sum((preds == cls) & (targets == cls)))
        fp = float(np.sum((preds == cls) & (targets != cls)))
        fn = float(np.sum((preds != cls) & (targets == cls)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return float(np.mean(scores))


def _autocast_context(device: torch.device, enabled: bool):
    """Return an autocast context manager appropriate for ``device``."""
    # AMP is meaningful on CUDA; on CPU/MPS we disable it to avoid slowdowns.
    use_amp = enabled and device.type == "cuda"
    return torch.amp.autocast(device_type=device.type, enabled=use_amp)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float = 0.0,
    epoch: int = 0,
    max_batches: int | None = None,
    log_every: int = 50,
) -> EpochResult:
    """Run one training epoch.

    Args:
        model: Model in train mode after this call.
        loader: Training dataloader.
        criterion: Loss function taking ``(logits, targets)``.
        optimizer: Optimizer stepped once per batch.
        device: Device the batches are moved to.
        scaler: AMP gradient scaler. Pass ``None`` to train in full precision.
        grad_clip: Max global gradient norm; ``0`` disables clipping.
        epoch: Epoch index, used only for log messages.
        max_batches: Stop after this many batches (used by ``--dry-run``).
        log_every: Emit a progress log line every N batches.

    Returns:
        An :class:`EpochResult` with loss, accuracy and macro-F1.
    """
    model.train()
    loss_meter = AverageMeter("loss")
    started = time.perf_counter()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    num_classes = 2

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        num_classes = max(num_classes, int(targets.max().item()) + 1)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, scaler is not None and scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        all_preds.append(logits.detach().argmax(dim=1).cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        if log_every and batch_index % log_every == 0:
            LOGGER.info(
                "epoch %d | batch %d/%d | loss %.4f (avg %.4f)",
                epoch,
                batch_index,
                len(loader),
                loss_meter.val,
                loss_meter.avg,
            )

    preds = np.concatenate(all_preds) if all_preds else np.empty(0, dtype=np.int64)
    truths = np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.int64)
    accuracy = float((preds == truths).mean()) if preds.size else 0.0
    return EpochResult(
        loss=loss_meter.avg,
        accuracy=accuracy,
        macro_f1=macro_f1_from_counts(preds, truths, num_classes),
        seconds=time.perf_counter() - started,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    amp: bool = False,
    max_batches: int | None = None,
) -> EpochResult:
    """Evaluate ``model`` over ``loader`` without updating any parameters.

    Args:
        model: Model placed in eval mode.
        loader: Validation or test dataloader.
        criterion: Optional loss; when ``None`` the reported loss is ``0``.
        device: Device the batches are moved to.
        amp: Enable autocast on CUDA for a faster pass.
        max_batches: Stop after this many batches (used by ``--dry-run``).

    Returns:
        An :class:`EpochResult` that also carries per-sample softmax
        probabilities and targets, for downstream threshold tuning.
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    started = time.perf_counter()
    prob_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_context(device, amp):
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, targets)
                loss_meter.update(loss.item(), images.size(0))

        probabilities = torch.softmax(logits.float(), dim=1)
        prob_chunks.append(probabilities.cpu().numpy())
        target_chunks.append(targets.cpu().numpy())

    if prob_chunks:
        probabilities_np = np.concatenate(prob_chunks, axis=0)
        targets_np = np.concatenate(target_chunks, axis=0)
        preds = probabilities_np.argmax(axis=1)
        accuracy = float((preds == targets_np).mean())
        macro_f1 = macro_f1_from_counts(preds, targets_np, probabilities_np.shape[1])
    else:
        probabilities_np = np.empty((0, 2), dtype=np.float32)
        targets_np = np.empty(0, dtype=np.int64)
        accuracy = 0.0
        macro_f1 = 0.0

    return EpochResult(
        loss=loss_meter.avg,
        accuracy=accuracy,
        macro_f1=macro_f1,
        seconds=time.perf_counter() - started,
        probabilities=probabilities_np,
        targets=targets_np,
    )


def build_criterion(
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> nn.Module:
    """Create a cross-entropy loss, optionally weighted and label-smoothed."""
    weight = class_weights.to(device) if (class_weights is not None and device) else class_weights
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def predict_logits(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and return ``(logits, probabilities)`` on CPU."""
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device))
        probabilities = torch.softmax(logits.float(), dim=1)
    return logits.detach().cpu(), probabilities.detach().cpu()
