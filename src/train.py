"""Training CLI: transfer learning with AMP, class balancing and early stopping.

Examples::

    python -m src.train --config configs/resnet50.yaml
    python -m src.train --model efficientnet_b0 --epochs 20 --balance sampler
    python -m src.train --dry-run            # two batches, verifies the pipeline
    python -m src.train --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from .config import Config, describe, load_config, resolve_path
from .dataset import build_dataloaders, compute_class_weights
from .engine import build_criterion, train_one_epoch, validate
from .model import available_models, build_model
from .utils import (
    format_seconds,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
    setup_logging,
)

LOGGER = logging.getLogger("src.train")


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line interface for training."""
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Train a benign/malignant skin-lesion classifier (research demo only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--data-dir", type=str, default=None, help="Override the dataset root.")
    parser.add_argument(
        "--model", type=str, default=None, choices=available_models(), help="Backbone architecture."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader workers (use 0 on Windows if needed)."
    )
    parser.add_argument(
        "--balance",
        type=str,
        default=None,
        choices=["sampler", "weights", "none"],
        help="Class-imbalance strategy.",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", help="Train only the classifier head."
    )
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Start from random weights (offline friendly)."
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument("--device", type=str, default=None, help="auto, cuda, mps or cpu.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from.")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Directory for best.pt and last.pt."
    )
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory.")
    parser.add_argument(
        "--patience", type=int, default=None, help="Early-stopping patience in epochs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only N training images (stratified) for a quick smoke run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run two train and two validation batches, then exit without checkpointing.",
    )
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Fold CLI flags into ``config``; flags win over YAML and environment."""
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.model:
        config.model.name = args.model
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.weight_decay is not None:
        config.train.weight_decay = args.weight_decay
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.balance:
        config.train.balance = args.balance
    if args.freeze_backbone:
        config.model.freeze_backbone = True
    if args.no_pretrained:
        config.model.pretrained = False
    if args.no_amp:
        config.train.amp = False
    if args.seed is not None:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    if args.checkpoint_dir:
        config.train.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.train.log_dir = args.log_dir
    if args.patience is not None:
        config.train.early_stopping_patience = args.patience
    return config


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> LambdaLR:
    """Linear warmup followed by cosine annealing, applied per epoch.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        epochs: Total number of epochs.
        warmup_epochs: Epochs spent ramping from ~0 to ``base_lr``.
        base_lr: Peak learning rate.
        min_lr: Floor the cosine decays towards.

    Returns:
        A :class:`~torch.optim.lr_scheduler.LambdaLR` stepped once per epoch.
    """
    warmup = max(0, min(warmup_epochs, max(0, epochs - 1)))
    floor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda)


def _make_writer(log_dir: Path):
    """Create a TensorBoard writer, returning ``None`` if TensorBoard is absent."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        LOGGER.warning("TensorBoard is not installed; skipping scalar logging")
        return None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))
    except OSError as exc:
        LOGGER.warning("Could not open TensorBoard log dir %s: %s", log_dir, exc)
        return None


def main(argv: list | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    setup_logging()

    try:
        config = apply_overrides(load_config(args.config), args)
    except ValueError as exc:
        LOGGER.error("Invalid configuration: %s", exc)
        return 2

    set_seed(config.seed)
    device = select_device(config.device)
    LOGGER.info("Effective configuration:\n%s", describe(config))

    try:
        bundle = build_dataloaders(config, balance=config.train.balance, limit=args.limit)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("Could not build dataloaders: %s", exc)
        return 1

    try:
        model = build_model(
            name=config.model.name,
            num_classes=len(bundle.class_names),
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone,
            dropout=config.model.dropout,
        ).to(device)
    except (ValueError, RuntimeError) as exc:
        LOGGER.error("Could not build model: %s", exc)
        return 1

    class_weights = None
    if config.train.balance == "weights":
        class_weights = compute_class_weights(bundle.train_targets, len(bundle.class_names))
        LOGGER.info("Class weights: %s", class_weights.tolist())
    criterion = build_criterion(class_weights, config.train.label_smoothing, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        config.train.epochs,
        config.train.warmup_epochs,
        config.train.lr,
        config.train.min_lr,
    )
    use_amp = bool(config.train.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if config.train.amp and not use_amp:
        LOGGER.info("AMP requested but only supported on CUDA; training in full precision")

    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        try:
            checkpoint = load_checkpoint(resolve_path(args.resume), map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if "scaler_state" in checkpoint and use_amp:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_metric = float(checkpoint.get("best_metric", 0.0))
            LOGGER.info("Resumed from %s at epoch %d", args.resume, start_epoch)
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            LOGGER.error("Could not resume: %s", exc)
            return 1

    if args.dry_run:
        LOGGER.info("Dry run: two training batches and two validation batches")
        train_result = train_one_epoch(
            model, bundle.train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip=config.train.grad_clip, epoch=0, max_batches=2, log_every=1,
        )
        val_result = validate(
            model, bundle.val_loader, criterion, device, amp=use_amp, max_batches=2
        )
        LOGGER.info("Dry-run train: %s", train_result.as_dict())
        LOGGER.info("Dry-run val:   %s", val_result.as_dict())
        LOGGER.info("Pipeline OK. No checkpoint was written.")
        return 0

    checkpoint_dir = resolve_path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(resolve_path(config.train.log_dir) / config.model.name)

    metadata = {
        "config": config.to_dict(),
        "class_names": bundle.class_names,
        "model_name": config.model.name,
        "image_size": config.data.image_size,
    }
    epochs_without_improvement = 0
    total_seconds = 0.0

    try:
        for epoch in range(start_epoch, config.train.epochs):
            train_result = train_one_epoch(
                model, bundle.train_loader, criterion, optimizer, device,
                scaler=scaler, grad_clip=config.train.grad_clip, epoch=epoch,
            )
            val_result = validate(model, bundle.val_loader, criterion, device, amp=use_amp)
            current_lr = optimizer.param_groups[0]["lr"]  # LR used for this epoch
            scheduler.step()
            total_seconds += train_result.seconds + val_result.seconds

            LOGGER.info(
                "epoch %d/%d | lr %.2e | train loss %.4f acc %.4f f1 %.4f | "
                "val loss %.4f acc %.4f f1 %.4f | elapsed %s",
                epoch + 1, config.train.epochs, current_lr,
                train_result.loss, train_result.accuracy, train_result.macro_f1,
                val_result.loss, val_result.accuracy, val_result.macro_f1,
                format_seconds(total_seconds),
            )

            if writer is not None:
                writer.add_scalar("loss/train", train_result.loss, epoch)
                writer.add_scalar("loss/val", val_result.loss, epoch)
                writer.add_scalar("accuracy/train", train_result.accuracy, epoch)
                writer.add_scalar("accuracy/val", val_result.accuracy, epoch)
                writer.add_scalar("macro_f1/train", train_result.macro_f1, epoch)
                writer.add_scalar("macro_f1/val", val_result.macro_f1, epoch)
                writer.add_scalar("lr", current_lr, epoch)

            improved = val_result.macro_f1 > best_metric
            if improved:
                best_metric = val_result.macro_f1
            save_checkpoint(
                checkpoint_dir / "last.pt", model, optimizer, scheduler,
                scaler if use_amp else None, epoch, best_metric, metadata,
            )

            if improved:
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt", model, optimizer, scheduler,
                    scaler if use_amp else None, epoch, best_metric,
                    {**metadata, "val_metrics": val_result.as_dict()},
                )
                LOGGER.info("New best val macro-F1: %.4f", best_metric)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.train.early_stopping_patience:
                    LOGGER.info(
                        "Early stopping after %d epochs without improvement",
                        epochs_without_improvement,
                    )
                    break
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user; last.pt holds the most recent epoch")
    finally:
        if writer is not None:
            writer.close()

    LOGGER.info(
        "Training finished. Best val macro-F1 %.4f. Checkpoints in %s",
        best_metric, checkpoint_dir,
    )
    LOGGER.info("Next: python -m src.evaluate --checkpoint %s", checkpoint_dir / "best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
