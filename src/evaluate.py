"""Held-out test-set evaluation with threshold analysis and figures.

Writes ``reports/metrics.json`` plus confusion-matrix, ROC and precision-recall
figures under ``reports/figures/``.

Two operating points are reported in addition to the default 0.5 threshold:

* ``best_f1`` - the threshold maximising F1 on the Malignant class.
* ``target_sensitivity`` - the highest threshold whose recall on Malignant is
  still at or above ``eval.target_sensitivity`` (0.95 by default).

For a screening task, a missed melanoma costs far more than a false alarm, so
the high-sensitivity point is the one worth quoting; the F1-optimal point is
included for comparison.

Example::

    python -m src.evaluate --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .config import POSITIVE_CLASS_INDEX, Config, load_config, resolve_path  # noqa: E402
from .dataset import build_test_loader  # noqa: E402
from .engine import validate  # noqa: E402
from .model import build_model  # noqa: E402
from .utils import (  # noqa: E402
    load_checkpoint,
    select_device,
    set_seed,
    setup_logging,
    write_json,
)

LOGGER = logging.getLogger("src.evaluate")


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line interface for evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate a trained checkpoint on the held-out test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to evaluate (default: CHECKPOINT_PATH)."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Override the dataset root.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="auto, cuda, mps or cpu.")
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=None,
        help="Minimum recall on the Malignant class for the high-sensitivity operating point.",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Destination JSON file for the metrics."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate on only N test images (smoke runs)."
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    return parser


def threshold_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute binary metrics for the positive class at a given threshold."""
    y_pred = (y_score >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": specificity,
        "f1": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def find_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the threshold maximising F1 on the positive class."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more element than thresholds.
    denominator = precision[:-1] + recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2 * precision[:-1] * recall[:-1] / denominator, 0.0)
    if f1.size == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def find_threshold_at_sensitivity(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float
) -> float | None:
    """Return the largest threshold whose recall is at least ``target_recall``.

    Returns ``None`` when no threshold reaches the requested sensitivity.
    """
    _, recall, thresholds = precision_recall_curve(y_true, y_score)
    eligible = [
        float(t) for t, r in zip(thresholds, recall[:-1], strict=False) if r >= target_recall
    ]
    if not eligible:
        return None
    return max(eligible)


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], destination: Path, title: str
) -> None:
    """Save an annotated confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=150)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:d}",
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    LOGGER.info("Wrote %s", destination)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, auc: float, destination: Path) -> None:
    """Save the ROC curve with the chance diagonal for reference."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=150)
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve - Malignant as positive class")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    LOGGER.info("Wrote %s", destination)


def plot_pr_curve(
    y_true: np.ndarray, y_score: np.ndarray, average_precision: float, destination: Path
) -> None:
    """Save the precision-recall curve with the positive-class baseline."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=150)
    ax.plot(recall, precision, label=f"PR (AP = {average_precision:.3f})", linewidth=2)
    ax.axhline(baseline, linestyle="--", linewidth=1, color="grey", label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve - Malignant as positive class")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    LOGGER.info("Wrote %s", destination)


def load_model_from_checkpoint(
    checkpoint_path: Path, config: Config, device
) -> tuple[object, list[str], dict]:
    """Rebuild the model architecture recorded in a checkpoint and load its weights.

    Returns:
        ``(model, class_names, metadata)``.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    metadata = checkpoint.get("extra", {}) or {}
    class_names = list(metadata.get("class_names") or ["Benign", "Malignant"])
    model_name = metadata.get("model_name", config.model.name)

    model = build_model(
        name=model_name,
        num_classes=len(class_names),
        pretrained=False,  # weights come from the checkpoint
        freeze_backbone=False,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    LOGGER.info("Loaded %s from %s", model_name, checkpoint_path)
    return model, class_names, metadata


def main(argv: list | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    setup_logging()

    try:
        config = load_config(args.config)
    except ValueError as exc:
        LOGGER.error("Invalid configuration: %s", exc)
        return 2

    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.device:
        config.device = args.device
    if args.target_sensitivity is not None:
        config.eval.target_sensitivity = args.target_sensitivity

    set_seed(config.seed)
    device = select_device(config.device)

    checkpoint_path = resolve_path(
        args.checkpoint or os.getenv("CHECKPOINT_PATH") or "checkpoints/best.pt"
    )

    try:
        model, class_names, metadata = load_model_from_checkpoint(checkpoint_path, config, device)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        LOGGER.error("Could not load checkpoint: %s", exc)
        return 1

    try:
        test_loader, dataset_classes = build_test_loader(config, limit=args.limit)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("Could not build the test loader: %s", exc)
        return 1

    if dataset_classes != class_names:
        LOGGER.warning(
            "Class order differs between checkpoint %s and dataset %s; using the dataset order",
            class_names, dataset_classes,
        )
        class_names = dataset_classes

    result = validate(model, test_loader, criterion=None, device=device, amp=False)
    if result.probabilities is None or result.probabilities.size == 0:
        LOGGER.error("Evaluation produced no predictions; is the test split empty?")
        return 1

    y_true = result.targets.astype(np.int64)
    y_score = result.probabilities[:, POSITIVE_CLASS_INDEX]
    y_pred = result.probabilities.argmax(axis=1)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
    except ValueError as exc:
        LOGGER.warning("Could not compute AUC/AP (%s); the test split may be single-class", exc)
        auc, ap = float("nan"), float("nan")

    operating_points = {"default": threshold_metrics(y_true, y_score, 0.5)}
    best_f1_threshold = find_best_f1_threshold(y_true, y_score)
    operating_points["best_f1"] = threshold_metrics(y_true, y_score, best_f1_threshold)

    sensitivity_threshold = find_threshold_at_sensitivity(
        y_true, y_score, config.eval.target_sensitivity
    )
    if sensitivity_threshold is None:
        LOGGER.warning(
            "No threshold reaches recall >= %.2f on %s",
            config.eval.target_sensitivity, class_names[POSITIVE_CLASS_INDEX],
        )
        operating_points["target_sensitivity"] = {
            "threshold": None,
            "target_recall": config.eval.target_sensitivity,
            "achieved": False,
        }
    else:
        point = threshold_metrics(y_true, y_score, sensitivity_threshold)
        point["target_recall"] = config.eval.target_sensitivity
        point["achieved"] = True
        operating_points["target_sensitivity"] = point

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "model_name": metadata.get("model_name", config.model.name),
        "device": str(device),
        "num_test_samples": int(y_true.size),
        "class_names": class_names,
        "positive_class": class_names[POSITIVE_CLASS_INDEX],
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": auc,
            "average_precision": ap,
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "operating_points": operating_points,
    }

    output_path = resolve_path(args.output or f"{config.eval.reports_dir}/metrics.json")
    try:
        write_json(output_path, payload)
    except OSError as exc:
        LOGGER.error("Could not write metrics: %s", exc)
        return 1

    if not args.no_figures:
        figures_dir = resolve_path(config.eval.figures_dir)
        try:
            figures_dir.mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(
                matrix, class_names, figures_dir / "confusion_matrix.png",
                "Confusion matrix (threshold 0.5)",
            )
            if not np.isnan(auc):
                plot_roc_curve(y_true, y_score, auc, figures_dir / "roc_curve.png")
                plot_pr_curve(y_true, y_score, ap, figures_dir / "pr_curve.png")
        except OSError as exc:
            LOGGER.error("Could not write figures: %s", exc)

    LOGGER.info(
        "Test accuracy %.4f | ROC-AUC %.4f | AP %.4f | macro-F1 %.4f",
        payload["metrics"]["accuracy"], auc, ap, payload["metrics"]["macro_f1"],
    )
    LOGGER.info("Full metrics written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
