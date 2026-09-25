"""Shared helpers: seeding, device selection, checkpoint I/O, metrics and logging."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, honouring ``$LOG_LEVEL``.

    Args:
        level: Explicit level name. Falls back to ``$LOG_LEVEL`` then ``INFO``.
    """
    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    numeric = getattr(logging, resolved, logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` for reproducible runs.

    Args:
        seed: Seed value shared by all libraries.
        deterministic: When ``True``, disable cuDNN autotuning so repeated runs
            produce identical results (at some cost in throughput).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    LOGGER.debug("Seeded RNGs with %d (deterministic=%s)", seed, deterministic)


def select_device(preference: str = "auto") -> torch.device:
    """Resolve a torch device from a preference string.

    Args:
        preference: ``auto``, ``cuda``, ``mps`` or ``cpu``. ``auto`` picks CUDA,
            then Apple MPS, then CPU. An unavailable explicit choice falls back
            to CPU with a warning rather than raising.

    Returns:
        The selected :class:`torch.device`.
    """
    pref = (preference or "auto").strip().lower()
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif mps_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif pref.startswith("cuda"):
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(pref)
    elif pref == "mps":
        if not mps_available:
            LOGGER.warning("MPS requested but unavailable; falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    else:
        device = torch.device("cpu")

    LOGGER.info("Using device: %s", device)
    return device


class AverageMeter:
    """Track a running average of a scalar (loss, accuracy, batch time)."""

    def __init__(self, name: str = "meter", fmt: str = ".4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """Add ``n`` observations of ``value`` to the running average."""
        if n <= 0:
            return
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name} {self.val:{self.fmt}} (avg {self.avg:{self.fmt}})"


def save_checkpoint(
    path: os.PathLike[str] | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Serialise model and training state to ``path``.

    Args:
        path: Destination ``.pt`` file. Parent directories are created.
        model: Model whose ``state_dict`` is stored.
        optimizer: Optional optimizer state, needed only to resume training.
        scheduler: Optional LR scheduler state.
        scaler: Optional AMP ``GradScaler`` state.
        epoch: Epoch index that produced this checkpoint.
        best_metric: Best validation metric seen so far.
        extra: Arbitrary JSON-serialisable metadata (config, class names, ...).

    Returns:
        The resolved path that was written.

    Raises:
        OSError: If the file cannot be written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()

    try:
        torch.save(payload, destination)
    except OSError as exc:
        raise OSError(f"Failed to write checkpoint {destination}: {exc}") from exc
    LOGGER.info("Saved checkpoint to %s (epoch %d, metric %.4f)", destination, epoch, best_metric)
    return destination


def load_checkpoint(
    path: os.PathLike[str] | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint file.
        map_location: Device the tensors are mapped onto.

    Returns:
        The checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If the file exists but cannot be deserialised.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {source}. Train a model first with "
            "'python -m src.train' or set CHECKPOINT_PATH."
        )
    try:
        # weights_only=False: our checkpoints carry config metadata, not only tensors.
        return torch.load(source, map_location=map_location, weights_only=False)
    except Exception as exc:  # torch raises assorted exception types here
        raise RuntimeError(f"Could not load checkpoint {source}: {exc}") from exc


def write_json(path: os.PathLike[str] | str, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as pretty-printed UTF-8 JSON, creating parent folders."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write JSON to {destination}: {exc}") from exc
    LOGGER.info("Wrote %s", destination)
    return destination


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters, optionally restricted to trainable ones."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def format_seconds(seconds: float) -> str:
    """Format a duration as ``MM:SS`` or ``HH:MM:SS``."""
    seconds = int(max(0.0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
