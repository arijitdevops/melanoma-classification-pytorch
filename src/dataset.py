"""Dataset construction: ImageFolder wrapping, stratified split and dataloaders.

Layout expected under ``DATA_DIR``::

    DATA_DIR/
        train/Benign/*.jpg
        train/Malignant/*.jpg
        test/Benign/*.jpg
        test/Malignant/*.jpg

The ``train`` folder is split into train and validation subsets by index, with
the class ratio preserved. The ``test`` folder is never touched during training
and is reserved for the final evaluation in :mod:`src.evaluate`.

Windows note: ``num_workers`` above 0 spawns worker processes. Every entry point
in this repository is guarded by ``if __name__ == "__main__":`` so this is safe,
but if you hit ``BrokenPipeError`` or slow start-up inside a notebook, set
``NUM_WORKERS=0`` (or ``--num-workers 0``).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from .config import CLASS_NAMES, Config
from .transforms import build_eval_transform, build_train_transform

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class DataBundle:
    """Everything the training loop needs from the data layer."""

    train_loader: DataLoader
    val_loader: DataLoader
    class_names: list[str]
    train_targets: list[int]
    val_targets: list[int]

    @property
    def num_classes(self) -> int:
        """Number of distinct classes."""
        return len(self.class_names)


def _check_split_dir(root: Path, split: str) -> Path:
    """Validate that ``root/split`` exists and contains the expected classes."""
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Expected dataset split at {split_dir}. Set DATA_DIR to the folder that "
            "contains train/ and test/, or run scripts/download_data.py."
        )
    missing = [name for name in CLASS_NAMES if not (split_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Dataset split {split_dir} is missing class folder(s): {', '.join(missing)}"
        )
    return split_dir


def build_image_folder(root: Path, split: str, transform=None) -> ImageFolder:
    """Create an :class:`~torchvision.datasets.ImageFolder` for one split.

    Args:
        root: Dataset root containing ``train`` and ``test``.
        split: Either ``"train"`` or ``"test"``.
        transform: Transform applied to each image.

    Returns:
        The constructed dataset.

    Raises:
        FileNotFoundError: If the split or a class folder is missing.
        RuntimeError: If the split contains no readable images.
    """
    split_dir = _check_split_dir(Path(root), split)
    dataset = ImageFolder(
        str(split_dir),
        transform=transform,
        is_valid_file=lambda p: p.lower().endswith(IMAGE_EXTENSIONS),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No images with extensions {IMAGE_EXTENSIONS} found under {split_dir}")
    LOGGER.info(
        "Loaded %d images from %s (classes: %s)",
        len(dataset),
        split_dir,
        ", ".join(dataset.classes),
    )
    return dataset


def stratified_split(
    targets: Sequence[int],
    val_split: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split indices into train/validation while preserving the class ratio.

    The split is fully determined by ``targets``, ``val_split`` and ``seed``:
    calling this twice with the same arguments returns identical index lists,
    regardless of global RNG state.

    Args:
        targets: Class label for every sample, in dataset order.
        val_split: Fraction of each class assigned to validation, in ``(0, 1)``.
        seed: Seed for the local permutation RNG.

    Returns:
        ``(train_indices, val_indices)``, each sorted ascending.

    Raises:
        ValueError: If ``val_split`` is outside ``(0, 1)`` or a class is too
            small to contribute at least one validation sample.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must lie strictly between 0 and 1, got {val_split}")

    labels = np.asarray(list(targets))
    if labels.size == 0:
        raise ValueError("Cannot split an empty target list")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        permuted = rng.permutation(class_indices)
        n_val = int(round(len(permuted) * val_split))
        n_val = max(1, min(n_val, len(permuted) - 1))
        if len(permuted) < 2:
            raise ValueError(
                f"Class {label} has only {len(permuted)} sample(s); cannot form a split"
            )
        val_indices.extend(int(i) for i in permuted[:n_val])
        train_indices.extend(int(i) for i in permuted[n_val:])

    train_indices.sort()
    val_indices.sort()
    LOGGER.info(
        "Stratified split: %d train / %d val (val_split=%.3f, seed=%d)",
        len(train_indices),
        len(val_indices),
        val_split,
        seed,
    )
    return train_indices, val_indices


def limit_indices(
    indices: Sequence[int],
    targets: Sequence[int],
    limit: int | None,
    seed: int = 42,
) -> list[int]:
    """Return a stratified, reproducible subset of ``indices`` of size ``limit``.

    Used by the ``--limit`` flags for quick smoke runs on a slice of the data.
    Each class keeps (roughly) its original share and at least one sample.

    Args:
        indices: Candidate dataset indices.
        targets: Label of every sample in the *full* dataset (indexed by ``indices``).
        limit: Maximum number of indices to keep. ``None`` or a value >= the
            number of candidates returns ``indices`` unchanged.
        seed: Seed for the local RNG.

    Returns:
        The kept indices, sorted ascending.
    """
    indices = list(indices)
    if limit is None or limit <= 0 or limit >= len(indices):
        return indices
    labels = np.asarray([targets[i] for i in indices])
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    classes = np.unique(labels)
    for label in classes:
        members = [indices[j] for j in np.flatnonzero(labels == label)]
        share = max(1, int(round(limit * len(members) / len(indices))))
        kept.extend(int(i) for i in rng.permutation(members)[:share])
    return sorted(kept)[: max(limit, len(classes))]


def class_distribution(targets: Sequence[int], class_names: Sequence[str]) -> dict[str, int]:
    """Return a ``{class_name: count}`` mapping for a list of integer labels."""
    counts = Counter(int(t) for t in targets)
    return {name: counts.get(index, 0) for index, name in enumerate(class_names)}


def compute_class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights normalised to mean 1.

    Args:
        targets: Integer labels of the training subset.
        num_classes: Total number of classes.

    Returns:
        A float tensor of shape ``(num_classes,)`` suitable for
        :class:`~torch.nn.CrossEntropyLoss`.
    """
    counts = np.bincount(np.asarray(list(targets), dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(targets: Sequence[int], num_classes: int) -> WeightedRandomSampler:
    """Build a sampler that draws each class with roughly equal probability."""
    counts = np.bincount(np.asarray(list(targets), dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    per_class = 1.0 / counts
    sample_weights = [float(per_class[int(t)]) for t in targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap a dataset in a :class:`~torch.utils.data.DataLoader`.

    ``persistent_workers`` is enabled only when workers are actually used, which
    avoids a noisy warning on Windows with ``num_workers=0``.
    """
    workers = max(0, int(num_workers))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=workers > 0,
    )


def build_dataloaders(
    config: Config, balance: str | None = None, limit: int | None = None
) -> DataBundle:
    """Build the training and validation dataloaders described by ``config``.

    Args:
        config: Loaded project configuration.
        balance: Override for ``config.train.balance``. When ``"sampler"``, a
            :class:`~torch.utils.data.WeightedRandomSampler` is attached to the
            training loader; otherwise the loader shuffles normally.
        limit: Optional cap on the number of training images (stratified). The
            validation subset is capped at ``max(limit // 4, 2)``. Intended for
            quick smoke runs, e.g. ``--limit 256``.

    Returns:
        A :class:`DataBundle` with both loaders and the split label lists.
    """
    root = config.data.resolved_data_dir()
    strategy = (balance or config.train.balance or "none").lower()

    train_tf = build_train_transform(config.data.image_size)
    eval_tf = build_eval_transform(config.data.image_size)

    # Two ImageFolder views over the same directory: one augmented, one not,
    # so the validation subset is never seen with random augmentation.
    train_full = build_image_folder(root, "train", transform=train_tf)
    val_full = build_image_folder(root, "train", transform=eval_tf)

    targets = list(train_full.targets)
    train_idx, val_idx = stratified_split(targets, config.data.val_split, config.seed)
    if limit:
        train_idx = limit_indices(train_idx, targets, limit, config.seed)
        val_idx = limit_indices(val_idx, targets, max(limit // 4, 2), config.seed)
        LOGGER.info("--limit %d: using %d train / %d val images", limit, len(train_idx), len(val_idx))

    train_targets = [targets[i] for i in train_idx]
    val_targets = [targets[i] for i in val_idx]

    LOGGER.info("Train distribution: %s", class_distribution(train_targets, train_full.classes))
    LOGGER.info("Val distribution:   %s", class_distribution(val_targets, train_full.classes))

    sampler = None
    if strategy == "sampler":
        sampler = build_weighted_sampler(train_targets, len(train_full.classes))
        LOGGER.info("Class imbalance handled with WeightedRandomSampler")

    train_loader = make_dataloader(
        Subset(train_full, train_idx),
        batch_size=config.data.batch_size,
        shuffle=True,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = make_dataloader(
        Subset(val_full, val_idx),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=list(train_full.classes),
        train_targets=train_targets,
        val_targets=val_targets,
    )


def build_test_loader(
    config: Config, limit: int | None = None
) -> tuple[DataLoader, list[str]]:
    """Build the held-out test loader used by :mod:`src.evaluate`.

    Args:
        config: Loaded project configuration.
        limit: Optional stratified cap on the number of test images (smoke runs).

    Returns:
        ``(loader, class_names)``.
    """
    root = config.data.resolved_data_dir()
    dataset = build_image_folder(root, "test", transform=build_eval_transform(config.data.image_size))
    class_names = list(dataset.classes)
    if limit:
        keep = limit_indices(range(len(dataset)), dataset.targets, limit, config.seed)
        LOGGER.info("--limit %d: evaluating on %d test images", limit, len(keep))
        dataset = Subset(dataset, keep)
    loader = make_dataloader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )
    return loader, class_names
