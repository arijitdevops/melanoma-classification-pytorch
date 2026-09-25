"""Shared pytest fixtures.

Every test here runs on CPU with tiny synthetic tensors, so the suite needs no
GPU, no dataset and no trained checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SyntheticImageDataset(Dataset):
    """A deterministic in-memory dataset of random tensors and alternating labels."""

    def __init__(self, size: int = 4, image_size: int = 32, num_classes: int = 2) -> None:
        generator = torch.Generator().manual_seed(0)
        self.images = torch.rand(size, 3, image_size, image_size, generator=generator)
        self.labels = torch.tensor([i % num_classes for i in range(size)], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.images.size(0))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


@pytest.fixture
def synthetic_loader() -> DataLoader:
    """A two-batch dataloader over four synthetic 32x32 samples."""
    return DataLoader(SyntheticImageDataset(size=4, image_size=32), batch_size=2, shuffle=False)


@pytest.fixture
def tiny_model() -> torch.nn.Module:
    """A minimal convolutional model with the same (logits) interface as the real one."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )


@pytest.fixture
def sample_image() -> Image.Image:
    """A small RGB PIL image for transform tests."""
    return Image.new("RGB", (300, 260), color=(180, 120, 110))


@pytest.fixture
def imbalanced_targets() -> list[int]:
    """A deliberately imbalanced label list: 70 zeros followed by 30 ones."""
    return [0] * 70 + [1] * 30


@pytest.fixture
def flask_app():
    """A Flask app configured to point at a checkpoint path that does not exist."""
    from app import create_app

    missing = PROJECT_ROOT / "checkpoints" / "__does_not_exist__.pt"
    app = create_app({"TESTING": True, "CHECKPOINT_PATH": str(missing), "SECRET_KEY": "test-key"})
    return app


@pytest.fixture
def client(flask_app):
    """A test client for :func:`flask_app`, with the model cache cleared."""
    from app.routes import reset_model_cache

    reset_model_cache()
    with flask_app.test_client() as test_client:
        yield test_client
    reset_model_cache()
