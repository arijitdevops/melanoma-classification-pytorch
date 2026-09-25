"""Typed configuration loaded from YAML with environment-variable overrides.

Precedence, lowest to highest:

1. The dataclass defaults in this module.
2. The YAML file at ``CONFIG_PATH`` (default ``configs/default.yaml``).
3. Environment variables (optionally sourced from a ``.env`` file).
4. Command-line flags applied by the individual CLI entry points.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv is a convenience, not a hard requirement.
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover - exercised only without python-dotenv
    pass

LOGGER = logging.getLogger(__name__)

#: Repository root, resolved relative to this file (``src/config.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

#: Class order matches ``torchvision.datasets.ImageFolder`` sorting of the
#: dataset directories: ``Benign`` < ``Malignant``. Index 1 is the positive
#: (malignant) class throughout the project.
CLASS_NAMES = ("Benign", "Malignant")
POSITIVE_CLASS_INDEX = 1


def _as_bool(value: str) -> bool:
    """Parse a truthy string from the environment."""
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class DataConfig:
    """Dataset location and dataloader behaviour."""

    data_dir: str = "../_datasets/melanoma_skin_cancer"
    val_split: float = 0.15
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    def resolved_data_dir(self) -> Path:
        """Return ``data_dir`` as an absolute path, anchored at the repo root."""
        candidate = Path(self.data_dir).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate


@dataclass
class ModelConfig:
    """Backbone selection and classifier head options."""

    name: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 2
    dropout: float = 0.2


@dataclass
class TrainConfig:
    """Optimisation schedule, regularisation and checkpointing."""

    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    min_lr: float = 1e-6
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    balance: str = "weights"
    early_stopping_patience: int = 5
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"


@dataclass
class EvalConfig:
    """Evaluation thresholds and output locations."""

    target_sensitivity: float = 0.95
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"


@dataclass
class Config:
    """Top-level configuration object passed around the codebase."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dictionary (useful for logging and checkpoints)."""
        return asdict(self)


def resolve_path(path_like: os.PathLike[str] | str) -> Path:
    """Resolve a possibly relative path against :data:`PROJECT_ROOT`.

    Absolute paths are returned unchanged, so callers can accept either form
    from the CLI, YAML or the environment.
    """
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _merge_section(section: Any, values: dict[str, Any] | None, path: str) -> None:
    """Overwrite dataclass attributes in place from a mapping, ignoring extras."""
    if not values:
        return
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{path}' must be a mapping, got {type(values)!r}")
    known = {f.name: f.type for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            LOGGER.warning("Ignoring unknown configuration key '%s.%s'", path, key)
            continue
        setattr(section, key, value)


def _apply_env_overrides(config: Config) -> Config:
    """Apply the documented environment-variable overrides to ``config``."""
    env_map = {
        "DATA_DIR": (config.data, "data_dir", str),
        "NUM_WORKERS": (config.data, "num_workers", int),
        "BATCH_SIZE": (config.data, "batch_size", int),
        "IMAGE_SIZE": (config.data, "image_size", int),
        "MODEL_NAME": (config.model, "name", str),
        "EPOCHS": (config.train, "epochs", int),
        "LEARNING_RATE": (config.train, "lr", float),
        "RANDOM_SEED": (config, "seed", int),
        "DEVICE": (config, "device", str),
    }
    for env_name, (section, attr, caster) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        try:
            setattr(section, attr, caster(raw))
        except (TypeError, ValueError):
            LOGGER.warning("Ignoring invalid value for %s: %r", env_name, raw)

    amp_raw = os.getenv("AMP")
    if amp_raw:
        config.train.amp = _as_bool(amp_raw)
    return config


def load_config(path: os.PathLike[str] | str | None = None) -> Config:
    """Build a :class:`Config` from YAML plus environment overrides.

    Args:
        path: Explicit YAML path. Falls back to ``$CONFIG_PATH`` and then to
            ``configs/default.yaml``. A missing file is not an error: the
            dataclass defaults are used and a warning is logged.

    Returns:
        A fully populated :class:`Config`.

    Raises:
        ValueError: If the YAML file exists but is not a mapping, or a section
            has the wrong shape.
    """
    config = Config()

    candidate = path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH
    config_path = Path(candidate).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Could not read configuration file {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration file {config_path} must contain a YAML mapping")

        _merge_section(config.data, raw.get("data"), "data")
        _merge_section(config.model, raw.get("model"), "model")
        _merge_section(config.train, raw.get("train"), "train")
        _merge_section(config.eval, raw.get("eval"), "eval")
        if "seed" in raw:
            config.seed = int(raw["seed"])
        if "device" in raw:
            config.device = str(raw["device"])
        LOGGER.info("Loaded configuration from %s", config_path)
    else:
        LOGGER.warning("Configuration file %s not found; using built-in defaults", config_path)

    return _apply_env_overrides(config)


def describe(config: Any, indent: int = 0) -> str:
    """Render a dataclass configuration as an indented, human-readable block."""
    lines = []
    prefix = " " * indent
    for f in fields(config):
        value = getattr(config, f.name)
        if is_dataclass(value):
            lines.append(f"{prefix}{f.name}:")
            lines.append(describe(value, indent + 2))
        else:
            lines.append(f"{prefix}{f.name}: {value}")
    return "\n".join(lines)
