"""Fetch the melanoma dataset (Kaggle) or register a local copy of it.

Dataset: ``ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant``
(https://www.kaggle.com/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant)

Three ways to get the data into place:

1. **kagglehub** (default, recommended)::

       pip install kagglehub
       python scripts/download_data.py

2. **Kaggle CLI**::

       pip install kaggle
       python scripts/download_data.py --method cli

   Both need a Kaggle API token: either ``kaggle.json`` in ``~/.kaggle/``
   (``%USERPROFILE%\\.kaggle\\kaggle.json`` on Windows) or the
   ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.

3. **A copy you already have** (folder or the downloaded ``.zip``)::

       python scripts/download_data.py --local D:\\sample_projects\\_datasets\\melanoma_skin_cancer
       python scripts/download_data.py --local D:\\Downloads\\archive.zip

   A folder that already has the right layout is only verified and nothing is
   copied; point ``DATA_DIR`` at it. Add ``--copy`` to copy it into
   ``--data-dir`` instead. A ``.zip`` is always extracted into ``--data-dir``.

The resulting layout must be::

    DATA_DIR/train/{Benign,Malignant}/*.jpg
    DATA_DIR/test/{Benign,Malignant}/*.jpg
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

LOGGER = logging.getLogger("scripts.download_data")

DATASET_SLUG = "ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "_datasets" / "melanoma_skin_cancer"
EXPECTED_SPLITS = ("train", "test")
EXPECTED_CLASSES = ("Benign", "Malignant")


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line interface for the download helper."""
    parser = argparse.ArgumentParser(
        prog="python scripts/download_data.py",
        description="Download the melanoma benign/malignant dataset from Kaggle, "
        "or verify/import a local copy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Destination directory. Defaults to $DATA_DIR, else ../_datasets/melanoma_skin_cancer "
        "(relative paths resolve against the repository root).",
    )
    parser.add_argument(
        "--method",
        choices=["kagglehub", "cli"],
        default="kagglehub",
        help="How to download from Kaggle.",
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Use an existing local copy (dataset folder or downloaded .zip) instead of downloading.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="With --local <folder>: copy the folder into --data-dir instead of using it in place.",
    )
    parser.add_argument("--slug", type=str, default=DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the equivalent kaggle CLI command and exit without downloading.",
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="Keep the downloaded archive after extraction."
    )
    return parser


def resolve_data_dir(raw: str | None) -> Path:
    """Resolve the destination; relative paths are anchored at the repository root."""
    path = Path(raw or os.getenv("DATA_DIR") or str(DEFAULT_DATA_DIR)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def kaggle_command(slug: str, destination: Path) -> list[str]:
    """Return the ``kaggle`` CLI invocation used to fetch the dataset."""
    return ["kaggle", "datasets", "download", "-d", slug, "-p", str(destination), "--unzip"]


def has_layout(root: Path) -> bool:
    """True when ``root`` contains every expected ``split/class`` folder."""
    return all((root / s / c).is_dir() for s in EXPECTED_SPLITS for c in EXPECTED_CLASSES)


def find_dataset_root(start: Path, max_depth: int = 3) -> Path | None:
    """Locate the folder holding ``train/`` and ``test/`` at or below ``start``.

    Kaggle archives sometimes wrap the data in an extra top-level folder, so a
    shallow search is done instead of assuming the exact nesting.
    """
    if has_layout(start):
        return start
    if max_depth == 0 or not start.is_dir():
        return None
    for child in sorted(p for p in start.iterdir() if p.is_dir()):
        found = find_dataset_root(child, max_depth - 1)
        if found is not None:
            return found
    return None


def verify_layout(data_dir: Path) -> bool:
    """Log the image count per split/class and report whether the layout is complete."""
    ok = True
    for split in EXPECTED_SPLITS:
        for class_name in EXPECTED_CLASSES:
            folder = data_dir / split / class_name
            if folder.is_dir():
                count = sum(1 for p in folder.iterdir() if p.is_file())
                LOGGER.info("%-40s %6d file(s)", f"{split}/{class_name}", count)
            else:
                LOGGER.error("Missing expected folder: %s", folder)
                ok = False
    return ok


def extract_zip(archive: Path, destination: Path) -> None:
    """Extract ``archive`` into ``destination``."""
    LOGGER.info("Extracting %s -> %s", archive, destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)


def copy_tree(source: Path, destination: Path) -> None:
    """Copy the expected split folders from ``source`` into ``destination``."""
    for split in EXPECTED_SPLITS:
        LOGGER.info("Copying %s -> %s", source / split, destination / split)
        shutil.copytree(source / split, destination / split, dirs_exist_ok=True)


def download_with_kagglehub(slug: str) -> Path:
    """Download (or reuse the kagglehub cache) and return the local dataset path."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run 'pip install kagglehub' or use --method cli."
        ) from exc
    LOGGER.info("Downloading %s with kagglehub (cached after the first run)", slug)
    return Path(kagglehub.dataset_download(slug))


def download_with_cli(slug: str, destination: Path) -> None:
    """Download and unzip with the ``kaggle`` CLI into ``destination``."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The 'kaggle' CLI was not found on PATH. Install it with 'pip install kaggle' "
            "and place your API token in ~/.kaggle/kaggle.json."
        )
    destination.mkdir(parents=True, exist_ok=True)
    command = kaggle_command(slug, destination)
    LOGGER.info("Running: %s", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"kaggle exited with code {completed.returncode}. Check your API credentials "
            "and that you have accepted the dataset's terms on the Kaggle website."
        )


def finish(root: Path | None, searched: Path) -> int:
    """Verify the final location and tell the user what to put in ``.env``."""
    if root is None or not verify_layout(root):
        LOGGER.error(
            "Could not find train/{Benign,Malignant} and test/{Benign,Malignant} under %s.",
            searched,
        )
        return 1
    LOGGER.info("Dataset ready at %s", root)
    LOGGER.info("Set DATA_DIR=%s in your .env file (or pass --data-dir to the scripts).", root)
    return 0


def main(argv: list | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    data_dir = resolve_data_dir(args.data_dir)

    if args.print_command:
        print(" ".join(kaggle_command(args.slug, data_dir)))
        return 0

    try:
        if args.local:
            source = Path(args.local).expanduser().resolve()
            if not source.exists():
                LOGGER.error("Local path does not exist: %s", source)
                return 1
            if source.is_file() and source.suffix.lower() == ".zip":
                extract_zip(source, data_dir)
                return finish(find_dataset_root(data_dir), data_dir)
            root = find_dataset_root(source)
            if root is not None and args.copy:
                copy_tree(root, data_dir)
                root = data_dir
            return finish(root, source)

        if args.method == "kagglehub":
            cached = download_with_kagglehub(args.slug)
            root = find_dataset_root(cached)
            if root is None:
                return finish(None, cached)
            if root != data_dir:
                copy_tree(root, data_dir)
            return finish(data_dir, data_dir)

        download_with_cli(args.slug, data_dir)
        for archive in sorted(data_dir.glob("*.zip")):  # older CLI versions ignore --unzip
            extract_zip(archive, data_dir)
            if not args.keep_zip:
                archive.unlink(missing_ok=True)
        return finish(find_dataset_root(data_dir), data_dir)
    except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
