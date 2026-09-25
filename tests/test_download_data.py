"""Tests for scripts/download_data.py (no network: Kaggle is mocked)."""

from __future__ import annotations

import importlib.util
import sys
import types
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_layout(root: Path) -> Path:
    for split in ("train", "test"):
        for cls in ("Benign", "Malignant"):
            folder = root / split / cls
            folder.mkdir(parents=True)
            (folder / "1.jpg").write_bytes(b"fake")
    return root


def test_local_folder_is_used_in_place(dl, tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    source = _make_layout(tmp_path / "wrapper" / "melanoma")
    dest = tmp_path / "dest"
    assert dl.main(["--local", str(tmp_path / "wrapper"), "--data-dir", str(dest)]) == 0
    assert not dest.exists()  # nothing copied without --copy
    assert dl.find_dataset_root(tmp_path / "wrapper") == source


def test_local_folder_copy(dl, tmp_path):
    _make_layout(tmp_path / "src")
    dest = tmp_path / "dest"
    assert dl.main(["--local", str(tmp_path / "src"), "--data-dir", str(dest), "--copy"]) == 0
    assert (dest / "test" / "Malignant" / "1.jpg").is_file()


def test_local_zip_is_extracted(dl, tmp_path):
    _make_layout(tmp_path / "raw")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for file in (tmp_path / "raw").rglob("*.jpg"):
            bundle.write(file, file.relative_to(tmp_path / "raw"))
    dest = tmp_path / "dest"
    assert dl.main(["--local", str(archive), "--data-dir", str(dest)]) == 0
    assert dl.has_layout(dest)


def test_kagglehub_download_is_copied_to_data_dir(dl, tmp_path, monkeypatch):
    cache = _make_layout(tmp_path / "cache")
    fake = types.SimpleNamespace(dataset_download=lambda slug: str(cache))
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    dest = tmp_path / "dest"
    assert dl.main(["--data-dir", str(dest)]) == 0
    assert dl.has_layout(dest)


def test_missing_layout_fails(dl, tmp_path):
    (tmp_path / "train" / "benign").mkdir(parents=True)  # wrong capitalisation
    assert dl.main(["--local", str(tmp_path)]) == 1
