"""Pre-import regressions; real Torch/CLIP forwards use the owned CPU stage."""

from pathlib import Path

import pytest

from reproduce.check_retrieval_clip_cpu import check
from reproduce.retrieval_clip_cpu import ClipCPUEncoder, verify_checkpoint


def test_cpu_guard_before_weight_read_or_output(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA visibility"):
        ClipCPUEncoder(tmp_path / "never-read")
    with pytest.raises(ValueError, match="CUDA visibility"):
        check(tmp_path / "never-read", tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_absent_checkpoint_not_downloaded(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(FileNotFoundError):
        ClipCPUEncoder(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_wrong_weight_rejected_before_optional_import(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    checkpoint = tmp_path / "untrusted.pt"
    checkpoint.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="official ViT-L/14"):
        verify_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="official ViT-L/14"):
        ClipCPUEncoder(checkpoint)


def test_existing_output_not_overwritten(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(FileExistsError):
        check(tmp_path / "never-read", tmp_path)
