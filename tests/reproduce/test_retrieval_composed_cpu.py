"""Composition input contracts, with optional Torch numerical checks."""

from pathlib import Path

import pytest

from reproduce.check_retrieval_composed_cpu import check
from reproduce.retrieval_composed_cpu import direct_fusion_query, rank_direct_composed


def test_cpu_guard_before_output_and_torch_import(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA visibility"):
        check(tmp_path / "missing.pt", tmp_path / "never-created")
    with pytest.raises(ValueError, match="CUDA visibility"):
        direct_fusion_query(None, None, 0.5)
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("bad_weight", [-0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_bad_weight_rejected_before_optional_torch(bad_weight, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="frozen image weight"):
        direct_fusion_query(None, None, bad_weight)


def test_reference_must_belong_to_complete_pool_before_optional_torch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="reference ID"):
        rank_direct_composed(None, ("a", "b"), "not-in-pool", None, 0.5)


def test_numeric_composition_on_authored_vectors(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    red = torch.zeros(768, dtype=torch.float32)
    blue = torch.zeros(768, dtype=torch.float32)
    red[0], blue[1] = 2.0, 3.0
    gallery = torch.stack([red, blue])
    modification = torch.zeros(768, dtype=torch.float32)
    modification[1] = 1.0
    query = direct_fusion_query(red, modification, 0.25)
    assert torch.isclose(torch.linalg.vector_norm(query), torch.tensor(1.0))
    assert rank_direct_composed(gallery, ("red", "blue"), "red", modification, 0.25) == (
        "blue",
        "red",
    )
    with pytest.raises(ValueError, match="already-unit"):
        direct_fusion_query(red, blue, 0.5)
