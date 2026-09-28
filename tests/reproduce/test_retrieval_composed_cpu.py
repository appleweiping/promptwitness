"""Composition input contracts, with optional Torch numerical checks."""

import sys
from pathlib import Path
from types import ModuleType

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


def test_checker_persists_authored_operation_receipts(tmp_path, monkeypatch):
    """A fake encoder exercises the checker wiring, not CLIP numerical behavior."""
    import reproduce.check_retrieval_composed_cpu as checker

    class Gallery(tuple):
        shape = (3, 768)
        dtype = "float32"
        device = "cpu"

    class FakeDraw:
        def rectangle(self, *_args, **_kwargs):
            pass

    class FakeEncoder:
        def __init__(self, _checkpoint):
            self.checkpoint_sha256 = "authored-fake"
            self.load_wall_seconds = 0.0
            self.costs = {"image_forward_calls": 0, "text_forward_calls": 0}

        def encode_image(self, image):
            self.costs["image_forward_calls"] += 1
            return image

        def encode_text(self, value):
            self.costs["text_forward_calls"] += 1
            return value

    torch = ModuleType("torch")
    torch.set_num_threads = lambda _count: None
    torch.manual_seed = lambda _seed: None
    torch.stack = lambda values: Gallery(values)
    torch.equal = lambda left, right: left == right
    image = ModuleType("Image")
    image.new = lambda mode, size, color: (mode, size, color)
    draw = ModuleType("ImageDraw")
    draw.Draw = lambda _image: FakeDraw()
    pil = ModuleType("PIL")
    pil.Image, pil.ImageDraw = image, draw
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(checker, "ClipCPUEncoder", FakeEncoder)
    monkeypatch.setattr(checker, "rank_direct_composed", lambda *_args: ("a", "b", "c"))
    output = tmp_path / "authored-only"
    report = checker.check(tmp_path / "fake-not-read.pt", output)
    assert report["status"] == "PASS_AUTHORED_COMPOSED_IMAGE_TEXT_CPU_ONLY"
    assert report["operation_ledger"]["attempts"] == 15
    assert report["operation_ledger"]["known_forward_calls"] == 10
    assert report["operation_ledger"]["forward_count_unmeasured_attempts"] == 5
    assert report["operation_ledger"]["unresolved"] == 0
    assert (output / "retrieval-work.sqlite").is_file()
