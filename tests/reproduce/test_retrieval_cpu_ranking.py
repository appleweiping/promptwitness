"""Pre-import checks only locally; Torch cases run in the documented CPU stage."""

from pathlib import Path

import pytest

from reproduce.check_retrieval_cpu_ranking import check
from reproduce.retrieval_cpu_ranking import rank_float32_cpu


def test_cpu_guard_precedes_import_or_output(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA visibility"):
        rank_float32_cpu(None, None, ("a",))
    with pytest.raises(ValueError, match="CUDA visibility"):
        check(tmp_path / "never-read.py", tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("ids", ((), [], ("",), ("a", "a"), (True,), (1,)))
def test_invalid_ids_before_optional_torch_import(ids, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="candidate IDs"):
        rank_float32_cpu(None, None, ids)
