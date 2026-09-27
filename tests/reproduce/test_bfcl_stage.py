"""Authored code-only inventory/controller mechanics, not official qualification."""

from __future__ import annotations

import json

import pytest

from reproduce import bfcl_native, bfcl_stage
from reproduce.strict_scoring import ScoringError


@pytest.fixture
def staged(monkeypatch, tmp_path):
    blobs = {
        "bfcl_eval/__init__.py": b"",
        "bfcl_eval/eval_checker/ast_eval/ast_checker.py": b"# authored fixture\n",
    }
    monkeypatch.setattr(bfcl_stage, "source_blobs", lambda _: blobs)
    root = tmp_path / "stage"
    bfcl_stage.stage_bfcl(tmp_path / "source", root)
    return root, blobs


@pytest.mark.parametrize(
    "name",
    [
        "bfcl_eval/data/answer.py",
        "bfcl_eval/data/gold.json",
        ".git/objects/object.py",
        "bfcl_eval/eval/out.py",
        "bfcl_eval/../outside.py",
        "/bfcl_eval/a.py",
        "bfcl_eval/__pycache__/a.py",
        "bfcl_eval/.env",
    ],
)
def test_noncode_paths_rejected(name):
    assert not bfcl_stage.code_path(name)


def test_stage_is_code_only_and_exclusive(staged, tmp_path):
    root, blobs = staged
    assert set(bfcl_stage.staged_inventory(root)["files"]) == set(blobs)
    with pytest.raises(FileExistsError):
        bfcl_stage.stage_bfcl(tmp_path / "source", root)


def test_unregistered_data_or_git_file_rejected(staged):
    root, _ = staged
    (root / "answer.json").write_text("{}")
    with pytest.raises(ScoringError, match="unregistered"):
        bfcl_stage.staged_inventory(root)


def test_changed_byte_rejected_by_controller(staged, tmp_path):
    root, _ = staged
    (root / "bfcl_eval/__init__.py").write_text("changed")
    with pytest.raises(ScoringError, match="differs"):
        bfcl_stage.verify_staged_bfcl(tmp_path / "source", root)


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "other"},
        {"files": []},
        {"files": ["bfcl_eval/data/gold.py"]},
        {"files": ["bfcl_eval/a.py", "bfcl_eval/a.py"]},
        {"files": [42]},
    ],
)
def test_invalid_receipt_rejected(staged, change):
    root, _ = staged
    receipt = json.loads((root / bfcl_stage.RECEIPT).read_text())
    receipt.update(change)
    (root / bfcl_stage.RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ScoringError, match="inventory"):
        bfcl_stage.staged_inventory(root)


def test_worker_binding_never_requests_upstream_git(staged, monkeypatch, tmp_path):
    root, _ = staged
    monkeypatch.setattr(bfcl_stage, "source_blobs", lambda _: pytest.fail("worker Git access"))
    calls = []

    def bind(*args):
        calls.append(args)
        return "authored-checker", {"revision": bfcl_native.BFCL_REVISION}

    monkeypatch.setattr(bfcl_native, "bind_bfcl_package", bind)
    checker, meta = bfcl_stage.load_staged_bfcl(root, tmp_path / "scratch", "task/model")
    assert checker == "authored-checker"
    assert calls == [(root / "bfcl_eval", tmp_path / "scratch", "task/model")]
    assert "no worker Git" in meta["source_binding"]
