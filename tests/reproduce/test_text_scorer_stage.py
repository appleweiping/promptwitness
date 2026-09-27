"""Authored staging/binding mechanics; not official dataset qualification."""

from __future__ import annotations

import json

import pytest

from reproduce import text_scorer_stage as stage
from reproduce.strict_scoring import ScoringError


@pytest.fixture
def staged(monkeypatch, tmp_path):
    source = tmp_path / "source"
    blobs = {name: b"# authored fixture\n" for name in stage.CODE_FILES}
    monkeypatch.setattr(stage, "source_blobs", lambda _: blobs)
    for root in stage.RESOURCE_DIRS:
        path = source / "nltk-data" / root / "fixture.txt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"authored resource")
    target = tmp_path / "runtime"
    stage.stage_native(source, target)
    return source, target


def test_code_resources_and_empty_import_directory_only(staged):
    source, target = staged
    receipt = stage.staged_inventory(target)
    assert len(receipt["code_files"]) == 11
    assert len(receipt["resource_files"]) == 4
    assert not list((target / stage.EMPTY_IMPORT_DIR).iterdir())
    stage.verify_stage(source, target)
    with pytest.raises(FileExistsError):
        stage.stage_native(source, target)


@pytest.mark.parametrize("extra", ["IFBench/IFBench_test.jsonl", ".git/HEAD", "answer.json"])
def test_data_and_git_never_part_of_runtime(staged, extra):
    _, target = staged
    path = target / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("authored extra")
    with pytest.raises(ScoringError, match="unregistered"):
        stage.staged_inventory(target)


@pytest.mark.parametrize("change", ["code", "resource"])
def test_controller_compares_code_and_resource_bytes(staged, change):
    source, target = staged
    name = stage.CODE_FILES[0] if change == "code" else stage.resource_paths(source)[0]
    (target / name).write_bytes(b"changed")
    with pytest.raises(ScoringError, match="differ"):
        stage.verify_stage(source, target)


@pytest.mark.parametrize(
    "change",
    [
        {"revisions": {}},
        {"code_files": []},
        {"resource_files": ["data/answer.json"]},
        {"resource_files": []},
    ],
)
def test_invalid_stage_receipt(staged, change):
    _, target = staged
    path = target / stage.RECEIPT
    receipt = json.loads(path.read_text())
    receipt.update(change)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ScoringError, match="inventory"):
        stage.staged_inventory(target)


def test_binding_does_not_request_original_checkout(staged, monkeypatch):
    _, target = staged
    monkeypatch.setattr(stage, "source_blobs", lambda _: pytest.fail("worker source access"))
    monkeypatch.setattr(stage.native, "verify_native_sources", lambda _: pytest.fail("worker Git"))
    monkeypatch.setattr(stage.native, "bind_native", lambda path: ("authored", path))
    assert stage.load_staged(target) == ("authored", target)


def test_missing_precreated_import_directory_fails(staged):
    _, target = staged
    (target / stage.EMPTY_IMPORT_DIR).rmdir()
    with pytest.raises(ScoringError, match="precreated"):
        stage.staged_inventory(target)


def test_missing_nltk_tree_is_not_an_implicit_download(tmp_path):
    with pytest.raises(ScoringError, match="missing declared"):
        stage.resource_paths(tmp_path)
