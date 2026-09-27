"""Authored preparations only, never real corpus/model performance."""

import hashlib
import json
from pathlib import Path

import pytest

from reproduce import prepare_model_transport as preparation
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, write_jsonl


def test_input_only_fit_prepare_preserves_complete_original_proposers(tmp_path, monkeypatch):
    store = tmp_path / "AUTHORED_store"
    (store / "fit/inputs").mkdir(parents=True)
    for family in preparation.TASK_CAPS:
        write_jsonl(
            store / "fit/inputs" / (family + ".jsonl"),
            ({"id": f"{family}-{i}", "AUTHORED": "complete original input"} for i in range(512)),
        )
    original = tmp_path / "AUTHORED_requests.jsonl"
    rows = [
        {
            "id": "typed-proposer:" + family,
            "role": "proposer",
            "family": family,
            "max_new_tokens": 2048,
            "messages": [{"role": "user", "content": "original native cost fixture"}],
        }
        for family in ("hotpotqa", "instruction_following")
    ]
    write_jsonl(original, rows)
    monkeypatch.setattr(
        preparation, "PROPOSER_SOURCE_SHA", hashlib.sha256(original.read_bytes()).hexdigest()
    )
    # Preparation has no path or code that opens gold or nonfit leaves.
    read_paths = []
    previous = Path.open

    def opened(path, *args, **kwargs):
        read_paths.append(path)
        return previous(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    prepared = preparation.prepare(store, original)
    assert len(prepared) == 10
    assert {r["model"] for r in prepared} == set(MODEL_REVISIONS)
    assert all("/fit/inputs/" in p.as_posix() or p == original for p in read_paths)
    for model in MODEL_REVISIONS:
        records = [r for r in prepared if r["model"] == model]
        assert len(records) == 5
        for record in records:
            if record["purpose"] == "proposer":
                assert record["original"] == next(r for r in rows if r["id"] == record["id"])
            else:
                family = record["family"]
                expected = min(
                    (f"{family}-{i}" for i in range(512)),
                    key=lambda u: hashlib.sha256(("backendtransport-v1:" + u).encode()).hexdigest(),
                )
                assert record["unit"]["id"] == expected
    assert json.loads(original.read_text().splitlines()[0]) == rows[0]
    changed = tmp_path / "AUTHORED_changed.jsonl"
    write_jsonl(changed, [{"id": "changed"}])
    with pytest.raises(ValueError, match="changed"):
        preparation.prepare(store, changed)
