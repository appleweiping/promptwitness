"""Authored record-selection mechanics; no corpus or model calls."""

import hashlib
import json

import pytest

from reproduce.prepare_bfcl_fit import MODEL_REVISIONS
from reproduce.run_model_access import training_records
from reproduce.torch_runtime import TASK_CAPS


def source():
    return [
        {"model": model, "id": model + ":" + family, "family": family, "purpose": "fit"}
        for model in MODEL_REVISIONS
        for family in TASK_CAPS
    ] + [
        {"model": model, "id": model + ":proposal:" + str(index), "purpose": "proposer"}
        for model in MODEL_REVISIONS
        for index in range(2)
    ]


def bound(rows):
    raw = "\n".join(json.dumps(row) for row in rows).encode()
    return raw, hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("model", MODEL_REVISIONS)
def test_only_all_original_fit_families_not_old_proposer_replay(model):
    raw, sha = bound(source())
    result = training_records(raw, sha, model)
    assert len(result) == 3
    assert {row["family"] for row in result} == set(TASK_CAPS)
    assert all(row["purpose"] == "fit" and row["model"] == model for row in result)


@pytest.mark.parametrize("change", ["digest", "missing_model", "missing_family", "duplicate"])
def test_never_silently_shrink_or_accept_changed_source(change):
    rows = source()
    if change == "missing_model":
        rows = rows[:-1]
    elif change == "missing_family":
        rows[0]["family"] = "hotpotqa"
    elif change == "duplicate":
        rows[0]["id"] = rows[1]["id"]
    raw, sha = bound(rows)
    if change == "digest":
        sha = "a" * 64
    with pytest.raises(ValueError):
        training_records(raw, sha, next(iter(MODEL_REVISIONS)))
