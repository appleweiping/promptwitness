"""Failure receipts for the pinned third-party metric-body differential.

Positive parity requires the private pinned SEARLE source and CPU torch run;
these negative tests download neither source nor benchmark data in CI.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from reproduce import check_retrieval_searle_metric_parity as parity


def test_wrong_source_is_retained_without_importing_torch(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    source = tmp_path / "wrong.py"
    source.write_text("def irrelevant(): pass\n", encoding="utf-8")
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parity.check(source, output)
    report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
    assert report["format"] == parity.FORMAT
    assert report["status"] == "FAILED_RETAINED"
    assert report["source_metric_function_bodies_executed"] is False
    assert report["authored_cpu_tie_fixture_checked"] is False
    assert report["model_forward_calls"] == 0


def test_missing_metric_body_is_retained_before_importing_torch(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    source = tmp_path / "missing.py"
    raw = b"def fiq_compute_val_metrics(): pass\n"
    source.write_bytes(raw)
    monkeypatch.setattr(parity, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="metric functions are missing"):
        parity.check(source, output)
    report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED_RETAINED"
    assert report["source_metric_function_bodies_executed"] is False
    assert report["authored_cpu_tie_fixture_checked"] is False


def test_compiled_but_unexecuted_metric_bodies_are_not_claimed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(parity, "_verified_functions", lambda _: {"torch": object()})

    def fail_before_metric_call(*_):
        raise MemoryError("authored allocation failure")

    monkeypatch.setattr(parity, "_features", fail_before_metric_call)
    output = tmp_path / "new-output"
    with pytest.raises(MemoryError, match="authored allocation failure"):
        parity.check(tmp_path / "unused-source.py", output)
    report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED_RETAINED"
    assert report["error_type"] == "MemoryError"
    assert report["source_metric_function_bodies_executed"] is False
    assert report["authored_cpu_tie_fixture_checked"] is False
