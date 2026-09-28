"""Durable authored retrieval operation costs; no benchmark/model claims."""

from __future__ import annotations

import pytest

from reproduce import retrieval_work_ledger as ledger_module
from reproduce.retrieval_work_ledger import MeteredClipEncoder, RetrievalWorkLedger


class FakeEncoder:
    def __init__(self):
        self.costs = {"image_forward_calls": 0, "text_forward_calls": 0}

    def encode_image(self, value):
        if value == "pre-fail":
            raise ValueError("preprocess failed")
        self.costs["image_forward_calls"] += 1
        if value == "post-fail":
            raise ValueError("forward failed")
        return ("image", value)

    def encode_text(self, value):
        self.costs["text_forward_calls"] += 1
        return ("text", value)


def test_forward_receipts_include_success_and_both_failure_points(tmp_path):
    path = tmp_path / "work.sqlite"
    ledger = RetrievalWorkLedger(path)
    encoder = MeteredClipEncoder(FakeEncoder(), ledger, "shared")
    try:
        assert ledger.summary()["attempts"] == 0
        assert encoder.encode_image("image-1", "good") == ("image", "good")
        with pytest.raises(ValueError, match="preprocess"):
            encoder.encode_image("image-2", "pre-fail")
        with pytest.raises(ValueError, match="forward"):
            encoder.encode_image("image-3", "post-fail")
        assert encoder.encode_text("text-1", "make it blue") == ("text", "make it blue")
        summary = ledger.summary()
        assert summary == {
            "attempts": 4,
            "completed": 2,
            "failed": 2,
            "unresolved": 0,
            "forward_count_unmeasured_attempts": 0,
            "wall_count_unmeasured_attempts": 0,
            "known_forward_calls": 3,
            "settled_operation_wall_seconds_sum_nonadditive": summary[
                "settled_operation_wall_seconds_sum_nonadditive"
            ],
        }
        assert summary["settled_operation_wall_seconds_sum_nonadditive"] >= 0
        with pytest.raises(ValueError, match="replayed"):
            encoder.encode_image("image-1", "again")
        assert encoder.encoder.costs["image_forward_calls"] == 2
        rows = ledger.connection.execute(
            "SELECT attempt_id,status,forward_calls FROM retrieval_work ORDER BY attempt_id"
        ).fetchall()
        assert rows == [
            ("image-1", "completed", 1),
            ("image-2", "failed", 0),
            ("image-3", "failed", 1),
            ("text-1", "completed", 1),
        ]
    finally:
        ledger.close()
    continued = RetrievalWorkLedger(path)
    try:
        assert continued.summary()["attempts"] == 4
        assert continued.unresolved() == ()
    finally:
        continued.close()


def test_nested_callback_preserves_separate_forward_and_outer_unknown(tmp_path):
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    encoder = MeteredClipEncoder(FakeEncoder(), ledger, "search")
    try:
        assert ledger.run(
            "rank-q1",
            "search",
            "rank_callback",
            lambda: encoder.encode_text("text-q1", "make it blue"),
        ) == ("text", "make it blue")
        summary = ledger.summary()
        assert summary["attempts"] == 2
        assert summary["completed"] == 2
        assert summary["known_forward_calls"] == 1
        assert summary["forward_count_unmeasured_attempts"] == 1
    finally:
        ledger.close()


def test_nested_duration_sum_is_not_labeled_as_allocated_time(tmp_path, monkeypatch):
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    ticks = iter((0.0, 1.0, 11.0, 12.0))
    try:
        with monkeypatch.context() as clock:
            clock.setattr(ledger_module.time, "monotonic", lambda: next(ticks))
            ledger.run(
                "outer",
                "search",
                "rank_callback",
                lambda: ledger.run("inner", "search", "text_encode", lambda: "done"),
            )
        summary = ledger.summary()
        assert summary["settled_operation_wall_seconds_sum_nonadditive"] == 22.0
        assert "measured_wall_seconds_lower_bound" not in summary
    finally:
        ledger.close()


def test_unresolved_crash_receipt_blocks_new_work_without_free_replay(tmp_path):
    path = tmp_path / "work.sqlite"
    ledger = RetrievalWorkLedger(path)
    try:
        with ledger.connection:
            ledger.connection.execute(
                "INSERT INTO retrieval_work VALUES (?,?,?,?,?,?,?)",
                ("interrupted", "search", "rank_callback", "reserved", 1.0, None, None),
            )
    finally:
        ledger.close()
    continued = RetrievalWorkLedger(path)
    invoked = []
    try:
        assert continued.unresolved() == ("interrupted",)
        with pytest.raises(ValueError, match="unresolved"):
            continued.run("next", "search", "rank_callback", lambda: invoked.append("called"))
        with pytest.raises(ValueError, match="unresolved"):
            continued.run(
                "interrupted", "search", "rank_callback", lambda: invoked.append("replay")
            )
        assert invoked == []
        assert continued.summary()["unresolved"] == 1
        assert continued.summary()["forward_count_unmeasured_attempts"] == 1
        assert continued.summary()["wall_count_unmeasured_attempts"] == 1
    finally:
        continued.close()


def test_invalid_operation_never_enters_ledger(tmp_path):
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    try:
        with pytest.raises(ValueError, match="registered"):
            ledger.run("x", "final", "restricted_score", lambda: None)
        with pytest.raises(ValueError, match="registered"):
            ledger.run("x", "search", "invented", lambda: None)
        assert ledger.summary()["attempts"] == 0
    finally:
        ledger.close()
