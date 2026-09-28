"""Persistent input-only rank process and real scorer on authored Linux data."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.gate import GateStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import check_retrieval_rank_session, retrieval_rank_role
from reproduce.process_access import LEAVES, AccessBoundaryError
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_rank_role import RestrictedRankerSession
from reproduce.retrieval_role_scoring import score_rankings_restricted
from reproduce.retrieval_work_ledger import RetrievalWorkLedger


def _authored_store(tmp_path):
    store = tmp_path / "store"
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    for leaf in ("fit/gold", "search/gold", "selection/gold", "final/gold"):
        (store / leaf / "sentinel.txt").write_text("AUTHORED-GOLD", encoding="utf-8")
    pool = ("reference", "target", *(f"other-{i}" for i in range(9)))
    rows = [
        {"id": f"q{i}", "reference_id": "reference", "modification": "make it blue", "category": ""}
        for i in range(64)
    ]
    inputs = store / "search/inputs"
    (inputs / "cirr.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (inputs / "cirr-gallery.json").write_text(json.dumps({"cirr": pool}), encoding="utf-8")
    (store / "search/gold/cirr.jsonl").write_text(
        "".join(
            json.dumps({"id": row["id"], "target_id": "target", "subset": ["reference", "target"]})
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    (inputs / "model.pt").write_bytes(b"authored-not-a-model")
    image_paths = {}
    for image_id in pool:
        relative = f"images/{image_id}.png"
        path = inputs / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"authored-not-an-image")
        image_paths[image_id] = relative
    return store, rows, pool, image_paths


def test_ranker_session_requires_linux_before_start(tmp_path):
    if sys.platform == "linux":
        pytest.skip("host-specific rejection is tested on non-Linux")
    store, _, _, image_paths = _authored_store(tmp_path)
    scratch = tmp_path / "ranker-scratch"
    scratch.mkdir()
    with pytest.raises(AccessBoundaryError, match="Linux"):
        RestrictedRankerSession(
            store=store,
            scratch=scratch,
            dataset="cirr",
            image_paths=image_paths,
            checkpoint="model.pt",
            image_weight=0.5,
            attempt_prefix="authored",
        )
    assert not (scratch / "ranker-stderr.log").exists()


def test_partial_output_frame_obeys_deadline(tmp_path):
    if sys.platform != "linux":
        pytest.skip("nonblocking pipe protocol is Linux-only")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('{'); sys.stdout.flush(); "
            "time.sleep(.4); sys.stdout.write('}\\n'); sys.stdout.flush()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert process.stdout is not None
        os.set_blocking(process.stdout.fileno(), False)
        assert select.select([process.stdout], [], [], 2)[0]
        session = object.__new__(RestrictedRankerSession)
        session.process = process
        session.stderr_path = tmp_path / "stderr.log"
        session._read_buffer = bytearray()
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="did not answer"):
            session._receive(start + 0.05)
        assert time.monotonic() - start < 0.3
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_input_backpressure_obeys_deadline():
    if sys.platform != "linux":
        pytest.skip("nonblocking pipe protocol is Linux-only")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert process.stdin is not None
        os.set_blocking(process.stdin.fileno(), False)
        session = object.__new__(RestrictedRankerSession)
        session.process = process
        session._closed = False
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="input timed out"):
            session._send({"large": "x" * 1_000_000}, start + 0.05)
        assert time.monotonic() - start < 0.3
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_timeout_aborts_ranker_process(tmp_path):
    if sys.platform != "linux":
        pytest.skip("nonblocking pipe protocol is Linux-only")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    os.set_blocking(process.stdin.fileno(), False)
    os.set_blocking(process.stdout.fileno(), False)
    session = object.__new__(RestrictedRankerSession)
    session.process = process
    session.timeout = 0.05
    session.stderr_path = tmp_path / "stderr.log"
    session._read_buffer = bytearray()
    session._closed = False
    with pytest.raises(TimeoutError, match="did not answer"):
        session.rank_one("q0")
    assert session._closed
    assert process.poll() is not None


def test_stalled_restricted_rank_keeps_failure_and_unknown_receipts(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("actual Landlock worker and persistent receipt require Linux")
    monkeypatch.setattr(
        retrieval_rank_role, "ENTRYPOINT", Path(check_retrieval_rank_session.__file__)
    )
    store, _, _, image_paths = _authored_store(tmp_path)
    scratch = tmp_path / "ranker-scratch"
    scratch.mkdir()
    outer = RetrievalWorkLedger(tmp_path / "outer-work.sqlite")
    try:
        with RestrictedRankerSession(
            store=store,
            scratch=scratch,
            dataset="cirr",
            image_paths=image_paths,
            checkpoint="model.pt",
            image_weight=0.5,
            attempt_prefix="authored-stall",
        ) as session:
            session.timeout = 1.0
            with pytest.raises(TimeoutError, match="did not answer"):
                outer.run("outer:q0", "search", "rank_callback", lambda: session.rank_one("q0"))
            assert session._closed
            assert session.process is not None and session.process.poll() is not None
        assert outer.summary()["failed"] == 1
        assert outer.summary()["unresolved"] == 0
        inner = RetrievalWorkLedger(scratch / "ranker-work.sqlite")
        try:
            assert inner.summary()["unresolved"] == 1
            assert inner.summary()["forward_count_unmeasured_attempts"] == 1
        finally:
            inner.close()
    finally:
        outer.close()


def test_restricted_live_ranker_to_scorer_and_gate(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("real Landlock ranker/scorer workers require Linux")
    monkeypatch.setattr(
        retrieval_rank_role, "ENTRYPOINT", Path(check_retrieval_rank_session.__file__)
    )
    store, rows, pool, image_paths = _authored_store(tmp_path)
    rank_scratch = tmp_path / "ranker-scratch"
    score_scratch = tmp_path / "score-scratch"
    rank_scratch.mkdir()
    score_scratch.mkdir()
    old_rank = ("reference", *pool[2:], "target")
    old = score_rankings_restricted(
        store, score_scratch, "cirr", "search", {row["id"]: old_rank for row in rows}
    )
    reference = {
        unit: observation["primary_hit"] for unit, observation in old["observations"].items()
    }
    assert set(reference.values()) == {0}
    plan = make_plan(
        reference,
        candidate_digest=digest("authored-rank-child"),
        execution_digest=digest("authored-rank-process"),
        fixture_seed=11,
    )
    journal = AuditJournal(
        tmp_path / "audit.sqlite",
        run_id="authored-rank-session",
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=plan.population,
    )
    outer_ledger = RetrievalWorkLedger(tmp_path / "outer-work.sqlite")
    try:
        with RestrictedRankerSession(
            store=store,
            scratch=rank_scratch,
            dataset="cirr",
            image_paths=image_paths,
            checkpoint="model.pt",
            image_weight=0.5,
            attempt_prefix="authored-rank",
        ) as session:
            assert session.ready["role"] == "retrieval_search_ranker"
            assert session.ready["pid"] != os.getpid()
            bridge = RetrievalAuditBridge(
                store=store,
                scratch_root=score_scratch,
                dataset="cirr",
                plan=plan,
                journal=journal,
                work_ledger=outer_ledger,
                unit_ids=tuple(reference),
                rank_one=session.rank_one,
            )
            verdict = bridge.evaluate(
                contract=ContractCheck(ContractStatus.VALID, ()),
                execution_scope="FROZEN_TABLE",
            )
            assert verdict.status == GateStatus.ELIGIBLE
            assert bridge.complete_survivor().scores == (1,) * 64
            with pytest.raises(ValueError, match="exited"):
                session.rank_one("q0")  # no uncharged retry of a settled query
        assert outer_ledger.summary()["attempts"] == 128
        inner_ledger = RetrievalWorkLedger(rank_scratch / "ranker-work.sqlite")
        try:
            assert inner_ledger.summary()["attempts"] == 64
            assert inner_ledger.summary()["unresolved"] == 0
        finally:
            inner_ledger.close()
    finally:
        outer_ledger.close()
        journal.close()
