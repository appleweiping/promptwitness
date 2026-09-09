"""Independent storage-boundary checks using only fresh temporary journals."""

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest
from interview_fixtures import completion_envelope

import promptwitness.interview_journal as storage
from promptwitness.interview_journal import (
    InterviewJournal,
    InterviewJournalError,
    verify_interview_export,
)
from promptwitness.interview_memory import MemoryEntry, MemorySnapshot, SourceHead
from promptwitness.interview_models import (
    BoundEvidence,
    CriterionAssessment,
    CriterionLink,
    InterviewContractError,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    MemoryRecord,
    ParticipantAnswer,
    contract_json,
)
from promptwitness.interview_stages import (
    RENDERER_VERSION,
    TEMPLATE_HASHES,
    AnalysisResult,
    InterviewStageContext,
    QuestionResult,
    build_analysis_request,
    build_question_request,
)
from promptwitness.interview_state import SCHEDULER_VERSION, InterviewCommand, schedule

IDENTITY = {
    "provider": "offline-review",
    "transport": "review/v1",
    "model": "review-model",
    "endpoint_sha256": "0" * 64,
    "headers_sha256": "0" * 64,
    "api_key_env": None,
    "timeout": 1,
}
LINK = CriterionLink("topic", "criterion")


def creation(interview="review", *, participant="participant", snapshot=None):
    plan = InterviewPlan(
        "plan",
        "1",
        "An independently scripted interview.",
        (InterviewTopic("topic", "Topic", (InterviewCriterion("criterion", "Explain."),)),),
    )
    return InterviewCommand(
        interview,
        "created",
        0,
        None,
        "created",
        {
            "participant_id": participant,
            "plan": plan.to_dict(),
            "provider_identities": {"analyst": IDENTITY, "questioner": IDENTITY},
            "template_hashes": TEMPLATE_HASHES,
            "renderer_version": RENDERER_VERSION,
            "scheduler_version": SCHEDULER_VERSION,
            "memory_snapshot": (snapshot or MemorySnapshot(participant, (), ())).to_dict(),
        },
    )


def command(state, kind, payload, *, command_id=None):
    return InterviewCommand(
        state.interview_id,
        command_id or f"command-{state.revision + 1}",
        state.revision,
        state.digest,
        kind,
        payload,
    )


def stage_request(state, *, snapshot=None, query=""):
    action = schedule(state)
    context = InterviewStageContext(
        state.interview_id,
        state.participant_id,
        state.revision,
        state.digest,
        f"request-{state.revision}",
        f"attempt-{state.revision}",
        state.plan,
        action.target,
        state.accepted_emergent,
        state.answers,
        state.questions,
        action.answer_id,
    )
    if action.kind == "question":
        return build_question_request(context, IDENTITY, snapshot=snapshot, query=query)
    assert action.kind == "analysis"
    return build_analysis_request(context, IDENTITY)


def completion_payload(request, result, **extra):
    return {
        "request_id": request.request_id,
        "attempt_id": request.attempt_id,
        "request_digest": request.digest,
        "result": result.to_dict(),
        "completion": completion_envelope(result),
        **extra,
    }


def pending_analysis(journal):
    state = journal.execute(creation())
    request = stage_request(state)
    state = journal.execute(command(state, "question_reserved", {"request": request.to_dict()}))
    state = journal.execute(
        command(
            state,
            "question_committed",
            completion_payload(request, QuestionResult(request, "What happened?"), question_id="q"),
        )
    )
    state = journal.execute(
        command(state, "answer_committed", {"question_id": "q", "answer_id": "a", "text": "A B"})
    )
    request = stage_request(state)
    state = journal.execute(command(state, "analysis_reserved", {"request": request.to_dict()}))
    source = state.answers[-1]
    evidence = BoundEvidence.from_answer(source, 0, len(source.text))
    assessment = CriterionAssessment("participant", "review", LINK, "covered", (evidence,))
    memory = MemoryRecord("participant", (evidence,), (LINK,))
    result = AnalysisResult(request, (assessment,), (memory,))
    return state, command(state, "analysis_committed", completion_payload(request, result))


def inventory(connection):
    return tuple(
        tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1,2"))
        for table in ("interview_events", "interview_heads", "interview_memory_index")
    )


def test_missing_read_open_does_not_create_any_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        InterviewJournal(tmp_path / "missing.sqlite")
    assert tuple(tmp_path.iterdir()) == ()


def test_unrelated_database_rejection_preserves_exact_file(tmp_path):
    path = tmp_path / "other.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE private_data (value TEXT)")
    connection.execute("INSERT INTO private_data VALUES ('unchanged')")
    connection.execute("PRAGMA user_version=77")
    connection.commit()
    connection.close()
    before = path.read_bytes()
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path, create=True)
    assert path.read_bytes() == before
    assert {entry.name for entry in tmp_path.iterdir()} == {path.name}


def test_unrelated_hot_rollback_database_is_not_recovered_before_rejection(tmp_path):
    path = tmp_path / "foreign.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA page_size=1024")
    connection.execute("CREATE TABLE private_data (value BLOB)")
    connection.executemany("INSERT INTO private_data VALUES (?)", [(b"x" * 500,)] * 200)
    connection.commit()
    connection.close()
    committed = path.read_bytes()
    # This intentionally crashes only a fresh fixture child after cache spill.
    # It leaves a genuine hot journal, not an imitation of SQLite internals.
    child = (
        "import os,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
        "c.execute('PRAGMA cache_size=4'); c.execute('BEGIN IMMEDIATE'); "
        "c.execute('UPDATE private_data SET value=zeroblob(500)'); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", child, str(path)], check=True, timeout=10)
    sidecar = tmp_path / "foreign.sqlite-journal"
    crashed = path.read_bytes()
    assert sidecar.exists() and crashed != committed
    journal_bytes = sidecar.read_bytes()
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path)
    assert path.read_bytes() == crashed
    assert sidecar.read_bytes() == journal_bytes


def test_initializer_failure_rolls_back_schema_and_closes_connection(tmp_path, monkeypatch):
    observed = []

    def fail(self):
        observed.append(self.connection)
        self.connection.execute(f"PRAGMA application_id={storage._APPLICATION_ID}")
        self.connection.execute(f"PRAGMA user_version={storage._SCHEMA_VERSION}")
        self.connection.execute(storage._SCHEMA[0])
        raise RuntimeError("injected initialization fault")

    path = tmp_path / "journal.sqlite"
    monkeypatch.setattr(InterviewJournal, "_initialize", fail)
    with pytest.raises(RuntimeError, match="injected"):
        InterviewJournal(path, create=True)
    with pytest.raises(sqlite3.ProgrammingError):
        observed[0].execute("SELECT 1")
    with sqlite3.connect(path) as check:
        assert check.execute("PRAGMA application_id").fetchone() == (0,)
        assert check.execute("PRAGMA user_version").fetchone() == (0,)
        assert check.execute("SELECT * FROM sqlite_master").fetchall() == []


def test_reference_schema_failure_never_opens_real_file(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError("injected reference fault")

    monkeypatch.setattr(storage, "_expected_schema", fail)
    with pytest.raises(RuntimeError, match="reference"):
        InterviewJournal(tmp_path / "journal.sqlite", create=True)
    assert tuple(tmp_path.iterdir()) == ()


def test_concurrent_initializers_observe_only_complete_schema(tmp_path):
    barrier = Barrier(2)
    path = tmp_path / "journal.sqlite"

    def initialize(_):
        barrier.wait(timeout=5)
        with InterviewJournal(path, create=True) as journal:
            return storage._schema_rows(journal.connection)

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(initialize, range(2)))
    assert values == [storage._expected_schema()] * 2


@pytest.mark.parametrize("kind", ["trigger", "extra_view", "identity", "version"])
def test_existing_schema_must_match_exact_trusted_ddl(tmp_path, kind):
    path = tmp_path / "journal.sqlite"
    with InterviewJournal(path, create=True):
        pass
    with sqlite3.connect(path) as connection:
        if kind == "trigger":
            connection.execute("DROP TRIGGER interview_events_no_update")
            connection.execute(
                "CREATE TRIGGER interview_events_no_update BEFORE UPDATE ON interview_events "
                "BEGIN SELECT 1; END"
            )
        elif kind == "extra_view":
            connection.execute("CREATE VIEW extra AS SELECT 1")
        else:
            pragma = "application_id" if kind == "identity" else "user_version"
            connection.execute(f"PRAGMA {pragma}=2")
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path)


def test_exact_command_receipt_survives_newer_head_without_extra_event(tmp_path):
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        create = creation()
        first = journal.execute(create)
        second = journal.execute(command(first, "finished", {"reason": "participant_stopped"}))
        before = inventory(journal.connection)
        replayed = journal.execute(create)
        assert replayed == first
        assert replayed.head != second.head
        assert journal.read("review") == second
        assert inventory(journal.connection) == before
        changed = replace(create, payload={**create.payload, "participant_id": "different"})
        with pytest.raises(InterviewJournalError, match="different bound inputs"):
            journal.execute(changed)
        assert inventory(journal.connection) == before


def test_same_head_competing_commands_have_only_one_committed_effect(tmp_path):
    path = tmp_path / "journal.sqlite"
    with InterviewJournal(path, create=True) as journal:
        first = journal.execute(creation())
    barrier = Barrier(2)

    def finish(index):
        with InterviewJournal(path) as journal:
            barrier.wait(timeout=5)
            try:
                result = journal.execute(
                    command(
                        first,
                        "finished",
                        {"reason": "participant_stopped"},
                        command_id=f"finish-{index}",
                    )
                )
                return result.revision
            except (InterviewContractError, InterviewJournalError):
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(finish, range(2)))
    assert sorted(results, key=str) == [2, "conflict"]
    with InterviewJournal(path) as journal:
        assert len(journal.events("review")) == 2


def test_same_reservation_command_only_one_submit_receipt_authorizes_external_effect(tmp_path):
    path = tmp_path / "journal.sqlite"
    with InterviewJournal(path, create=True) as journal:
        first = journal.execute(creation())
    request = stage_request(first)
    reserve = command(first, "question_reserved", {"request": request.to_dict()})
    barrier = Barrier(2)

    def submit(_):
        with InterviewJournal(path) as journal:
            barrier.wait(timeout=5)
            return journal.submit(reserve)

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(submit, range(2)))
    assert sorted(receipt.applied for receipt in receipts) == [False, True]
    assert receipts[0].state == receipts[1].state
    with InterviewJournal(path) as journal:
        assert len(journal.events("review")) == 2
        recovered = journal.submit(reserve)
        assert recovered.applied is False
        assert recovered.state == receipts[0].state
        assert journal.execute(reserve) == recovered.state
        assert len(journal.events("review")) == 2


def test_old_submit_receipt_returns_original_state_without_external_effect_permission(tmp_path):
    path = tmp_path / "journal.sqlite"
    create = creation()
    with InterviewJournal(path, create=True) as journal:
        receipt = journal.submit(create)
        assert receipt.applied is True
        ended = journal.execute(
            command(receipt.state, "finished", {"reason": "participant_stopped"})
        )
    with InterviewJournal(path) as journal:
        recovered = journal.submit(create)
        assert recovered.applied is False
        assert recovered.state == receipt.state
        assert journal.read("review") == ended
        assert len(journal.events("review")) == 2


@pytest.mark.parametrize("table", ["interview_heads", "interview_memory_index"])
def test_analysis_failure_rolls_back_event_head_memory_and_receipt(tmp_path, table):
    path = tmp_path / "journal.sqlite"
    with InterviewJournal(path, create=True) as journal:
        state, complete = pending_analysis(journal)
        before = inventory(journal.connection)

        def deny(action, first, _second, _database, _trigger):
            return (
                sqlite3.SQLITE_DENY
                if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE) and first == table
                else sqlite3.SQLITE_OK
            )

        journal.connection.set_authorizer(deny)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                journal.execute(complete)
        finally:
            journal.connection.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        assert inventory(journal.connection) == before
        assert journal.read("review") == state
    with InterviewJournal(path) as journal:
        after = journal.execute(complete)
        assert after.revision == state.revision + 1
        assert len(after.memories) == 1
        assert journal.execute(complete) == after
        assert len(journal.events("review")) == after.revision


def test_created_snapshot_must_be_from_same_verified_database(tmp_path):
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        absent = MemorySnapshot("participant", (SourceHead("absent", 1, "0" * 64),), ())
        with pytest.raises(KeyError):
            journal.execute(creation(snapshot=absent))
        assert inventory(journal.connection) == ((), (), ())


def test_question_reservation_cannot_inject_self_declared_foreign_memory(tmp_path):
    source = ParticipantAnswer("absent", "participant", "a", "q", "fabricated recall", 1)
    evidence = BoundEvidence.from_answer(source, 0, len(source.text))
    memory = MemoryRecord("participant", (evidence,), (LINK,))
    forged = MemorySnapshot(
        "participant", (SourceHead("absent", 2, "1" * 64),), (MemoryEntry(memory, "absent", 2),)
    )
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        state = journal.execute(creation())
        request = stage_request(state, snapshot=forged, query="fabricated")
        assert request.selected_memory_ids == (memory.memory_id,)
        before = inventory(journal.connection)
        with pytest.raises(InterviewContractError, match="recall"):
            journal.execute(command(state, "question_reserved", {"request": request.to_dict()}))
        assert inventory(journal.connection) == before


def test_same_database_memory_selection_is_exact_revision_bound_and_replayable(tmp_path):
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        before_memory, complete = pending_analysis(journal)
        committed = journal.execute(complete)
        old = journal.memory_snapshot("participant", (before_memory.head,))
        current = journal.memory_snapshot("participant", (committed.head,))
        assert old.entries == ()
        assert current.entries == committed.memories
        second = journal.execute(creation("next", snapshot=current))
        assert contract_json(second.memory_snapshot.to_dict()) == contract_json(current.to_dict())
        assert journal.read("review", revision=before_memory.revision) == before_memory
        with pytest.raises(InterviewJournalError):
            journal.memory_snapshot("other-participant", (committed.head,))
        with pytest.raises(InterviewJournalError):
            journal.memory_snapshot("participant", (replace(committed.head, digest="f" * 64),))


def test_commit_failure_rolls_back_complete_analysis_transition(tmp_path):
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        before, complete = pending_analysis(journal)
        original = inventory(journal.connection)

        def deny_commit(action, first, _second, _database, _trigger):
            return (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT"
                else sqlite3.SQLITE_OK
            )

        journal.connection.set_authorizer(deny_commit)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                journal.execute(complete)
        finally:
            journal.connection.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        assert inventory(journal.connection) == original
        assert journal.read("review") == before
        assert journal.execute(complete).revision == before.revision + 1


def test_completed_export_requires_separately_trusted_head(tmp_path):
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        first = journal.execute(creation())
        last = journal.execute(command(first, "finished", {"reason": "participant_stopped"}))
        exported = journal.export("review")
        assert verify_interview_export(exported, expected_head=last.head) == last
        assert verify_interview_export(contract_json(exported), expected_head=last.head) == last
        with pytest.raises(TypeError):
            verify_interview_export(exported)
        with pytest.raises(InterviewJournalError):
            verify_interview_export(exported, expected_head=first.head)
        exported["events"].pop()
        with pytest.raises(InterviewJournalError):
            verify_interview_export(exported, expected_head=last.head)


def test_history_event_budget_rejects_append_without_receipt(tmp_path, monkeypatch):
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        first = journal.execute(creation())
        before = inventory(journal.connection)
        monkeypatch.setattr(storage, "MAX_HISTORY_EVENTS", 1)
        with pytest.raises(InterviewJournalError, match="bounds"):
            journal.execute(command(first, "finished", {"reason": "participant_stopped"}))
        assert inventory(journal.connection) == before
        assert journal.read("review") == first


@pytest.mark.parametrize("table", ["interview_heads", "interview_memory_index"])
def test_tampered_cached_materialization_is_rejected_after_exact_guards_restored(tmp_path, table):
    path = tmp_path / "journal.sqlite"
    with InterviewJournal(path, create=True) as journal:
        _, complete = pending_analysis(journal)
        committed = journal.execute(complete)
    with sqlite3.connect(path) as connection:
        if table == "interview_heads":
            # Equal-size, valid JSON is still different from the replayed count.
            original = contract_json(committed.to_dict())
            changed = original.replace('"participant_turns":1', '"participant_turns":0', 1)
            assert changed != original and len(changed) == len(original)
            connection.execute("UPDATE interview_heads SET state_json=?", (changed,))
        else:
            connection.execute("DROP TRIGGER interview_memory_no_update")
            connection.execute("UPDATE interview_memory_index SET memory_digest=?", ("0" * 64,))
            guard = next(
                statement
                for statement in storage._SCHEMA
                if "CREATE TRIGGER interview_memory_no_update " in statement
            )
            connection.execute(guard)
    with InterviewJournal(path) as journal, pytest.raises(InterviewJournalError):
        journal.read("review")


def test_orphan_sidecar_rejected_without_creating_database(tmp_path):
    path = tmp_path / "journal.sqlite"
    sidecar = tmp_path / "journal.sqlite-wal"
    sidecar.write_bytes(b"unrelated existing sidecar")
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path, create=True)
    assert not path.exists()
    assert sidecar.read_bytes() == b"unrelated existing sidecar"


def test_database_hardlink_alias_is_rejected_without_modification(tmp_path):
    path = tmp_path / "journal.sqlite"
    with InterviewJournal(path, create=True):
        pass
    alias = tmp_path / "alias.sqlite"
    try:
        alias.hardlink_to(path)
    except OSError:
        pytest.skip("filesystem does not support creating hard links")
    before = path.read_bytes()
    with pytest.raises(InterviewJournalError, match="hardlink"):
        InterviewJournal(path)
    assert path.read_bytes() == alias.read_bytes() == before


def test_secret_bearing_nonidentity_mapping_cannot_be_committed(tmp_path):
    create = creation()
    malformed = replace(
        create,
        payload={
            **create.payload,
            "provider_identities": {
                "analyst": {"api_key": "FAKE_DO_NOT_PERSIST"},
                "questioner": IDENTITY,
            },
        },
    )
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        with pytest.raises(InterviewContractError):
            journal.execute(malformed)
        assert inventory(journal.connection) == ((), (), ())


@pytest.mark.parametrize("stage", ["question", "analysis"])
@pytest.mark.parametrize("change", ["different_result", "incomplete"])
def test_completed_transition_must_reparse_stored_envelope_atomically(tmp_path, stage, change):
    with InterviewJournal(tmp_path / "journal.sqlite", create=True) as journal:
        if stage == "analysis":
            state, complete = pending_analysis(journal)
            payload = complete.to_dict()["payload"]
        else:
            state = journal.execute(creation())
            request = stage_request(state)
            state = journal.execute(
                command(state, "question_reserved", {"request": request.to_dict()})
            )
            payload = completion_payload(
                request, QuestionResult(request, "What happened?"), question_id="q"
            )
        if change == "incomplete":
            payload["completion"]["choices"][0]["finish_reason"] = "length"
        elif stage == "question":
            payload["result"]["text"] = "A valid but different question?"
        else:
            payload["result"]["assessments"][0]["status"] = "partial"
        before = inventory(journal.connection)
        # The command and event are freshly hashed. Matching hashes alone must
        # not replace reparsing the original positive completion's actual text.
        with pytest.raises(InterviewContractError):
            journal.execute(command(state, stage + "_committed", payload))
        assert inventory(journal.connection) == before
        assert journal.read("review") == state
