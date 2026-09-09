"""Real SQLite tests for domain history ownership, receipts and atomic writes."""

import copy
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from interview_fixtures import completion_envelope

import promptwitness.interview_journal as journal_module
from promptwitness.interview_journal import (
    InterviewJournal,
    InterviewJournalConflict,
    InterviewJournalError,
    InterviewNotFoundError,
    verify_interview_export,
)
from promptwitness.interview_memory import MemorySnapshot, SourceHead
from promptwitness.interview_models import (
    BoundEvidence,
    CriterionAssessment,
    InterviewContractError,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    MemoryRecord,
    contract_digest,
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
from promptwitness.interview_state import (
    InterviewCommand,
    current_memory_snapshot,
    schedule,
)


def creation(interview_id="interview", participant_id="participant", *, snapshot=None):
    plan = InterviewPlan(
        "plan",
        "v1",
        "Understand an original workshop experience",
        (InterviewTopic("workshop", "Workshop", (InterviewCriterion("task", "Task performed"),)),),
    )
    identity = {
        "provider": "scripted-interview-test/v1",
        "transport": "in-process-test/v1",
        "model": "test-only",
        "endpoint_sha256": contract_digest("no-network-test"),
        "headers_sha256": contract_digest({}),
        "api_key_env": None,
        "timeout": 1,
    }
    payload = {
        "participant_id": participant_id,
        "plan": plan.to_dict(),
        "provider_identities": {
            "analyst": identity,
            "questioner": identity,
        },
        "template_hashes": dict(TEMPLATE_HASHES),
        "renderer_version": RENDERER_VERSION,
        "scheduler_version": "scheduler/v1",
        "memory_snapshot": (
            snapshot if snapshot is not None else MemorySnapshot(participant_id, (), ())
        ).to_dict(),
    }
    return InterviewCommand(interview_id, "create", 0, None, "created", payload)


def stop(state, command_id="stop"):
    return InterviewCommand(
        state.interview_id,
        command_id,
        state.revision,
        state.digest,
        "finished",
        {"reason": "participant_stopped"},
    )


def head(state):
    return SourceHead(state.interview_id, state.revision, state.digest)


def perform(journal, state, kind, payload):
    return journal.execute(
        InterviewCommand(
            state.interview_id,
            f"command-{state.revision + 1}",
            state.revision,
            state.digest,
            kind,
            payload,
        )
    )


def request_for(state):
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
    if action.kind == "analysis":
        return build_analysis_request(context, state.provider_identities["analyst"])
    return build_question_request(
        context,
        state.provider_identities["questioner"],
        snapshot=current_memory_snapshot(state),
        query="sensor",
    )


def completion(request, result, **extra):
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
    question_request = request_for(state)
    state = perform(journal, state, "question_reserved", {"request": question_request.to_dict()})
    state = perform(
        journal,
        state,
        "question_committed",
        completion(
            question_request,
            QuestionResult(question_request, "What task did you perform?"),
            question_id="question-one",
        ),
    )
    state = perform(
        journal,
        state,
        "answer_committed",
        {
            "question_id": "question-one",
            "answer_id": "answer-one",
            "text": "I built a solar sensor 🛰️.\r\nIt measured light.",
        },
    )
    analysis_request = request_for(state)
    state = perform(journal, state, "analysis_reserved", {"request": analysis_request.to_dict()})
    source = analysis_request.context.analysis_answer
    evidence = BoundEvidence.from_answer(source, 0, len(source.text))
    link = analysis_request.target.link
    result = AnalysisResult(
        analysis_request,
        (CriterionAssessment("participant", "interview", link, "covered", (evidence,)),),
        (
            MemoryRecord("participant", (evidence,), (link,), "Participant described a sensor."),
            MemoryRecord(
                "participant", (evidence,), (link,), "Participant described light measurement."
            ),
        ),
    )
    return state, analysis_request, result


def test_atomic_analysis_reopen_then_second_interview_recall(tmp_path):
    path = tmp_path / "journal.db"
    with InterviewJournal(path, create=True) as journal:
        pending, request, result = pending_analysis(journal)
        state = perform(journal, pending, "analysis_committed", completion(request, result))
        assert state.revision == 6 and len(state.memories) == 2
        assert schedule(state).reason == "agenda_completed"
        rows = journal.connection.execute(
            "SELECT revision,memory_id,memory_json FROM interview_memory_index ORDER BY memory_id"
        ).fetchall()
        assert len(rows) == 2 and {row[0] for row in rows} == {6}
        assert {row[1] for row in rows} == {memory.memory_id for memory in result.memories}
        assert journal.read("interview") == state
    with InterviewJournal(path) as journal:
        before_analysis = journal.memory_snapshot("participant", (head(pending),))
        assert before_analysis.entries == ()
        recalled = journal.memory_snapshot("participant", (head(state),))
        assert recalled.entries == tuple(
            sorted(state.memories, key=lambda item: item.memory.memory_id)
        )
        second = journal.execute(creation("second", snapshot=recalled))
        assert not second.assessments and not second.memories
        second_request = request_for(second)
        assert set(second_request.selected_memory_ids) == {
            memory.memory_id for memory in result.memories
        }
        assert schedule(second).kind == "question"
        assert (
            verify_interview_export(journal.export("interview"), expected_head=head(state)) == state
        )


@pytest.mark.parametrize("table", ["interview_events", "interview_heads", "interview_memory_index"])
def test_analysis_sql_failure_leaves_no_partial_assessment_memory_or_receipt(tmp_path, table):
    path = tmp_path / "journal.db"
    with InterviewJournal(path, create=True) as journal:
        pending, request, result = pending_analysis(journal)
        before = journal.export("interview")

        def deny(action, name, _second, _database, _trigger):
            return (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_INSERT and name == table
                else sqlite3.SQLITE_OK
            )

        journal.connection.set_authorizer(deny)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                perform(journal, pending, "analysis_committed", completion(request, result))
        finally:
            journal.connection.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        assert journal.export("interview") == before
        assert journal.read("interview") == pending
        assert (
            journal.connection.execute("SELECT COUNT(*) FROM interview_memory_index").fetchone()[0]
            == 0
        )
    with InterviewJournal(path) as journal:
        assert journal.read("interview") == pending
        committed = perform(journal, pending, "analysis_committed", completion(request, result))
        assert len(committed.memories) == 2 and len(committed.assessments) == 1


def test_exception_between_memory_inserts_rolls_back_entire_analysis(tmp_path, monkeypatch):
    path = tmp_path / "journal.db"
    with InterviewJournal(path, create=True) as journal:
        pending, request, result = pending_analysis(journal)
        original = journal_module._memory_row
        calls = 0

        def fail_second(entry):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("fault between memory rows")
            return original(entry)

        with monkeypatch.context() as patch:
            patch.setattr(journal_module, "_memory_row", fail_second)
            with pytest.raises(RuntimeError, match="between memory rows"):
                perform(journal, pending, "analysis_committed", completion(request, result))
        assert journal.read("interview") == pending
        assert (
            journal.connection.execute("SELECT COUNT(*) FROM interview_memory_index").fetchone()[0]
            == 0
        )
        assert not journal.connection.in_transaction
    with InterviewJournal(path) as journal:
        assert journal.read("interview") == pending


def test_creation_cannot_omit_committed_memory_from_selected_source_head(tmp_path):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        pending, request, result = pending_analysis(journal)
        source = perform(journal, pending, "analysis_committed", completion(request, result))
        incomplete = MemorySnapshot("participant", (head(source),), ())
        with pytest.raises(InterviewJournalConflict, match="imported memories"):
            journal.execute(creation("second", snapshot=incomplete))
        with pytest.raises(InterviewNotFoundError):
            journal.read("second")


def test_reopen_receipts_historical_reads_and_trusted_export(tmp_path):
    path = tmp_path / "interviews.db"
    command = creation()
    original = command.to_dict()
    with InterviewJournal(path, create=True) as journal:
        initial = journal.execute(command)
        finished = journal.execute(stop(initial))
        assert finished.revision == 2
        assert journal.execute(command) == initial
        assert journal.execute(stop(initial)) == finished
        assert journal.read("interview") == finished
        assert journal.read("interview", revision=1, expected_head=head(initial)) == initial
        exported = journal.export("interview")
        assert verify_interview_export(exported, expected_head=head(finished)) == finished
        assert (
            verify_interview_export(contract_json(exported), expected_head=head(finished))
            == finished
        )
        assert command.to_dict() == original
    with InterviewJournal(path) as journal:
        assert journal.read("interview", expected_head=head(finished)) == finished
        assert len(journal.events("interview")) == 2
        assert journal.execute(command) == initial


def test_changed_command_inputs_and_stale_heads_do_not_write(tmp_path):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        initial = journal.execute(creation())
        with pytest.raises(InterviewJournalConflict):
            journal.execute(creation(participant_id="someone-else"))
        journal.execute(stop(initial))
        with pytest.raises(ValueError):
            journal.execute(stop(initial, "different-stop"))
        assert len(journal.events("interview")) == 2
        with pytest.raises(InterviewJournalConflict):
            journal.read("interview", expected_head=head(initial))
        with pytest.raises(InterviewJournalConflict):
            journal.read("interview", revision=3)
        with pytest.raises(InterviewContractError):
            journal.read("interview", revision=True)
        with pytest.raises(ValueError):
            journal.read("interview", expected_head={})
        with pytest.raises(InterviewNotFoundError):
            journal.read("missing")
        with pytest.raises(InterviewContractError):
            journal.execute({})


@pytest.mark.parametrize("table", ["interview_events", "interview_heads", "commit"])
def test_injected_create_failure_rolls_back_event_head_and_receipt(tmp_path, table):
    path = tmp_path / "journal.db"
    with InterviewJournal(path, create=True) as journal:

        def deny(action, name, _second, _database, _trigger):
            if (action == sqlite3.SQLITE_INSERT and name == table) or (
                table == "commit" and action == sqlite3.SQLITE_TRANSACTION and name == "COMMIT"
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        journal.connection.set_authorizer(deny)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                journal.execute(creation())
        finally:
            # None cannot disable an authorizer before Python 3.11.
            journal.connection.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        for name in ("interview_events", "interview_heads", "interview_memory_index"):
            assert journal.connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] == 0
        assert not journal.connection.in_transaction
    with InterviewJournal(path) as journal:
        with pytest.raises(InterviewNotFoundError):
            journal.read("interview")
        assert journal.execute(creation()).revision == 1


def test_two_connections_race_to_create_exact_same_command(tmp_path):
    path = tmp_path / "journal.db"
    with InterviewJournal(path, create=True):
        pass
    barrier = Barrier(2)

    def writer(_index):
        with InterviewJournal(path) as journal:
            barrier.wait(timeout=10)
            return journal.execute(creation()).digest

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(writer, range(2)))
    assert results[0] == results[1]
    with InterviewJournal(path) as journal:
        assert len(journal.events("interview")) == 1


def test_concurrent_initializers_recheck_ownership_under_lock(tmp_path):
    path = tmp_path / "initialization.db"
    barrier = Barrier(2)

    def initialize(_index):
        barrier.wait(timeout=10)
        with InterviewJournal(path, create=True) as journal:
            return journal.connection.execute("PRAGMA application_id").fetchone()[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert len(set(executor.map(initialize, range(2)))) == 1
    with InterviewJournal(path):
        pass


def test_initialization_failure_rolls_back_ddl_and_closes_connection(tmp_path, monkeypatch):
    path = tmp_path / "initialization.db"
    connections = []

    def fail(self):
        connections.append(self.connection)
        self.connection.execute("PRAGMA application_id=42")
        self.connection.execute("CREATE TABLE partial (value TEXT)")
        raise RuntimeError("injected initialization failure")

    monkeypatch.setattr(InterviewJournal, "_initialize", fail)
    with pytest.raises(RuntimeError, match="injected"):
        InterviewJournal(path, create=True)
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []
    finally:
        connection.close()


def test_schema_reference_failure_does_not_open_real_database(tmp_path, monkeypatch):
    path = tmp_path / "journal.db"

    def fail():
        raise RuntimeError("reference failure")

    monkeypatch.setattr(journal_module, "_expected_schema", fail)
    with pytest.raises(RuntimeError, match="reference failure"):
        InterviewJournal(path, create=True)
    assert not path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "PRAGMA application_id=42",
        "PRAGMA user_version=2",
        "CREATE VIEW extra AS SELECT 1",
        "DROP TRIGGER interview_events_no_update",
        "CREATE TABLE extra (value TEXT)",
    ],
)
def test_schema_mutations_are_rejected_on_reopen(tmp_path, mutation):
    path = tmp_path / "journal.db"
    with InterviewJournal(path, create=True) as journal:
        journal.execute(creation())
        journal.connection.execute(mutation)
    before = path.read_bytes()
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path)
    assert path.read_bytes() == before


def test_noop_append_guard_is_not_accepted_by_name(tmp_path):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        journal.execute(creation())
        journal.connection.execute("DROP TRIGGER interview_events_no_update")
        journal.connection.execute(
            "CREATE TRIGGER interview_events_no_update BEFORE UPDATE ON interview_events "
            "BEGIN SELECT 1; END"
        )
        with pytest.raises(InterviewJournalError):
            journal.read("interview")


@pytest.mark.parametrize(
    "column,value",
    [
        ("revision", 9),
        ("event_digest", "f" * 64),
        ("state_digest", "f" * 64),
        ("state_json", "{}"),
    ],
)
def test_head_cache_is_not_authoritative(tmp_path, column, value):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        journal.execute(creation())
        if column == "revision":
            journal.connection.execute("PRAGMA foreign_keys=OFF")
        journal.connection.execute(f"UPDATE interview_heads SET {column}=?", (value,))
        if column == "revision":
            journal.connection.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(InterviewJournalError):
            journal.read("interview")


def test_orphaned_event_and_index_inventory_are_not_missing_interviews(tmp_path):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        journal.execute(creation())
        journal.connection.execute("DELETE FROM interview_heads")
        with pytest.raises(InterviewJournalError, match="orphaned"):
            journal.execute(creation())


def test_unexpected_memory_index_row_is_rejected_before_loading_content(tmp_path):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        journal.execute(creation())
        journal.connection.execute(
            "INSERT INTO interview_memory_index VALUES (?,?,?,?,?)",
            ("interview", 1, "invented", "f" * 64, "{}"),
        )
        with pytest.raises(InterviewJournalError, match="index inventory"):
            journal.read("interview")


@pytest.mark.parametrize("name,limit", [("MAX_HISTORY_EVENTS", 0), ("MAX_HISTORY_BYTES", 1)])
def test_history_admission_rolls_back_without_publishing_receipt(
    tmp_path, monkeypatch, name, limit
):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        monkeypatch.setattr(journal_module, name, limit)
        with pytest.raises(InterviewJournalError, match="history"):
            journal.execute(creation())
        assert (
            journal.connection.execute("SELECT COUNT(*) FROM interview_events").fetchone()[0] == 0
        )


def test_corrupt_history_and_cache_sizes_rejected_before_deserialization(tmp_path, monkeypatch):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        journal.execute(creation())
        with monkeypatch.context() as patch:
            patch.setattr(journal_module, "MAX_HISTORY_BYTES", 1)
            with pytest.raises(InterviewJournalError, match="history"):
                journal.read("interview")
        with monkeypatch.context() as patch:
            patch.setattr(journal_module, "MAX_CONTRACT_BYTES", 1)
            with pytest.raises(InterviewJournalError, match="cached head"):
                journal.read("interview")


def test_imported_source_heads_must_exist_match_and_share_participant(tmp_path):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        initial = journal.execute(creation())
        snapshot = journal.memory_snapshot("participant", (head(initial),))
        assert snapshot.heads == (head(initial),) and snapshot.entries == ()
        assert journal.execute(creation("second", snapshot=snapshot)).revision == 1
        with pytest.raises(InterviewJournalConflict):
            journal.memory_snapshot("someone-else", (head(initial),))
        with pytest.raises(InterviewJournalConflict):
            journal.memory_snapshot("participant", (SourceHead("interview", 2, initial.digest),))
        with pytest.raises(InterviewJournalConflict):
            journal.memory_snapshot("participant", (SourceHead("interview", 1, "f" * 64),))
        with pytest.raises(InterviewNotFoundError):
            journal.memory_snapshot("participant", (SourceHead("missing", 1, "f" * 64),))


@pytest.mark.parametrize("change", ["truncate", "digest", "extra", "empty", "wrong_interview"])
def test_export_tampering_fails_against_independent_head(tmp_path, change):
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        initial = journal.execute(creation())
        finished = journal.execute(stop(initial))
        value = copy.deepcopy(journal.export("interview"))
    if change == "truncate":
        value["events"].pop()
    elif change == "digest":
        value["events"][0]["digest"] = "f" * 64
    elif change == "extra":
        value["extra"] = True
    elif change == "empty":
        value["events"] = []
    else:
        value["interview_id"] = "other"
    with pytest.raises(ValueError):
        verify_interview_export(value, expected_head=head(finished))


def test_create_false_does_not_initialize_missing_or_empty_database(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        InterviewJournal(path)
    assert not path.exists()
    path.touch()
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path)
    assert path.stat().st_size == 0
    with pytest.raises(ValueError):
        InterviewJournal(":memory:", create=True)
    with pytest.raises(ValueError):
        InterviewJournal(tmp_path, create=True)
    with pytest.raises(ValueError):
        InterviewJournal(path, create=1)


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_orphan_sidecars_preserved_without_creating_database(tmp_path, suffix):
    path = tmp_path / "missing.db"
    sidecar = tmp_path / (path.name + suffix)
    sidecar.write_bytes(b"original unrelated bytes")
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path, create=True)
    assert not path.exists()
    assert sidecar.read_bytes() == b"original unrelated bytes"


@pytest.mark.parametrize("suffix", ["", "-journal", "-wal", "-shm"])
def test_hardlink_aliases_refused_before_sqlite_open(tmp_path, suffix):
    path = tmp_path / "interview.db"
    original = tmp_path / "original.bin"
    original.write_bytes(b"preserve me")
    if suffix:
        with InterviewJournal(path, create=True):
            pass
    alias = tmp_path / (path.name + suffix)
    alias.hardlink_to(original)
    before = path.read_bytes()
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path, create=True)
    assert original.read_bytes() == b"preserve me"
    assert path.read_bytes() == before


def test_dangling_sidecar_symbolic_link_is_not_invisible(tmp_path):
    path = tmp_path / "missing.db"
    sidecar = tmp_path / "missing.db-wal"
    try:
        sidecar.symlink_to(tmp_path / "absent-target")
    except OSError:
        pytest.skip("host does not permit creating symbolic links")
    with pytest.raises(InterviewJournalError):
        InterviewJournal(path, create=True)
    assert not path.exists() and sidecar.is_symlink()
