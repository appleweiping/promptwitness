"""Single-database interview events, verified heads and rebuildable memory index.

This module performs no provider calls. A caller must commit its reservation and
release the transaction before external work. Hash chains detect disagreement
with a trusted head; they do not authenticate a database against its owner.
"""

from __future__ import annotations

import sqlite3
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interview_memory import MemoryEntry, MemorySnapshot, SourceHead
from .interview_models import (
    MAX_CONTRACT_BYTES,
    InterviewContractError,
    _closed,
    _identifier,
    _integer,
    _versioned,
    contract_digest,
    contract_json,
    load_interview_json,
)
from .interview_state import (
    InterviewCommand,
    InterviewEvent,
    InterviewState,
    apply_event,
    make_event,
    replay_interview,
)

_APPLICATION_ID = 1347897674
_SCHEMA_VERSION = 1
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_HISTORY_EVENTS = 100_000
_SCHEMA = (
    """CREATE TABLE interview_events (
        interview_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        command_id TEXT NOT NULL,
        command_digest TEXT NOT NULL,
        previous_digest TEXT,
        event_digest TEXT NOT NULL,
        state_digest TEXT NOT NULL,
        event_json TEXT NOT NULL,
        PRIMARY KEY(interview_id, revision),
        UNIQUE(interview_id, command_id)
    )""",
    """CREATE TABLE interview_heads (
        interview_id TEXT PRIMARY KEY NOT NULL,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        event_digest TEXT NOT NULL,
        state_digest TEXT NOT NULL,
        state_json TEXT NOT NULL,
        FOREIGN KEY(interview_id, revision)
            REFERENCES interview_events(interview_id, revision)
    )""",
    """CREATE TABLE interview_memory_index (
        interview_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        memory_id TEXT NOT NULL,
        memory_digest TEXT NOT NULL,
        memory_json TEXT NOT NULL,
        PRIMARY KEY(interview_id, memory_id),
        FOREIGN KEY(interview_id, revision)
            REFERENCES interview_events(interview_id, revision)
    )""",
    """CREATE TRIGGER interview_events_no_update BEFORE UPDATE ON interview_events
        BEGIN SELECT RAISE(ABORT, 'append-only interview events'); END""",
    """CREATE TRIGGER interview_events_no_delete BEFORE DELETE ON interview_events
        BEGIN SELECT RAISE(ABORT, 'append-only interview events'); END""",
    """CREATE TRIGGER interview_memory_no_update BEFORE UPDATE ON interview_memory_index
        BEGIN SELECT RAISE(ABORT, 'append-only interview memory'); END""",
    """CREATE TRIGGER interview_memory_no_delete BEFORE DELETE ON interview_memory_index
        BEGIN SELECT RAISE(ABORT, 'append-only interview memory'); END""",
)


class InterviewJournalError(ValueError):
    """An unsupported or inconsistent interview database was rejected."""


class InterviewJournalConflict(InterviewJournalError):
    """A command ID or expected head does not match the durable history."""


class InterviewNotFoundError(KeyError):
    """No event, head or memory index exists for the requested interview."""


@dataclass(frozen=True, slots=True)
class InterviewReceipt:
    """Historical command result and whether this transaction newly applied it.

    Only the caller that receives ``applied=True`` owns a new reservation's
    first invocation. This is an in-process decision signal, not an external
    exactly-once token, a lease, or proof that a call actually occurred.
    """

    state: InterviewState
    applied: bool


@dataclass(frozen=True, slots=True)
class _History:
    events: tuple[InterviewEvent, ...]
    state: InterviewState
    selected: tuple[InterviewEvent, InterviewState] | None


def _schema_rows(connection: sqlite3.Connection) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        )
    )


def _expected_schema() -> tuple[tuple[Any, ...], ...]:
    connection = sqlite3.connect(":memory:")
    try:
        for statement in _SCHEMA:
            connection.execute(statement)
        return _schema_rows(connection)
    finally:
        connection.close()


def _memory_row(entry: MemoryEntry) -> tuple[Any, ...]:
    value = entry.to_dict()
    return (
        entry.interview_id,
        entry.revision,
        entry.memory.memory_id,
        contract_digest(value),
        contract_json(value),
    )


def _head(state: InterviewState) -> SourceHead:
    return SourceHead(state.interview_id, state.revision, state.digest)


class InterviewJournal:
    """Validate and atomically persist a domain reducer's complete transitions.

    Opening defaults to an existing database. Pass ``create=True`` explicitly to
    initialize an empty/new file. Neither mode migrates unrelated schemas. Every
    read replays the event chain and checks derived head/index caches in one read
    transaction; every write repeats these checks under the SQLite writer lock.
    The per-interview event history is bounded, not silently truncated.
    """

    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        if type(create) is not bool:
            raise ValueError("create must be boolean")
        if str(path) == ":memory:":
            raise ValueError("interview journal requires an explicit filesystem path")
        supplied = Path(path)
        if supplied.is_symlink():
            raise InterviewJournalError("interview journal must not be a symbolic link")
        target = supplied.resolve()
        exists = target.exists()
        if exists and not target.is_file():
            raise ValueError("interview journal path is not a regular file")
        if not create and not exists:
            raise FileNotFoundError("interview journal does not exist")
        if exists and target.stat().st_nlink != 1:
            raise InterviewJournalError("interview journal must not have hardlink aliases")
        sidecars = []
        for suffix in ("-journal", "-wal", "-shm"):
            candidate = Path(str(target) + suffix)
            try:
                information = candidate.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
                raise InterviewJournalError("SQLite sidecars must be unaliased regular files")
            sidecars.append(candidate)
        if (not exists or target.stat().st_size == 0) and sidecars:
            raise InterviewJournalError("refusing a new journal beside pre-existing sidecars")
        if exists and target.stat().st_size:
            # The first SQLite query can recover a hot rollback journal before
            # returning PRAGMA application_id. Check the raw, fixed main-file
            # ownership marker first so rejecting a foreign hot database does
            # not recover its uncommitted pages or delete its sidecar.
            with target.open("rb") as stream:
                header = stream.read(100)
            if (
                len(header) != 100
                or header[:16] != b"SQLite format 3\x00"
                or int.from_bytes(header[68:72], "big") != _APPLICATION_ID
                or int.from_bytes(header[60:64], "big") != _SCHEMA_VERSION
            ):
                raise InterviewJournalError("unsupported main-file interview ownership marker")
        self.path = target
        mode = "rwc" if create else "rw"
        self._schema = _expected_schema()
        self.connection = sqlite3.connect(
            target.as_uri() + "?mode=" + mode, uri=True, isolation_level=None, timeout=10
        )
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            if self._pristine():
                if not create:
                    raise InterviewJournalError(
                        "existing file is not an initialized interview journal"
                    )
                self.connection.execute("BEGIN IMMEDIATE")
                if self._pristine():
                    self._initialize()
                else:
                    self._validate_schema()
                self.connection.commit()
            else:
                self._validate_schema()
        except BaseException:
            self.connection.rollback()
            self.connection.close()
            raise

    def _pristine(self) -> bool:
        return (
            self.connection.execute("PRAGMA application_id").fetchone()[0] == 0
            and self.connection.execute("PRAGMA user_version").fetchone()[0] == 0
            and not _schema_rows(self.connection)
        )

    def _initialize(self) -> None:
        self.connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
        self.connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        for statement in _SCHEMA:
            self.connection.execute(statement)
        self._validate_schema()

    def _validate_schema(self) -> None:
        if (
            self.connection.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID
            or self.connection.execute("PRAGMA user_version").fetchone()[0] != _SCHEMA_VERSION
            or self.connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1
            or _schema_rows(self.connection) != self._schema
        ):
            raise InterviewJournalError("unsupported interview journal identity, schema or guards")

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            self._validate_schema()
            yield
            if write:
                self.connection.commit()
            else:
                self.connection.rollback()
        except BaseException:
            self.connection.rollback()
            raise

    def __enter__(self) -> InterviewJournal:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _read(
        self,
        interview_id: str,
        *,
        revision: int | None = None,
        command_id: str | None = None,
    ) -> _History:
        _identifier(interview_id, "interview ID")
        head_size = self.connection.execute(
            "SELECT length(CAST(state_json AS BLOB)),length(CAST(event_digest AS BLOB)),"
            "length(CAST(state_digest AS BLOB)),typeof(revision) "
            "FROM interview_heads WHERE interview_id=?",
            (interview_id,),
        ).fetchone()
        counts = self.connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(length(CAST(event_json AS BLOB)) "
            "+length(CAST(command_id AS BLOB))+length(CAST(command_digest AS BLOB)) "
            "+COALESCE(length(CAST(previous_digest AS BLOB)),0) "
            "+length(CAST(event_digest AS BLOB))+length(CAST(state_digest AS BLOB))),0) "
            "FROM interview_events WHERE interview_id=?",
            (interview_id,),
        ).fetchone()
        if head_size is None:
            if (
                counts[0]
                or self.connection.execute(
                    "SELECT 1 FROM interview_memory_index WHERE interview_id=? LIMIT 1",
                    (interview_id,),
                ).fetchone()
            ):
                raise InterviewJournalError("interview has orphaned events or memory index")
            raise InterviewNotFoundError("interview does not exist")
        if not 1 <= counts[0] <= MAX_HISTORY_EVENTS or counts[1] > MAX_HISTORY_BYTES:
            raise InterviewJournalError("interview history is empty or exceeds its bounds")
        if (
            head_size[0] > MAX_CONTRACT_BYTES
            or head_size[1] != 64
            or head_size[2] != 64
            or head_size[3] != "integer"
        ):
            raise InterviewJournalError("invalid or oversized interview cached head")
        head = self.connection.execute(
            "SELECT * FROM interview_heads WHERE interview_id=?", (interview_id,)
        ).fetchone()
        events = []
        selected = None
        memories = []
        state = None
        for row in self.connection.execute(
            "SELECT * FROM interview_events WHERE interview_id=? ORDER BY revision", (interview_id,)
        ):
            event = InterviewEvent.from_dict(load_interview_json(row["event_json"]))
            transition = apply_event(state, event)
            state = transition.state
            encoded = contract_json(event.to_dict())
            if (
                event.interview_id != interview_id
                or event.sequence != row["revision"]
                or event.command_id != row["command_id"]
                or event.command_digest != row["command_digest"]
                or event.previous_digest != row["previous_digest"]
                or event.digest != row["event_digest"]
                or contract_digest(state.to_dict()) != row["state_digest"]
                or encoded != row["event_json"]
            ):
                raise InterviewJournalError("interview event metadata or derived state mismatch")
            for entry in transition.memory_additions:
                if entry.interview_id != interview_id or entry.revision != event.sequence:
                    raise InterviewJournalError("memory index entry does not belong to its event")
                memories.append(_memory_row(entry))
            events.append(event)
            if event.sequence == revision or event.command_id == command_id:
                selected = (event, state)
        if state is None or (
            head["revision"] != state.revision
            or head["event_digest"] != state.digest
            or head["state_digest"] != contract_digest(state.to_dict())
            or head["state_json"] != contract_json(state.to_dict())
        ):
            raise InterviewJournalError("interview cached head differs from replayed events")
        expected = tuple(sorted(memories, key=lambda row: row[2]))
        index_size = self.connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(length(CAST(memory_id AS BLOB)) "
            "+length(CAST(memory_digest AS BLOB))+length(CAST(memory_json AS BLOB))),0) "
            "FROM interview_memory_index WHERE interview_id=?",
            (interview_id,),
        ).fetchone()
        if index_size[0] != len(expected) or index_size[1] != sum(
            len((row[2] + row[3] + row[4]).encode("utf-8")) for row in expected
        ):
            raise InterviewJournalError("interview memory index inventory or size mismatch")
        actual = tuple(
            tuple(row)
            for row in self.connection.execute(
                "SELECT interview_id,revision,memory_id,memory_digest,memory_json "
                "FROM interview_memory_index WHERE interview_id=? ORDER BY memory_id",
                (interview_id,),
            )
        )
        if actual != expected or expected != tuple(
            sorted((_memory_row(entry) for entry in state.memories), key=lambda row: row[2])
        ):
            raise InterviewJournalError("interview memory index differs from source events")
        return _History(tuple(events), state, selected)

    def read(
        self,
        interview_id: str,
        *,
        revision: int | None = None,
        expected_head: SourceHead | None = None,
    ) -> InterviewState:
        if revision is not None:
            _integer(revision, "requested revision", minimum=1)
        if expected_head is not None and not isinstance(expected_head, SourceHead):
            raise ValueError("expected_head requires a SourceHead")
        with self._transaction():
            history = self._read(interview_id, revision=revision)
            if revision is not None and history.selected is None:
                raise InterviewJournalConflict("requested revision is not in the interview history")
            state = history.state if history.selected is None else history.selected[1]
            if expected_head is not None and _head(state) != expected_head:
                raise InterviewJournalConflict("interview does not match the expected trusted head")
            return state

    def events(self, interview_id: str) -> tuple[InterviewEvent, ...]:
        with self._transaction():
            return self._read(interview_id).events

    def _snapshot(self, participant_id: str, heads: Sequence[SourceHead]) -> MemorySnapshot:
        # Validate the closed head inventory even when no memory happens to exist.
        selected = MemorySnapshot(participant_id, tuple(heads), ())
        entries: list[MemoryEntry] = []
        for head in selected.heads:
            history = self._read(head.interview_id, revision=head.revision)
            if history.selected is None:
                raise InterviewJournalConflict("selected source revision is missing")
            state = history.selected[1]
            if _head(state) != head or state.participant_id != participant_id:
                raise InterviewJournalConflict("source head or participant does not match")
            entries.extend(state.memories)
        return MemorySnapshot.build(participant_id, selected.heads, entries)

    def memory_snapshot(self, participant_id: str, heads: Sequence[SourceHead]) -> MemorySnapshot:
        """Select only explicitly anchored histories verified in this same database."""
        with self._transaction():
            return self._snapshot(participant_id, heads)

    def execute(self, command: InterviewCommand) -> InterviewState:
        """Commit a transition or return its exact historical state, without invoking work."""
        return self.submit(command).state

    def submit(self, command: InterviewCommand) -> InterviewReceipt:
        """Commit one complete transition and distinguish newly applied reservations.

        No external calls occur here. A successful reservation can be abandoned
        before any network request; resumption must not interpret it as safe to
        repeat. A caller needs an explicit fresh retry command for another call.
        """
        if not isinstance(command, InterviewCommand):
            raise InterviewContractError("execute requires an InterviewCommand")
        with self._transaction(write=True):
            try:
                history = self._read(command.interview_id, command_id=command.command_id)
            except InterviewNotFoundError:
                history = None
            if history is not None and history.selected is not None:
                event, recorded = history.selected
                if event.command_digest != command.digest:
                    raise InterviewJournalConflict("command ID has different bound inputs")
                return InterviewReceipt(recorded, False)
            previous = None if history is None else history.state
            events = () if history is None else history.events
            actual_head = (0, None) if previous is None else (previous.revision, previous.digest)
            if (command.expected_revision, command.expected_digest) != actual_head:
                raise InterviewJournalConflict("command does not extend the current interview head")
            if command.kind == "created":
                snapshot = MemorySnapshot.from_dict(command.payload["memory_snapshot"])
                verified = self._snapshot(snapshot.participant_id, snapshot.heads)
                if contract_json(snapshot.to_dict()) != contract_json(verified.to_dict()):
                    raise InterviewJournalConflict(
                        "imported memories differ from committed sources"
                    )
            event = make_event(previous, command)
            transition = apply_event(previous, event)
            state = transition.state
            encoded_event = contract_json(event.to_dict())

            def stored_size(item: InterviewEvent) -> int:
                return (
                    len(contract_json(item.to_dict()).encode("utf-8"))
                    + len(item.command_id.encode("utf-8"))
                    + len(item.command_digest)
                    + len(item.previous_digest or "")
                    + len(item.digest)
                    + 64
                )

            history_bytes = sum(stored_size(item) for item in events)
            if (
                len(events) + 1 > MAX_HISTORY_EVENTS
                or history_bytes + stored_size(event) > MAX_HISTORY_BYTES
            ):
                raise InterviewJournalError("interview history would exceed its bounds")
            state_json = contract_json(state.to_dict())
            state_digest = contract_digest(state.to_dict())
            self.connection.execute(
                "INSERT INTO interview_events VALUES (?,?,?,?,?,?,?,?)",
                (
                    event.interview_id,
                    event.sequence,
                    event.command_id,
                    event.command_digest,
                    event.previous_digest,
                    event.digest,
                    state_digest,
                    encoded_event,
                ),
            )
            self.connection.execute(
                "INSERT INTO interview_heads VALUES (?,?,?,?,?) "
                "ON CONFLICT(interview_id) DO UPDATE SET revision=excluded.revision, "
                "event_digest=excluded.event_digest,state_digest=excluded.state_digest, "
                "state_json=excluded.state_json",
                (state.interview_id, state.revision, state.digest, state_digest, state_json),
            )
            for entry in transition.memory_additions:
                if entry.interview_id != state.interview_id or entry.revision != state.revision:
                    raise InterviewJournalError(
                        "memory addition does not belong to this transition"
                    )
                self.connection.execute(
                    "INSERT INTO interview_memory_index VALUES (?,?,?,?,?)", _memory_row(entry)
                )
            return InterviewReceipt(state, True)

    def export(self, interview_id: str) -> dict[str, Any]:
        """Return a bounded private transcript export; caller owns safe publication."""
        with self._transaction():
            history = self._read(interview_id)
            value = {
                "format": "promptwitness.interview-export/v1",
                "interview_id": interview_id,
                "head": _head(history.state).to_dict(),
                "events": [event.to_dict() for event in history.events],
            }
            contract_json(value)
            return value


def verify_interview_export(
    value: Mapping[str, Any] | str | bytes, *, expected_head: SourceHead
) -> InterviewState:
    """Pure replay against a separately trusted head; never imports into a database."""
    if not isinstance(expected_head, SourceHead):
        raise ValueError("export verification requires a trusted SourceHead")
    if isinstance(value, (str, bytes)):
        value = load_interview_json(value)
    contract_json(value)
    data = _versioned(value, "interview-export", {"interview_id", "head", "events"})
    _identifier(data["interview_id"], "interview ID")
    head = SourceHead.from_dict(
        _closed(data["head"], {"interview_id", "revision", "digest"}, "head")
    )
    if head != expected_head or data["interview_id"] != head.interview_id:
        raise InterviewJournalConflict("export does not match the trusted head")
    if (
        not isinstance(data["events"], (list, tuple))
        or not 1 <= len(data["events"]) <= MAX_HISTORY_EVENTS
    ):
        raise InterviewJournalError("invalid exported event inventory")
    events = tuple(InterviewEvent.from_dict(item) for item in data["events"])
    state = replay_interview(events)
    if _head(state) != expected_head:
        raise InterviewJournalConflict("replayed export differs from the trusted head")
    return state
