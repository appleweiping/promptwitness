"""Local version registry and deterministic prompt replay."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from .adapters import prompt_to_dict
from .diff import compare_prompts
from .models import ContentBlock, DiffReport, Message, PromptDocument, ToolSpec
from .parser import parse_prompt
from .variables import inspect_variables, render_template


@dataclass(frozen=True, slots=True)
class PromptVersion:
    """One immutable stored version and its content digest."""

    prompt_id: str
    version: int
    digest: str
    created_at: str
    approved_by: str | None


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Rendered messages plus the original tool contract."""

    prompt_id: str
    version: int
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    variables: tuple[str, ...]


class PromptRegistry:
    """Transactional version registry backed by a schema-tagged SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self._connection = sqlite3.connect(str(path))
        self._closed = False
        try:
            with self._connection:
                tables = self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if not tables:
                    self._connection.executescript(
                        "CREATE TABLE prompts (prompt_id TEXT NOT NULL, version INTEGER NOT NULL, "
                        "document TEXT NOT NULL, digest TEXT NOT NULL, created_at TEXT NOT NULL, "
                        "approved_by TEXT, PRIMARY KEY(prompt_id, version));"
                        "CREATE TABLE approvals (prompt_id TEXT NOT NULL, "
                        "version INTEGER NOT NULL, "
                        "actor TEXT NOT NULL, approved_at TEXT NOT NULL, "
                        "PRIMARY KEY(prompt_id, version));"
                        "PRAGMA user_version=1;"
                    )
                if self._connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise ValueError("not a supported PromptWitness registry")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> PromptRegistry:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("prompt registry is closed")

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    @staticmethod
    def _wire(document: PromptDocument) -> tuple[str, str]:
        rendered = json.dumps(
            prompt_to_dict(document),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return rendered, hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def put(
        self, document: PromptDocument, *, expected_previous_digest: str | None = None
    ) -> PromptVersion:
        """Append a version, optionally enforcing optimistic concurrency."""
        self._ensure_open()
        if not isinstance(document, PromptDocument):
            raise TypeError("document must be a PromptDocument")
        body, digest = self._wire(document)
        latest = self._connection.execute(
            "SELECT version, digest FROM prompts WHERE prompt_id=? ORDER BY version DESC LIMIT 1",
            (document.prompt_id,),
        ).fetchone()
        if expected_previous_digest is not None and (
            latest is None or latest[1] != expected_previous_digest
        ):
            raise ValueError("expected previous prompt digest does not match latest version")
        version = int(latest[0]) + 1 if latest else 1
        created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connection:
            self._connection.execute(
                "INSERT INTO prompts(prompt_id,version,document,digest,created_at,approved_by) "
                "VALUES(?,?,?,?,?,NULL)",
                (document.prompt_id, version, body, digest, created),
            )
        return PromptVersion(document.prompt_id, version, digest, created, None)

    def versions(self, prompt_id: str) -> tuple[PromptVersion, ...]:
        """List versions oldest first without loading full documents."""
        self._ensure_open()
        rows = self._connection.execute(
            "SELECT prompt_id,version,digest,created_at,approved_by FROM prompts "
            "WHERE prompt_id=? ORDER BY version",
            (prompt_id,),
        ).fetchall()
        return tuple(PromptVersion(*row) for row in rows)

    def get(self, prompt_id: str, version: int | None = None) -> PromptDocument:
        """Load a detached native document; default is the latest version."""
        self._ensure_open()
        if version is None:
            row = self._connection.execute(
                "SELECT document FROM prompts WHERE prompt_id=? ORDER BY version DESC LIMIT 1",
                (prompt_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT document FROM prompts WHERE prompt_id=? AND version=?", (prompt_id, version)
            ).fetchone()
        if row is None:
            raise KeyError((prompt_id, version))
        return parse_prompt(json.loads(row[0]))

    def approve(self, prompt_id: str, version: int, actor: str) -> PromptVersion:
        """Record a human-readable approval for an existing version."""
        self._ensure_open()
        if not isinstance(actor, str) or not actor.strip() or any(ord(c) < 32 for c in actor):
            raise ValueError("approval actor must be a non-empty printable string")
        row = self._connection.execute(
            "SELECT digest,created_at FROM prompts WHERE prompt_id=? AND version=?",
            (prompt_id, version),
        ).fetchone()
        if row is None:
            raise KeyError((prompt_id, version))
        approved = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connection:
            self._connection.execute(
                "INSERT INTO approvals(prompt_id,version,actor,approved_at) VALUES(?,?,?,?) "
                "ON CONFLICT(prompt_id,version) DO UPDATE SET "
                "actor=excluded.actor,approved_at=excluded.approved_at",
                (prompt_id, version, actor, approved),
            )
            self._connection.execute(
                "UPDATE prompts SET approved_by=? WHERE prompt_id=? AND version=?",
                (actor, prompt_id, version),
            )
        return PromptVersion(prompt_id, version, row[0], row[1], actor)

    def diff(self, prompt_id: str, before: int, after: int) -> DiffReport:
        """Compare two stored immutable versions using PromptWitness rules."""
        return compare_prompts(self.get(prompt_id, before), self.get(prompt_id, after))

    def replay(
        self,
        prompt_id: str,
        values: Mapping[str, object],
        *,
        version: int | None = None,
        strict: bool = True,
    ) -> RenderedPrompt:
        """Render variables without evaluating expressions or calling a model."""
        document = self.get(prompt_id, version)
        rendered = tuple(
            Message(
                message.role,
                render_template(message.content, values, strict=strict),
                message.name,
                message.message_id,
                tuple(
                    ContentBlock(
                        part.type,
                        {
                            **dict(part.data),
                            "text": render_template(part.data["text"], values, strict=strict),
                        },
                    )
                    if isinstance(part.data.get("text"), str)
                    else part
                    for part in message.content_parts
                ),
            )
            for message in document.messages
        )
        names = tuple(
            sorted(
                {
                    name
                    for message in document.messages
                    for name in inspect_variables(message.content).names
                }
            )
        )
        active = version if version is not None else self.versions(prompt_id)[-1].version
        return RenderedPrompt(prompt_id, active, rendered, document.tools, names)
