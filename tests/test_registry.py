import json
import sqlite3
from pathlib import Path

import pytest

from promptwitness import Message, PromptDocument, PromptRegistry, ToolSpec


def document(content: str = "Hello {{ name }}") -> PromptDocument:
    return PromptDocument(
        "welcome",
        (Message("system", "You are concise."), Message("user", content)),
        (ToolSpec("lookup", "Look up a value", {"type": "object", "properties": {}}, ()),),
    )


def test_versions_approval_diff_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "prompts.sqlite"
    with PromptRegistry(path) as registry:
        first = registry.put(document())
        registry.put(document("Hi {{ name }}"), expected_previous_digest=first.digest)
        assert [item.version for item in registry.versions("welcome")] == [1, 2]
        assert registry.get("welcome", 1) == document()
        approved = registry.approve("welcome", 2, "reviewer@example.test")
        assert approved.approved_by == "reviewer@example.test"
        assert registry.versions("welcome")[-1].approved_by == "reviewer@example.test"
        report = registry.diff("welcome", 1, 2)
        assert report.warning_count == 1
        rendered = registry.replay("welcome", {"name": "Ada"})
        assert rendered.version == 2 and rendered.messages[-1].content == "Hi Ada"
        assert rendered.tools[0].name == "lookup" and rendered.variables == ("name",)
        with pytest.raises(KeyError):
            registry.get("missing")


def test_optimistic_concurrency_and_stable_storage(tmp_path: Path) -> None:
    with PromptRegistry(tmp_path / "registry.db") as registry:
        first = registry.put(document())
        with pytest.raises(ValueError, match="digest"):
            registry.put(document("other"), expected_previous_digest="0" * 64)
        second = registry.put(document("other"), expected_previous_digest=first.digest)
        assert second.version == 2
        with pytest.raises(ValueError, match="actor"):
            registry.approve("welcome", 1, "bad\nactor")


def test_strict_replay_and_closed_registry(tmp_path: Path) -> None:
    registry = PromptRegistry(tmp_path / "registry.db")
    registry.put(document())
    with pytest.raises(KeyError, match="missing"):
        registry.replay("welcome", {})
    loose = registry.replay("welcome", {}, strict=False)
    assert "{{ name }}" in loose.messages[-1].content
    registry.close()
    registry.close()
    with pytest.raises(ValueError, match="closed"):
        registry.versions("welcome")


def test_rejects_unrelated_database_without_modifying_it(tmp_path: Path) -> None:
    path = tmp_path / "other.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE important(value TEXT)")
        connection.execute("INSERT INTO important VALUES ('keep')")
    with pytest.raises(ValueError, match="supported"):
        PromptRegistry(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM important").fetchall() == [("keep",)]
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)


def test_invalid_json_document_is_not_accepted(tmp_path: Path) -> None:
    registry = PromptRegistry(tmp_path / "registry.db")
    path = tmp_path / "tamper.json"
    path.write_text(
        '{"schema_version":1,"id":"x","messages":[],"tools":[],"metadata":{}}', encoding="utf-8"
    )
    assert json.loads(path.read_text(encoding="utf-8"))["id"] == "x"
    with pytest.raises(KeyError):
        registry.get("x")
    registry.close()
