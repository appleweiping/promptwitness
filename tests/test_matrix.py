import pytest

from promptwitness import (
    ContentBlock,
    Message,
    PromptDocument,
    Scenario,
    compare_matrices,
    render_matrix,
    save_matrix,
)


def document() -> PromptDocument:
    return PromptDocument("demo", (Message("user", "Hello {{name}}"),))


def test_render_matrix_is_deterministic_and_diffable() -> None:
    scenarios = (Scenario("a", {"name": "Ada"}), Scenario("b", {"name": "Grace"}))
    first = render_matrix(document(), scenarios)
    second = render_matrix(document(), scenarios)
    assert first == second
    assert first[0].messages[0].content == "Hello Ada"
    assert first[0].variables == ("name",)
    assert not any(item.changed for item in compare_matrices(first, second))
    changed = render_matrix(PromptDocument("demo", (Message("user", "Hi {{name}}"),)), scenarios)
    assert all(item.changed for item in compare_matrices(first, changed))


def test_render_matrix_renders_text_inside_multimodal_blocks() -> None:
    source = PromptDocument(
        "vision",
        (
            Message(
                "user",
                "Describe {{name}}",
                content_parts=(
                    ContentBlock("input_text", {"text": "Describe {{name}}"}),
                    ContentBlock("image_url", {"image_url": {"url": "https://example.test/a"}}),
                ),
            ),
        ),
    )
    row = render_matrix(source, (Scenario("ada", {"name": "Ada"}),))[0]
    assert row.messages[0].content_parts[0].data["text"] == "Describe Ada"
    assert row.messages[0].content_parts[1].data["image_url"]["url"] == "https://example.test/a"


def test_scenario_validation_and_missing_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        render_matrix(document(), (Scenario("a", {}), Scenario("a", {})))
    with pytest.raises(KeyError):
        render_matrix(document(), (Scenario("a", {}),))
    with pytest.raises(ValueError, match="scenario_id"):
        Scenario("", {})
    with pytest.raises(ValueError, match="tags"):
        Scenario("a", {}, ("x", "x"))


def test_matrix_artifact_round_trip_and_tamper_detection(tmp_path) -> None:
    rows = render_matrix(document(), (Scenario("a", {"name": "Ada"}),))
    path = tmp_path / "matrix.json"
    artifact = save_matrix(rows, str(path))
    assert artifact == artifact.load(str(path))
    loaded = artifact.load(str(path))
    assert loaded.rows[0].messages[0].content == "Hello Ada"
    payload = path.read_text(encoding="utf-8").replace(artifact.digest, "0" * 64)
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        artifact.load(str(path))
