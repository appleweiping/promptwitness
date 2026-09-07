import pytest

from promptwitness import Message, PromptDocument, Scenario, compare_matrices, render_matrix


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


def test_scenario_validation_and_missing_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        render_matrix(document(), (Scenario("a", {}), Scenario("a", {})))
    with pytest.raises(KeyError):
        render_matrix(document(), (Scenario("a", {}),))
    with pytest.raises(ValueError, match="scenario_id"):
        Scenario("", {})
    with pytest.raises(ValueError, match="tags"):
        Scenario("a", {}, ("x", "x"))
