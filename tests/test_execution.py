import pytest

from promptwitness import Message, PromptDocument, Scenario, ScenarioExecutor


def document() -> PromptDocument:
    return PromptDocument("demo", (Message("user", "Hello {{name}}"),))


def test_executor_preserves_order_and_retries_provider_failures() -> None:
    calls: dict[str, int] = {}

    def provider(row):
        calls[row.scenario_id] = calls.get(row.scenario_id, 0) + 1
        if row.scenario_id == "b" and calls[row.scenario_id] == 1:
            raise RuntimeError("temporary")
        return {"text": row.messages[0].content.upper()}

    report = ScenarioExecutor().run(
        document(),
        (Scenario("a", {"name": "Ada"}), Scenario("b", {"name": "Bob"})),
        provider,
        workers=2,
        retries=1,
    )
    assert report.complete and report.succeeded == 2
    assert [row.scenario_id for row in report.rows] == ["a", "b"]
    assert report.rows[1].attempts == 2 and calls["b"] == 2
    assert all(row.output_digest for row in report.rows)


def test_executor_reports_failures_or_raises_when_requested() -> None:
    def provider(_row):
        raise RuntimeError("no service")

    report = ScenarioExecutor().run(document(), (Scenario("a", {"name": "Ada"}),), provider)
    assert not report.complete and report.failed == 1
    assert "RuntimeError" in (report.rows[0].error or "")
    with pytest.raises(ValueError, match="provider failed"):
        ScenarioExecutor().run(
            document(), (Scenario("a", {"name": "Ada"}),), provider, fail_fast=True
        )
