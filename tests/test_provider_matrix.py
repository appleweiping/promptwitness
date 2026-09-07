from __future__ import annotations

import json

import pytest

from promptwitness import (
    ProviderMatrix,
    ProviderSpec,
    Scenario,
    load_prompt,
    load_replay_providers,
    render_matrix,
)
from promptwitness.cli import main


def _inputs(tmp_path):  # type: ignore[no-untyped-def]
    prompt = tmp_path / "prompt.json"
    prompt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "demo",
                "messages": [{"role": "user", "content": "Hello {{name}}"}],
            }
        ),
        encoding="utf-8",
    )
    scenarios = (Scenario("a", {"name": "Ada"}), Scenario("b", {"name": "Bob"}))
    rows = render_matrix(load_prompt(prompt), scenarios)
    return prompt, scenarios, rows


def test_provider_matrix_runs_replays_and_reports_digest_agreement(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt, scenarios, rows = _inputs(tmp_path)
    responses = {row.digest: {"text": row.messages[0].content} for row in rows}
    matrix = ProviderMatrix(
        (
            ProviderSpec("same", lambda row: responses[row.digest]),
            ProviderSpec("same-copy", lambda row: responses[row.digest], retries=1),
        )
    )
    report = matrix.run(load_prompt(prompt), scenarios, workers=2, scenario_workers=2)
    assert report.providers == ("same", "same-copy")
    assert report.pairwise_agreement == 1.0
    assert all(run.report.complete for run in report.runs)
    assert report.to_dict()["runs"][0]["rows"][0]["output_digest"]


def test_provider_matrix_validates_specs_and_failure_agreement(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt, scenarios, rows = _inputs(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        ProviderMatrix((ProviderSpec("x", lambda _: "a"), ProviderSpec("x", lambda _: "b")))
    with pytest.raises(ValueError, match="at least one"):
        ProviderMatrix(())
    with pytest.raises(ValueError, match="positive"):
        ProviderMatrix((ProviderSpec("x", lambda _: "a"),)).run(
            load_prompt(prompt), scenarios, workers=0
        )
    responses = {rows[0].digest: "one", rows[1].digest: "two"}
    failing = ProviderMatrix(
        (
            ProviderSpec("ok", lambda row: responses[row.digest]),
            ProviderSpec("missing", lambda _row: (_ for _ in ()).throw(RuntimeError("offline"))),
        )
    ).run(load_prompt(prompt), scenarios)
    assert failing.pairwise_agreement is None
    assert failing.runs[1].report.failed == 2


def test_provider_matrix_rejects_bad_spec_and_run_inputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt, scenarios, _ = _inputs(tmp_path)
    with pytest.raises(ValueError, match="name"):
        ProviderSpec("bad name", lambda _: "x")
    with pytest.raises(TypeError, match="callable"):
        ProviderSpec("bad", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="retries"):
        ProviderSpec("bad", lambda _: "x", retries=-1)
    matrix = ProviderMatrix((ProviderSpec("ok", lambda _: "x"),))
    with pytest.raises(TypeError, match="PromptDocument"):
        matrix.run(None, scenarios)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scenario"):
        matrix.run(load_prompt(prompt), (), scenario_workers=1)
    with pytest.raises(ValueError, match="positive"):
        matrix.run(load_prompt(prompt), scenarios, scenario_workers=0)


def test_replay_provider_loader_and_cli(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt, scenarios, rows = _inputs(tmp_path)
    providers_path = tmp_path / "providers.json"
    providers_path.write_text(
        json.dumps(
            {
                "format": "promptwitness.replay-providers.v1",
                "providers": [{"name": "replay", "responses": {row.digest: "ok" for row in rows}}],
            }
        ),
        encoding="utf-8",
    )
    assert load_replay_providers(providers_path)[0].name == "replay"
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": case.scenario_id, "values": dict(case.values)} for case in scenarios]),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "provider-matrix",
                str(prompt),
                str(scenarios_path),
                str(providers_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["providers"] == ["replay"]
    bad = tmp_path / "bad.json"
    bad.write_text('{"format":"wrong","providers":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_replay_providers(bad)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "unsupported"),
        ({"format": "promptwitness.replay-providers.v1", "providers": []}, "non-empty"),
        (
            {"format": "promptwitness.replay-providers.v1", "providers": [1]},
            "object",
        ),
        (
            {
                "format": "promptwitness.replay-providers.v1",
                "providers": [{"name": "x", "responses": [], "extra": True}],
            },
            "unknown",
        ),
        (
            {
                "format": "promptwitness.replay-providers.v1",
                "providers": [{"name": "x", "responses": []}],
            },
            "responses",
        ),
    ],
)
def test_replay_provider_loader_rejects_shapes(tmp_path, payload, message) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_replay_providers(path)
