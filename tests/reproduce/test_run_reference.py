"""Authored CLI control tests, never actual inference or official score claims."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from reproduce import run_reference as cli
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS
from reproduce.strict_scoring import ScoringError


def setup(tmp_path, monkeypatch):
    args = SimpleNamespace(
        store=tmp_path / "store",
        bfcl_runtime=tmp_path,
        text_runtime=tmp_path,
        output=tmp_path / "output",
        snapshot=tmp_path,
        model_python=tmp_path / "python",
        history=tmp_path / "history",
        history_sha256="a" * 64,
        ledger=tmp_path / "ledger",
        family="hotpotqa",
        model=next(iter(MODEL_REVISIONS)),
        run_id="authored-reference",
    )
    pools = {
        pool: {f"{pool}{i}": {"id": f"{pool}{i}"} for i in range(count)}
        for pool, count in (("fit", 512), ("search", 256), ("selection", 256))
    }
    monkeypatch.setattr(cli, "rows", lambda store, leaf, family: pools[leaf.split("/")[0]])
    monkeypatch.setattr(cli, "execution_bindings", lambda *args: {"data_digest": "b" * 64})
    wire = Mock()
    monkeypatch.setattr(cli, "task_wire", wire)
    history = Mock(return_value={"AUTHORED": True})
    monkeypatch.setattr(cli, "closed_history", history)
    account = Mock()
    account.usage.side_effect = [
        {"calls": 914, "input_tokens": 10, "output_tokens": 2, "gpu_hours": 4},
        {"calls": 1170, "input_tokens": 266, "output_tokens": 258, "gpu_hours": 5},
    ]
    continuation = Mock(return_value=account)
    monkeypatch.setattr(cli, "continue_ledger", continuation)
    model = MagicMock()
    process = model.return_value.__enter__.return_value
    process.profile, process.access = {"backend_version": "AUTHORED"}, {"AUTHORED": True}
    monkeypatch.setattr(cli, "PersistentModel", model)
    controller = Mock()
    monkeypatch.setattr(cli, "PipelineController", controller)
    pipeline = Mock()
    pipeline.return_value.reference.return_value = {
        "reference": {unit: 0 for unit in pools["search"]},
    }
    monkeypatch.setattr(cli, "RealPipeline", pipeline)
    return args, pools, wire, history, continuation, account, model, controller, pipeline


def test_reference_cli_keeps_entire_original_population_and_cost_history(tmp_path, monkeypatch):
    args, pools, wire, history, continued, account, model, controller, pipeline = setup(
        tmp_path, monkeypatch
    )
    report = cli.run(args)
    assert wire.call_count == 256
    assert report["units"] == report["new_calls"] == 256
    assert report["before"]["calls"] == 914 and report["after"]["calls"] == 1170
    assert report["history"] == {"AUTHORED": True}
    assert "local-only" in report["unknown_cost_count_scope"]
    assert report["M1"] == "UNFITTED" and report["Pilot"] == "NOT_RUN"
    assert report["ONLINE_PINNED"] == "NOT_VALIDATED"
    assert report["final_content_read"] is False
    mapping = pipeline.return_value.reference.call_args.args[0]
    assert set(mapping) == set(pools["search"])
    assert all(value.startswith("authored-reference:reference:") for value in mapping.values())
    spec = controller.call_args.args[2]
    assert spec["configuration"]["max_evaluation_episodes"] == 2048
    assert set(spec["search_ids"]) == set(pools["search"])
    history.assert_called_once_with(args.history, args.history_sha256)
    continued.assert_called_once_with(args.ledger, {"AUTHORED": True})
    assert model.call_args.kwargs["data_root"] == args.store
    account.close.assert_called_once()


@pytest.mark.parametrize("defect", ["smaller_population", "overlap", "unsupported_input"])
def test_reference_preflight_does_not_allocate_for_invalid_population(
    tmp_path, monkeypatch, defect
):
    args, pools, wire, history, _, _, model, _, _ = setup(tmp_path, monkeypatch)
    if defect == "smaller_population":
        pools["search"].pop("search0")
    elif defect == "overlap":
        pools["search"]["fit0"] = pools["search"].pop("search0")
    else:
        wire.side_effect = ValueError("unsupported original task")
    with pytest.raises((ScoringError, ValueError)):
        cli.run(args)
    history.assert_not_called()
    model.assert_not_called()


def test_reference_original_scoring_failure_closes_owned_model_and_ledger(tmp_path, monkeypatch):
    args, _, _, _, _, account, model, _, pipeline = setup(tmp_path, monkeypatch)
    pipeline.return_value.reference.side_effect = ScoringError("actual scorer failure witness")
    with pytest.raises(ScoringError, match="failure"):
        cli.run(args)
    assert model.return_value.__exit__.call_args.args[0] is ScoringError
    account.close.assert_called_once()
    assert not (args.output / "summary.json").exists()
