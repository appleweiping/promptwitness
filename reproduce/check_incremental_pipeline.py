"""Authored-only native CPU qualification of the incremental controller route.

No real dataset, inference, M1, online randomness, native optimizer engine or
method result. Canned execution and token counters live in a separate authored
ledger; they must NEVER be imported into the actual cross-version cost ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptwitness.incremental.budget import ResourceLedger, ResourceLimit
from promptwitness.incremental.gate import GateStatus
from reproduce.check_role_pipeline import fixtures
from reproduce.incremental_pipeline import IncrementalPipeline
from reproduce.pipeline_controller import PipelineController
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, write_jsonl
from reproduce.process_access import LEAVES
from reproduce.strict_scoring import ScoringError


def check_case(root: Path, case: str, runtimes: dict[str, Path]) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    store = root / "store"
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    fixture = fixtures("hotpotqa")[0]
    search = [f"s-{i}" for i in range(64)]
    for stage, units in (("search", search), ("selection", ["v"])):
        write_jsonl(
            store / f"{stage}/inputs/hotpotqa.jsonl", ({"id": u, **fixture["input"]} for u in units)
        )
        write_jsonl(
            store / f"{stage}/gold/hotpotqa.jsonl", ({"id": u, **fixture["gold"]} for u in units)
        )

    def prompt(identifier):
        return {
            "schema_version": 1,
            "id": identifier,
            "messages": [
                {
                    "id": "instruction",
                    "role": "system",
                    "content": "Authored fixture; not used for generation",
                }
            ],
        }

    def response(unit, output):
        return {
            "id": unit,
            "status": "completed",
            "output": output,
            "replicate": "AUTHORED_NO_MODEL_RANDOMNESS",
            "input_tokens": 12,
            "output_tokens": 3,
            "allocated_seconds": None,
        }

    model = next(iter(MODEL_REVISIONS))
    spec = {
        "run_id": "authored-" + case,
        "family": "hotpotqa",
        "model": model,
        "execution": {
            "model_revision": MODEL_REVISIONS[model],
            "tokenizer_revision": "AUTHORED",
            "backend_version": "NO_INFERENCE_AUTHORED",
            **{
                k: "a" * 64
                for k in (
                    "backend_config_digest",
                    "template_digest",
                    "decoding_digest",
                    "scorer_digest",
                    "data_digest",
                    "tool_environment_digest",
                )
            },
        },
        "configuration": {
            "purpose": "AUTHORED_ONLY_NOT_SCIENTIFIC_FREEZE",
            "max_evaluation_episodes": 2048,
        },
        "seed_prompt": prompt("seed"),
        "search_ids": search,
        "selection_ids": ["v"],
    }
    controller = PipelineController(store, root / "controller", spec, runtimes)
    old = "London" if case == "eligible" else "paris"
    controller.reference([response(u, old) for u in search])
    controller.freeze_candidate(
        spec["seed_prompt"],
        prompt("candidate"),
        {"format": "promptwitness.delta.predictor/v1.1", "status": "UNFITTED"},
        structured=True,
    )
    controller.predict("candidate")
    ceiling = ResourceLimit(800000, 2000000000, 200000000, 1000)
    ledger = ResourceLedger(
        root / "AUTHORED_NOT_REAL_COST.sqlite",
        historical_usage={"calls": 0, "input_tokens": 0, "output_tokens": 0, "gpu_hours": 0},
        historical_digest="a" * 64,
        global_limit=ceiling,
        stage_limits={"AUTHORED": ceiling},
        gpu_uuid="GPU-AUTHORED-NO-ALLOCATION",
    )
    route = IncrementalPipeline(
        controller, ledger, resource_stage="AUTHORED", input_cap=32768, output_cap=64
    )
    plan = route.freeze(
        "candidate",
        replicates={u: "AUTHORED_NO_MODEL_RANDOMNESS" for u in search},
        execution_scope="FROZEN_TABLE",
        fixture_seed=7,
    )
    queried = []

    def execute(request):
        queried.append(request["unit"]["id"])
        if case == "failed":
            raise RuntimeError("authored executor failure, no model call")
        return response(queried[-1], "paris" if case == "eligible" else "London")

    gate = route.evaluate("candidate", execute)
    expected = {
        "eligible": GateStatus.ELIGIBLE,
        "rejected": GateStatus.INELIGIBLE,
        "failed": GateStatus.INCONCLUSIVE,
    }[case]
    if gate.status != expected or len(queried) >= 64:
        raise ScoringError("authored gate status/early prefix differs")
    early = len(queried)
    if case == "eligible":
        vector = route.complete_survivor("candidate", execute)
        if vector.scores != (1,) * 64 or len(queried) != len(set(queried)) or len(queried) != 64:
            raise ScoringError("native survivor did not complete actual unknown units only")
        controller.end_search(["seed", "candidate"])
        controller.selection("seed", [response("v", old)])
        controller.selection("candidate", [response("v", "paris")])
    else:
        if case == "failed":
            if (
                route.journal.observations(plan)
                or route.evaluate("candidate", execute).status != expected
            ):
                raise ScoringError("failed observation zero-filled or automatically retried")
            if len(queried) != 1:
                raise ScoringError("failed random unit silently regenerated")
        controller.end_search(["seed"])
        controller.selection("seed", [response("v", old)])
    ended = controller.end_selection("seed")
    resumed = PipelineController.resume(store, controller.directory, runtimes)
    reports = [e["report"] for e in resumed.events if e["kind"] == "result"]
    pids = {r["worker_pid"] for r in reports}
    if (
        len(pids) != len(reports)
        or os.getpid() in pids
        or any(r["landlock_abi"] < 1 for r in reports)
    ):
        raise ScoringError("actual restricted independent workers missing")
    result = {
        "case": case,
        "gate": gate.to_dict(),
        "population": 64,
        "early_authored_executor_attempts": early,
        "total_authored_executor_attempts": len(queried),
        "logical_episodes": route.journal.consumed_episodes(),
        "authored_ledger_not_real_resource_usage": ledger.usage(),
        "observed_actual_scores": len(route.journal.observations(plan)),
        "workers": len(reports),
        "worker_PIDs": sorted(pids),
        "controller_PID": os.getpid(),
        "landlock_ABIs": [r["landlock_abi"] for r in reports],
        "CPU_worker_wall_seconds": sum(
            e["CPU_wall_seconds"] for e in resumed.events if e["kind"] == "result"
        ),
        "seed_choice_retained": ended["frozen_prompt"] == spec["seed_prompt"],
        "final_admission": ended["final_admission"],
    }
    route.close()
    ledger.close()
    return result


def check(root: Path, runtimes: dict[str, Path]) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    result = {
        "format": "promptwitness.incremental-pipeline-qualification/v1",
        "status": "PASS_AUTHORED_ONLY",
        "family": "hotpotqa",
        "cases": [
            check_case(root / case, case, runtimes) for case in ("eligible", "rejected", "failed")
        ],
        "new_model_calls": 0,
        "allocated_GPU_hours": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "paid_API_usd": 0,
        "M1": "UNFITTED",
        "Pilot": "NOT_RUN",
        "real_online_or_native_optimizer_engine_admitted": False,
        "whole_scientific_admission": "PARTIAL_NOT_PASSED",
        "AUTHORED_ledger_must_not_be_imported_into_real_history": True,
    }
    with (root / "qualification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("bfcl_runtime", type=Path)
    parser.add_argument("text_runtime", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.root, {"bfcl": args.bfcl_runtime, "text": args.text_runtime})))
