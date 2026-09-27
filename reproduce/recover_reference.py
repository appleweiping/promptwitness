"""Score original completed reference responses without replaying model calls.

This explicit CPU recovery follows a known scorer failure. It does not replay
failed/unresolved generations, change frozen scorer/data bytes, or grant final.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduce.online_resources import closed_history
from reproduce.pipeline_controller import PipelineController, copied
from reproduce.real_pipeline import execution_bindings
from reproduce.role_pipeline import validate
from reproduce.strict_scoring import ScoringError


def original_responses(controller):
    """Use only each original, uniquely settled input-only generation."""
    controller._no_pending()
    spec = controller.spec
    vectors = [
        e for e in controller.events if e["kind"] == "vector_frozen" and e["purpose"] == "reference"
    ]
    if len(vectors) != 1 or vectors[0]["identifier"] != spec["seed_prompt"]["id"]:
        raise ScoringError("one original reference vector freeze required")
    requests = vectors[0]["requests"]
    if [r["unit"]["id"] for r in requests] != spec["search_ids"]:
        raise ScoringError("original complete reference population required")
    responses = []
    for request in requests:
        original = [
            e
            for e in controller.events
            if e["kind"] == "generation_request"
            and e.get("purpose") == "reference"
            and e["identifier"] == vectors[0]["identifier"]
            and e["unit"] == request["unit"]["id"]
        ]
        if len(original) != 1 or original[0]["request"] != request:
            raise ScoringError("original generation request differs")
        settled = [
            e
            for e in controller.events
            if e.get("call_id") == original[0]["call_id"]
            and e["kind"] in {"generation_result", "generation_failed"}
        ]
        if len(settled) != 1 or settled[0]["kind"] != "generation_result":
            raise ScoringError("failed/unresolved model generation cannot be recovered by scoring")
        response = copied(settled[0]["response"])
        validate(
            controller._message("score", spec["seed_prompt"], responses=[response]),
            "search_scorer",
            "search",
        )
        if response["id"] != request["unit"]["id"] or response["replicate"] != request["replicate"]:
            raise ScoringError("original random unit differs")
        responses.append(response)
    return responses


def run(args):
    history = closed_history(args.ledger, args.ledger_sha256)
    runtimes = {"bfcl": args.bfcl_runtime, "text": args.text_runtime}
    controller = PipelineController.resume(args.store, args.controller, runtimes)
    bindings = execution_bindings(args.store, controller.spec["family"], runtimes)
    if any(controller.spec["execution"].get(k) != v for k, v in bindings.items()):
        raise ScoringError("frozen data/scorer/tool bytes differ; do not re-score")
    responses = original_responses(controller)
    completed = [e for e in controller.events if e["kind"] == "reference_complete"]
    result = controller._one("reference_complete") if completed else controller.reference(responses)
    scorer = [e for e in controller.events if e["kind"] == "result" and e["tag"] == "reference"]
    if len(scorer) != 1:
        raise ScoringError("one original successful scorer receipt required")
    receipt = scorer[0]["report"]
    report = {
        "format": "promptwitness.original-reference-recovery/v1",
        "status": "ACTUAL_REFERENCE_SCORED_ORIGINAL_GENERATIONS_NOT_SCIENCE",
        "family": controller.spec["family"],
        "model": controller.spec["model"],
        "units": len(result["reference"]),
        "correct": sum(result["reference"].values()),
        "new_model_calls": 0,
        "new_GPU_allocation": False,
        "history": history,
        "scorer_worker_pid": receipt["worker_pid"],
        "landlock_abi": receipt["landlock_abi"],
        "original_failed_workers_retained": True,
        "original_generation_replayed": False,
        "M1": "UNFITTED",
        "Pilot": "NOT_RUN",
        "ONLINE_PINNED": "NOT_VALIDATED",
        "method_effects": "NOT_MEASURED",
        "final_content_read": False,
    }
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "controller", "bfcl-runtime", "text-runtime", "ledger", "summary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    print(json.dumps(run(parser.parse_args()), allow_nan=False))
