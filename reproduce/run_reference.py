"""Generate and officially score an entire original search reference population.

This component run is not M1, a Pilot, native optimization or online acceptance.
No final data is opened and no candidate is chosen from reference outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduce.online_resources import closed_history, continue_ledger
from reproduce.persistent_model import PersistentModel
from reproduce.pipeline_controller import PipelineController
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS
from reproduce.prepare_preflight_requests import SEEDS
from reproduce.real_pipeline import RealPipeline, execution_bindings
from reproduce.role_pipeline import FAMILIES, rows
from reproduce.strict_scoring import ScoringError
from reproduce.torch_runtime import task_wire


def run(args):
    runtimes = {"bfcl": args.bfcl_runtime, "text": args.text_runtime}
    populations = {
        pool: rows(args.store, pool + "/inputs", args.family)
        for pool in ("fit", "search", "selection")
    }
    if {pool: len(units) for pool, units in populations.items()} != {
        "fit": 512,
        "search": 256,
        "selection": 256,
    }:
        raise ScoringError("original complete proposed-v2 training populations required")
    if any(
        set(populations[a]) & set(populations[b])
        for a, b in (("fit", "search"), ("fit", "selection"), ("search", "selection"))
    ):
        raise ScoringError("training populations overlap")
    bindings = execution_bindings(args.store, args.family, runtimes)
    seed = {
        "schema_version": 1,
        "id": "seed-" + args.family,
        "messages": [{"id": "instruction", "role": "system", "content": SEEDS[args.family]}],
    }
    configuration = {
        "purpose": "REAL_REFERENCE_COMPONENT_NOT_PILOT",
        "max_evaluation_episodes": 2048,
        "reference": "complete unchanged256 search units; no outcome-selected subsampling",
    }
    # No GPU is allocated for an unsupported/malformed original task input.
    for unit in populations["search"].values():
        task_wire(
            {
                "candidate": seed,
                "execution": {},
                "configuration": configuration,
                "family": args.family,
                "model": args.model,
                "unit": unit,
                "replicate": args.run_id + ":reference:" + unit["id"],
            }
        )
    history = closed_history(args.history, args.history_sha256)
    account = continue_ledger(args.ledger, history)
    before = account.usage()
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        with PersistentModel(
            args.output / "model",
            args.snapshot,
            args.model,
            account,
            args.run_id,
            model_python=args.model_python,
            data_root=args.store,
        ) as process:
            spec = {
                "run_id": args.run_id,
                "family": args.family,
                "model": args.model,
                "execution": {**process.profile, **bindings},
                "configuration": configuration,
                "seed_prompt": seed,
                "search_ids": list(populations["search"]),
                "selection_ids": list(populations["selection"]),
            }
            controller = PipelineController(args.store, args.output / "controller", spec, runtimes)
            route = RealPipeline(controller, process)
            # Runtime/model/source and all random-unit IDs are frozen before any
            # response. The replicate labels do not confer ONLINE_PINNED.
            reference = route.reference(
                {unit: args.run_id + ":reference:" + unit for unit in spec["search_ids"]}
            )
            access, execution = process.access, spec["execution"]
        after = account.usage()
        report = {
            "format": "promptwitness.real-reference-component/v1",
            "status": "ACTUAL_COMPLETE_REFERENCE_NOT_SCIENTIFIC_ACCEPTANCE",
            "family": args.family,
            "model": args.model,
            "units": len(reference["reference"]),
            "correct": sum(reference["reference"].values()),
            "scoring": "original annotations, existing restricted official worker",
            "execution": execution,
            "access": access,
            "new_calls": after["calls"] - before["calls"],
            "new_input_tokens": after["input_tokens"] - before["input_tokens"],
            "new_output_tokens": after["output_tokens"] - before["output_tokens"],
            "allocated_GPU_hours": after["gpu_hours"] - before["gpu_hours"],
            "history": history,
            "unknown_cost_count_scope": (
                "usage count is local-only; inherited disclosure is in history"
            ),
            "before": before,
            "after": after,
            "selection_generation": "NOT_RUN_THIS_COMPONENT",
            "native_optimizer": "NOT_RUN",
            "ONLINE_PINNED": "NOT_VALIDATED",
            "M1": "UNFITTED",
            "Pilot": "NOT_RUN",
            "method_effects": "NOT_MEASURED",
            "final_content_read": False,
        }
        with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        return report
    finally:
        account.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "store",
        "bfcl-runtime",
        "text-runtime",
        "output",
        "snapshot",
        "model-python",
        "history",
        "ledger",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--model", choices=MODEL_REVISIONS, required=True)
    for name in ("run-id", "history-sha256"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(parser.parse_args()), allow_nan=False))
