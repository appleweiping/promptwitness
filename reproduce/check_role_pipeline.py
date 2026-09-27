"""Authored native CPU qualification of real role IPC and controller stages.

No benchmark record, model call, real reference or M1 fit. Inputs/outputs below
are authored transport witnesses, not responses to changed prompts or results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reproduce.check_bfcl_scorer import authored_fixtures
from reproduce.pipeline_controller import PipelineController
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, write_jsonl
from reproduce.process_access import LEAVES
from reproduce.role_pipeline import FAMILIES
from reproduce.strict_scoring import ScoringError


def fixtures(family):
    if family == "bfcl":
        return [
            {
                "name": r["case"],
                "input": {
                    "category": r["category"],
                    "function": r["functions"],
                    "question": [[{"role": "user", "content": "Authored IPC fixture"}]],
                },
                "gold": {"ground_truth": r["answers"] if r["category"] != "irrelevance" else None},
                "output": r["response"],
                "expected": r["expected"],
                "candidate_output": "",
                "candidate_expected": int(r["category"] == "irrelevance"),
            }
            for r in authored_fixtures()
            if r["error"] is None
        ]
    outputs = ("The Paris!", "London", "") if family == "hotpotqa" else ("one two", "one, two", "")
    return [
        {
            "name": str(index),
            "input": {"messages": [{"role": "user", "content": "Authored IPC input only"}]},
            "gold": {"answer": "paris"}
            if family == "hotpotqa"
            else {"instruction_id": ["punctuation:no_comma"], "kwargs": [{}]},
            "output": output,
            "expected": int(index == 0),
            "candidate_output": "paris" if family == "hotpotqa" else "no comma here",
            "candidate_expected": 1,
        }
        for index, output in enumerate(outputs)
    ]


def check(root: Path, family: str, runtimes: dict[str, Path]) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    store = root / "store"
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    cases = fixtures(family)
    for stage, prefix in (("search", "s-"), ("selection", "v-")):
        write_jsonl(
            store / f"{stage}/inputs/{family}.jsonl",
            ({"id": prefix + r["name"], **r["input"]} for r in cases),
        )
        write_jsonl(
            store / f"{stage}/gold/{family}.jsonl",
            ({"id": prefix + r["name"], **r["gold"]} for r in cases),
        )

    def prompt(identifier, text):
        return {
            "schema_version": 1,
            "id": identifier,
            "messages": [{"id": "instruction", "role": "system", "content": text}],
        }

    model = next(iter(MODEL_REVISIONS))
    spec = {
        "run_id": "authored-" + family,
        "family": family,
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
        "configuration": {"purpose": "AUTHORED_IPC_ONLY_NOT_SCIENTIFIC_CONFIG_FREEZE"},
        "seed_prompt": prompt("seed", "Authored seed, not used for generation"),
        "search_ids": ["s-" + r["name"] for r in cases],
        "selection_ids": ["v-" + r["name"] for r in cases],
    }

    def responses(prefix, candidate=False):
        return [
            {
                "id": prefix + r["name"],
                "status": "completed",
                "output": r["candidate_output"] if candidate else r["output"],
                "replicate": "AUTHORED_NO_RANDOM_MODEL_EXECUTION",
                "input_tokens": None,
                "output_tokens": None,
                "allocated_seconds": None,
            }
            for r in cases
        ]

    controller = PipelineController(store, root / "controller", spec, runtimes)
    reference = controller.reference(responses("s-"))
    if reference["reference"] != {"s-" + r["name"]: r["expected"] for r in cases}:
        raise ScoringError("authored original scorer vector differs")
    controller.freeze_candidate(
        spec["seed_prompt"],
        prompt("candidate", "Authored changed text"),
        {"format": "promptwitness.delta.predictor/v1.1", "status": "UNFITTED"},
        structured=True,
    )
    prediction = controller.predict("candidate")
    if prediction["predictor_status"] != "UNFITTED" or any(
        row["regression_probability"] is not None or row["improvement_probability"] is not None
        for row in prediction["predictions"].values()
    ):
        raise ScoringError("authored unfitted predictor invented probabilities")
    candidate = controller.score_candidate("candidate", responses("s-", True))
    controller.end_search(["seed", "candidate"])
    selection = controller.selection("candidate", responses("v-", True))
    seed_selection = controller.selection("seed", responses("v-"))
    ended = controller.end_selection("seed")
    if ended["frozen_prompt"] != spec["seed_prompt"] or {
        unit: r["score"] for unit, r in seed_selection["observations"].items()
    } != {"v-" + r["name"]: r["expected"] for r in cases}:
        raise ScoringError("original seed identity or actual selection vector changed")
    for report, prefix in ((candidate, "s-"), (selection, "v-")):
        if {unit: r["score"] for unit, r in report["observations"].items()} != {
            prefix + r["name"]: r["candidate_expected"] for r in cases
        }:
            raise ScoringError("authored changed-output scorer vector differs")
    resumed = PipelineController.resume(store, root / "controller", runtimes)
    workers = [e["report"] for e in resumed.events if e["kind"] == "result"]
    pids = {r["worker_pid"] for r in workers}
    if len(pids) != 5 or os.getpid() in pids or any(r["landlock_abi"] < 1 for r in workers):
        raise ScoringError("actual independent restricted worker evidence missing")
    result = {
        "format": "promptwitness.role-pipeline-qualification/v1",
        "status": "PASS_AUTHORED_ONLY",
        "family": family,
        "units": len(cases),
        "native_score_checks": len(cases) * 4,
        "workers": 5,
        "distinct_worker_PIDs": sorted(pids),
        "controller_PID": os.getpid(),
        "landlock_ABIs": [r["landlock_abi"] for r in workers],
        "predictor": "UNFITTED_NO_IMPUTED_PROBABILITIES",
        "actual_stage_order": [e["kind"] for e in resumed.events],
        "CPU_worker_wall_seconds": sum(
            e["CPU_wall_seconds"] for e in resumed.events if e["kind"] == "result"
        ),
        "selection_ended": ended["kind"],
        "caller_chosen_seed_retained": True,
        "final_admission": ended["final_admission"],
        "new_model_calls": 0,
        "allocated_GPU_hours": 0,
        "tokens": 0,
        "paid_API_usd": 0,
        "real_training_search_reference_generated": False,
        "real_M1_or_Pilot": False,
        "online_random_unit_semantics_or_native_optimizer_admitted": False,
        "whole_scientific_admission": "PARTIAL_NOT_PASSED",
    }
    with (root / "qualification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("bfcl_runtime", type=Path)
    parser.add_argument("text_runtime", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            check(args.root, args.family, {"bfcl": args.bfcl_runtime, "text": args.text_runtime})
        )
    )
