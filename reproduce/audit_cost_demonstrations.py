"""Audit four preselected fit-side responses, not a task-accuracy experiment."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import random
import sys
from pathlib import Path


def audit(
    source: Path,
    upstream: Path,
    pools_path: Path,
    *,
    fit_context: bool = False,
    response_file: Path | None = None,
) -> dict:
    import nltk
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    from langdetect import DetectorFactory

    nltk.data.path.insert(0, str(source / "nltk-data"))
    sys.path.insert(0, str(upstream))
    from open_instruct.IFEvalG import instructions_registry

    plan = json.loads(pools_path.read_text(encoding="utf-8"))
    reserved = plan["proposer_context_training_inputs_reserved_for_fit"]["instruction_following"]
    fit_ids = {row["id"] for row in plan["instruction_following"]["pools"]["fit"]}
    if len(reserved) != 4 or not set(reserved) <= fit_ids:
        raise ValueError("only the four preselected fit-side cost-demo inputs may be scored")
    response_path = response_file or source / "infra1/qwen-infra1.jsonl"
    rows = [json.loads(line) for line in response_path.read_text(encoding="utf-8").splitlines()]
    ids = (
        sorted(
            {
                row["id"].split(":", 1)[1]
                for row in rows
                if row["family"] == "instruction_following"
                and row["role"] == "task"
                and row.get("profile", "base") == "base"
                and row["id"].split(":", 1)[1] in fit_ids
            },
            key=lambda unit: hashlib.sha256(
                ("native-cost-context-v1:" + unit).encode()
            ).hexdigest(),
        )
        if fit_context
        else reserved
    )
    table = parquet.read_table(source / "if-train.parquet", columns=["key", "ground_truth"])
    table = table.filter(arrow.compute.is_in(table["key"], value_set=arrow.array(ids)))
    annotations = {
        row["key"]: ast.literal_eval(row["ground_truth"])[0] for row in table.to_pylist()
    }
    responses = {
        row["id"].split(":", 1)[1]: row
        for row in rows
        if row["id"] in {"instruction_following:" + unit for unit in ids}
    }
    if set(annotations) != set(ids) or set(responses) != set(ids):
        raise ValueError("missing cost-demonstration annotation or prior output")
    checks = []
    random.seed(11)
    DetectorFactory.seed = 11
    for unit in ids:
        annotation, response = annotations[unit], responses[unit]
        if (
            response["status"] != "completed"
            or response["scoring_executed"] is not False
            or response["model"] != "Qwen/Qwen3.5-9B"
            or response["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        ):
            raise ValueError("changed prior response provenance")
        if len(annotation["instruction_id"]) != len(annotation["kwargs"]):
            raise ValueError("misaligned constraints")
        flags = []
        for identifier, arguments in zip(
            annotation["instruction_id"], annotation["kwargs"], strict=True
        ):
            checker = instructions_registry.INSTRUCTION_DICT[identifier](identifier)
            arguments = {
                key: value for key, value in (arguments or {}).items() if value is not None
            }
            checker.build_description(**copy.deepcopy(arguments))
            flags.append(
                bool(response["output"].strip()) and checker.check_following(response["output"])
            )
        checks.append({"unit": unit, "strict_all_constraints_pass": all(flags)})
    return {
        "format": "promptwitness.delta.cost-demonstration-audit/v1",
        "purpose": (
            "Check whether the frozen repeated unscored-response stress context is a "
            "native accepted demonstration proxy."
        ),
        "source_commit": "99b1ee970490a2d0d5664663eb0b1acc410b944c",
        "language_detector_seed": 11,
        "fit_only_units": checks,
        "scoped_to_all_prior_fit_responses": fit_context,
        "first_four_accepted_for_cost_context": [
            row["unit"] for row in checks if row["strict_all_constraints_pass"]
        ][:4]
        if fit_context
        else None,
        "accepted_count": sum(row["strict_all_constraints_pass"] for row in checks),
        "prior_response_file_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
        "sentence_resource_sha256": hashlib.sha256(
            (source / "nltk-data/tokenizers/punkt_tab.zip").read_bytes()
        ).hexdigest(),
        "pool_proposal_sha256": hashlib.sha256(pools_path.read_bytes()).hexdigest(),
        "new_model_calls": 0,
        "final_test_accessed": False,
        "baseline_accuracy_or_method_effect_measured": False,
        "limitations": [
            "The candidate audit unit set is determined only by previously frozen IDs, "
            "fit membership and hash order, before looking at the scores.",
            "For the optional cost context, taking the first four passing responses resembles "
            "native bootstrap acceptance but is not a reproduction of its shuffle/teacher. "
            "All failed checks and prior inference costs remain recorded.",
            "Not the actual randomized native MIPRO bootstrap sequence; acceptance here does "
            "not establish the final demonstration distribution or optimization benefit.",
            "Binary training checks require all constraints, rather than upstream fractional "
            "reward. No thinking-text removal or loose whitespace transformation is applied.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("pools", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-context", action="store_true")
    parser.add_argument("--response-file", type=Path)
    args = parser.parse_args()
    result = audit(
        args.source,
        args.upstream,
        args.pools,
        fit_context=args.fit_context,
        response_file=args.response_file,
    )
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("accepted_count", "new_model_calls")}, indent=2))
