"""Prepare the last bounded G0 calibration, without search/final scoring."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from reproduce.prepare_preflight_requests import SEEDS


def validate_context_audit(accepted: dict, fit_ids: set[str]) -> list[str]:
    """Only distinct, strictly accepted fit responses may become cost demos."""
    if (
        accepted.get("format") != "promptwitness.delta.cost-demonstration-audit/v1"
        or accepted.get("source_commit") != "99b1ee970490a2d0d5664663eb0b1acc410b944c"
        or accepted.get("scoped_to_all_prior_fit_responses") is not True
        or accepted.get("final_test_accessed") is not False
        or accepted.get("new_model_calls") != 0
        or accepted.get("language_detector_seed") != 11
    ):
        raise ValueError("unsupported demonstration audit provenance")
    checks = accepted["fit_only_units"]
    ids = [row["unit"] for row in checks]
    if (
        len(ids) != len(set(ids))
        or not set(ids) <= fit_ids
        or any(type(row["strict_all_constraints_pass"]) is not bool for row in checks)
    ):
        raise ValueError("invalid fit-only strict checks")
    passing = [row["unit"] for row in checks if row["strict_all_constraints_pass"]]
    demo_ids = accepted["first_four_accepted_for_cost_context"]
    if accepted["accepted_count"] != len(passing) or demo_ids != passing[:4]:
        raise ValueError("passing checks and selected demonstrations disagree")
    if len(demo_ids) != 4 or len(set(demo_ids)) != 4:
        raise ValueError(
            "four distinct verified fit responses are required; no repetition or invention"
        )
    return demo_ids


def task_messages(family: str, row: dict) -> list[dict]:
    if family == "bfcl":
        text = json.dumps(
            {"question": row["question"], "functions": row["function"]}, ensure_ascii=False
        )
    elif family == "hotpotqa":
        context = "\n\n".join(
            f"{title}: {''.join(sentences)}"
            for title, sentences in zip(
                row["context"]["title"], row["context"]["sentences"], strict=True
            )
        )
        text = f"Context:\n{context}\n\nQuestion: {row['question']}"
    elif family == "instruction_following":
        if len(row["messages"]) != 1 or row["messages"][0]["role"] != "user":
            raise ValueError("preserve original input; unsupported multi-message cost fixture")
        return copy.deepcopy(row["messages"])
    else:
        raise ValueError("unsupported family")
    return [{"role": "user", "content": text}]


def prepare(source: Path, pools_path: Path, accepted_path: Path, response_path: Path) -> list[dict]:
    import pyarrow.parquet as parquet
    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.propose.grounded_proposer import generate_instruction_class

    pools = json.loads(pools_path.read_text(encoding="utf-8"))
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    if (
        accepted["pool_proposal_sha256"] != hashlib.sha256(pools_path.read_bytes()).hexdigest()
        or accepted["prior_response_file_sha256"]
        != hashlib.sha256(response_path.read_bytes()).hexdigest()
        or accepted["scoped_to_all_prior_fit_responses"] is not True
    ):
        raise ValueError("changed fit context provenance")
    fit_ids = {row["id"] for row in pools["instruction_following"]["pools"]["fit"]}
    demo_ids = validate_context_audit(accepted, fit_ids)
    bfcl = {}
    for path in [
        source / "bfcl-simple-python.jsonl",
        *sorted((source / "bfcl-offline").glob("*.json")),
    ]:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            bfcl[row["id"]] = row
    hotpot = {
        row["id"]: row
        for name in ("hotpot-train-0.parquet", "hotpot-train-1.parquet")
        for row in parquet.read_table(source / name).to_pylist()
    }
    instruction = {
        row["key"]: row for row in parquet.read_table(source / "if-train.parquet").to_pylist()
    }
    responses = {
        row["id"].split(":", 1)[1]: row
        for line in response_path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))["family"] == "instruction_following"
        and row["role"] == "task"
        and row.get("profile", "base") == "base"
    }
    instruction_demos = []
    for unit in demo_ids:
        response = responses[unit]
        if (
            response["status"] != "completed"
            or response["scoring_executed"] is not False
            or response["model"] != "Qwen/Qwen3.5-9B"
            or response["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
            or not response["output"].strip()
        ):
            raise ValueError("invalid accepted prior response")
        instruction_demos.extend(
            [
                *task_messages("instruction_following", instruction[unit]),
                {"role": "assistant", "content": response["output"]},
            ]
        )
    requests = []
    for family, table, cap in (
        ("bfcl", bfcl, 1024),
        ("hotpotqa", hotpot, 64),
        ("instruction_following", instruction, 4096),
    ):
        ordered = sorted(
            pools[family]["pools"]["search"],
            key=lambda row: hashlib.sha256(("native-context-v1:" + row["id"]).encode()).hexdigest(),
        )
        if family == "bfcl":
            chosen = [
                row
                for category, count in (
                    ("simple_python", 3),
                    ("multiple", 3),
                    ("parallel", 3),
                    ("parallel_multiple", 4),
                    ("irrelevance", 3),
                )
                for row in [item for item in ordered if item["category"] == category][:count]
            ]
        else:
            chosen = ordered[:16]
        if len(chosen) != 16 or any(row["id"] in demo_ids for row in chosen):
            raise ValueError("search coverage or fit/search isolation broken")
        for metadata in chosen:
            for profile in (
                ("base", "four_valid_demo_cost_context")
                if family == "instruction_following"
                else ("base",)
            ):
                requests.append(
                    {
                        "id": f"{family}:{profile}:{metadata['id']}",
                        "family": family,
                        "role": "task",
                        "profile": profile,
                        "messages": [
                            {"role": "system", "content": SEEDS[family]},
                            *(instruction_demos if profile != "base" else []),
                            *task_messages(family, table[metadata["id"]]),
                        ],
                        "max_new_tokens": cap,
                    }
                )
    # Use the actual pinned official proposer signature/JSON formatter, not
    # invented predicted instructions. Static summaries here are cost fixtures,
    # not a claim to have executed the full native data-aware proposal workflow.
    generator = generate_instruction_class()
    adapter = JSONAdapter(use_native_function_calling=False)
    for family in ("instruction_following", "hotpotqa"):
        demos = (
            instruction_demos
            if family == "instruction_following"
            else [
                message
                for metadata in pools["hotpotqa"]["pools"]["fit"][:4]
                for message in [
                    *task_messages("hotpotqa", hotpot[metadata["id"]]),
                    {"role": "assistant", "content": hotpot[metadata["id"]]["answer"]},
                ]
            ]
        )
        messages = adapter.format(
            generator.signature,
            [],
            {
                "dataset_description": "Training-side cost fixture: " + family,
                "program_code": "Predict(task_input -> task_output)",
                "program_description": "One LM task call with a fixed external task interface.",
                "module": "task_predictor",
                "module_description": "Return one task response.",
                "task_demos": json.dumps(demos, ensure_ascii=False),
                "previous_instructions": "No previous candidate instruction or score is used.",
                "basic_instruction": SEEDS[family],
                "tip": "Preserve the required task interface and every input requirement.",
            },
        )
        requests.append(
            {
                "id": f"typed-proposer:{family}",
                "family": family,
                "role": "proposer",
                "profile": "official_json_proposer_cost_fixture",
                "messages": messages,
                "max_new_tokens": 2048,
            }
        )
    for family, profile in (
        ("bfcl", "base"),
        ("hotpotqa", "base"),
        ("instruction_following", "base"),
        ("instruction_following", "four_valid_demo_cost_context"),
    ):
        original = next(
            row for row in requests if row["family"] == family and row["profile"] == profile
        )
        repeated = copy.deepcopy(original)
        repeated["repeat_of"] = original["id"]
        repeated["id"] = "repeat:" + original["id"]
        requests.append(repeated)
    if len(requests) != 70 or len({row["id"] for row in requests}) != 70:
        raise ValueError("fixed all-role accounting changed")
    return requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("pools", type=Path)
    parser.add_argument("accepted", type=Path)
    parser.add_argument("responses", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = prepare(args.source, args.pools, args.accepted, args.responses)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "requests": len(rows),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "method_effects_or_final_scores_measured": False,
            },
            indent=2,
        )
    )
