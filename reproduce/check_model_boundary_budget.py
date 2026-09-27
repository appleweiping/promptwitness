"""Check measured completed-model cost without imputing missing model profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from reproduce.summarize_cost_envelope import finite_nonnegative


def check(plan: dict, measurements: dict) -> dict:
    models = measurements["models_native_context_phase"]
    if not models or any(model["requests"] != 70 for model in models):
        raise ValueError("only completed, fully reported 70-request model boundaries are supported")
    if plan["main"]["runs"] != 180 or plan["mechanism"]["pairs_per_cell"] != 24:
        raise ValueError("fixed scientific matrix changed")
    totals = {"input_tokens": 0.0, "output_tokens": 0.0, "allocated_gpu_hours": 0.0}
    stages = []
    for model in models:
        # A third-family transfer-only measurement cannot be charged as a main model.
        if model["model"].startswith("mistralai/"):
            continue
        profiles = {(p["family"], p["role"], p["profile"]): p for p in model["profiles"]}
        for family in ("bfcl", "hotpotqa", "instruction_following"):
            base = profiles[family, "task", "base"]
            context = (
                profiles[family, "task", "four_valid_demo_cost_context"]
                if family == "instruction_following"
                else base
            )
            for stage, count, fraction in (
                (
                    "mechanism_full_tables",
                    (
                        plan["mechanism"]["independent_lineages_per_cell"]
                        + plan["mechanism"]["pairs_per_cell"]
                    )
                    * plan["mechanism"]["examples_per_pair"],
                    0,
                ),
                (
                    "main_nonreference_evaluation",
                    30
                    * (
                        plan["main"]["max_episodes_per_run"]
                        - plan["main"]["reference_pool_per_run"]
                    ),
                    1 / 3,
                ),
                (
                    "full_seed_reference_with_optimistic_cache",
                    plan["main"]["reference_pool_per_run"],
                    0,
                ),
                (
                    "all_selection_checkpoints",
                    30
                    * len(plan["selection"]["budget_checkpoints"])
                    * plan["selection"]["examples"],
                    1 / 3,
                ),
                ("one_final_prompt_per_run", 30 * plan["final"]["examples"][family], 1 / 3),
            ):
                cost = {}
                for field, key in (
                    ("mean_input_tokens", "input_tokens"),
                    ("mean_output_tokens", "output_tokens"),
                    ("mean_warm_allocated_seconds", "allocated_gpu_hours"),
                ):
                    a = finite_nonnegative(base[field], field)
                    b = finite_nonnegative(context[field], field)
                    cost[key] = (
                        count
                        * ((1 - fraction) * a + fraction * b)
                        / (3600 if key == "allocated_gpu_hours" else 1)
                    )
                    totals[key] += cost[key]
                stages.append(
                    {
                        "model": model["model"],
                        "family": family,
                        "stage": stage,
                        "requests": count,
                        "four_demo_instruction_mixture_fraction": fraction
                        if family == "instruction_following"
                        else 0,
                        **cost,
                    }
                )
    if not stages:
        raise ValueError("at least one completed main-model measurement is required")
    reserve = max(0.2, plan["reserve_fraction"])
    with_reserve = {
        key: (
            value * (1 + reserve)
            if key == "allocated_gpu_hours"
            else math.ceil(value * (1 + reserve))
        )
        for key, value in totals.items()
    }
    ceilings = {"input_tokens": 2000000000, "output_tokens": 200000000, "allocated_gpu_hours": 1000}
    exceeded = [key for key in ceilings if with_reserve[key] > ceilings[key]]
    return {
        "format": "promptwitness.delta.model-boundary-budget-check/v1",
        "status": "MEASURED_SUBSET_PLANNING_CHECK_NOT_A_CERTIFIED_LOWER_BOUND",
        "g0_passed": False,
        "measured_main_model_task_stages": stages,
        "before_reserve_measured_subset": totals,
        "with_reserve_measured_subset": with_reserve,
        "protective_ceilings": ceilings,
        "exceeded_by_this_planning_subset": exceeded,
        "resource_gate_failed_under_frozen_count_and_cost_assumptions": bool(exceeded),
        "costs_deliberately_not_priced": [
            "All unmeasured main-model stages and transfer-model stages",
            "All proposer calls, cold reloads, pilot/G1, retries and future failures",
            "Already consumed G0 costs, which remain in the separate consumption ledger",
            "BFCL/HotpotQA four-demonstration task contexts beyond their base-only placeholder",
        ],
        "limitations": [
            "This forecasts the declared maximum episode allowances, not actual consumption "
            "or proof that every possible run spends its maximum.",
            "A non-exceeded subset cannot approve G0 or price missing stages as zero; a full "
            "feasible forecast and remaining readiness checks are still required.",
            "Instruction context uses measured four-demo cost for two MIPRO configurations "
            "of six. Other task contexts use a disclosed base-only placeholder.",
            "Cost means and 20% reserve are not statistical bounds. Shared load, output "
            "distributions and legitimate early stopping can change actual costs.",
            "Failure applies to this model/execution/count plan, not every alternate protocol. "
            "No scientific effectiveness claim is measured.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(
        *[json.loads(p.read_text(encoding="utf-8")) for p in (args.plan, args.measurements)]
    )
    result["input_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.plan, args.measurements)
    }
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("with_reserve_measured_subset", "exceeded_by_this_planning_subset")
            },
            indent=2,
        )
    )
