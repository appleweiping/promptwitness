"""Forecast all roles from measured cost profiles; proxies do not establish G0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

FAMILIES = ("bfcl", "hotpotqa", "instruction_following")
CEILINGS = {
    "requests": 800000,
    "input_tokens": 2000000000,
    "output_tokens": 200000000,
    "allocated_gpu_hours": 1000,
}


def forecast(
    plan: dict,
    measurements: dict,
    *,
    reference_cache_valid: bool,
    mechanism_stress_fraction: float,
    optimization_stress_fraction: float,
) -> dict:
    """An explicitly assumed planning scenario, not an execution-cost guarantee."""
    models = measurements["models_envelope_phase"]
    if len(models) != 3 or len({row["model"] for row in models}) != 3:
        raise ValueError("both main models and the independent transfer model must be measured")
    if any(
        not 0 <= value <= 1 for value in (mechanism_stress_fraction, optimization_stress_fraction)
    ):
        raise ValueError("invalid context-profile mixture")
    mains = [row for row in models if not row["model"].startswith("mistralai/")]
    transfer = [row for row in models if row["model"].startswith("mistralai/")]
    if len(mains) != 2 or len(transfer) != 1 or plan["main"]["runs"] != 180:
        raise ValueError("fixed experiment/model matrix changed")
    totals = {
        "requests": measurements["completed_requests_all_g0_phases"],
        "input_tokens": measurements["input_tokens_all_g0_phases"],
        "output_tokens": measurements["output_tokens_all_g0_phases"],
        "allocated_gpu_hours": measurements["allocated_gpu_hours_all_g0_phases"],
    }
    stages = []

    def add(model, family, role, stage, count, fraction):
        profiles = {(row["family"], row["role"], row["profile"]): row for row in model["profiles"]}
        if role == "task":
            base = profiles[family, role, "base"]
            stress = profiles[family, role, "eight_demo_stress_proxy"]
        else:
            base = stress = profiles[family, role, "serial_training_context_proxy"]
        cost = {"requests": count}
        for key, field in (
            ("input_tokens", "mean_input_tokens"),
            ("output_tokens", "mean_output_tokens"),
            ("allocated_gpu_hours", "mean_warm_allocated_seconds"),
        ):
            a, b = base[field], stress[field]
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in (a, b)
            ):
                raise ValueError("missing/non-finite measured cost")
            cost[key] = (
                count
                * ((1 - fraction) * a + fraction * b)
                / (3600 if key == "allocated_gpu_hours" else 1)
            )
        for key, value in cost.items():
            totals[key] += value
        stages.append(
            {
                "model": model["model"],
                "family": family,
                "role": role,
                "stage": stage,
                "eight_demo_proxy_fraction": fraction if role == "task" else None,
                **cost,
            }
        )

    for model in mains:
        for family in FAMILIES:
            mechanism = (
                plan["mechanism"]["independent_lineages_per_cell"]
                + plan["mechanism"]["pairs_per_cell"]
            ) * plan["mechanism"]["examples_per_pair"]
            add(
                model, family, "task", "mechanism_full_tables", mechanism, mechanism_stress_fraction
            )
            reference = plan["main"]["reference_pool_per_run"]
            # References still consume each run's episode budget, even if reused.
            main = 30 * (plan["main"]["max_episodes_per_run"] - reference)
            add(
                model,
                family,
                "task",
                "main_nonreference_evaluation",
                main,
                optimization_stress_fraction,
            )
            add(
                model,
                family,
                "task",
                "full_seed_reference",
                reference * (1 if reference_cache_valid else 30),
                0,
            )
            add(
                model,
                family,
                "proposer",
                "main_all_proposer_roles",
                30 * plan["main"]["max_proposer_calls_per_run"],
                0,
            )
            add(
                model,
                family,
                "task",
                "all_selection_checkpoints",
                30 * 3 * plan["selection"]["examples"],
                optimization_stress_fraction,
            )
            add(
                model,
                family,
                "task",
                "one_final_prompt_per_run",
                30 * plan["final"]["examples"][family],
                optimization_stress_fraction,
            )
    for family in FAMILIES:
        add(
            transfer[0],
            family,
            "task",
            "frozen_prompt_transfer_only",
            plan["transfer"]["frozen_prompts_per_family"] * plan["transfer"]["examples"][family],
            optimization_stress_fraction,
        )
    # Pilot/G1 are an explicit upper resource allocation, not free validation.
    # Chosen two-family pilot uses BFCL/HotpotQA on Qwen; no final/IFBench access.
    pilot_model = next(row for row in mains if row["model"].startswith("Qwen/"))
    pilot_profiles = [
        row for row in pilot_model["profiles"] if row["family"] in ("bfcl", "hotpotqa")
    ]
    pilot_count = plan["pilot_and_g1_max_requests"]
    pilot = {
        "stage": "pilot_and_g1_all_roles",
        "requests": pilot_count,
        "input_tokens": pilot_count * max(row["mean_input_tokens"] for row in pilot_profiles),
        "output_tokens": pilot_count * max(row["mean_output_tokens"] for row in pilot_profiles),
        "allocated_gpu_hours": 30,
    }
    for key in totals:
        totals[key] += pilot[key]
    stages.append(pilot)
    # Upper planned reload count: main runs, mechanism setup, transfers and repairs.
    cold_count = 400
    cold_seconds = max(row["cold_start_seconds"] for row in models) * cold_count
    totals["allocated_gpu_hours"] += cold_seconds / 3600
    stages.append(
        {
            "stage": "planned_cold_reloads",
            "loads": cold_count,
            "allocated_gpu_hours": cold_seconds / 3600,
            "basis": (
                "400 loads at slowest measured cold start; already consumed G0 cold starts "
                "separately retained"
            ),
        }
    )
    expected_requests = (
        plan["mechanism"]["maximum_unique_model_requests"]
        + plan["main"]["maximum_unique_evaluation_model_requests"]
        + plan["main"]["maximum_proposer_requests"]
        + plan["selection"]["maximum_requests"]
        + plan["final"]["maximum_requests"]
        + plan["transfer"]["maximum_requests"]
        + pilot_count
        + measurements["completed_requests_all_g0_phases"]
        + (0 if reference_cache_valid else 6 * 29 * 256)
    )
    if totals["requests"] != expected_requests:
        raise ValueError("stage accounting disagrees with fixed experiment counts")
    reserve = max(0.2, plan["reserve_fraction"])
    with_reserve = {
        key: math.ceil(value * (1 + reserve))
        if key != "allocated_gpu_hours"
        else value * (1 + reserve)
        for key, value in totals.items()
    }
    exceeded = [key for key in CEILINGS if with_reserve[key] > CEILINGS[key]]
    return {
        "before_reserve": totals,
        "with_reserve": with_reserve,
        "ceilings": CEILINGS,
        "exceeded": exceeded,
        "within_protective_ceilings_under_assumptions": not exceeded,
        "reference_cache_valid_assumption": reference_cache_valid,
        "mechanism_stress_fraction": mechanism_stress_fraction,
        "optimization_stress_fraction": optimization_stress_fraction,
        "stages": stages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    measurements = json.loads(args.measurements.read_text(encoding="utf-8"))
    scenarios = {
        "base_only_optimistic_not_native_context": forecast(
            plan,
            measurements,
            reference_cache_valid=True,
            mechanism_stress_fraction=0,
            optimization_stress_fraction=0,
        ),
        "proposed_mixed_context_proxy": forecast(
            plan,
            measurements,
            reference_cache_valid=True,
            mechanism_stress_fraction=0.25,
            optimization_stress_fraction=1 / 3,
        ),
        "all_stress_proxy": forecast(
            plan,
            measurements,
            reference_cache_valid=True,
            mechanism_stress_fraction=1,
            optimization_stress_fraction=1,
        ),
        "no_cross_run_reference_cache": forecast(
            plan,
            measurements,
            reference_cache_valid=False,
            mechanism_stress_fraction=0.25,
            optimization_stress_fraction=1 / 3,
        ),
    }
    result = {
        "format": "promptwitness.delta.resource-forecast/v1",
        "status": "PLANNING_SCENARIOS_NOT_G0_APPROVAL",
        "g0_passed": False,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "measurements_sha256": hashlib.sha256(args.measurements.read_bytes()).hexdigest(),
        "scenarios": scenarios,
        "unresolved": [
            "Full production cache-key identity and accounting have not been implemented.",
            "Eight-demo repeated-response/proposer proxies are not accepted native optimizer "
            "contexts; do not equate proxy cost with native achieved accuracy.",
            "Instruction training examples have verifier annotations but no labeled output "
            "strings; native MIPRO cannot receive invented gold demonstrations.",
            "Actual proposal decoding/caps and grouped final pools are not yet frozen.",
            "Task output distribution may change across candidate prompts; the 20% reserve "
            "is a planning allowance, not a coverage guarantee.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                name: {key: scenario[key] for key in ("with_reserve", "exceeded")}
                for name, scenario in scenarios.items()
            },
            indent=2,
        )
    )
