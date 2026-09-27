"""Cost scenarios using the bounded single-unit calibration, not effect results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from reproduce.forecast_campaign import forecast


def scenarios(plan: dict, envelope: dict, calibration: dict) -> dict:
    native_models = calibration["models_native_context_phase"]
    if len(native_models) != 3 or any(len(row["profiles"]) != 6 for row in native_models):
        raise ValueError("all frozen native-context cost profiles are required")
    old_models = {row["model"]: row for row in envelope["models_envelope_phase"]}
    if set(old_models) != {row["model"] for row in native_models}:
        raise ValueError("model matrix differs between cost phases")
    consumption = calibration["consumed_all_g0_phases"]
    measured = {
        "completed_requests_all_g0_phases": consumption["requests"],
        "input_tokens_all_g0_phases": consumption["input_tokens"],
        "output_tokens_all_g0_phases": consumption["output_tokens"],
        "allocated_gpu_hours_all_g0_phases": consumption["allocated_gpu_hours"],
        "models_envelope_phase": [],
    }
    basis = []
    for model in native_models:
        profiles = {(row["family"], row["role"], row["profile"]): row for row in model["profiles"]}
        converted = []
        for family in ("bfcl", "hotpotqa", "instruction_following"):
            base = profiles[family, "task", "base"]
            context = (
                profiles[family, "task", "four_valid_demo_cost_context"]
                if family == "instruction_following"
                else base
            )
            for profile, row in (("base", base), ("eight_demo_stress_proxy", context)):
                converted.append({**row, "profile": profile})
            # BFCL typed proposer cost is not directly sampled. Use the measured
            # HotpotQA official-format fixture for a disclosed planning surrogate.
            source_family = "hotpotqa" if family == "bfcl" else family
            proposer = profiles[source_family, "proposer", "official_json_proposer_cost_fixture"]
            converted.append(
                {**proposer, "family": family, "profile": "serial_training_context_proxy"}
            )
            basis.append(
                {
                    "model": model["model"],
                    "family": family,
                    "base_task": "single-unit original input, no demonstrations",
                    "context_task": (
                        "four distinct strictly accepted fit-side demonstration cost fixture"
                        if family == "instruction_following"
                        else "base-only optimistic placeholder"
                    ),
                    "proposer": "official JSON formatter/static-summary cost fixture: "
                    + source_family,
                }
            )
        measured["models_envelope_phase"].append(
            {
                "model": model["model"],
                "cold_start_seconds": max(
                    model["cold_start_seconds"], old_models[model["model"]]["cold_start_seconds"]
                ),
                "profiles": converted,
            }
        )
    result = {
        "base_only_optimistic_context": forecast(
            plan,
            measured,
            reference_cache_valid=True,
            mechanism_stress_fraction=0,
            optimization_stress_fraction=0,
        ),
        "mipro_four_demo_instruction_context_other_tasks_base_only": forecast(
            plan,
            measured,
            reference_cache_valid=True,
            mechanism_stress_fraction=0,
            optimization_stress_fraction=1 / 3,
        ),
        "same_without_cross_run_reference_cache": forecast(
            plan,
            measured,
            reference_cache_valid=False,
            mechanism_stress_fraction=0,
            optimization_stress_fraction=1 / 3,
        ),
    }
    # This retains the earlier pressure envelope solely as a sensitivity test.
    # It is NOT silently relabeled as native four-example BFCL/HotpotQA cost.
    pressure = copy.deepcopy(measured)
    for model in pressure["models_envelope_phase"]:
        old_profiles = old_models[model["model"]]["profiles"]
        for row in model["profiles"]:
            if (
                row["role"] == "task"
                and row["profile"] == "eight_demo_stress_proxy"
                and row["family"] != "instruction_following"
            ):
                old = next(
                    item
                    for item in old_profiles
                    if item["family"] == row["family"]
                    and item["role"] == "task"
                    and item["profile"] == "eight_demo_stress_proxy"
                )
                for field in (
                    "mean_input_tokens",
                    "mean_output_tokens",
                    "mean_warm_allocated_seconds",
                ):
                    row[field] = max(row[field], old[field])
    result["bfcl_hotpot_old_eight_demo_pressure_sensitivity_not_native"] = forecast(
        plan,
        pressure,
        reference_cache_valid=True,
        mechanism_stress_fraction=0,
        optimization_stress_fraction=1 / 3,
    )
    for scenario in result.values():
        for stage in scenario["stages"]:
            if "eight_demo_proxy_fraction" in stage:
                stage["context_mixture_fraction"] = stage.pop("eight_demo_proxy_fraction")
        scenario["context_mixture_fraction"] = scenario.pop("optimization_stress_fraction")
    return {
        "format": "promptwitness.delta.calibrated-resource-forecast/v1",
        "status": "PLANNING_FORECAST_NOT_CERTIFIED_RESOURCE_BOUND_OR_G0_APPROVAL",
        "g0_passed": False,
        "scenarios": result,
        "cost_profile_basis": basis,
        "limitations": [
            "All fixed maximum stage counts and prior consumption remain charged, "
            "with 20% reserve.",
            "The 1/3 context mixture corresponds to the two MIPRO configurations among six; "
            "actual accepted demonstration counts and candidate output lengths remain uncertain.",
            "BFCL/HotpotQA four-demonstration task contexts are not directly measured. Their "
            "base-only scenario is optimistic; the old eight-demo scenario is a pressure proxy.",
            "Static-summary official-format proposer fixtures are not a full native proposal "
            "workflow. BFCL uses a disclosed HotpotQA cost surrogate.",
            "Cross-run reference reuse still needs a complete request/unit cache implementation "
            "and must preserve each run's episode charges.",
            "Sample means with a reserve are a planning forecast, not a mathematical resource "
            "bound or coverage guarantee. Shared load and candidate outputs can change cost.",
            "This file contains no task accuracy, method benefits or hypothesis-test results.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("envelope", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = scenarios(
        *[
            json.loads(path.read_text(encoding="utf-8"))
            for path in (args.plan, args.envelope, args.calibration)
        ]
    )
    result["input_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.plan, args.envelope, args.calibration)
    }
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                name: {key: row[key] for key in ("with_reserve", "exceeded")}
                for name, row in result["scenarios"].items()
            },
            indent=2,
        )
    )
