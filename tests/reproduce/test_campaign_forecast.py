"""Synthetic costs exercise accounting, not the research performance claims."""

import copy
import json
from pathlib import Path

import pytest

from reproduce.check_model_boundary_budget import check
from reproduce.forecast_calibrated_campaign import scenarios
from reproduce.forecast_campaign import FAMILIES, forecast


@pytest.fixture
def plan():
    return json.loads(
        (
            Path(__file__).parents[2] / "configs/research/campaign-cost-counts.proposed.json"
        ).read_text(encoding="utf-8")
    )


@pytest.fixture
def measurements():
    models = []
    for model in ("Qwen/test", "allenai/test", "mistralai/test"):
        profiles = []
        for family in FAMILIES:
            for role, profile, seconds in (
                ("task", "base", 1),
                ("task", "eight_demo_stress_proxy", 4),
                ("proposer", "serial_training_context_proxy", 10),
            ):
                profiles.append(
                    {
                        "family": family,
                        "role": role,
                        "profile": profile,
                        "mean_input_tokens": 100,
                        "mean_output_tokens": 10,
                        "mean_warm_allocated_seconds": seconds,
                    }
                )
        models.append({"model": model, "profiles": profiles, "cold_start_seconds": 20})
    return {
        "models_envelope_phase": models,
        "completed_requests_all_g0_phases": 700,
        "input_tokens_all_g0_phases": 70000,
        "output_tokens_all_g0_phases": 7000,
        "allocated_gpu_hours_all_g0_phases": 1,
    }


def scenario(plan, measurements, cache=True, fraction=0):
    return forecast(
        plan,
        measurements,
        reference_cache_valid=cache,
        mechanism_stress_fraction=fraction,
        optimization_stress_fraction=fraction,
    )


def test_all_roles_and_twenty_percent_reserve(plan, measurements):
    result = scenario(plan, measurements)
    assert result["before_reserve"]["requests"] == 654844
    assert result["with_reserve"]["requests"] == 785813
    assert result["within_protective_ceilings_under_assumptions"]
    assert sum(row.get("requests", 0) for row in result["stages"]) + 700 == 654844
    assert "one_final_prompt_per_run" in {row["stage"] for row in result["stages"]}
    assert "frozen_prompt_transfer_only" in {row["stage"] for row in result["stages"]}


def test_unproven_reference_cache_does_not_get_free_calls(plan, measurements):
    result = scenario(plan, measurements, cache=False)
    assert result["before_reserve"]["requests"] == 654844 + 44544
    assert "requests" in result["exceeded"]


def test_stress_cost_is_not_replaced_with_base_cost(plan, measurements):
    base, stress = scenario(plan, measurements), scenario(plan, measurements, fraction=1)
    assert (
        stress["before_reserve"]["allocated_gpu_hours"]
        > base["before_reserve"]["allocated_gpu_hours"]
    )
    assert stress["before_reserve"]["requests"] == base["before_reserve"]["requests"]


def test_reserve_cannot_be_reduced_by_configuration(plan, measurements):
    plan["reserve_fraction"] = 0
    assert scenario(plan, measurements)["with_reserve"]["requests"] == 785813


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_bad_cost_fails_closed(plan, measurements, value):
    measurements["models_envelope_phase"][0]["profiles"][0]["mean_output_tokens"] = value
    with pytest.raises(ValueError, match="measured cost"):
        scenario(plan, measurements)


def test_missing_transfer_model_fails_closed(plan, measurements):
    measurements["models_envelope_phase"].pop()
    with pytest.raises(ValueError, match="transfer model"):
        scenario(plan, measurements)


def test_changed_matrix_and_bad_mixture_fail_closed(plan, measurements):
    changed = copy.deepcopy(plan)
    changed["main"]["runs"] = 179
    with pytest.raises(ValueError, match="matrix"):
        scenario(changed, measurements)
    with pytest.raises(ValueError, match="mixture"):
        scenario(plan, measurements, fraction=1.01)


def test_inconsistent_fixed_counts_fail_closed(plan, measurements):
    plan["final"]["maximum_requests"] += 1
    with pytest.raises(ValueError, match="accounting"):
        scenario(plan, measurements)


@pytest.fixture
def calibration(measurements):
    models = []
    for model in measurements["models_envelope_phase"]:
        profiles = [
            copy.deepcopy(r)
            for r in model["profiles"]
            if r["role"] == "task" and r["profile"] == "base"
        ]
        instruction_context = copy.deepcopy(
            next(r for r in profiles if r["family"] == "instruction_following")
        )
        instruction_context.update(
            profile="four_valid_demo_cost_context", mean_warm_allocated_seconds=12
        )
        profiles.append(instruction_context)
        for family in ("hotpotqa", "instruction_following"):
            row = copy.deepcopy(
                next(
                    r
                    for r in model["profiles"]
                    if r["family"] == family and r["role"] == "proposer"
                )
            )
            row["profile"] = "official_json_proposer_cost_fixture"
            profiles.append(row)
        models.append({"model": model["model"], "profiles": profiles, "cold_start_seconds": 15})
    return {
        "models_native_context_phase": models,
        "consumed_all_g0_phases": {
            "requests": 910,
            "input_tokens": 1200000,
            "output_tokens": 200000,
            "allocated_gpu_hours": 4,
        },
    }


def test_calibration_retains_all_costs_and_context_basis(plan, measurements, calibration):
    result = scenarios(plan, measurements, calibration)
    assert result["g0_passed"] is False
    rows = result["scenarios"]
    base = rows["base_only_optimistic_context"]
    native = rows["mipro_four_demo_instruction_context_other_tasks_base_only"]
    assert base["before_reserve"]["requests"] == 655054
    assert base["with_reserve"]["requests"] == 786065
    assert (
        native["before_reserve"]["allocated_gpu_hours"]
        > base["before_reserve"]["allocated_gpu_hours"]
    )
    assert len(result["cost_profile_basis"]) == 9
    assert all("context_mixture_fraction" in row for row in rows.values())
    assert "requests" in rows["same_without_cross_run_reference_cache"]["exceeded"]


def test_pressure_profiles_not_silently_called_native(plan, measurements, calibration):
    result = scenarios(plan, measurements, calibration)["scenarios"]
    mixed = result["mipro_four_demo_instruction_context_other_tasks_base_only"]
    pressure = result["bfcl_hotpot_old_eight_demo_pressure_sensitivity_not_native"]
    assert (
        pressure["before_reserve"]["allocated_gpu_hours"]
        > mixed["before_reserve"]["allocated_gpu_hours"]
    )
    assert pressure["before_reserve"]["requests"] == mixed["before_reserve"]["requests"]


def test_missing_calibration_cell_rejected(plan, measurements, calibration):
    calibration["models_native_context_phase"][0]["profiles"].pop()
    with pytest.raises(ValueError, match="all frozen"):
        scenarios(plan, measurements, calibration)


def test_small_measured_subset_cannot_approve_g0(plan, calibration):
    calibration["models_native_context_phase"] = calibration["models_native_context_phase"][:1]
    calibration["models_native_context_phase"][0]["requests"] = 70
    result = check(plan, calibration)
    assert not result["resource_gate_failed_under_frozen_count_and_cost_assumptions"]
    assert result["g0_passed"] is False
    assert any("unmeasured" in item for item in result["costs_deliberately_not_priced"])


def test_measured_single_model_task_share_can_fail_protective_ceiling(plan, calibration):
    calibration["models_native_context_phase"] = calibration["models_native_context_phase"][:1]
    model = calibration["models_native_context_phase"][0]
    model["requests"] = 70
    for profile in model["profiles"]:
        if profile["family"] == "instruction_following" and profile["role"] == "task":
            profile["mean_warm_allocated_seconds"] = 100
    result = check(plan, calibration)
    assert "allocated_gpu_hours" in result["exceeded_by_this_planning_subset"]
    assert result["resource_gate_failed_under_frozen_count_and_cost_assumptions"]
    assert all(
        row["model"].startswith("Qwen/") for row in result["measured_main_model_task_stages"]
    )


def test_incomplete_model_boundary_cannot_supply_cost_mean(plan, calibration):
    for model in calibration["models_native_context_phase"]:
        model["requests"] = 69
    with pytest.raises(ValueError, match="completed"):
        check(plan, calibration)
