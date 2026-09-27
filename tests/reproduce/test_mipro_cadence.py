"""Authored cadence regressions; not installed optimizer/scientific evidence."""

import copy
import logging
from types import SimpleNamespace

import pytest

from reproduce.mipro_cadence import cadence_optimizer


class Pruned(Exception):
    pass


class Program:
    def __init__(self, name="seed"):
        self.name = name

    def deepcopy(self):
        return copy.deepcopy(self)

    def predictors(self):
        return [None]


class Study:
    def __init__(self):
        self.trials = []

    def add_trial(self, trial):
        self.trials.append(trial)

    def optimize(self, objective, n_trials):
        for _ in range(n_trials):
            trial = SimpleNamespace(number=len(self.trials), state="RUNNING", value=None)
            self.trials.append(trial)
            try:
                value = objective(trial)
                state = "COMPLETE"
            except Pruned:
                value, state = None, "PRUNED"
            trial.state, trial.value = state, value


def exercise(*, prune, minibatch=True, all_rejected=False, failure=False, choices=None):
    study, evaluations, chosen_pools = Study(), [], []
    choices = ["bad"] * 3 if all_rejected else choices or ["good", "good", "bad"]
    num_trials, sequence = len(choices), iter(choices)
    optuna = SimpleNamespace(
        TrialPruned=Pruned,
        logging=SimpleNamespace(WARNING=1, set_verbosity=lambda _: None),
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: None),
        trial=SimpleNamespace(
            create_trial=lambda **kwargs: SimpleNamespace(state="COMPLETE", **kwargs)
        ),
        create_study=lambda **kwargs: study,
    )

    def evaluate(size, valset, program, evaluator, rng):
        evaluations.append((program.name, size))
        if program.name == "bad":
            if failure:
                raise RuntimeError("actual evaluation failure")
            if prune:
                raise Pruned("actual gate rejection")
        return SimpleNamespace(score={"seed": 50, "good": 100, "bad": 0}[program.name])

    native = SimpleNamespace(
        logger=logging.getLogger("authored-cadence"),
        GREEN="",
        ENDC="",
        _import_optuna=lambda: optuna,
        eval_candidate_program=evaluate,
        save_candidate_program=lambda *args, **kwargs: None,
    )

    def select(program, instructions, demos, trial, logs, number):
        program.name = next(sequence)
        return [program.name], {"choice": program.name}

    owner = SimpleNamespace(
        rng=None,
        verbose=False,
        log_dir=None,
        track_stats=True,
        prompt_model_total_calls=0,
        total_calls=0,
        _select_and_insert_instructions_and_demos=select,
        _log_minibatch_eval=lambda *args: None,
        _log_normal_eval=lambda *args: None,
        _get_param_distributions=lambda *args: {},
    )

    def full(
        number,
        adjusted,
        pool,
        seen,
        evaluator,
        valset,
        logs,
        total,
        data,
        best_score,
        best_program,
        native_study,
        instructions,
        demos,
    ):
        chosen_pools.append(dict(pool))
        available = [key for key in pool if key not in seen]
        if not available:
            raise ValueError("No valid program found in param_score_dict")
        key = max(available, key=lambda k: sum(row[0] for row in pool[k]) / len(pool[k]))
        _, program, params = pool[key][0]
        actual = evaluate(len(valset), valset, program, evaluator, None).score
        data.append({"score": actual, "program": program, "full_eval": True})
        native_study.add_trial(optuna.trial.create_trial(value=actual, params=params))
        seen[key] = {"program": program, "score": actual}
        if actual > best_score:
            best_score, best_program = actual, program.deepcopy()
        return best_score, best_program, total + len(valset)

    owner._perform_full_evaluation = full
    winner = cadence_optimizer(native)(
        owner,
        Program(),
        {0: ["seed", "good", "bad"]},
        None,
        None,
        list(range(64)),
        num_trials,
        minibatch,
        8,
        3,
        11,
    )
    return winner, study, evaluations, chosen_pools


def test_prune_at_due_boundary_preserves_actual_survivor_promotion():
    control, _, control_calls, _ = exercise(prune=False)
    winner, study, calls, pools = exercise(prune=True)
    assert control.name == winner.name == "good"
    assert (
        control_calls == calls == [("seed", 64), ("good", 8), ("good", 8), ("bad", 8), ("good", 64)]
    )
    rejected = [t for t in study.trials if t.state == "PRUNED"]
    assert len(rejected) == 1 and rejected[0].value is None
    assert "bad" not in pools[0]
    assert winner.trial_logs[4] == {
        "pruned": True,
        "scheduled_full_evaluation": "actual_survivor",
        "native_trial_number": 3,
        "objective_call": 3,
    }
    assert all(row["program"].name != "bad" for row in winner.mb_candidate_programs)


def test_all_pruned_has_no_fabricated_full_candidate_or_numeric_trial():
    winner, study, calls, pools = exercise(prune=True, all_rejected=True)
    assert winner.name == "seed" and winner.score == 50
    assert calls == [("seed", 64), ("bad", 8), ("bad", 8), ("bad", 8)]
    assert pools == []
    assert [t.state for t in study.trials] == ["COMPLETE", "PRUNED", "PRUNED", "PRUNED"]
    assert all(t.value is None for t in study.trials[1:])
    assert winner.trial_logs[4]["scheduled_full_evaluation"] == "no_actual_candidate"
    assert winner.mb_candidate_programs == []


def test_full_search_preserves_actual_best_without_extra_periodic_evaluation():
    winner, study, calls, pools = exercise(prune=True, minibatch=False)
    assert winner.name == "good" and winner.score == 100
    assert pools == [] and len(calls) == 4 and all(size == 64 for _, size in calls)
    assert study.trials[-1].state == "PRUNED" and study.trials[-1].value is None


def test_evaluation_failure_is_not_gate_rejection():
    with pytest.raises(RuntimeError, match="actual evaluation failure"):
        exercise(prune=True, failure=True)


def test_empty_first_boundary_does_not_hide_final_actual_survivor():
    winner, study, calls, pools = exercise(prune=True, choices=["bad", "bad", "bad", "good"])
    assert winner.name == "good" and winner.score == 100
    assert calls[-2:] == [("good", 8), ("good", 64)] and len(pools) == 1
    assert winner.trial_logs[5]["objective_call"] == 4
    assert winner.trial_logs[5]["scheduled_full_evaluation"] == "actual_survivor"
    assert [t.value for t in study.trials if t.state == "PRUNED"] == [None] * 3


def test_already_full_combos_do_not_mask_later_prune_with_selector_failure():
    winner, study, calls, pools = exercise(
        prune=True, choices=["good", "bad", "bad", "bad", "bad", "bad"]
    )
    assert winner.name == "good" and winner.score == 100
    assert calls.count(("good", 64)) == 1 and len(pools) == 1
    assert winner.trial_logs[8]["scheduled_full_evaluation"] == "no_unseen_actual_candidate"
    assert [t.state for t in study.trials].count("PRUNED") == 5
    assert all(t.value is None for t in study.trials if t.state == "PRUNED")
