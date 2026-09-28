"""Qualify the pinned MIPRO retrieval adapter on authored rankings only.

This exercises the original proposer, bootstrap, Optuna selector and the shared
strict cadence. Rank generation and scoring are in-process authored fixtures:
not CIRR data, not a restricted scorer worker, and not model/cost evidence.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
from collections import Counter
from pathlib import Path
from unittest.mock import patch

# DSPy's lazy NumPy proxy requires this import order in the fixed environment.
import numpy  # noqa: F401

# isort: split
import dspy
import optuna
from dspy.evaluate import Evaluate
from dspy.teleprompt import mipro_optimizer_v2 as native
from dspy.teleprompt import utils

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import ActualScoreVector
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import retrieval_audit_bridge as bridge_module
from reproduce.check_mipro_native import FixtureSingleResponseAdapter
from reproduce.check_mipro_search import SearchFixtureLM
from reproduce.mipro_search import mipro_prompt, strict_mipro_search
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_mipro_search import RetrievalMIPROGateEvaluator
from reproduce.retrieval_scoring import RetrievalGold, score_ranking
from reproduce.retrieval_work_ledger import RetrievalWorkLedger
from reproduce.strict_scoring import ScoringError

FORMAT = "promptwitness.authored-retrieval-mipro-native/v1"
UNITS = tuple(f"s{index}" for index in range(64))
POOL = ("reference", "target", *(f"other-{index}" for index in range(10)))


def authored_ranking(instruction: str, unit: str) -> tuple[str, ...]:
    hit = instruction == "authored retain" or (
        instruction == "authored seed" and int(unit[1:]) % 4 != 0
    )
    return ("target", "reference", *POOL[2:]) if hit else ("reference", *POOL[2:], "target")


def authored_hit(unit: str, ranking: tuple[str, ...]) -> int:
    gold = RetrievalGold(unit, "cirr", "reference", "target", category="", subset=POOL)
    return score_ranking(gold, candidate_ids=POOL, ranking=ranking).primary_hit


def authored_score_worker(_store, _scratch, dataset, stage, rankings):
    if (dataset, stage) != ("cirr", "search"):
        raise ScoringError("authored search scorer received a different dataset or stage")
    return {
        "observations": {
            unit: {"primary_hit": authored_hit(unit, ranking)} for unit, ranking in rankings.items()
        }
    }


def _scenario(output: Path, *, minibatch: bool) -> dict:
    output.mkdir()
    scratch = output / "scratch"
    scratch.mkdir()
    reference = ActualScoreVector(
        UNITS, tuple(authored_hit(unit, authored_ranking("authored seed", unit)) for unit in UNITS)
    )
    if sum(reference.scores) != 48:
        raise ScoringError("authored 75% reference changed")
    reference_map = dict(zip(UNITS, reference.scores, strict=True))
    journal = AuditJournal(
        output / "audit.sqlite",
        run_id="authored-retrieval-native",
        reference_digest=digest(reference_map),
        execution_digest=digest("authored-retrieval-execution"),
        reference_episodes=len(UNITS),
    )
    ledger = RetrievalWorkLedger(output / "work.sqlite")
    try:
        lm = SearchFixtureLM()
        student = dspy.Predict(
            dspy.Signature("task_input -> task_output", instructions="authored seed")
        )
        documents: list[tuple[str, dict, RetrievalAuditBridge]] = []

        def freeze(program):
            for identifier, document, bridge in documents:
                if mipro_prompt(program, identifier) == document:
                    return identifier, document, bridge
            identifier = f"candidate-{len(documents)}"
            document = mipro_prompt(program, identifier)
            plan = make_plan(
                reference_map,
                candidate_digest=digest(document),
                execution_digest=digest("authored-retrieval-execution"),
                fixture_seed=11,
            )
            journal.freeze(plan)  # Persist before this candidate's first rank.
            instruction = document["messages"][0]["content"]
            bridge = RetrievalAuditBridge(
                store=output,
                scratch_root=scratch,
                dataset="cirr",
                plan=plan,
                journal=journal,
                work_ledger=ledger,
                unit_ids=UNITS,
                rank_one=lambda unit, instruction=instruction: authored_ranking(instruction, unit),
            )
            documents.append((identifier, document, bridge))
            return identifier, document, bridge

        evaluator = RetrievalMIPROGateEvaluator(
            reference_document=mipro_prompt(student, "seed"),
            reference_scores=reference,
            freeze=freeze,
            contract=ContractCheck(ContractStatus.VALID, ()),
            execution_scope="FROZEN_TABLE",
        )
        optimizer = dspy.MIPROv2(
            metric=lambda example, prediction, trace=None: float(
                prediction.task_output == example.task_output
            ),
            prompt_model=lm,
            task_model=lm,
            auto=None,
            num_candidates=4,
            num_threads=1,
            max_errors=1,
            seed=11,
            init_temperature=0,
            verbose=False,
        )
        fit = [
            dspy.Example(task_input=f"fit-{index}", task_output="authored answer").with_inputs(
                "task_input"
            )
            for index in range(12)
        ]
        search = [
            dspy.Example(unit_id=unit, task_input=unit).with_inputs("task_input") for unit in UNITS
        ]
        if any("task_output" in example.toDict() for example in search):
            raise ScoringError("search gold entered native optimizer examples")
        studies = []
        original_create = optuna.create_study

        def capture(*args, **kwargs):
            study = original_create(*args, **kwargs)
            studies.append(study)
            return study

        with (
            patch.object(bridge_module, "score_rankings_restricted", authored_score_worker),
            patch.object(optuna, "create_study", capture),
            dspy.context(lm=lm, adapter=FixtureSingleResponseAdapter()),
            strict_mipro_search(evaluator),
        ):
            selected = optimizer.compile(
                student,
                trainset=fit,
                valset=search,
                num_trials=12,
                minibatch=minibatch,
                minibatch_size=8,
                minibatch_full_eval_steps=3,
                program_aware_proposer=False,
                data_aware_proposer=False,
                tip_aware_proposer=False,
                fewshot_aware_proposer=True,
            )
        states = Counter(trial.state.name for trial in studies[0].trials)
        if (
            selected.signature.instructions != "authored retain"
            or selected.score != 100
            or not states["PRUNED"]
            or not states["COMPLETE"]
            or states["FAIL"]
            or any(
                trial.value is not None
                for trial in studies[0].trials
                if trial.state.name == "PRUNED"
            )
            or ledger.summary()["unresolved"]
        ):
            raise ScoringError("native authored retrieval compile did not retain complete outcomes")
        return {
            "minibatch": minibatch,
            "trial_states": dict(states),
            "selected_instruction": selected.signature.instructions,
            "selected_score": selected.score,
            "candidate_outer_rank_and_score_attempts": ledger.summary()["attempts"],
            "reference_authored_scores_precomputed_outside_ledger": len(UNITS),
            "logical_episodes": journal.consumed_episodes(),
            "candidate_documents": len(documents),
            "fixture_lm_calls": dict(Counter(call["role"] for call in lm.calls)),
        }
    finally:
        ledger.close()
        journal.close()


def check(source: Path, output: Path) -> dict:
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != "da1736e21ffda8cc4b86379d4748b011764d507c":
        raise ScoringError("fixed original DSPy revision changed")
    for module, relative in (
        (native, "teleprompt/mipro_optimizer_v2.py"),
        (utils, "teleprompt/utils.py"),
        (inspect.getmodule(Evaluate), "evaluate/evaluate.py"),
    ):
        if Path(module.__file__).read_bytes() != (source / "dspy" / relative).read_bytes():
            raise ScoringError("installed DSPy source differs from fixed checkout")
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("qualification output must be private and outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    results = [
        _scenario(output / name, minibatch=value)
        for name, value in (("full", False), ("minibatch", True))
    ]
    report = {
        "format": FORMAT,
        "status": "PASS_AUTHORED_NATIVE_RETRIEVAL_COMPILE",
        "dspy_revision": revision,
        "source_byte_equal": True,
        "scorer": "in_process_authored_rankings_with_production_metric_not_restricted_worker",
        "search_gold_in_optimizer_examples": False,
        "real_generator_calls": 0,
        "official_benchmark_data": False,
        "scientific_admission": False,
        "full_role_physical_cost_measured": False,
        "results": results,
    }
    with (output / "qualification.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing fixed original DSPy checkout")
    parser.add_argument("output", type=Path, help="new private result directory outside repo")
    args = parser.parse_args()
    print(json.dumps(check(args.source, args.output), indent=2, allow_nan=False))
