"""Qualify the fixed original GEPA engine on authored retrieval rankings.

The scorer is patched to an in-process authored fixture but uses the project's
actual ranking metric. No official images, real generators, scorer isolation,
physical model costs, or scientific effects are established by this command.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import gepa
from gepa.core import adapter as native_adapter
from gepa.core import engine as native_engine
from gepa.core.adapter import EvaluationBatch
from gepa.proposer.reflective_mutation import reflective_mutation as native_proposer
from gepa.strategies import acceptance as native_acceptance
from gepa.strategies import proposal_selection as native_selection
from gepa.strategies.proposal_selection import AllImprovements

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import ActualScoreVector
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import retrieval_audit_bridge as bridge_module
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_gepa_search import RetrievalGEPAGateController, gepa_prompt
from reproduce.retrieval_scoring import RetrievalGold, score_ranking
from reproduce.retrieval_work_ledger import RetrievalWorkLedger
from reproduce.strict_scoring import ScoringError

FORMAT = "promptwitness.authored-retrieval-gepa-native/v1"
REVISION = "d771eb21b5dd3228bc3f567293d2ccfc423fc900"
SEARCH = tuple(f"search-{index}" for index in range(64))
SELECTION = tuple(f"selection-{index}" for index in range(16))
POOL = ("reference", "target", *(f"other-{index}" for index in range(10)))


def _ranking(candidate: dict[str, str]) -> tuple[str, ...]:
    hit = candidate == {"instruction": "authored retain"}
    return ("target", "reference", *POOL[2:]) if hit else ("reference", *POOL[2:], "target")


def _hit(unit: str, ranking: tuple[str, ...]) -> int:
    gold = RetrievalGold(unit, "cirr", "reference", "target", category="", subset=POOL)
    return score_ranking(gold, candidate_ids=POOL, ranking=ranking).primary_hit


class AuthoredAdapter:
    propose_new_texts = None  # GEPA's custom proposer owns pre-outcome freeze.

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.attempts = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        if not batch or any(
            not isinstance(example, dict)
            or set(example) != {"unit_id"}
            or example["unit_id"] not in (*SEARCH, *SELECTION)
            for example in batch
        ):
            raise ScoringError("authored native data must contain only query identities")
        if candidate == {"instruction": "authored retain"}:
            self.events.append("native-child-eval")
        rankings = [_ranking(candidate) for _ in batch]
        scores = [
            float(_hit(example["unit_id"], ranking))
            for example, ranking in zip(batch, rankings, strict=True)
        ]
        self.attempts += len(batch)
        traces = (
            [
                {"unit_id": example["unit_id"], "ranking": ranking, "score": score}
                for example, ranking, score in zip(batch, rankings, scores, strict=True)
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(
            outputs=rankings,
            scores=scores,
            trajectories=traces,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if components_to_update != ["instruction"] or eval_batch.trajectories is None:
            raise ScoringError("authored GEPA reflection has unexpected components")
        return {
            "instruction": [
                {
                    "Inputs": {"unit_id": row["unit_id"]},
                    "Generated Outputs": str(row["ranking"]),
                    "Feedback": f"actual primary_hit={row['score']}",
                }
                for row in eval_batch.trajectories
            ]
        }


class FixedFirstBatch:
    """Make the authored regression example's native minibatch explicit."""

    def next_minibatch_ids(self, loader, state):
        del state
        return list(loader.all_ids())[:8]


class RegressionAdapter(AuthoredAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        if candidate not in ({"instruction": "authored seed"}, {"instruction": "authored trap"}):
            raise ScoringError("unexpected authored regression candidate")
        if candidate["instruction"] == "authored trap":
            self.events.append("native-trap-eval")
        else:
            self.events.append("native-parent-eval")
        outputs = []
        scores = []
        traces = []
        for example in batch:
            unit = example["unit_id"]
            hit = (unit in SEARCH[:8]) == (candidate["instruction"] == "authored trap")
            ranking = (
                ("target", "reference", *POOL[2:]) if hit else ("reference", *POOL[2:], "target")
            )
            score = float(_hit(unit, ranking))
            outputs.append(ranking)
            scores.append(score)
            traces.append({"unit_id": unit, "ranking": ranking, "score": score})
        self.attempts += len(batch)
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces if capture_traces else None,
            num_metric_calls=len(batch),
        )


def _rejected_scenario(output: Path, scorer) -> dict:
    """Exercise native rejection after an improving but harmful minibatch."""
    output.mkdir()
    scratch = output / "scratch"
    scratch.mkdir()
    ledger = RetrievalWorkLedger(output / "audit-work.sqlite")
    reference = ActualScoreVector(SEARCH, tuple(int(unit not in SEARCH[:8]) for unit in SEARCH))
    journal = AuditJournal(
        output / "audit.sqlite",
        run_id="authored-gepa-regression",
        reference_digest=digest(dict(zip(SEARCH, reference.scores, strict=True))),
        execution_digest=digest("authored-gepa-regression-execution"),
        reference_episodes=len(SEARCH),
    )
    seed = {"instruction": "authored seed"}
    events: list[str] = []
    adapter = RegressionAdapter(events)

    def freeze(candidate, parent_document, parent_scores):
        events.append("freeze")
        if parent_document != gepa_prompt(seed, "seed") or parent_scores != reference:
            raise ScoringError("regression parent identity changed")
        document = gepa_prompt(candidate, "trap")
        plan = make_plan(
            dict(zip(SEARCH, reference.scores, strict=True)),
            candidate_digest=digest(document),
            execution_digest=digest("authored-gepa-regression-execution"),
            fixture_seed=11,
        )
        journal.freeze(plan)

        def rank_one(unit):
            events.append("audit-rank")
            hit = unit in SEARCH[:8]
            return ("target", "reference", *POOL[2:]) if hit else ("reference", *POOL[2:], "target")

        bridge = RetrievalAuditBridge(
            store=output,
            scratch_root=scratch,
            dataset="cirr",
            plan=plan,
            journal=journal,
            work_ledger=ledger,
            unit_ids=SEARCH,
            rank_one=rank_one,
        )
        return "trap", document, bridge

    controller = RetrievalGEPAGateController(
        seed_candidate=seed,
        seed_identifier="seed",
        seed_scores=reference,
        propose=lambda _parent, _feedback, _components: {"instruction": "authored trap"},
        freeze=freeze,
        contract=ContractCheck(ContractStatus.VALID, ()),
        execution_scope="FROZEN_TABLE",
    )
    try:
        with patch.object(bridge_module, "score_rankings_restricted", scorer):
            result = gepa.optimize(
                seed_candidate=seed,
                trainset=[{"unit_id": unit} for unit in SEARCH],
                valset=[{"unit_id": unit} for unit in SELECTION],
                adapter=adapter,
                custom_candidate_proposer=controller.propose_new_texts,
                acceptance_criterion=controller,
                selection_strategy=AllImprovements(),
                use_merge=False,
                raise_on_exception=True,
                skip_perfect_score=False,
                batch_sampler=FixedFirstBatch(),
                seed=11,
                max_metric_calls=1000,
                stop_callbacks=lambda state: state.i >= 0,
                run_dir=str(output / "native-run"),
            )
        ranked = events.count("audit-rank")
        if (
            result.best_candidate != seed
            or result.num_candidates != 1
            or not 0 < ranked < len(SEARCH)
            or "native-trap-eval" not in events
            or events.index("freeze") > events.index("native-trap-eval")
            or ledger.summary()["unresolved"]
            or result.total_metric_calls != adapter.attempts
        ):
            raise ScoringError("native GEPA failed to reject an actually harmful proposal")
        try:
            controller.search_scores({"instruction": "authored trap"})
        except KeyError:
            pass
        else:
            raise ScoringError("native rejected child exposed a selector vector")
        return {
            "native_minibatch_before": 0,
            "native_minibatch_after": 8,
            "gate_rejected_before_full_selection_eval": True,
            "audit_rank_units_before_rejection": ranked,
            "native_metric_calls_exclude_audit": result.total_metric_calls,
            "audit_outer_rank_and_score_attempts": ledger.summary()["attempts"],
            "survivor_vector_exposed": False,
            "fixed_authored_minibatch_not_default_sampler": True,
            "scripted_proposer_calls_not_real_lm": 1,
        }
    finally:
        ledger.close()
        journal.close()


def _source_check(source: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != REVISION:
        raise ScoringError("fixed original GEPA revision changed")
    for installed, relative in (
        (inspect.getfile(gepa.optimize), "api.py"),
        (native_adapter.__file__, "core/adapter.py"),
        (native_engine.__file__, "core/engine.py"),
        (native_proposer.__file__, "proposer/reflective_mutation/reflective_mutation.py"),
        (native_acceptance.__file__, "strategies/acceptance.py"),
        (native_selection.__file__, "strategies/proposal_selection.py"),
    ):
        if Path(installed).read_bytes() != (source / "src/gepa" / relative).read_bytes():
            raise ScoringError("installed GEPA source differs from fixed checkout")


def check(source: Path, output: Path) -> dict:
    _source_check(source)
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ScoringError("qualification output must be private and outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    native_run = output / "native-run"
    ledger = RetrievalWorkLedger(output / "audit-work.sqlite")
    journal = AuditJournal(
        output / "audit.sqlite",
        run_id="authored-gepa-native",
        reference_digest=digest(dict.fromkeys(SEARCH, 0)),
        execution_digest=digest("authored-gepa-execution"),
        reference_episodes=len(SEARCH),
    )
    events: list[str] = []
    adapter = AuthoredAdapter(events)
    seed = {"instruction": "authored seed"}
    reference = ActualScoreVector(SEARCH, (0,) * len(SEARCH))
    plan_digests = []

    def authored_score_worker(_store, _scratch, dataset, stage, rankings):
        if (dataset, stage) != ("cirr", "search"):
            raise ScoringError("authored scorer received wrong stage")
        return {
            "observations": {
                unit: {"primary_hit": _hit(unit, ranking)} for unit, ranking in rankings.items()
            }
        }

    def freeze(candidate, parent_document, parent_scores):
        events.append("freeze")
        if parent_document != gepa_prompt(seed, "seed") or parent_scores != reference:
            raise ScoringError("authored GEPA parent identity changed")
        document = gepa_prompt(candidate, "child")
        plan = make_plan(
            dict(zip(parent_scores.unit_ids, parent_scores.scores, strict=True)),
            candidate_digest=digest(document),
            execution_digest=digest("authored-gepa-execution"),
            fixture_seed=11,
        )
        journal.freeze(plan)
        plan_digests.append(plan.sha256)

        def rank_one(unit):
            events.append("audit-rank")
            return _ranking(candidate)

        bridge = RetrievalAuditBridge(
            store=output,
            scratch_root=scratch,
            dataset="cirr",
            plan=plan,
            journal=journal,
            work_ledger=ledger,
            unit_ids=SEARCH,
            rank_one=rank_one,
        )
        return "child", document, bridge

    controller = RetrievalGEPAGateController(
        seed_candidate=seed,
        seed_identifier="seed",
        seed_scores=reference,
        propose=lambda _parent, _feedback, _components: {"instruction": "authored retain"},
        freeze=freeze,
        contract=ContractCheck(ContractStatus.VALID, ()),
        execution_scope="FROZEN_TABLE",
    )
    started = time.perf_counter()
    try:
        with patch.object(bridge_module, "score_rankings_restricted", authored_score_worker):
            result = gepa.optimize(
                seed_candidate=seed,
                trainset=[{"unit_id": unit} for unit in SEARCH],
                valset=[{"unit_id": unit} for unit in SELECTION],
                adapter=adapter,
                custom_candidate_proposer=controller.propose_new_texts,
                acceptance_criterion=controller,
                selection_strategy=AllImprovements(),
                use_merge=False,
                raise_on_exception=True,
                skip_perfect_score=False,
                reflection_minibatch_size=8,
                seed=11,
                max_metric_calls=1000,
                stop_callbacks=lambda state: state.i >= 0,
                run_dir=str(native_run),
            )
        positive_elapsed = time.perf_counter() - started
        if (
            result.best_candidate != {"instruction": "authored retain"}
            or result.num_candidates != 2
            or result.best_score != 1.0
            or controller.search_scores(result.best_candidate).scores != (1,) * len(SEARCH)
            or len(plan_digests) != 1
            or events.count("freeze") != 1
            or events.count("audit-rank") != len(SEARCH)
            or "native-child-eval" not in events
            or events.index("freeze") > events.index("native-child-eval")
            or ledger.summary()["unresolved"]
            or result.total_metric_calls != adapter.attempts
        ):
            raise ScoringError("authored original GEPA search did not retain complete outcomes")
        rejected = _rejected_scenario(output / "rejected", authored_score_worker)
        total_elapsed = time.perf_counter() - started
        report = {
            "format": FORMAT,
            "status": "PASS_AUTHORED_NATIVE_GEPA_RETRIEVAL_SEARCH",
            "gepa_revision": REVISION,
            "source_byte_equal": True,
            "original_engine_and_reflective_proposer": True,
            "native_selection_strategy": "AllImprovements",
            "adapted_custom_proposal_and_acceptance_hooks": True,
            "merge_disabled_to_prevent_gate_bypass": True,
            "search_units": len(SEARCH),
            "selection_units": len(SELECTION),
            "optimizer_examples_contain_gold": False,
            "freeze_precedes_first_child_outcome": True,
            "actual_complete_search_survivor": len(SEARCH),
            "native_metric_calls_exclude_audit": result.total_metric_calls,
            "native_adapter_metric_attempts": adapter.attempts,
            "audit_outer_rank_and_score_attempts": ledger.summary()["attempts"],
            "audit_logical_episodes_including_precomputed_reference": journal.consumed_episodes(),
            "authored_reference_scores_given_by_fixture_rule_not_scored": len(SEARCH),
            "positive_scripted_proposer_calls_not_real_lm": 1,
            "real_model_calls": 0,
            "gpu_hours": 0,
            "official_images_or_annotations_accessed": False,
            "restricted_scorer_worker_exercised": False,
            "positive_wall_seconds": positive_elapsed,
            "complete_checker_wall_seconds": total_elapsed,
            "native_rejected_scenario": rejected,
            "fresh_run_only": True,
        }
        (output / "qualification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report
    finally:
        ledger.close()
        journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gepa_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.gepa_source, args.output), sort_keys=True))
