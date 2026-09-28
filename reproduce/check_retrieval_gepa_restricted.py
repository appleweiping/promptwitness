"""Run fixed native GEPA with real Landlock scorer workers on authored CIRR IDs.

This closes the in-process scorer substitution in the previous GEPA interface
checker. The optimizer/controller process itself remains trusted and can see the
authored store path: this is not full optimizer-versus-gold process isolation,
real CLIP, Qwen/OLMo, official benchmark access, or a scientific result.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.proposal_selection import AllImprovements

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import ActualScoreVector
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import retrieval_audit_bridge as bridge_module
from reproduce.check_retrieval_gepa_search import (
    POOL,
    SEARCH,
    SELECTION,
    AuthoredAdapter,
    FixedFirstBatch,
    _source_check,
)
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_gepa_restricted import (
    RestrictedGEPAQualifierScoring,
    create_authored_store,
    qualification_report_header,
    scenario_progress_receipt,
)
from reproduce.retrieval_gepa_search import RetrievalGEPAGateController, gepa_prompt
from reproduce.retrieval_work_ledger import RetrievalWorkLedger
from reproduce.strict_scoring import ScoringError

FORMAT = "promptwitness.authored-native-gepa-restricted-scorer/v1"
WIN_RANK = ("target", "reference", *POOL[2:])
LOSE_RANK = ("reference", *POOL[2:], "target")


def _rank(candidate: dict[str, str], unit: str, scenario: str) -> tuple[str, ...]:
    if scenario == "positive":
        if candidate not in ({"instruction": "authored seed"}, {"instruction": "authored retain"}):
            raise ScoringError("unknown authored positive candidate")
        return WIN_RANK if candidate["instruction"] == "authored retain" else LOSE_RANK
    if scenario == "negative":
        if candidate not in ({"instruction": "authored seed"}, {"instruction": "authored trap"}):
            raise ScoringError("unknown authored negative candidate")
        first = unit in SEARCH[:8]
        hit = first if candidate["instruction"] == "authored trap" else not first
        return WIN_RANK if hit else LOSE_RANK
    raise ScoringError("unknown authored scenario")


class RestrictedAuthoredAdapter(AuthoredAdapter):
    """Native GEPA evaluation routed through an actual restricted scorer."""

    def __init__(self, events: list[str], scoring: RestrictedGEPAQualifierScoring, scenario: str):
        super().__init__(events)
        self.scoring = scoring
        self.scenario = scenario

    def evaluate(self, batch, candidate, capture_traces=False):
        if not batch or any(not isinstance(row, dict) or set(row) != {"unit_id"} for row in batch):
            raise ScoringError("native examples must contain only query identity")
        units = [row["unit_id"] for row in batch]
        if len(set(units)) != len(units):
            raise ScoringError("native batch has repeated query identities")
        if set(units) <= self.scoring.search_ids:
            stage = "search"
        elif set(units) == self.scoring.selection_ids:
            stage = "selection"
        else:
            raise ScoringError("native batch crosses stages or truncates selection")
        rankings = {unit: _rank(candidate, unit, self.scenario) for unit in units}
        if candidate["instruction"] != "authored seed":
            self.events.append("native-child-eval")
        report = self.scoring.score(stage, rankings, purpose="native")
        scores = [float(report["observations"][unit]["primary_hit"]) for unit in units]
        self.attempts += len(batch)
        trajectories = (
            [
                {"unit_id": unit, "ranking": rankings[unit], "score": score}
                for unit, score in zip(units, scores, strict=True)
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(
            outputs=[rankings[unit] for unit in units],
            scores=scores,
            trajectories=trajectories,
            num_metric_calls=len(batch),
        )


def _run_scenario(output: Path, scenario: str) -> dict:
    output.mkdir()
    store = output / "store"
    create_authored_store(store, search_ids=SEARCH, selection_ids=SELECTION, pool=POOL)
    scratch = output / "scratch"
    scratch.mkdir()
    native_ledger = RetrievalWorkLedger(output / "native-score-work.sqlite")
    audit_ledger = RetrievalWorkLedger(output / "audit-work.sqlite")
    scoring = RestrictedGEPAQualifierScoring(
        store=store,
        scratch_root=scratch,
        ledger=native_ledger,
        search_ids=SEARCH,
        selection_ids=SELECTION,
    )
    seed = {"instruction": "authored seed"}
    child = {"instruction": "authored retain" if scenario == "positive" else "authored trap"}
    events: list[str] = []
    journal: AuditJournal | None = None
    report: dict = {"scenario": scenario, "status": "FAILED_RETAINED"}
    started = time.monotonic()
    try:
        reference_rankings = {unit: _rank(seed, unit, scenario) for unit in SEARCH}
        reference_report = scoring.score("search", reference_rankings, purpose="reference")
        reference = ActualScoreVector(
            SEARCH,
            tuple(reference_report["observations"][unit]["primary_hit"] for unit in SEARCH),
        )
        expected_reference = (
            (0,) * len(SEARCH)
            if scenario == "positive"
            else tuple(int(unit not in SEARCH[:8]) for unit in SEARCH)
        )
        if reference.scores != expected_reference:
            raise ScoringError("actual restricted reference differs from authored rule")
        execution_digest = digest(f"authored-gepa-restricted-{scenario}")
        journal = AuditJournal(
            output / "audit.sqlite",
            run_id=f"authored-gepa-restricted-{scenario}",
            reference_digest=digest(dict(zip(SEARCH, reference.scores, strict=True))),
            execution_digest=execution_digest,
            reference_episodes=len(SEARCH),
        )

        def freeze(candidate, parent_document, parent_scores):
            events.append("freeze")
            if parent_document != gepa_prompt(seed, "seed") or parent_scores != reference:
                raise ScoringError("native parent changed before restricted freeze")
            document = gepa_prompt(candidate, "child")
            plan = make_plan(
                dict(zip(SEARCH, reference.scores, strict=True)),
                candidate_digest=digest(document),
                execution_digest=execution_digest,
                fixture_seed=11,
            )
            journal.freeze(plan)

            def rank_one(unit):
                events.append("audit-rank")
                return _rank(candidate, unit, scenario)

            return (
                "child",
                document,
                RetrievalAuditBridge(
                    store=store,
                    scratch_root=scratch,
                    dataset="cirr",
                    plan=plan,
                    journal=journal,
                    work_ledger=audit_ledger,
                    unit_ids=SEARCH,
                    rank_one=rank_one,
                ),
            )

        adapter = RestrictedAuthoredAdapter(events, scoring, scenario)
        controller = RetrievalGEPAGateController(
            seed_candidate=seed,
            seed_identifier="seed",
            seed_scores=reference,
            propose=lambda _parent, _feedback, _components: dict(child),
            freeze=freeze,
            contract=ContractCheck(ContractStatus.VALID, ()),
            execution_scope="FROZEN_TABLE",
        )
        options = (
            {"batch_sampler": FixedFirstBatch()}
            if scenario == "negative"
            else {"reflection_minibatch_size": 8}
        )
        with patch.object(bridge_module, "score_rankings_restricted", scoring.audit_score):
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
                seed=11,
                max_metric_calls=1000,
                stop_callbacks=lambda state: state.i >= 0,
                run_dir=str(output / "native-run"),
                **options,
            )
        ranked = events.count("audit-rank")
        if (
            events.count("freeze") != 1
            or "native-child-eval" not in events
            or events.index("freeze") > events.index("native-child-eval")
            or result.total_metric_calls != adapter.attempts
            or audit_ledger.summary()["unresolved"]
            or native_ledger.summary()["unresolved"]
            or scoring.reference_units != len(SEARCH)
            or len(scoring.worker_pids)
            != native_ledger.summary()["completed"] + scoring.audit_units
            or any(pid == os.getpid() for pid in scoring.worker_pids)
        ):
            raise ScoringError("incomplete native or restricted scorer execution")
        if scenario == "positive":
            if (
                result.best_candidate != child
                or result.num_candidates != 2
                or result.best_score != 1.0
                or controller.search_scores(child).scores != (1,) * len(SEARCH)
                or ranked != len(SEARCH)
                or scoring.audit_units != len(SEARCH)
                or scoring.native_search_units + scoring.native_selection_units
                != result.total_metric_calls
            ):
                raise ScoringError("restricted survivor was not completed before selection")
            report["actual_complete_search_survivor"] = len(SEARCH)
        else:
            if (
                result.best_candidate != seed
                or result.num_candidates != 1
                or not 0 < ranked < len(SEARCH)
                or scoring.native_selection_units != len(SELECTION)
            ):
                raise ScoringError("harmful candidate was not rejected before selection")
            try:
                controller.search_scores(child)
            except KeyError:
                pass
            else:
                raise ScoringError("rejected candidate exposed a completed search vector")
            report["actual_complete_search_survivor"] = 0
        report.update(
            status="PASS_AUTHORED_RESTRICTED_NATIVE_GEPA",
            native_metric_attempts_excluding_audit=result.total_metric_calls,
        )
        return report
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(
            scenario_progress_receipt(
                scoring,
                native_ledger,
                audit_ledger,
                audit_rank_units=events.count("audit-rank"),
            )
        )
        report["wall_seconds"] = time.monotonic() - started
        (output / "scenario.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        native_ledger.close()
        audit_ledger.close()
        if journal is not None:
            journal.close()


def check(source: Path, output: Path) -> dict:
    _source_check(source)
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ScoringError("qualification output must be private and outside checkout")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict = qualification_report_header(FORMAT)
    try:
        report["positive"] = _run_scenario(output / "positive", "positive")
        report["negative"] = _run_scenario(output / "negative", "negative")
        report["score_worker_is_real_landlock_child"] = True
        report["status"] = "PASS_AUTHORED_NATIVE_GEPA_RESTRICTED_SCORER"
        return report
    finally:
        report["wall_seconds"] = time.monotonic() - started
        (output / "qualification.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gepa_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.gepa_source, args.output), sort_keys=True))
