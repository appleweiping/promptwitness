"""Authored CPU ranking -> ground-truth hit -> original paired-gate check.

Hand-authored vectors are NOT image/text model embeddings or dataset scores.
The SQLite files are logical authored audit journals, not actual cost ledgers.
No network, model, GPU, benchmark data or final-test access is involved.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.gate import GateStatus, evaluate_candidate
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import complete_survivor_scores
from promptwitness.incremental.sampling import digest, make_plan
from reproduce.retrieval_scoring import RetrievalGold, rank_cosine, score_ranking


def check_case(root: Path, case: str) -> dict:
    pool = {
        "reference": (1, 0, 0),
        "target": (0, 1, 0),
        **{f"d{i}": (0, 0, 1) for i in range(8)},
    }
    units = tuple(f"authored-q-{i}" for i in range(64))

    def score(unit, vector):
        gold = RetrievalGold(unit, "cirr", "reference", "target", subset=("reference", "target"))
        return score_ranking(
            gold, candidate_ids=tuple(pool), ranking=rank_cosine(vector, pool)
        ).primary_hit

    old = {u: score(u, (0, 0, 1) if case == "eligible" else (0, 1, 0)) for u in units}
    plan = make_plan(
        old, candidate_digest=digest([case]), execution_digest=digest(pool), fixture_seed=11
    )
    journal = AuditJournal(
        root / (case + "-AUTHORED.sqlite"),
        run_id=case,
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=len(units),
    )
    queries = []

    def query(unit):
        queries.append(unit)
        if case == "failed":
            # Actual malformed ranking must fail in scorer, never return zero.
            gold = RetrievalGold(
                unit, "cirr", "reference", "target", subset=("reference", "target")
            )
            return score_ranking(gold, candidate_ids=tuple(pool), ranking=()).primary_hit
        return score(unit, (0, 1, 0) if case == "eligible" else (0, 0, 1))

    try:
        result = evaluate_candidate(
            plan,
            journal,
            query,
            contract=ContractCheck(ContractStatus.VALID, ()),
            execution_scope="FROZEN_TABLE",
        )
        expected = {
            "eligible": GateStatus.ELIGIBLE,
            "rejected": GateStatus.INELIGIBLE,
            "failed": GateStatus.INCONCLUSIVE,
        }[case]
        if result.status != expected or not 0 < len(queries) < len(units):
            raise ValueError("authored gate status/prefix differs")
        early = len(queries)
        if case == "eligible":

            def complete(unit):
                attempt = journal.reserve(plan, unit)
                actual = query(unit)
                journal.settle(plan, unit, attempt, score=actual)
                return actual

            vector = complete_survivor_scores(units, journal.observations(plan), complete)
            if vector.scores != (1,) * len(units) or len(queries) != len(set(queries)):
                raise ValueError("survivor omitted or repeated actual scores")
            if len(queries) != len(units):
                raise ValueError("survivor incomplete")
        elif case == "failed" and journal.observations(plan):
            raise ValueError("failed score imputed")
        return {
            "case": case,
            "status": result.status.value,
            "certificate_scope": result.certificate_scope,
            "population": len(units),
            "early_authored_query_attempts": early,
            "total_authored_query_attempts": len(queries),
            "observed_scores": len(journal.observations(plan)),
            "logical_episodes": journal.consumed_episodes(),
        }
    finally:
        journal.close()


def check(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cases = [check_case(root, case) for case in ("eligible", "rejected", "failed")]
    result = {
        "format": "promptwitness.icmr-retrieval-kernel-check/v0.1",
        "status": "PASS_AUTHORED_CPU_ONLY",
        "cases": cases,
        "CPU_wall_seconds": time.perf_counter() - started,
        "real_model_calls": 0,
        "new_GPU_allocations": 0,
        "image_or_text_encoder_executed": False,
        "official_native_scorer_parity_established": False,
        "real_M1_or_Pilot": False,
        "scientific_admission": False,
    }
    (root / "qualification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.output), allow_nan=False))


if __name__ == "__main__":
    main()
