"""Offline recorded-table plan/run/report; not a live optimization launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import ContractCheck, ContractStatus
from .gate import evaluate_candidate
from .journal import AuditJournal
from .sampling import AuditPlan, make_plan


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys are forbidden")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    return _decode(path.read_bytes())


def _decode(raw: bytes) -> dict[str, Any]:
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("offline audit JSON exceeds the 16 MiB input envelope")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _emit(result: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)


def _plan(args: argparse.Namespace) -> int:
    plan = make_plan(
        _read(args.reference),
        candidate_digest=args.candidate_digest,
        execution_digest=args.execution_digest,
        risk=None if args.risk is None else _read(args.risk),
        fixture_seed=args.fixture_seed,
    )
    _emit(plan.to_dict(), args.output)
    return 0


def _run(args: argparse.Namespace) -> int:
    plan = AuditPlan.from_dict(_read(args.plan))
    journal = AuditJournal(
        args.ledger,
        run_id=args.run_id,
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=plan.population,
        max_episodes=args.max_episodes,
    )
    try:
        journal.freeze(plan)  # Persist allocation before opening candidate outcomes.
        raw_outcomes = args.outcomes.read_bytes()
        journal.bind_outcome_source(plan, hashlib.sha256(raw_outcomes).hexdigest())
        outcomes = _decode(raw_outcomes)
        expected = {x for stratum in plan.strata for x in stratum.permutation}
        if set(outcomes) != expected or any(
            type(score) is not int or score not in (0, 1) for score in outcomes.values()
        ):
            raise ValueError("complete actual binary frozen-table outcomes required; no imputation")
        # This command is specifically a recorded-table mechanism tool, not a
        # claim that a static tool contract or online generation was verified.
        result = evaluate_candidate(
            plan,
            journal,
            outcomes.__getitem__,
            contract=ContractCheck(
                ContractStatus.VALID, ("recorded binary table; no dispatched tools",)
            ),
            execution_scope="FROZEN_TABLE",
        )
        payload = result.to_dict() | {
            "mode": "RECORDED_TABLE_MECHANISM_ONLY",
            "new_real_model_calls": 0,
            "logical_evaluation_episodes": journal.consumed_episodes(),
        }
        _emit(payload, args.output)
        return 0 if result.status.value == "ELIGIBLE" else 2
    finally:
        journal.close()


def _report(args: argparse.Namespace) -> int:
    # Read-only inspection cannot create/reset a database or extend a grant.
    connection = sqlite3.connect(args.ledger.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT reference_episodes,max_episodes FROM delta_runs WHERE run_id=?", (args.run_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown run")
        payload = {
            "run_id": args.run_id,
            "reference_episodes": row[0],
            "max_episodes": row[1],
            "charged_candidate_query_attempts": connection.execute(
                "SELECT count(*) FROM delta_queries WHERE run_id=?", (args.run_id,)
            ).fetchone()[0],
            "events": [
                json.loads(r[0])
                for r in connection.execute(
                    "SELECT payload FROM delta_events WHERE run_id=? ORDER BY event_id",
                    (args.run_id,),
                )
            ],
            "physical_model_calls_or_gpu_costs_inferred": False,
        }
        _emit(payload, args.output)
        return 0
    finally:
        connection.close()


def add_delta_commands(subparsers: Any) -> None:
    group = subparsers.add_parser(
        "delta", help="experimental recorded-table finite-suite plan/run/report"
    )
    commands = group.add_subparsers(dest="delta_command", required=True)
    plan = commands.add_parser("plan", help="freeze homogeneous strata and audit permutations")
    plan.add_argument("reference", type=Path)
    plan.add_argument("--candidate-digest", required=True)
    plan.add_argument("--execution-digest", required=True)
    plan.add_argument("--risk", type=Path)
    plan.add_argument(
        "--fixture-seed",
        type=int,
        help="mechanical fixture only; not live adaptive audit randomness",
    )
    plan.add_argument("--output", type=Path)
    plan.set_defaults(handler=_plan)
    run = commands.add_parser(
        "run", help="query a complete recorded oracle table after plan freeze"
    )
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--outcomes", type=Path, required=True)
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--max-episodes", type=int, default=2048)
    run.add_argument("--output", type=Path)
    run.set_defaults(handler=_run)
    report = commands.add_parser("report", help="read an existing audit ledger without mutation")
    report.add_argument("--ledger", type=Path, required=True)
    report.add_argument("--run-id", required=True)
    report.add_argument("--output", type=Path)
    report.set_defaults(handler=_report)
