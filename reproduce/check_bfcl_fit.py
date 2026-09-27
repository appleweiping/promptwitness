"""Real archived fit-response scoring in a restricted Linux fit worker.

Public stdout contains aggregate evidence only. Actual per-unit scores remain
in private scratch. This audit is neither real M1 fitting nor a Pilot.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# process_access uses python -I / runpy, not an installed reproduce package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproduce.bfcl_stage import load_staged_bfcl, staged_inventory
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, unique_rows
from reproduce.process_access import launch_role
from reproduce.strict_scoring import ScoringError, score_bfcl


def score_fit(store: Path, runtime: Path, scratch: Path) -> dict:
    if (
        os.environ.get("PW_ACCESS_ROLE") != "fit_learner"
        or os.environ.get("PW_ACCESS_STAGE") != "fit"
    ):
        raise ScoringError("fit scoring requires an already restricted fit worker")

    def read(leaf, filename="bfcl.jsonl"):
        with (store / leaf / filename).open(encoding="utf-8") as stream:
            return unique_rows(json.loads(line) for line in stream if line.strip())

    inputs, gold = read("fit/inputs"), read("fit/gold")
    if not inputs or set(inputs) != set(gold):
        raise ScoringError("fit inputs/gold/records are incomplete or inconsistent")
    models, private = [], []
    for index, (model, revision) in enumerate(MODEL_REVISIONS.items()):
        records = read("fit/records", f"bfcl-{index}.jsonl")
        if not records or not set(records) <= set(inputs):
            raise ScoringError("fit response set incomplete")
        checker, metadata = load_staged_bfcl(runtime, scratch / "official", model)
        checks = []
        for unit in sorted(records):
            record, row = records[unit], inputs[unit]
            if (
                record["model"] != model
                or record["revision"] != revision
                or record["original_scoring_executed"] is not False
            ):
                raise ScoringError("fit archive profile changed")
            score = score_bfcl(
                record["output"],
                record["status"],
                row["category"],
                row["function"],
                gold[unit]["ground_truth"],
                checker,
            )
            checks.append({"unit": unit, "score": score.value, "profile": score.profile})
        private.append({"native": metadata, "checks": checks})
        models.append(
            {
                "model": model,
                "prior_fit_responses_scored": len(checks),
                "correct": sum(row["score"] for row in checks),
            }
        )
    with (scratch / "fit-scores.json").open("x", encoding="utf-8") as stream:
        json.dump({"models": private}, stream, ensure_ascii=False, indent=2)
    return {
        "format": "promptwitness.bfcl-fit-rescore/v1",
        "status": "COMPLETED_FIT_AUDIT_ONLY",
        "evaluation_type": "real_gt",
        "fit_population": len(inputs),
        "models": models,
        "source_revision": metadata["revision"],
        "worker_role": os.environ["PW_ACCESS_ROLE"],
        "landlock_abi": int(os.environ["PW_LANDLOCK_ABI"]),
        "worker_pid": os.getpid(),
        "new_model_calls": 0,
        "new_allocated_GPU_hours": 0,
        "final_samples_or_scores_read_by_worker": False,
        "native_module": metadata["native_module"],
        "no_inference_handler_instantiated": True,
        "method_effect_or_pilot_measured": False,
        "full_scorer_isolation_online_gate": "PARTIAL_NOT_PASSED",
    }


def check(store: Path, runtime: Path, scratch: Path, *, timeout: float) -> dict:
    staged_inventory(runtime)
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        result = launch_role(
            "fit_learner",
            "fit",
            store,
            scratch,
            Path(__file__),
            [str(store), str(runtime), str(scratch), "--worker"],
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run already kills/waits its own timed-out child. Retain
        # exact captured bytes, then propagate; no score or completed receipt.
        for name, content in (("worker.stdout", exc.stdout), ("worker.stderr", exc.stderr)):
            with (scratch / name).open("xb") as stream:
                stream.write(
                    content.encode("utf-8") if isinstance(content, str) else content or b""
                )
        raise
    with (scratch / "worker.stdout").open("x", encoding="utf-8") as stream:
        stream.write(result.stdout)
    with (scratch / "worker.stderr").open("x", encoding="utf-8") as stream:
        stream.write(result.stderr)
    if result.returncode:
        raise ScoringError(
            f"restricted BFCL fit worker failed with exit {result.returncode}; see private stderr"
        )
    report = json.loads(result.stdout)
    if report["worker_pid"] == os.getpid() or report["status"] != "COMPLETED_FIT_AUDIT_ONLY":
        raise ScoringError("missing independent fit worker receipt")
    report["controller_pid"] = os.getpid()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="execution observation deadline, not experiment quota",
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(score_fit(args.store, args.runtime, args.scratch)))
    else:
        if args.output is None:
            parser.error("controller requires an exclusive result path")
        with args.output.open("x", encoding="utf-8") as stream:
            report = check(args.store, args.runtime, args.scratch, timeout=args.timeout)
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(report))
