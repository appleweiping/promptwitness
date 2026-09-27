"""Official Hotpot/IFTrain fit-only rescore in a process-restricted worker.

No model calls, no nonfit data, no Pilot or M1 claim. Per-ID scores are private;
only aggregate evidence is returned through stdout to the trusted controller.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproduce.check_strict_scorers import construction_rng, mechanical_checks
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, unique_rows
from reproduce.prepare_text_fit import FAMILIES
from reproduce.process_access import launch_role
from reproduce.strict_scoring import ScoringError, score_hotpotqa, score_iftrain
from reproduce.text_scorer_stage import REVISIONS, load_staged, staged_inventory


def score_fit(store: Path, runtime: Path, scratch: Path) -> dict:
    if (
        os.environ.get("PW_ACCESS_ROLE") != "fit_learner"
        or os.environ.get("PW_ACCESS_STAGE") != "fit"
    ):
        raise ScoringError("text fit scoring requires an already restricted fit worker")
    registry, bench_strict, hotpot_em = load_staged(runtime)
    fixtures = mechanical_checks(registry, bench_strict, hotpot_em)

    def read(leaf, name):
        with (store / leaf / name).open(encoding="utf-8") as stream:
            return unique_rows(json.loads(line) for line in stream if line.strip())

    aggregates, private = {}, {}
    for family in FAMILIES:
        inputs, gold = read("fit/inputs", family + ".jsonl"), read("fit/gold", family + ".jsonl")
        if not inputs or set(inputs) != set(gold):
            raise ScoringError("text fit input and annotation sets inconsistent")
        models, model_checks = [], []
        for index, (model, revision) in enumerate(MODEL_REVISIONS.items()):
            records = read("fit/records", f"{family}-{index}.jsonl")
            if not records or not set(records) <= set(inputs):
                raise ScoringError("missing text fit response set")
            checks = []
            for unit in sorted(records):
                row, annotation = records[unit], gold[unit]
                if (
                    row["model"] != model
                    or row["revision"] != revision
                    or row["original_scoring_executed"] is not False
                ):
                    raise ScoringError("changed text fit archive profile")
                with construction_rng():
                    score = (
                        score_hotpotqa(
                            row["output"], row["status"], annotation["answer"], hotpot_em
                        )
                        if family == "hotpotqa"
                        else score_iftrain(
                            row["output"],
                            row["status"],
                            annotation["instruction_id"],
                            annotation["kwargs"],
                            registry,
                        )
                    )
                checks.append({"unit": unit, "score": score.value, "profile": score.profile})
            models.append(
                {
                    "model": model,
                    "prior_fit_responses_scored": len(checks),
                    "correct": sum(row["score"] for row in checks),
                }
            )
            model_checks.append({"model": model, "checks": checks})
        aggregates[family] = {"fit_population": len(inputs), "models": models}
        private[family] = model_checks
    with (scratch / "text-fit-scores.json").open("x", encoding="utf-8") as stream:
        json.dump(private, stream, ensure_ascii=False, indent=2)
    return {
        "format": "promptwitness.text-fit-rescore/v1",
        "status": "COMPLETED_FIT_AUDIT_ONLY",
        "evaluation_type": "real_gt",
        "families": aggregates,
        "authored_mechanical_checks": len(fixtures),
        "source_revisions": REVISIONS,
        "worker_pid": os.getpid(),
        "worker_role": os.environ["PW_ACCESS_ROLE"],
        "landlock_abi": int(os.environ["PW_LANDLOCK_ABI"]),
        "new_model_calls": 0,
        "new_allocated_GPU_hours": 0,
        "final_samples_or_scores_read_by_worker": False,
        "IFBench_scope": "authored mechanics only, no final records",
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
        for name, content in (("worker.stdout", exc.stdout), ("worker.stderr", exc.stderr)):
            with (scratch / name).open("xb") as stream:
                stream.write(
                    content.encode("utf-8") if isinstance(content, str) else content or b""
                )
        raise
    for name, content in (("worker.stdout", result.stdout), ("worker.stderr", result.stderr)):
        with (scratch / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    if result.returncode:
        raise ScoringError(
            f"restricted text fit worker failed with exit {result.returncode}; see stderr"
        )
    report = json.loads(result.stdout)
    if report["worker_pid"] == os.getpid() or report["status"] != "COMPLETED_FIT_AUDIT_ONLY":
        raise ScoringError("missing independent text fit worker receipt")
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
        help="environment observation deadline, not experiment quota",
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
