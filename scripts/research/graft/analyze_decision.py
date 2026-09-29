"""Paired hierarchical bootstrap for decision-study records (H-fork validation).

Each bootstrap replicate resamples train rows (with replacement) for every predictor and
dev questions for the target, jointly across predictors (paired), recomputes each
predictor per edit and its Spearman rho with the dev-accuracy change, and records the
differences fork - answer, fork - fresh8 and fork - fresh_all. Runs are pooled by
averaging rho over runs within a replicate (each run keeps its own proposal pool).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from fidelity_study import spearman


def predictors_for(raw: dict, names: list[str], rows: list[int], questions: list[int]) -> tuple[dict, list[float]]:
    base_fresh = raw["base_fresh"]
    fork_rows = set(raw["fork_rows"])
    fork_index = {row: k for k, row in enumerate(raw["fork_rows"])}  # position in per-edit fork lists
    sampled_fork = [fork_index[r] for r in rows if r in fork_rows]
    first8 = [r for r in rows if r < 8]
    out: dict[str, list[float]] = {}
    for key in ("answer_exact", "answer_patch"):
        out[key] = [-statistics.mean(raw[key][n][r] for r in rows) for n in names]
    for key in ("fork_exact", "fork_patch"):
        out[key] = [-statistics.mean(raw[key][n][k] for k in sampled_fork) if sampled_fork else 0.0
                    for n in names]
    for key, subset in (("fresh8", first8), ("fresh_all", rows)):
        out[key] = [statistics.mean(raw["fresh"][n][r] - base_fresh[r] for r in subset) if subset else 0.0
                    for n in names]
    base_dev = raw["base_dev"]
    target = [statistics.mean(raw["dev"][n][q] - base_dev[q] for q in questions) for n in names]
    return out, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in args.files]
    runs = [r for r in runs if "raw" in r]
    if not runs:
        raise SystemExit("no records with raw per-row data (decision_study v3)")
    rng = random.Random(args.seed)
    keys = ("answer_exact", "answer_patch", "fork_exact", "fork_patch", "fresh8", "fresh_all")
    point: dict[str, float] = {}
    boot: dict[str, list[float]] = {k: [] for k in keys}
    diffs: dict[str, list[float]] = {"fork_patch-answer_exact": [], "fork_patch-fresh8": [],
                                     "fork_patch-fresh_all": [], "fork_exact-answer_exact": []}
    for replicate in range(args.replicates + 1):
        per_key: dict[str, list[float]] = {k: [] for k in keys}
        for run in runs:
            raw, names = run["raw"], run["edits"]
            n_rows, n_q = len(raw["base_fresh"]), len(raw["base_dev"])
            if replicate == 0:
                rows, questions = list(range(n_rows)), list(range(n_q))
            else:
                rows = [rng.randrange(n_rows) for _ in range(n_rows)]
                questions = [rng.randrange(n_q) for _ in range(n_q)]
            preds, target = predictors_for(raw, names, rows, questions)
            for k in keys:
                rho = spearman(preds[k], target)
                per_key[k].append(0.0 if rho is None else rho)
        means = {k: statistics.mean(v) for k, v in per_key.items()}
        if replicate == 0:
            point = means
            continue
        for k in keys:
            boot[k].append(means[k])
        diffs["fork_patch-answer_exact"].append(means["fork_patch"] - means["answer_exact"])
        diffs["fork_patch-fresh8"].append(means["fork_patch"] - means["fresh8"])
        diffs["fork_patch-fresh_all"].append(means["fork_patch"] - means["fresh_all"])
        diffs["fork_exact-answer_exact"].append(means["fork_exact"] - means["answer_exact"])

    def interval(values: list[float]) -> str:
        ordered = sorted(values)
        lo, hi = ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]
        return f"[{lo:+.2f}, {hi:+.2f}]"

    print(f"runs: {len(runs)} ({', '.join(r['task'] + '/' + Path(r['model_path']).parts[-3].split('--')[-1] for r in runs)})")
    for k in keys:
        print(f"{k:14s} mean rho {point[k]:+.3f}  95% CI {interval(boot[k])}")
    for k, v in diffs.items():
        print(f"{k:24s} diff {statistics.mean(v):+.3f}  95% CI {interval(v)}  P(diff>0) {sum(x > 0 for x in v) / len(v):.3f}")


if __name__ == "__main__":
    main()
