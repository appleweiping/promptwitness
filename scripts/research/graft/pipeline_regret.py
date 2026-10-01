"""What is a shortlist worth when its candidates are verified anyway? (zero-GPU analysis)

On each validity pool (decision_study record: per-edit fresh greedy correctness on the 24
training rows, per-edit held-out correctness on 200 questions) this simulates one round of
the shortlist-then-verify loop used by GReaTer and by the Stage C search:

  shortlist the top-mu edits by a predictor (scored on the 24 rows), verify each by fresh
  greedy accuracy on the minibatch (training rows 0-7, as in the search), accept the best
  if it beats the incumbent's minibatch accuracy (ties: no edit), else keep the incumbent.

The realized gain is the held-out change of the accepted edit (0 if none). Predictors:
GReaTer's answer loss (exact, patched), fresh accuracy on 24 rows, |answer-loss change|
(magnitude only), a uniformly random shortlist (exact expectation over all mu-subsets) and
verify-all (no shortlist). Also reported per pool: pool-level tests over pools and the
rank-normalized wrong-row/right-row split of the answer-loss signal (each row's changes are
ranked across edits before averaging, which removes scale differences between rows).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from fidelity_study import spearman


def minibatch(raw: dict, name: str, rows: list[int]) -> float:
    return statistics.mean(raw["fresh"][name][r] for r in rows)


def realized(raw: dict, names: list[str], shortlist: list[str], rows: list[int], target: dict[str, float],
             margin: int = 0) -> float:
    """Held-out change of the accepted edit; acceptance needs more than ``margin`` net rows."""
    base = sum(raw["base_fresh"][r] for r in rows)
    scores = {n: sum(raw["fresh"][n][r] for r in rows) for n in shortlist}
    best = max(shortlist, key=lambda n: (scores[n], -names.index(n)))
    return target[best] if scores[best] > base + margin else 0.0


def grid(run: dict) -> dict[str, float]:
    """Verify-all over all edits with 8 / 16 / 24 verification rows and net-row margins 0-3."""
    raw, names = run["raw"], run["edits"]
    target = {n: statistics.mean(d - b for d, b in zip(raw["dev"][n], raw["base_dev"])) for n in names}
    out = {}
    for size in (8, 16, 24):
        for margin in (0, 1, 2, 3):
            out[f"verify_all_rows{size}_margin{margin}"] = realized(raw, names, names, list(range(size)), target, margin)
    return out


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def analyze(run: dict, mu: int) -> dict:
    raw, names = run["raw"], run["edits"]
    rows_all = list(range(len(raw["base_fresh"])))
    verify_rows = [r for r in rows_all if r < 8]
    base_dev = raw["base_dev"]
    target = {n: statistics.mean(d - b for d, b in zip(raw["dev"][n], base_dev)) for n in names}
    preds = {
        "answer_exact": {n: -statistics.mean(raw["answer_exact"][n]) for n in names},
        "answer_patch": {n: -statistics.mean(raw["answer_patch"][n]) for n in names},
        "answer_magnitude": {n: statistics.mean(abs(v) for v in raw["answer_exact"][n]) for n in names},
        "fresh_all": {n: minibatch(raw, n, rows_all) for n in names},
    }
    out: dict = {"oracle_best": max(target.values()), "mean_edit": statistics.mean(target.values())}
    for key, pred in preds.items():
        shortlist = sorted(names, key=lambda n: (-pred[n], names.index(n)))[:mu]
        out[key] = realized(raw, names, shortlist, verify_rows, target)
    subsets = list(itertools.combinations(names, mu))
    out["random"] = statistics.mean(realized(raw, names, list(s), verify_rows, target) for s in subsets)
    out["verify_all"] = realized(raw, names, names, verify_rows, target)
    # Rank-normalized answer-loss signal on wrong vs right rows.
    dev_vec = [target[n] for n in names]
    for label, value in (("wrong", 0), ("right", 1)):
        rows = [r for r in rows_all if raw["base_fresh"][r] == value]
        if len(rows) < 3:
            continue
        per_row = {r: ranks([-raw["answer_exact"][n][r] for n in names]) for r in rows}
        rank_score = [statistics.mean(per_row[r][i] for r in rows) for i in range(len(names))]
        mean_score = [-statistics.mean(raw["answer_exact"][n][r] for r in rows) for n in names]
        out[f"answer_{label}_rows_rho"] = spearman(mean_score, dev_vec)
        out[f"answer_{label}_rows_rank_rho"] = spearman(rank_score, dev_vec)
        out[f"{label}_rows"] = len(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = {}
    for path in args.files:
        run = json.loads(path.read_text(encoding="utf-8"))
        if "raw" not in run or run.get("edit_kind") == "token":
            continue
        results[path.stem] = analyze(run, args.mu)
    keys = ("answer_exact", "answer_patch", "answer_magnitude", "fresh_all", "random", "verify_all", "oracle_best")
    print(f"{'pool':55s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
    for name, res in results.items():
        print(f"{name:55s} " + " ".join(f"{100 * res[k]:12.2f}" for k in keys))
    pooled = {k: statistics.mean(r[k] for r in results.values()) for k in keys}
    print(f"{'mean (points of held-out accuracy)':55s} " + " ".join(f"{100 * pooled[k]:12.2f}" for k in keys))
    comparisons = {}
    for a, b in (("answer_exact", "random"), ("answer_patch", "random"), ("fresh_all", "random"),
                 ("answer_magnitude", "random"), ("verify_all", "random"), ("answer_patch", "answer_exact")):
        diffs = [r[a] - r[b] for r in results.values()]
        sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
        t = statistics.mean(diffs) / (sd / len(diffs) ** 0.5) if sd and sd == sd and sd > 0 else float("nan")
        comparisons[f"{a}-{b}"] = {"mean": statistics.mean(diffs), "pools_better": sum(d > 0 for d in diffs),
                                   "pools_worse": sum(d < 0 for d in diffs), "t_over_pools": t}
        print(f"  {a} - {b}: {100 * statistics.mean(diffs):+.2f} points; better in "
              f"{comparisons[f'{a}-{b}']['pools_better']}, worse in {comparisons[f'{a}-{b}']['pools_worse']} "
              f"of {len(diffs)} pools; t = {t:+.2f}")
    grids = {}
    for path in args.files:
        run = json.loads(path.read_text(encoding="utf-8"))
        if "raw" in run and run.get("edit_kind") != "token":
            grids[path.stem] = grid(run)
    print("verify-all: mean realized held-out change (points) by verification rows x acceptance margin (net rows)")
    for size in (8, 16, 24):
        cells = []
        for margin in (0, 1, 2, 3):
            key = f"verify_all_rows{size}_margin{margin}"
            vals = [g[key] for g in grids.values()]
            cells.append(f"m{margin} {100 * statistics.mean(vals):+5.2f} (+{sum(v > 0 for v in vals)}/-{sum(v < 0 for v in vals)})")
        print(f"  rows {size:2d}: " + "   ".join(cells))
    print("rank-normalized answer-loss signal (rho with held-out change): wrong rows / right rows")
    for name, res in results.items():
        cells = []
        for label in ("wrong", "right"):
            if f"answer_{label}_rows_rho" in res:
                cells.append(f"{label} n={res[f'{label}_rows']:2d} mean {res[f'answer_{label}_rows_rho']:+.2f} "
                             f"rank {res[f'answer_{label}_rows_rank_rho']:+.2f}")
        print(f"  {name:55s} " + " | ".join(cells))
    if args.json:
        args.json.write_text(json.dumps({"pools": results, "pooled": pooled, "comparisons": comparisons}, indent=1),
                             encoding="utf-8")


if __name__ == "__main__":
    main()
