"""Stage C analysis, prespecified in the plan amendment of 2026-09-30 (before any outcome).

Inputs: the ``select_and_test.py`` output (dev selection and one test read per search run)
and the fixed-prompt baselines (``evaluate_prompts.py``) for the same models. For each
model and task, a method's accuracy is the mean over its seeds. Comparisons are paired at
the task level: the difference of two methods is averaged over seeds within a task and
then over tasks; its interval comes from a hierarchical bootstrap that resamples seeds
within each task and test questions (paired across methods, which share the test set).
Also reported: wins/ties/losses over tasks, accuracy on the clean subset (test rows that
GReaTer's optimization never saw), and measured search-plus-selection seconds per run.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from promptwitness import graft_tasks

FIXED = ("zs_cot", "greater_init", "greater_published")
PAIRS = (("patch", "random"), ("patch", "exact"), ("patch", "textgrad"), ("patch", "greater_init"),
         ("patch", "greater_published"), ("random", "greater_init"), ("textgrad", "greater_init"))


def model_key(path: str) -> str:
    name = path.lower()
    return "llama3" if "llama" in name else "gemma2" if "gemma" in name else "qwen3" if "qwen" in name else name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selected", type=Path, help="select_and_test.py output")
    parser.add_argument("--baselines", type=Path, nargs="+", default=[], help="evaluate_prompts outputs")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--clean", action="store_true", help="restrict to clean test rows (BBH)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    # vectors[model][task][method] = list over seeds of per-question correctness
    vectors: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cost: dict = defaultdict(list)
    for entry in json.loads(args.selected.read_text(encoding="utf-8")).values():
        key = model_key(entry["model_path"])
        vectors[key][entry["task"]][entry["method"]].append(entry["test_correct"])
        cost[(key, entry["method"])].append((entry.get("search_seconds") or 0.0) +
                                            (entry.get("eval_seconds") or 0.0))
    for path in args.baselines:
        results = json.loads(path.read_text(encoding="utf-8"))
        key = path.stem
        for name, entry in results.items():
            task, method = name.split("/")
            if method in FIXED and entry.get("test_correct") and task in vectors.get(key, {}):
                vectors[key][task][method].append(entry["test_correct"])

    masks = {}
    for key in vectors:
        for task in vectors[key]:
            test = graft_tasks.load_splits(args.data_dir, task)["test"]
            masks[task] = [(not args.clean) or task not in graft_tasks.BBH_TASKS
                           or int(e.example_id.rsplit(":", 1)[1]) >= 100 for e in test]

    def acc(vec: list[int], mask: list[bool], pick: list[int] | None = None) -> float:
        idx = pick if pick is not None else range(len(vec))
        kept = [vec[i] for i in idx if mask[i]]
        return statistics.mean(kept) if kept else float("nan")

    rng = random.Random(0)
    summary: dict = {"models": {}}
    for key in sorted(vectors):
        tasks = sorted(vectors[key])
        methods = sorted({m for t in tasks for m in vectors[key][t]})
        table = {t: {m: statistics.mean(acc(v, masks[t]) for v in vectors[key][t][m])
                     for m in vectors[key][t]} for t in tasks}
        print(f"== {key} ({'clean' if args.clean else 'full'} test)")
        print(f"{'task':40s} " + " ".join(f"{m[:10]:>10s}" for m in methods))
        for t in tasks:
            print(f"{t:40s} " + " ".join(f"{100 * table[t][m]:10.1f}" if m in table[t] else f"{'-':>10s}"
                                         for m in methods))
        means = {m: statistics.mean(table[t][m] for t in tasks if m in table[t]) for m in methods}
        print(f"{'mean':40s} " + " ".join(f"{100 * means[m]:10.1f}" for m in methods))
        model_out = {"table": table, "means": means, "pairs": {}, "cost_seconds": {}}
        for a, b in PAIRS:
            shared = [t for t in tasks if a in table[t] and b in table[t]]
            if not shared:
                continue
            point = statistics.mean(table[t][a] - table[t][b] for t in shared)
            boots = []
            for _ in range(args.replicates):
                diffs = []
                for t in shared:
                    n = len(vectors[key][t][a][0])
                    pick = [rng.randrange(n) for _ in range(n)]
                    va = [rng.choice(vectors[key][t][a]) for _ in vectors[key][t][a]]
                    vb = [rng.choice(vectors[key][t][b]) for _ in vectors[key][t][b]]
                    diffs.append(statistics.mean(acc(v, masks[t], pick) for v in va) -
                                 statistics.mean(acc(v, masks[t], pick) for v in vb))
                boots.append(statistics.mean(diffs))
            boots.sort()
            lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
            wins = sum(table[t][a] > table[t][b] for t in shared)
            ties = sum(table[t][a] == table[t][b] for t in shared)
            model_out["pairs"][f"{a}-{b}"] = {"diff": point, "ci": [lo, hi], "tasks": len(shared),
                                             "wins": wins, "ties": ties, "p_gt0": sum(x > 0 for x in boots) / len(boots)}
            print(f"  {a:>10s} - {b:<18s} {100 * point:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}] "
                  f"W/T/L {wins}/{ties}/{len(shared) - wins - ties}  P>0 {model_out['pairs'][f'{a}-{b}']['p_gt0']:.3f}")
        for (k, m), secs in cost.items():
            if k == key:
                model_out["cost_seconds"][m] = statistics.mean(secs)
        print("  search+selection minutes per run: " +
              ", ".join(f"{m} {s / 60:.0f}" for m, s in sorted(model_out["cost_seconds"].items())))
        summary["models"][key] = model_out
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
