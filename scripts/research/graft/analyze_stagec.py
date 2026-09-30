"""Stage C analysis, prespecified in the plan amendments of 2026-09-30 (before any outcome).

Inputs: the ``select_and_test.py`` output (dev selection and one test read per search run)
and the fixed-prompt baselines (``evaluate_prompts.py``) for the same models. Seeds are
paired across methods by seed index (for a seed, all methods see the same minibatches and
proposal questions); a fixed prompt is paired with every seed. For each model and task a
method difference is averaged over paired seeds, then over tasks. Two intervals:
  conditional  bootstrap of test questions (paired across methods, which share the test
               set) and of seed pairs within each task; tasks fixed
  task-level   bootstrap over tasks of the task-level differences (the population claim)
Also: wins/ties/losses over tasks, the clean subset (test rows GReaTer's optimization
never saw), and measured search-plus-selection minutes per run.
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
         ("patch", "greater_published"), ("random", "greater_init"), ("textgrad", "greater_init"),
         ("patch", "zs_cot"))


def model_key(path: str) -> str:
    name = path.lower()
    return "llama3" if "llama" in name else "gemma2" if "gemma" in name else "qwen3" if "qwen" in name else name


def interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selected", type=Path, help="select_and_test.py output")
    parser.add_argument("--baselines", type=Path, nargs="+", default=[], help="evaluate_prompts outputs")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--clean", action="store_true", help="restrict to clean test rows (BBH)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    # runs[model][task][method][seed] = per-question correctness (seed 0 for fixed prompts)
    runs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    cost: dict = defaultdict(list)
    for entry in json.loads(args.selected.read_text(encoding="utf-8")).values():
        key = model_key(entry["model_path"])
        runs[key][entry["task"]][entry["method"]][entry["seed"]] = entry["test_correct"]
        cost[(key, entry["method"])].append((entry.get("search_seconds") or 0.0) + (entry.get("eval_seconds") or 0.0))
    for path in args.baselines:
        key = path.stem
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items():
            task, method = name.split("/")
            if method in FIXED and entry.get("test_correct") and task in runs.get(key, {}):
                runs[key][task][method][0] = entry["test_correct"]

    masks = {}
    for key in runs:
        for task in runs[key]:
            test = graft_tasks.load_splits(args.data_dir, task)["test"]
            masks[task] = [(not args.clean) or task not in graft_tasks.BBH_TASKS
                           or int(e.example_id.rsplit(":", 1)[1]) >= 100 for e in test]

    def acc(vec: list[int], task: str, pick: list[int] | None = None) -> float:
        mask = masks[task]
        idx = pick if pick is not None else range(len(vec))
        kept = [vec[i] for i in idx if mask[i]]
        return statistics.mean(kept) if kept else float("nan")

    def seed_pairs(task_runs: dict, a: str, b: str) -> list[tuple[list[int], list[int]]]:
        sa, sb = task_runs[a], task_runs[b]
        if 0 in sa and 0 in sb:
            return [(sa[0], sb[0])]
        if 0 in sb:
            return [(sa[s], sb[0]) for s in sorted(sa)]
        if 0 in sa:
            return [(sa[0], sb[s]) for s in sorted(sb)]
        return [(sa[s], sb[s]) for s in sorted(set(sa) & set(sb))]

    rng = random.Random(0)
    summary: dict = {"subset": "clean" if args.clean else "full", "models": {}}
    for key in sorted(runs):
        tasks = sorted(runs[key])
        methods = sorted({m for t in tasks for m in runs[key][t]})
        table = {t: {m: statistics.mean(acc(v, t) for v in runs[key][t][m].values()) for m in runs[key][t]}
                 for t in tasks}
        print(f"== {key} ({summary['subset']} test)")
        print(f"{'task':40s} " + " ".join(f"{m[:10]:>10s}" for m in methods))
        for t in tasks:
            print(f"{t:40s} " + " ".join(f"{100 * table[t][m]:10.1f}" if m in table[t] else f"{'-':>10s}"
                                         for m in methods))
        means = {m: statistics.mean(table[t][m] for t in tasks if m in table[t]) for m in methods}
        print(f"{'mean':40s} " + " ".join(f"{100 * means[m]:10.1f}" for m in methods))
        model_out: dict = {"table": table, "means": means, "pairs": {}, "cost_minutes": {}}
        for a, b in PAIRS:
            shared = [t for t in tasks if a in runs[key][t] and b in runs[key][t] and seed_pairs(runs[key][t], a, b)]
            if not shared:
                continue
            pairs = {t: seed_pairs(runs[key][t], a, b) for t in shared}
            task_diff = {t: statistics.mean(acc(x, t) - acc(y, t) for x, y in pairs[t]) for t in shared}
            point = statistics.mean(task_diff.values())
            conditional, task_level = [], []
            for _ in range(args.replicates):
                diffs = []
                for t in shared:
                    n = len(pairs[t][0][0])
                    pick = [rng.randrange(n) for _ in range(n)]
                    chosen = [rng.choice(pairs[t]) for _ in pairs[t]]
                    diffs.append(statistics.mean(acc(x, t, pick) - acc(y, t, pick) for x, y in chosen))
                conditional.append(statistics.mean(diffs))
                task_level.append(statistics.mean(task_diff[rng.choice(shared)] for _ in shared))
            clo, chi = interval(conditional)
            tlo, thi = interval(task_level)
            wins = sum(d > 0 for d in task_diff.values())
            ties = sum(d == 0 for d in task_diff.values())
            model_out["pairs"][f"{a}-{b}"] = {"diff": point, "conditional_ci": [clo, chi], "task_ci": [tlo, thi],
                                             "tasks": len(shared), "wins": wins, "ties": ties,
                                             "seed_pairs": sum(len(p) for p in pairs.values())}
            print(f"  {a:>10s} - {b:<18s} {100 * point:+5.1f}  cond [{100 * clo:+5.1f}, {100 * chi:+5.1f}]  "
                  f"tasks [{100 * tlo:+5.1f}, {100 * thi:+5.1f}]  W/T/L {wins}/{ties}/{len(shared) - wins - ties}")
        for (k, m), secs in cost.items():
            if k == key:
                model_out["cost_minutes"][m] = statistics.mean(secs) / 60
        print("  search+selection minutes per run: " +
              ", ".join(f"{m} {s:.0f}" for m, s in sorted(model_out["cost_minutes"].items())))
        summary["models"][key] = model_out
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
