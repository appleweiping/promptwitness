"""Aggregate GRAFT run records and fixed-prompt evaluations into tables.

Inputs: run JSONs from ``run_graft.py`` (any directory tree) and optional
``evaluate_prompts.py`` outputs. Outputs markdown (stdout) and, with --latex, a
LaTeX table body. Numbers are read from files; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def model_key(path: str) -> str:
    lowered = path.lower()
    for key in ("llama", "gemma", "olmo"):
        if key in lowered:
            return key
    return Path(path).name


def load_runs(paths: list[Path]) -> dict[tuple[str, str, str], dict[int, dict]]:
    runs: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for root in paths:
        for file in sorted(root.rglob("*.json")) if root.is_dir() else [root]:
            data = json.loads(file.read_text(encoding="utf-8"))
            if data.get("format") != "promptwitness.graft-run/v1":
                continue
            runs[(model_key(data["model_path"]), data["task"], data["method"])][data["seed"]] = data
    return runs


def load_fixed(paths: list[Path]) -> dict[tuple[str, str, str], float]:
    fixed: dict[tuple[str, str, str], float] = {}
    for file in paths:
        data = json.loads(file.read_text(encoding="utf-8"))
        model = file.stem.split("-")[0]
        for key, value in data.items():
            if "test_accuracy" in value:
                fixed[(model, value["task"], value["set"])] = value["test_accuracy"]
    return fixed


def gpu_seconds(run: dict) -> float:
    return sum(phase["seconds"] for name, phase in run["cost"].items()
               if not name.startswith(("dev", "test")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--fixed", nargs="*", type=Path, default=[])
    parser.add_argument("--methods", nargs="+", default=["patch", "exact", "random", "textgrad", "gate"])
    parser.add_argument("--latex", type=Path)
    args = parser.parse_args()
    runs = load_runs(args.runs)
    fixed = load_fixed(args.fixed)
    models = sorted({m for m, _, _ in runs} | {m for m, _, _ in fixed})
    tasks = sorted({t for _, t, _ in runs} | {t for _, t, _ in fixed})
    fixed_sets = sorted({s for _, _, s in fixed})
    columns = fixed_sets + args.methods
    lines = []
    for model in models:
        print(f"\n### {model}\n")
        print("| task | " + " | ".join(columns) + " |")
        print("|---|" + "---|" * len(columns))
        averages: dict[str, list[float]] = defaultdict(list)
        for task in tasks:
            cells = []
            for column in columns:
                if column in fixed_sets:
                    value = fixed.get((model, task, column))
                    cells.append("–" if value is None else f"{100 * value:.1f}")
                    if value is not None:
                        averages[column].append(value)
                    continue
                seeds = runs.get((model, task, column), {})
                accs = [r["test_accuracy"] for r in seeds.values()]
                if not accs:
                    cells.append("–")
                    continue
                mean = statistics.mean(accs)
                averages[column].append(mean)
                sd = statistics.pstdev(accs) if len(accs) > 1 else 0.0
                cells.append(f"{100 * mean:.1f}±{100 * sd:.1f} (n={len(accs)})")
            print(f"| {task} | " + " | ".join(cells) + " |")
            lines.append((model, task, cells))
        print("| **average** | " + " | ".join(
            f"{100 * statistics.mean(averages[c]):.1f} ({len(averages[c])})" if averages[c] else "–"
            for c in columns) + " |")
        # Search cost per method (search phases only, excluding dev/test evaluation).
        print("\n| method | runs | mean search GPU-s | mean accepted edits |")
        print("|---|---|---|---|")
        for method in args.methods:
            chosen = [r for (m, _, meth), seeds in runs.items() if m == model and meth == method
                      for r in seeds.values()]
            if chosen:
                print(f"| {method} | {len(chosen)} | {statistics.mean(map(gpu_seconds, chosen)):.0f} | "
                      f"{statistics.mean(len(r['accepted']) - 1 for r in chosen):.1f} |")
        # Paired comparison of GRAFT against each control over (task, seed).
        for other in args.methods[1:]:
            wins = ties = losses = 0
            for task in tasks:
                a = runs.get((model, task, "patch"), {})
                b = runs.get((model, task, other), {})
                for seed in set(a) & set(b):
                    delta = a[seed]["test_accuracy"] - b[seed]["test_accuracy"]
                    wins += delta > 0
                    ties += delta == 0
                    losses += delta < 0
            if wins + ties + losses:
                print(f"\npatch vs {other}: {wins} wins / {ties} ties / {losses} losses")
    if args.latex:
        body = [" & ".join([task.replace("_", r"\_")] + [c.replace("±", r"$\pm$").split(" (")[0] for c in cells]) + r" \\"
                for _, task, cells in lines]
        args.latex.write_text("\n".join(body) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
