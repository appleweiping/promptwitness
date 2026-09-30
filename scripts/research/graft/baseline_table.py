"""Fixed-prompt baselines per task: full test split and the clean subset.

The clean subset excludes test rows that GReaTer's optimization saw (rows 0-99 of each
released file: 0-49 trained on, 50-99 monitored), so published GReaTer prompts are also
compared on data they were not tuned on. Paired bootstrap over test questions within
tasks for the published-minus-initial difference.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from promptwitness import graft_tasks

SETS = ("zs_cot", "greater_init", "greater_published")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    tasks = sorted({k.split("/")[0] for k in results})
    rows, pairs = [], {"full": [], "clean": []}
    for task in tasks:
        test = graft_tasks.load_splits(args.data_dir, task)["test"]
        clean = [int(e.example_id.rsplit(":", 1)[1]) >= 100 for e in test] if task in graft_tasks.BBH_TASKS \
            else [True] * len(test)
        line = {"task": task}
        for name in SETS:
            entry = results.get(f"{task}/{name}", {})
            correct = entry.get("test_correct")
            if not correct:
                continue
            line[name] = statistics.mean(correct)
            kept = [c for c, k in zip(correct, clean) if k]
            line[name + "_clean"] = statistics.mean(kept) if kept else None
            line[name + "_vec"] = correct
        if "greater_published" in line and "greater_init" in line:
            for subset in ("full", "clean"):
                keep = clean if subset == "clean" else [True] * len(test)
                pairs[subset].append([(p, i) for p, i, k in zip(line["greater_published_vec"],
                                                                line["greater_init_vec"], keep) if k])
        rows.append(line)
    fmt = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
    print(f"{'task':40s} " + " ".join(f"{s[:8]:>8s} {'clean':>6s}" for s in SETS))
    for line in rows:
        print(f"{line['task']:40s} " + " ".join(f"{fmt(line.get(s)):>8s} {fmt(line.get(s + '_clean')):>6s}" for s in SETS))
    for s in SETS:
        vals = [l[s] for l in rows if s in l]
        cvals = [l[s + "_clean"] for l in rows if l.get(s + "_clean") is not None]
        if vals:
            print(f"mean {s:18s} full {100 * statistics.mean(vals):5.1f} ({len(vals)} tasks)  clean {100 * statistics.mean(cvals):5.1f}")
    rng = random.Random(0)
    for subset, per_task in pairs.items():
        if not per_task:
            continue
        point = statistics.mean(statistics.mean(p - i for p, i in t) for t in per_task)
        boots = []
        for _ in range(args.replicates):
            boots.append(statistics.mean(statistics.mean(p - i for p, i in (rng.choice(t) for _ in t))
                                         for t in per_task))
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
        wins = sum(statistics.mean(p - i for p, i in t) > 0 for t in per_task)
        print(f"published - init ({subset}): {100 * point:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}] "
              f"over {len(per_task)} tasks; wins {wins}")


if __name__ == "__main__":
    main()
