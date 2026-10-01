"""Official GReaTer: gradient vs random shortlist (shared deterministic reader).

Inputs: per-run evaluations written by ops/job_greater_eval.sh (eval2/greater/<name>.json,
keys "<task>/greater_published") for run names llama3_<task>_<gradient|random>[_r<rep>], and the
fixed-prompt baselines (eval2/baselines_llama3.json: greater_init, zs_cot, greater_published).
Per task: test accuracy of each arm and repetition, gradient - random paired on test questions
(repetitions paired by index), each arm against GReaTer's initial prompt; full and clean subsets
(BBH test questions with file row >= 100 were never seen by GReaTer's own optimization).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from promptwitness import graft_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evals", type=Path, help="directory with eval2/greater/*.json")
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    arms: dict = defaultdict(lambda: defaultdict(dict))  # task -> arm -> rep -> correct vector
    for path in sorted(args.evals.glob("llama3_*.json")):
        m = re.fullmatch(r"llama3_(.+)_(gradient|random)(?:_r(\d+))?", path.stem)
        if not m:
            continue
        task, arm, rep = m.group(1), m.group(2), int(m.group(3) or 1)
        entry = json.loads(path.read_text(encoding="utf-8")).get(f"{task}/greater_published", {})
        if entry.get("test_correct"):
            arms[task][arm][rep] = entry["test_correct"]
    base = json.loads(args.baselines.read_text(encoding="utf-8"))
    rng = random.Random(0)
    for subset in ("full", "clean"):
        print(f"== {subset} test")
        for task in sorted(arms):
            test = graft_tasks.load_splits(args.data_dir, task)["test"]
            keep = [subset == "full" or task not in graft_tasks.BBH_TASKS or int(e.example_id.rsplit(":", 1)[1]) >= 100
                    for e in test]
            idx = [i for i, k in enumerate(keep) if k]

            def acc(vec: list[int], pick: list[int]) -> float:
                return statistics.mean(vec[i] for i in pick)

            init = base.get(f"{task}/greater_init", {}).get("test_correct")
            cells = []
            for arm in ("gradient", "random"):
                reps = arms[task].get(arm, {})
                cells.append(f"{arm} " + "/".join(f"{100 * acc(v, idx):.1f}" for _, v in sorted(reps.items())))
            line = f"  {task:40s} " + "  ".join(cells)
            if init:
                line += f"  init {100 * acc(init, idx):.1f}"
            pairs = [(arms[task]["gradient"][r], arms[task]["random"][r])
                     for r in sorted(set(arms[task].get("gradient", {})) & set(arms[task].get("random", {})))]
            if pairs:
                diff = statistics.mean(acc(g, idx) - acc(r, idx) for g, r in pairs)
                boots = []
                for _ in range(args.replicates):
                    pick = [rng.choice(idx) for _ in idx]
                    boots.append(statistics.mean(acc(g, pick) - acc(r, pick) for g, r in pairs))
                boots.sort()
                line += (f"  | gradient-random {100 * diff:+.1f} "
                         f"[{100 * boots[int(0.025 * len(boots))]:+.1f}, {100 * boots[int(0.975 * len(boots)) - 1]:+.1f}]"
                         f" (questions resampled, {len(pairs)} rep pair(s))")
            print(line)


if __name__ == "__main__":
    main()
