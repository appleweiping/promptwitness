"""Write the preregistered Stage C / Stage D run plans (see research/GRAFT_EXPERIMENT_PLAN.md)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGE_C_TASKS = ("date_understanding", "formal_fallacies", "object_counting", "navigate",
                 "tracking_shuffled_objects_five_objects", "movie_recommendation", "folio")
MODELS = ("llama3", "gemma2")


def stage_c(extra: list[str]) -> list[dict]:
    runs = []
    for seed in (1, 2, 3):
        for task in STAGE_C_TASKS:
            for model in MODELS:
                for method in ("patch", "exact", "random", "textgrad"):
                    runs.append({"task": task, "model": model, "method": method, "seed": seed,
                                 "extra": extra})
                if seed == 1:
                    runs.append({"task": task, "model": model, "method": "gate", "seed": 1,
                                 "extra": extra})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    runs = stage_c(args.extra)
    args.output.write_text(json.dumps({"stage": "C", "runs": runs}, indent=1) + "\n")
    print(len(runs), "runs")


if __name__ == "__main__":
    main()
