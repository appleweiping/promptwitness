"""Write the preregistered Stage C / Stage D run plans (see research/GRAFT_EXPERIMENT_PLAN.md)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGE_C_TASKS = ("date_understanding", "formal_fallacies", "object_counting", "navigate",
                 "tracking_shuffled_objects_five_objects", "movie_recommendation", "folio")
MODELS = ("llama3", "gemma2")
# name -> run_graft arguments. Variants share proposals, budget, dev checkpoints and the
# deferred (engine-shared) dev selection and test read; only scoring/acceptance differ.
VARIANTS = {
    "patch": ["--method", "patch"],
    "exact": ["--method", "exact"],
    "random": ["--method", "random"],
    "textgrad": ["--method", "textgrad"],
    "gate": ["--method", "gate"],
    "fork-patch-margin": ["--method", "patch", "--objective", "fork", "--accept", "margin"],
    "fork-patch-fresh": ["--method", "patch", "--objective", "fork", "--accept", "fresh"],
    "fork-exact-margin": ["--method", "exact", "--objective", "fork", "--accept", "margin"],
    "answer-patch-fresh": ["--method", "patch", "--objective", "answer", "--accept", "fresh"],
}


def stage_c(variants: list[str], models: list[str], seeds: list[int], single_seed: list[str],
            extra: list[str]) -> list[dict]:
    runs = []
    for seed in seeds:
        for task in STAGE_C_TASKS:
            for model in models:
                for name in variants + (single_seed if seed == seeds[0] else []):
                    runs.append({"task": task, "model": model, "method": name, "seed": seed,
                                 "extra": VARIANTS[name] + ["--defer-eval", *extra]})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["patch", "exact", "random", "textgrad"])
    parser.add_argument("--single-seed", nargs="*", default=["gate"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    runs = stage_c(args.variants, args.models, args.seeds, args.single_seed, args.extra)
    args.output.write_text(json.dumps({"stage": "C", "runs": runs}, indent=1) + "\n")
    print(len(runs), "runs")


if __name__ == "__main__":
    main()
