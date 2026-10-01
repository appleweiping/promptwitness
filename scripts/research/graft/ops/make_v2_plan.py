"""Stage C v2 plan for one model (seed-major order), per the 2026-10-01 amendment.

usage: make_v2_plan.py <model> <out.json> <seeds comma-separated> [dist]
"""
import json
import sys

sys.path.insert(0, "/media/lenovo/data2/promptwitness-graft/scripts/research/graft")
from make_stage_plan import STAGE_C_TASKS, VARIANTS

model, out = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3].split(",")]
with_dist = len(sys.argv) > 4 and sys.argv[4] == "dist"
runs = []
for seed in seeds:
    for task in STAGE_C_TASKS:
        arms = ["patch-v50", "random-v50"] + (["dist-v50"] if with_dist else []) + ["random-v8"]
        if seed <= 2:
            arms.append("textgrad-v50")
        for name in arms:
            runs.append({"task": task, "model": model, "method": name, "seed": seed,
                         "extra": VARIANTS[name] + ["--defer-eval"]})
json.dump({"stage": "C2", "runs": runs}, open(out, "w"), indent=1)
print(len(runs), "runs ->", out)
