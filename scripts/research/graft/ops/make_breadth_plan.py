"""Breadth plan (GReaTer-scale table): one structural-search arm on GReaTer's 21 BBH tasks, GSM8K and FOLIO.

usage: make_breadth_plan.py <model> <arm> <out.json> [seed, default 1]

The arm is the configuration chosen end to end in Stage C v2 (e.g. random-v50). For a model whose Stage C v2
runs already cover a task with this arm and seed, the task is skipped (the Stage C v2 record is reused).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/media/lenovo/data2/promptwitness-graft/scripts/research/graft")
sys.path.insert(0, "/media/lenovo/data2/promptwitness-graft/src")
from make_stage_plan import STAGE_C_TASKS, VARIANTS  # noqa: E402

from promptwitness import graft_tasks  # noqa: E402

model, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
tasks = sorted(graft_tasks.BBH_TASKS) + ["gsm8k", "folio"]
done = Path("/media/lenovo/data2/promptwitness-graft-runtime/stageC2/runs")
runs, reused = [], []
for task in tasks:
    if task in STAGE_C_TASKS and (done / f"{model}-{task}-{arm}-s{seed}.json").exists():
        reused.append(task)
        continue
    runs.append({"task": task, "model": model, "method": arm, "seed": seed, "extra": VARIANTS[arm] + ["--defer-eval"]})
json.dump({"stage": "breadth", "runs": runs, "reused_from_stageC2": reused}, open(out, "w"), indent=1)
print(len(tasks), "tasks:", len(runs), "runs ->", out, "| reused from Stage C v2:", reused)
