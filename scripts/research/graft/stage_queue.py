"""Crash-safe sequential queue for GRAFT runs (Stage C/D) on one GPU.

Reads a JSON plan {"runs": [{"task", "model", "method", "seed", "extra": [...]}, ...]},
skips runs whose output exists or that another queue has claimed (``<run>.lock``), records the git commit and exit status of each run
in a JSONL journal, and optionally waits while another process (the frozen pilot)
needs the GPU. Never kills other processes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODELS = {
    "llama3": ("/media/lenovo/data2/graft-hf-cache/models--NousResearch--Meta-Llama-3-8B-Instruct/"
               "snapshots/53346005fb0ef11d3b6a83b12c895cca40156b6c", "sdpa"),
    "gemma2": ("/media/lenovo/data2/graft-hf-cache/models--unsloth--gemma-2-9b-it/snapshots/"
               "fc7d4737cda11c3a19af2b722319e846670b4d89", "eager"),
    "qwen3": ("/media/lenovo/data2/graft-hf-cache/models--Qwen--Qwen3-8B/snapshots/"
              "b968826d9c46dd6066d109eabc6255188de91218", "sdpa"),
}


def pilot_active() -> bool:
    state = Path("/media/lenovo/data2/promptwitness-structured-gradient-pilot-runtime/"
                 "pilot-full-v1/batch_state.json")
    try:
        status = json.loads(state.read_text())["status"]
    except (OSError, ValueError, KeyError):
        return False
    return status not in ("completed", "complete", "done", "finished")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--yield-to-pilot", action="store_true")
    parser.add_argument("--model", help="only runs of this model key (one queue per GPU and model)")
    parser.add_argument("--gen-server", help="host:port of a graft_genserver serving this model")
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="stop the queue after this many failed runs in a row")
    args = parser.parse_args()
    failures = 0
    repo = Path(__file__).resolve().parents[3]
    plan = json.loads(args.plan.read_text())
    journal = args.out / "journal.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    for run in plan["runs"]:
        if args.model and run["model"] != args.model:
            continue
        name = f"{run['model']}-{run['task']}-{run['method']}-s{run['seed']}{run.get('suffix', '')}"
        output = args.out / f"{name}.json"
        if output.exists():
            continue
        try:  # claim the run, so queues on several GPUs can share one plan without duplicates
            os.close(os.open(args.out / f"{name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue  # running elsewhere, or failed earlier (delete the lock to retry)
        while args.yield_to_pilot and pilot_active():
            time.sleep(120)
        model_path, attn = MODELS[run["model"]]
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        extra = run.get("extra", [])
        method = [] if "--method" in extra else ["--method", run["method"]]  # variants carry their own
        command = [sys.executable, str(repo / "scripts/research/graft/run_graft.py"),
                   "--task", run["task"], "--model-path", model_path, "--data-dir", str(args.data_dir),
                   *method, "--seed", str(run["seed"]), "--attn", attn,
                   "--output", str(output), *extra,
                   *(["--gen-server", args.gen_server] if args.gen_server else [])]
        started = time.time()
        with (args.out / f"{name}.log").open("w") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=repo).returncode
        with journal.open("a") as handle:
            handle.write(json.dumps({"run": name, "commit": commit, "exit": code,
                                     "seconds": time.time() - started,
                                     "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        print(json.dumps({"run": name, "exit": code}), flush=True)
        failures = failures + 1 if code else 0
        if failures >= args.max_consecutive_failures:
            print(json.dumps({"stopped": "consecutive failures", "last": name}), flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
