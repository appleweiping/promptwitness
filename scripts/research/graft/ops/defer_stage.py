"""Defer (or release) the not-yet-started runs of a stage plan without touching a running one.

usage: defer_stage.py <plan.json> <out_dir> defer|release
defer:   create <run>.lock (content MARK) for every run with neither output nor lock, so that the
         running stage_queue.py skips them after its current run and the GPU queue moves on.
release: delete exactly the locks carrying MARK (real claims and failure locks are left alone).
"""
import json
import os
import sys
from pathlib import Path

MARK = b"deferred-by-cc\n"
plan, out, action = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), sys.argv[3]
for run in plan["runs"]:
    name = f"{run['model']}-{run['task']}-{run['method']}-s{run['seed']}{run.get('suffix', '')}"
    lock = out / f"{name}.lock"
    if action == "defer":
        if (out / f"{name}.json").exists():
            continue
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print("claimed", name)
            continue
        os.write(fd, MARK)
        os.close(fd)
        print("deferred", name)
    elif action == "release":
        if lock.exists() and lock.read_bytes() == MARK:
            lock.unlink()
            print("released", name)
