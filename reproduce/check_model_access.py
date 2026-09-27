"""Actual Linux inference access witness, authored sentinels, zero CUDA calls.

Uses the same isolated interpreter/bootstrap/environment as PersistentModel.
Native inference under this boundary is a separate required qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from reproduce.persistent_model import model_environment
from reproduce.process_access import LEAVES


def check(directory: Path, snapshot: Path, model_python: Path) -> dict:
    if not model_python.is_absolute() or not model_python.is_file():
        raise ValueError("existing absolute model interpreter required")
    directory.mkdir(parents=True, exist_ok=False)
    store, scratch = directory / "authored-store", directory / "scratch"
    scratch.mkdir()
    for leaf in LEAVES:
        folder = store / leaf
        folder.mkdir(parents=True)
        (folder / "MODEL_ACCESS_AUTHORED.txt").write_text("AUTHORED_ONLY", encoding="utf-8")
    message = {
        "snapshot": str(snapshot.resolve(strict=True)),
        "model": "ACCESS_PROBE_NO_MODEL_LOAD",
        "data_root": str(store.resolve()),
        "scratch": str(scratch.resolve()),
    }
    with (
        (directory / "worker.stdout").open("x", encoding="utf-8") as stdout,
        (directory / "worker.stderr").open("x", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            [
                str(model_python),
                "-u",
                "-I",
                str(Path(__file__).with_name("model_access.py")),
                "--probe",
            ],
            input=json.dumps(message) + "\n",
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
            close_fds=True,
            env=model_environment(model_python, scratch),
            cwd=scratch,
        )
    if result.returncode:
        raise RuntimeError(f"actual inference access probe failed {result.returncode}; see stderr")
    report = json.loads((directory / "worker.stdout").read_text(encoding="utf-8"))
    if (
        report["kind"] != "access_probe"
        or report["access"]["worker_pid"] == os.getpid()
        or report["access"]["landlock_abi"] < 1
        or report["access"]["benchmark_data_read_grants"] != 0
        or len(report["observations"]) != 2 * len(LEAVES)
        or not all(row["denied"] for row in report["observations"])
        or not all(
            (store / leaf / "MODEL_ACCESS_AUTHORED.txt").read_text() == "AUTHORED_ONLY"
            for leaf in LEAVES
        )
    ):
        raise RuntimeError("missing actual restricted model access witness")
    report.update(
        controller_pid=os.getpid(),
        model_calls=0,
        allocated_GPU_hours=0,
        real_data_or_scores_read=False,
        native_inference_under_boundary="NOT_RUN",
    )
    with (directory / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("model_python", type=Path)
    print(json.dumps(check(**vars(parser.parse_args()))))
