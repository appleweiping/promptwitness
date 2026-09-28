"""Run the frozen 24-search Pilot serially on one local GPU, then holdout once.

The caller sets CUDA_VISIBLE_DEVICES, HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE.
Raw records, process logs and holdout outputs belong in a private sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from promptwitness.structured_data import TASKS

METHODS = ("A", "B", "C", "D")
SEEDS = (1, 2, 3)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _gpu_memory_used_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _wait_for_gpu(state: dict[str, Any], state_path: Path) -> None:
    while True:
        used = _gpu_memory_used_mib()
        if used < 2048:
            return
        state["status"] = "waiting_for_gpu_0"
        state["gpu_0_memory_used_mib"] = used
        _write(state_path, state)
        sleep(60)


def _require_clean_revision(repo: Path, revision: str) -> None:
    current = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != revision or dirty:
        raise RuntimeError("pilot source revision changed or working tree is dirty")


def _run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def _verify_search_record(
    record: dict[str, Any],
    task: str,
    method: str,
    seed: int,
    model_path: Path,
    max_new_tokens: int,
    budget: float,
    tolerance: float,
) -> None:
    if (
        record.get("status") != "frozen_validation_selected"
        or record.get("task") != task
        or record.get("method") != method
        or record.get("seed") != seed
        or record.get("model_path") != str(model_path)
        or record.get("max_new_tokens") != max_new_tokens
        or record.get("round_limit") != (0 if method == "A" else 8)
        or record.get("optimization_budget_seconds") != (0 if method == "A" else budget)
        or record.get("comparison_mode") != ("calibration" if method == "A" else "budget_capped")
    ):
        raise ValueError(
            f"existing search record differs from frozen protocol: {task}-{method}-seed{seed}"
        )
    if method != "A" and float(record["budget_overrun_seconds"]) > tolerance:
        raise ValueError(f"budget overrun exceeded tolerance: {task}-{method}-seed{seed}")


def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    budget = float(config["optimization_budget_seconds"])
    overrun_tolerance = float(config["budget_overrun_tolerance_seconds"])
    max_new_tokens = int(config["max_new_tokens"])
    if budget <= 0 or overrun_tolerance <= 0 or int(config["rounds"]) != 8:
        raise ValueError("expected a positive frozen budget, tolerance, and eight-round ceiling")
    if (
        config.get("tasks") != list(TASKS)
        or config.get("methods") != list(METHODS)
        or config.get("seeds") != list(SEEDS)
        or args.model_path.name != config.get("model_snapshot")
    ):
        raise ValueError("batch arguments differ from frozen task, arm, seed or model protocol")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("batch requires CUDA_VISIBLE_DEVICES=0 for the authorized free GPU")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("batch requires offline model loading")
    repo = Path(__file__).resolve().parents[2]
    records_dir = args.output_dir / "search"
    logs_dir = args.output_dir / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "batch_state.json"
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require_clean_revision(repo, revision)
    protocol = {
        "config": config,
        "model_path": str(args.model_path),
        "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        "search_shard_manifest_sha256": hashlib.sha256(
            (args.shard_dir / "shard_manifest.json").read_bytes()
        ).hexdigest(),
        "holdout_shard_manifest_sha256": hashlib.sha256(
            (args.holdout_dir / "shard_manifest.json").read_bytes()
        ).hexdigest(),
        "source_revision": revision,
    }
    if state_path.exists():
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("protocol") != protocol:
            raise ValueError("existing batch state differs from frozen protocol")
        state["run_attempts"] = state.get("run_attempts", 1) + 1
        state["prior_process_wall_seconds"] = state.get("process_wall_seconds", 0)
        state["unknown_prior_wall_seconds"] = (
            state.get("unknown_prior_wall_seconds", False)
            or "process_wall_seconds" not in state
        )
    else:
        state = {
            "format": "promptwitness.structured-gradient-batch/v1",
            "status": "running",
            "protocol": protocol,
            "completed": [],
            "run_attempts": 1,
            "prior_process_wall_seconds": 0.0,
            "unknown_prior_wall_seconds": False,
        }
    # Check the entire existing result set before any new model call or state write.
    for task in TASKS:
        for method in METHODS:
            for seed in SEEDS:
                output = records_dir / f"{task}-{method}-seed{seed}.json"
                if output.exists():
                    _verify_search_record(
                        json.loads(output.read_text(encoding="utf-8")),
                        task,
                        method,
                        seed,
                        args.model_path,
                        max_new_tokens,
                        budget,
                        overrun_tolerance,
                    )
    started = perf_counter()
    _write(state_path, state)
    try:
        for task in TASKS:
            for method in METHODS:
                for seed in SEEDS:
                    _require_clean_revision(repo, revision)
                    key = f"{task}-{method}-seed{seed}"
                    output = records_dir / f"{key}.json"
                    if output.exists():
                        prior = json.loads(output.read_text(encoding="utf-8"))
                    else:
                        _wait_for_gpu(state, state_path)
                        state["status"] = "running"
                        state["current"] = key
                        _write(state_path, state)
                        command = [
                            sys.executable,
                            str(repo / "scripts/research/structured_gradient_search.py"),
                            "--method",
                            method,
                            "--task",
                            task,
                            "--seed",
                            str(seed),
                            "--model-path",
                            str(args.model_path),
                            "--shard-dir",
                            str(args.shard_dir),
                            "--manifest",
                            str(args.split_manifest),
                            "--max-new-tokens",
                            str(max_new_tokens),
                            "--rounds",
                            "0" if method == "A" else "8",
                            "--budget-seconds",
                            "0" if method == "A" else str(budget),
                            "--output",
                            str(output),
                        ]
                        _run_command(command, logs_dir / f"{key}.log")
                        prior = json.loads(output.read_text(encoding="utf-8"))
                    _verify_search_record(
                        prior,
                        task,
                        method,
                        seed,
                        args.model_path,
                        max_new_tokens,
                        budget,
                        overrun_tolerance,
                    )
                    if key not in state["completed"]:
                        state["completed"].append(key)
                    state.pop("current", None)
                    state["status"] = "running"
                    _write(state_path, state)
        _wait_for_gpu(state, state_path)
        state["status"] = "running_holdout"
        _write(state_path, state)
        holdout_dir = args.output_dir / "holdout"
        _run_command(
            [
                sys.executable,
                str(repo / "scripts/research/structured_gradient_holdout.py"),
                "--records-dir",
                str(records_dir),
                "--holdout-dir",
                str(args.holdout_dir),
                "--split-manifest",
                str(args.split_manifest),
                "--model-path",
                str(args.model_path),
                "--output-dir",
                str(holdout_dir),
                "--budget-seconds",
                str(budget),
                "--max-new-tokens",
                str(max_new_tokens),
            ],
            logs_dir / "holdout.log",
        )
        state["status"] = "running_report"
        _write(state_path, state)
        _run_command(
            [
                sys.executable,
                str(repo / "scripts/research/structured_gradient_report.py"),
                "--records-dir",
                str(records_dir),
                "--holdout-dir",
                str(holdout_dir),
                "--csv",
                str(args.output_dir / "four_arm_table.csv"),
                "--markdown",
                str(args.output_dir / "four_arm_table.md"),
                "--english",
                str(args.output_dir / "mihai_preliminary_note.md"),
            ],
            logs_dir / "report.log",
        )
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        raise
    finally:
        state["process_wall_seconds"] = (
            state["prior_process_wall_seconds"] + perf_counter() - started
        )
        _write(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
