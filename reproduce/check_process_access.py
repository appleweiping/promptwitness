"""Authored split-access sentinels on actual Linux processes; no dataset/inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Independent expected access matrix, not computed from the policy under test.
EXPECTED = {
    "fit_learner": {"fit/inputs", "fit/gold", "fit/records"},
    "optimizer": {"fit/inputs", "fit/gold", "search/inputs", "search/reference"},
    "predictor": {
        "fit/inputs",
        "fit/gold",
        "fit/records",
        "search/inputs",
        "search/reference",
        "search/parent",
    },
    "search_scorer": {"search/inputs", "search/gold"},
    "selection_scorer": {"selection/inputs", "selection/gold"},
    "final_scorer": {"final/inputs", "final/gold"},
}


@contextmanager
def private_env_witness() -> Iterator[None]:
    previous = os.environ.get("PW_SENTINEL_PRIVATE")
    os.environ["PW_SENTINEL_PRIVATE"] = "AUTHORED_NOT_A_CREDENTIAL"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PW_SENTINEL_PRIVATE", None)
        else:
            os.environ["PW_SENTINEL_PRIVATE"] = previous


def probe(root: Path) -> None:
    """The application attempts actual file opens after entering Landlock."""
    # Exercise the same restricted namespace-package import as real scorers.
    from reproduce.process_access import role_leaves

    imported_role_leaves = role_leaves(os.environ["PW_ACCESS_ROLE"], os.environ["PW_ACCESS_STAGE"])
    observations = []
    for leaf in sorted(set.union(*EXPECTED.values())):
        target = root / leaf / "sentinel.txt"
        try:
            content = target.read_text(encoding="utf-8")
            read = "allowed" if content == f"AUTHORED:{leaf}\n" else "content_mismatch"
        except PermissionError:
            read = "denied"
        try:
            with target.open("a", encoding="utf-8") as stream:
                stream.write("UNEXPECTED_WRITE\n")
            write = "allowed"
        except PermissionError:
            write = "denied"
        observations.append({"leaf": leaf, "read": read, "write": write})
    scratch = Path.cwd() / "scratch-witness.txt"
    scratch.write_text("authored scratch\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "role": os.environ["PW_ACCESS_ROLE"],
                "pid": os.getpid(),
                "abi": int(os.environ["PW_LANDLOCK_ABI"]),
                "checks": observations,
                "scratch_write": scratch.read_text(encoding="utf-8") == "authored scratch\n",
                "private_env_inherited": "PW_SENTINEL_PRIVATE" in os.environ,
                "helper_import_after_restriction": set(imported_role_leaves)
                == EXPECTED[os.environ["PW_ACCESS_ROLE"]],
            }
        )
    )


def check(work: Path, output: Path) -> dict[str, Any]:
    from reproduce.process_access import LEAVES, AccessBoundaryError, landlock_abi, launch_role

    # Exclusive artifacts preserve failed runs and prevent accidental rerun over
    # the previous witness. Every file below is authored here, not benchmark GT.
    with output.open("x", encoding="utf-8") as stream, private_env_witness():
        abi = landlock_abi()
        work.mkdir(parents=True, exist_ok=False)
        root = work / "store"
        for leaf in LEAVES:
            directory = root / leaf
            directory.mkdir(parents=True)
            (directory / "sentinel.txt").write_text(f"AUTHORED:{leaf}\n", encoding="utf-8")
        records = []
        for role in EXPECTED:
            scratch = work / f"scratch-{role}"
            scratch.mkdir()
            stage = (
                "fit"
                if role == "fit_learner"
                else (
                    "selection"
                    if role == "selection_scorer"
                    else "final"
                    if role == "final_scorer"
                    else "search"
                )
            )
            invocation = launch_role(
                role, stage, root, scratch, Path(__file__), ["--probe", str(root)]
            )
            if invocation.returncode != 0:
                raise AccessBoundaryError(f"worker {role} failed: {invocation.stderr}")
            record = json.loads(invocation.stdout)
            if record["pid"] == os.getpid() or not record["scratch_write"]:
                raise AccessBoundaryError("worker/scratch witness failed")
            if record["private_env_inherited"] or record["abi"] < 1:
                raise AccessBoundaryError("worker environment/boundary witness failed")
            if len(record["checks"]) != len(LEAVES):
                raise AccessBoundaryError("incomplete split checks")
            for row in record["checks"]:
                expected = "allowed" if row["leaf"] in EXPECTED[role] else "denied"
                if row["read"] != expected or row["write"] != "denied":
                    raise AccessBoundaryError(f"split access mismatch: {role}/{row['leaf']}")
            records.append(record)
        result = {
            "format": "promptwitness.process-access-witness/v1",
            "status": "PASS_AUTHORED_ONLY",
            "abi": abi,
            "controller_pid": os.getpid(),
            "workers": records,
            "read_checks": 66,
            "write_checks": 66,
            "evaluation_type": "authored_access_mechanics_not_performance",
            "model_calls": 0,
            "allocated_gpu_hours": 0,
            "real_dataset_store": "NOT_PREPARED",
            "pipeline_role_wiring": "NOT_COMPLETE",
            "scientific_admission": "PARTIAL_NOT_PASSED",
            "limitations": [
                "trusted controller and cooperating local operator",
                "not an air gap or restriction of unsandboxed same-user processes",
                "ABI1 does not constrain network/stat/truncate syscalls",
                "previous final-card preview remains disclosed",
            ],
        }
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        probe(Path(sys.argv[2]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("work", type=Path)
        parser.add_argument("output", type=Path)
        args = parser.parse_args()
        report = check(args.work, args.output)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "abi",
                        "read_checks",
                        "write_checks",
                        "scientific_admission",
                    )
                }
            )
        )
