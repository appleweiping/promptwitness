"""Read-only resource inventory; never installs, launches a model or reads secrets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def inventory(storage: tuple[str, ...] = ()) -> dict[str, object]:
    """Return package versions, disk capacity and current GPU occupancy."""
    versions = {}
    for name in ("torch", "transformers", "vllm", "huggingface-hub", "scipy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    disks = []
    for name in storage:
        path = Path(name)
        if path.is_dir():
            usage = shutil.disk_usage(path)
            disks.append(
                {
                    "path": name,
                    "free_bytes": usage.free,
                    "total_bytes": usage.total,
                    "writable": os.access(path, os.W_OK),
                }
            )
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        probe = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        gpu = {"returncode": probe.returncode, "rows": probe.stdout.strip().splitlines()}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = {"error_type": type(error).__name__}
    return {
        "format": "promptwitness.delta.resource-inventory/v1",
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": versions,
        "disks": disks,
        "gpu": gpu,
        "model_inference_executed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(inventory(tuple(args.storage)), indent=2, sort_keys=True))
