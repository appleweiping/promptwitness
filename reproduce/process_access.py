"""Linux process-enforced research split access, not a malicious-user sandbox.

Workers restrict themselves before importing application code. There is no
fallback to an unrestricted process. The controller remains trusted and must
advance stages only from the research gates; this module cannot judge science.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import runpy
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class AccessBoundaryError(RuntimeError):
    """A requested role/stage or native boundary cannot be enforced."""


LEAVES = (
    "fit/inputs",
    "fit/gold",
    "fit/records",
    "search/inputs",
    "search/reference",
    "search/parent",
    "search/gold",
    "selection/inputs",
    "selection/gold",
    "final/inputs",
    "final/gold",
)
ROLE_STAGES = {
    "fit_learner": frozenset({"fit"}),
    "optimizer": frozenset({"search"}),
    "predictor": frozenset({"search"}),
    "search_scorer": frozenset({"search"}),
    "selection_scorer": frozenset({"selection"}),
    "final_scorer": frozenset({"final"}),
}
ROLE_LEAVES = {
    "fit_learner": ("fit/inputs", "fit/gold", "fit/records"),
    "optimizer": ("fit/inputs", "fit/gold", "search/inputs", "search/reference"),
    "predictor": (
        "fit/inputs",
        "fit/gold",
        "fit/records",
        "search/inputs",
        "search/reference",
        "search/parent",
    ),
    "search_scorer": ("search/inputs", "search/gold"),
    "selection_scorer": ("selection/inputs", "selection/gold"),
    "final_scorer": ("final/inputs", "final/gold"),
}


def role_leaves(role: str, stage: str) -> tuple[str, ...]:
    if role not in ROLE_STAGES or stage not in ROLE_STAGES[role]:
        raise AccessBoundaryError("role is not authorized in this stage")
    return ROLE_LEAVES[role]


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _directory(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise AccessBoundaryError("access roots must be existing directories")
    return resolved


def build_policy(
    role: str, stage: str, data_root: Path, runtime_roots: Sequence[Path], scratch: Path
) -> dict[str, Any]:
    leaves = role_leaves(role, stage)
    root = _directory(data_root)
    # Existing split leaves cannot alias each other or escape the dataset store.
    all_leaves = {name: _directory(root / name) for name in LEAVES}
    for name, path in all_leaves.items():
        if path != root / name:
            raise AccessBoundaryError("split leaves must not be symlinks")
    paths = list(all_leaves.values())
    if any(_overlaps(a, b) for index, a in enumerate(paths) for b in paths[index + 1 :]):
        raise AccessBoundaryError("split leaves overlap")
    runtime = tuple(dict.fromkeys(_directory(path) for path in runtime_roots))
    work = _directory(scratch)
    if not runtime or any(_overlaps(path, root) for path in (*runtime, work)):
        raise AccessBoundaryError("runtime/scratch must be separate from the data store")
    if any(_overlaps(path, work) for path in runtime):
        raise AccessBoundaryError("writable scratch must be separate from runtime code")
    return {
        "format": "promptwitness.process-access/v1",
        "role": role,
        "stage": stage,
        "data_root": str(root),
        "data_read": [str(all_leaves[name]) for name in leaves],
        "runtime_read_execute": [str(path) for path in runtime],
        "scratch_read_write": str(work),
    }


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def landlock_abi() -> int:
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise AccessBoundaryError("Linux x86_64/aarch64 Landlock is required")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = int(libc.syscall(444, 0, 0, 1))
    if abi < 1:
        raise AccessBoundaryError(f"Landlock unavailable: errno {ctypes.get_errno()}")
    return abi


def _checked(result: int, operation: str) -> int:
    if result < 0:
        raise AccessBoundaryError(f"{operation} failed: errno {ctypes.get_errno()}")
    return result


def _path_flags() -> int:
    if sys.platform == "linux":
        return os.O_PATH | os.O_CLOEXEC
    raise AccessBoundaryError("Linux path flags are required")


def enforce_policy(policy: dict[str, Any]) -> int:
    """Irreversibly restrict a fresh single-threaded worker and its children.

    ABI1 has no network or truncate syscall restriction. Not an air gap, and
    not protection against another unsandboxed process of the same user.
    """
    abi = landlock_abi()
    # These are the complete filesystem rights of ABI1, plus supported newer
    # rename/reparent (ABI2) and truncate (ABI3) rights. No best-effort bypass.
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13
    if abi >= 3:
        handled |= 1 << 14
    read = (1 << 2) | (1 << 3)
    libc = ctypes.CDLL(None, use_errno=True)
    ruleset = _Ruleset(handled)
    fd = _checked(
        int(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)), "create ruleset"
    )
    try:
        grants = [
            *((Path(path), read) for path in policy["data_read"]),
            *((Path(path), read | 1) for path in policy["runtime_read_execute"]),
            (Path(policy["scratch_read_write"]), handled & ~1),
        ]
        # Dynamic loader/time-zone files only, not /etc or the user's home.
        for name in ("/etc/ld.so.cache", "/etc/localtime"):
            path = Path(name)
            if path.exists():
                grants.append((path, 1 << 2))
        for path, rights in grants:
            parent_fd = os.open(path, _path_flags())
            try:
                rule = _PathBeneath(rights, parent_fd)
                _checked(int(libc.syscall(445, fd, 1, ctypes.byref(rule), 0)), "add path rule")
            finally:
                os.close(parent_fd)
        _checked(int(libc.prctl(38, 1, 0, 0, 0)), "no_new_privs")
        _checked(int(libc.syscall(446, fd, 0)), "restrict self")
    finally:
        os.close(fd)
    return abi


def runtime_roots() -> tuple[Path, ...]:
    """Runtime only: interpreter installations, system libraries, helper code.

    Corpus files must not live inside these directories. The policy rejects a
    runtime ancestor/descendant of the registered dataset store or scratch.
    """
    roots = [Path(sys.prefix), Path(sys.base_prefix), Path(__file__).resolve().parent]
    roots.extend(Path(name) for name in ("/usr", "/lib", "/lib64") if Path(name).is_dir())
    return tuple(dict.fromkeys(path.resolve() for path in roots))


def launch_role(
    role: str,
    stage: str,
    data_root: Path,
    scratch: Path,
    entrypoint: Path,
    arguments: Sequence[str] = (),
    *,
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    """Launch approved application code with no inherited data descriptors/env.

    Stdio pipes are the only inherited descriptors; the caller owns messages
    passed through those pipes. App exceptions/nonzero exits are not zero scores.
    """
    policy = build_policy(role, stage, data_root, runtime_roots(), scratch)
    entry = entrypoint.resolve(strict=True)
    if not any(entry.is_relative_to(Path(root)) for root in policy["runtime_read_execute"]):
        raise AccessBoundaryError("entrypoint is outside approved runtime roots")
    policy["entrypoint"] = str(entry)
    policy["arguments"] = list(arguments)
    environment = {
        "PATH": os.pathsep.join((str(Path(sys.executable).parent), "/usr/bin", "/bin")),
        "LANG": "C.UTF-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TMPDIR": str(scratch.resolve()),
    }
    return subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(policy),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        close_fds=True,
        env=environment,
        cwd=scratch,
    )


def worker() -> None:
    policy = json.load(sys.stdin)
    if policy.get("format") != "promptwitness.process-access/v1":
        raise AccessBoundaryError("invalid worker policy")
    # Rebuild grants from role/stage, never trust arbitrary JSON path grants.
    checked = build_policy(
        policy["role"],
        policy["stage"],
        Path(policy["data_root"]),
        [Path(path) for path in policy["runtime_read_execute"]],
        Path(policy["scratch_read_write"]),
    )
    if checked != {
        key: value for key, value in policy.items() if key not in {"entrypoint", "arguments"}
    }:
        raise AccessBoundaryError("worker grants disagree with role policy")
    entry = Path(policy["entrypoint"]).resolve(strict=True)
    if not any(entry.is_relative_to(Path(root)) for root in checked["runtime_read_execute"]):
        raise AccessBoundaryError("entrypoint is outside approved runtime roots")
    if len(os.listdir("/proc/self/task")) != 1:
        raise AccessBoundaryError("restriction must precede application threads")
    sys.dont_write_bytecode = True
    abi = enforce_policy(checked)
    os.chdir(checked["scratch_read_write"])
    sys.argv = [str(entry), *policy["arguments"]]
    # Only source paths are returned to application imports; dataset paths are
    # never added to sys.path. Native libraries remain supplied by the runtime.
    sys.path.insert(0, str(entry.parent))
    os.environ["PW_ACCESS_ROLE"] = checked["role"]
    os.environ["PW_ACCESS_STAGE"] = checked["stage"]
    os.environ["PW_LANDLOCK_ABI"] = str(abi)
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    worker()
