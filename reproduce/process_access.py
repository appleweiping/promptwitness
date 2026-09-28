"""Linux process-enforced research split access, not a malicious-user sandbox.

Workers restrict themselves before importing application code. There is no
fallback to an unrestricted process. The controller remains trusted and must
advance stages only from the research gates; this module cannot judge science.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.machinery
import importlib.util
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
    # CIR target identities and subsets are scorer-only. Legacy text roles
    # retain their original grants; CIR callers must use these distinct roles.
    "retrieval_fit_learner": frozenset({"fit"}),
    "retrieval_fit_scorer": frozenset({"fit"}),
    "retrieval_optimizer": frozenset({"search"}),
    "retrieval_predictor": frozenset({"search"}),
    "retrieval_fit_ranker": frozenset({"fit"}),
    "retrieval_search_ranker": frozenset({"search"}),
    "retrieval_selection_ranker": frozenset({"selection"}),
    "retrieval_search_scorer": frozenset({"search"}),
    "retrieval_selection_scorer": frozenset({"selection"}),
    "retrieval_final_scorer": frozenset({"final"}),
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
    "retrieval_fit_learner": ("fit/inputs", "fit/records"),
    "retrieval_fit_scorer": ("fit/inputs", "fit/gold"),
    "retrieval_optimizer": ("fit/inputs", "search/inputs", "search/reference"),
    "retrieval_predictor": (
        "fit/inputs",
        "fit/records",
        "search/inputs",
        "search/reference",
        "search/parent",
    ),
    "retrieval_fit_ranker": ("fit/inputs",),
    "retrieval_search_ranker": ("search/inputs",),
    "retrieval_selection_ranker": ("selection/inputs",),
    "retrieval_search_scorer": ("search/inputs", "search/gold"),
    "retrieval_selection_scorer": ("selection/inputs", "selection/gold"),
    "retrieval_final_scorer": ("final/inputs", "final/gold"),
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


def enforce_policy(
    policy: dict[str, Any],
    *,
    read_files: Sequence[Path] = (),
    read_write_files: Sequence[Path] = (),
    write_file_trees: Sequence[Path] = (),
) -> int:
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
            *((path, 1 << 2) for path in read_files),
            *((path, (1 << 1) | (1 << 2)) for path in read_write_files),
            # Existing/future files only: no create/remove/reparent/execute.
            *((path, (1 << 1) | ((1 << 14) if abi >= 3 else 0)) for path in write_file_trees),
        ]
        # Dynamic loader/time-zone files only, not /etc or the user's home.
        for name in ("/etc/ld.so.cache", "/etc/localtime"):
            path = Path(name)
            if path.exists():
                grants.append((path, 1 << 2))
        # PyTorch's CPU runtime opens this device and CPU metadata while
        # loading CLIP. Only the ranker needs these file-level read grants;
        # neither /dev nor /proc is traversable through a broad directory rule.
        if policy["role"] in {
            "retrieval_fit_ranker",
            "retrieval_search_ranker",
            "retrieval_selection_ranker",
        }:
            grants.append((Path("/dev/urandom"), 1 << 2))
            grants.append((Path("/proc/cpuinfo"), 1 << 2))
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
    package = Path(__file__).resolve().parents[1] / "src/promptwitness"
    if package.is_dir():
        roots.append(package)  # Package code only, never the checkout/data parent.
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
    message: dict[str, Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Launch approved application code with no inherited data descriptors/env.

    Stdio pipes are the only inherited descriptors; the caller owns messages
    passed through those pipes. App exceptions/nonzero exits are not zero scores.
    """
    policy, environment = _role_launch(role, stage, data_root, scratch, entrypoint, arguments)
    return subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(policy)
        + "\n"
        + (json.dumps(message, allow_nan=False) if message is not None else ""),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        close_fds=True,
        env=environment,
        cwd=scratch,
    )


def _role_launch(
    role: str,
    stage: str,
    data_root: Path,
    scratch: Path,
    entrypoint: Path,
    arguments: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str]]:
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
        # Vendor imports call Path.home(). Do not query the real user database
        # or inherit a credential/cache home outside the granted scratch.
        "HOME": str(scratch.resolve()),
    }
    if role in {"retrieval_fit_ranker", "retrieval_search_ranker", "retrieval_selection_ranker"}:
        environment["CUDA_VISIBLE_DEVICES"] = ""  # Fixed CPU CLIP/ranking backend.
    return policy, environment


def start_role(
    role: str,
    stage: str,
    data_root: Path,
    scratch: Path,
    entrypoint: Path,
    arguments: Sequence[str] = (),
    *,
    stderr: Any,
) -> subprocess.Popen[str]:
    """Start a persistent role worker; application reads JSONL after restriction.

    The caller owns the returned process and must close or terminate it. Its
    stdout is a protocol stream, not a source of trusted scoring labels.
    """
    landlock_abi()
    policy, environment = _role_launch(role, stage, data_root, scratch, entrypoint, arguments)
    process = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
        bufsize=1,
        close_fds=True,
        env=environment,
        cwd=scratch,
    )
    if process.stdin is None:
        process.terminate()
        raise AccessBoundaryError("role worker has no input pipe")
    try:
        process.stdin.write(json.dumps(policy, allow_nan=False) + "\n")
        process.stdin.flush()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    return process


def worker() -> None:
    # Read only the launcher policy. Application JSON remains unread on the pipe
    # until after restriction and is consumed by the role's application itself.
    policy = json.loads(sys.stdin.readline())
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
    # Namespace discovery would list the ungranted parent of reproduce/. Bind
    # this known helper package explicitly without widening any directory grant.
    spec = importlib.machinery.ModuleSpec("reproduce", loader=None, is_package=True)
    spec.submodule_search_locations = [str(Path(__file__).resolve().parent)]
    sys.modules["reproduce"] = importlib.util.module_from_spec(spec)
    package = Path(__file__).resolve().parents[1] / "src/promptwitness"
    if package.is_dir():
        package_spec = importlib.util.spec_from_file_location(
            "promptwitness", package / "__init__.py", submodule_search_locations=[str(package)]
        )
        if package_spec is None or package_spec.loader is None:
            raise AccessBoundaryError("known package code cannot be loaded")
        module = importlib.util.module_from_spec(package_spec)
        sys.modules["promptwitness"] = module
        package_spec.loader.exec_module(module)
    os.environ["PW_ACCESS_ROLE"] = checked["role"]
    os.environ["PW_ACCESS_STAGE"] = checked["stage"]
    os.environ["PW_LANDLOCK_ABI"] = str(abi)
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    worker()
