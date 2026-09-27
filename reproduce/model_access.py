"""Restrict the inference child before application imports or CUDA threads.

Same cooperating-operator boundary as process_access; ABI1 is not an air gap.
No benchmark directory is granted. Only complete stdin messages reach the model.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

# Python -I does not add the script directory to sys.path. Load only the known
# standard-library boundary helper, not an application package or repo parent.
_helper_spec = importlib.util.spec_from_file_location(
    "_model_process_access", Path(__file__).resolve().with_name("process_access.py")
)
if _helper_spec is None or _helper_spec.loader is None:
    raise RuntimeError("known process access helper cannot be loaded")
_helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)
LEAVES, AccessBoundaryError = _helper.LEAVES, _helper.AccessBoundaryError
_directory, _overlaps = _helper._directory, _helper._overlaps
enforce_policy, runtime_roots = _helper.enforce_policy, _helper.runtime_roots

DEVICES = ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools")


def native_files() -> tuple[list[Path], list[Path]]:
    read = [Path("/dev/urandom")]
    write = [Path("/dev/null"), *(Path(name) for name in DEVICES)]
    if any(not stat.S_ISCHR(path.stat().st_mode) for path in (*read, *write)):
        raise AccessBoundaryError("fixed CUDA/random/null device is not a character device")
    # Actual restricted CUDA startup denied these two OS metadata reads.
    # Grant exact files only, never all of /proc or process write access.
    information = [Path("/proc/cpuinfo"), Path("/proc/sys/vm/mmap_min_addr")]
    if any(not path.is_file() for path in information):
        raise AccessBoundaryError("registered CPU/CUDA system metadata files unavailable")
    read.extend(information)
    return read, write


def thread_metadata_root() -> Path:
    # CUDA names a newly created driver thread through this procfs hierarchy.
    # Resolve /self in this still single-threaded child, not in the controller.
    return _directory(Path("/proc/self/task"))


def model_policy(snapshot: Path, data_root: Path, scratch: Path) -> dict:
    """Build grants from this interpreter, exact snapshot and fixed native paths.

    Real HF snapshots are symlink farms: grant each referenced blob FILE, not
    the cache ancestor or other revisions. Reject data/runtime/scratch overlap.
    """
    root, weights, work = map(_directory, (data_root, snapshot, scratch))
    for leaf in LEAVES:
        if _directory(root / leaf) != root / leaf:
            raise AccessBoundaryError("registered benchmark leaves must not be symlinks")
    runtime = runtime_roots()
    readonly = [weights]
    readonly.extend(
        Path(name).resolve()
        for name in ("/proc/self", "/proc/driver/nvidia", "/sys")
        if Path(name).is_dir()
    )
    files, devices = native_files()
    thread_metadata = thread_metadata_root()
    blobs = weights.parent.parent / "blobs"
    for path in weights.rglob("*"):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file() or not target.is_relative_to(blobs):
                raise AccessBoundaryError("snapshot link is not an original model blob file")
            files.append(target)
    grants = (*runtime, *readonly, *files, *devices, thread_metadata)
    if any(_overlaps(path, root) or _overlaps(path, work) for path in grants):
        raise AccessBoundaryError("model grants overlap benchmark data or writable scratch")
    if _overlaps(root, work):
        raise AccessBoundaryError("scratch overlaps benchmark store")
    return {
        "format": "promptwitness.model-process-access/v1",
        "role": "inference_message_only",
        "benchmark_data_root": str(root),
        "benchmark_data_read": [],
        "data_read": [str(path) for path in dict.fromkeys(readonly)],
        "runtime_read_execute": [str(path) for path in runtime],
        "scratch_read_write": str(work),
        "read_files": [str(path) for path in dict.fromkeys(files)],
        "read_write_files": [str(path) for path in devices],
        "write_file_trees": [str(thread_metadata)],
    }


def bind_source():
    """Bind known source leaves, without granting/listing the checkout parent."""
    helper = Path(__file__).resolve().parent
    spec = importlib.machinery.ModuleSpec("reproduce", loader=None, is_package=True)
    spec.submodule_search_locations = [str(helper)]
    sys.modules["reproduce"] = importlib.util.module_from_spec(spec)
    package = helper.parent / "src/promptwitness"
    package_spec = importlib.util.spec_from_file_location(
        "promptwitness", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    if package_spec is None or package_spec.loader is None:
        raise AccessBoundaryError("known source package cannot be loaded")
    module = importlib.util.module_from_spec(package_spec)
    sys.modules["promptwitness"] = module
    package_spec.loader.exec_module(module)


def restrict(message: dict) -> dict:
    if set(message) != {"snapshot", "model", "data_root", "scratch"}:
        raise AccessBoundaryError("literal model launch fields required")
    policy = model_policy(*(Path(message[k]) for k in ("snapshot", "data_root", "scratch")))
    if len(os.listdir("/proc/self/task")) != 1:
        raise AccessBoundaryError("model restriction must precede application threads")
    if "torch" in sys.modules or "promptwitness" in sys.modules:
        raise AccessBoundaryError("application loaded before model restriction")
    abi = enforce_policy(
        policy,
        read_files=[Path(path) for path in policy["read_files"]],
        read_write_files=[Path(path) for path in policy["read_write_files"]],
        write_file_trees=[Path(path) for path in policy["write_file_trees"]],
    )
    os.chdir(policy["scratch_read_write"])
    sys.dont_write_bytecode = True
    return {
        "role": policy["role"],
        "landlock_abi": abi,
        "benchmark_data_read_grants": 0,
        "restriction_before_application_imports": True,
        "worker_pid": os.getpid(),
        "thread_metadata_write_scope": "own_proc_task_only",
    }


def probe(message: dict, access: dict):
    """Only authored sentinel contents: never open a real benchmark record."""
    root = Path(message["data_root"])
    observations = []
    for leaf in LEAVES:
        path = root / leaf / "MODEL_ACCESS_AUTHORED.txt"
        for mode in ("r", "a"):
            try:
                with path.open(mode, encoding="utf-8") as stream:
                    if mode == "r":
                        stream.read()
                    else:
                        stream.write("AUTHORED_WRITE_ATTEMPT")
            except PermissionError:
                observations.append({"leaf": leaf, "mode": mode, "denied": True})
            else:
                raise AccessBoundaryError("model process can open a benchmark sentinel")
    with (Path(message["scratch"]) / "scratch-witness").open("x", encoding="utf-8") as stream:
        stream.write("AUTHORED_ONLY")
    # Opening original metadata through the actual snapshot symlink exercises
    # the referenced-file grant; do not return any model file contents.
    with (Path(message["snapshot"]) / "config.json").open("rb") as stream:
        nonempty = bool(stream.read(1))
    if not nonempty:
        raise AccessBoundaryError("model snapshot metadata is empty")
    return {
        "kind": "access_probe",
        "access": access,
        "observations": observations,
        "original_snapshot_metadata_read": True,
        "application_import_after_restriction": True,
        "scratch_writable": True,
        "model_loaded": False,
    }


def worker(*, probe_only=False):
    message = json.loads(sys.stdin.readline())
    access = restrict(message)
    if probe_only:
        bind_source()
        print(json.dumps(probe(message, access)), flush=True)
        return
    bind_source()
    from reproduce.torch_runtime import worker as model_worker

    model_worker(Path(message["snapshot"]), message["model"], access=access)


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--probe"]):
        raise SystemExit("usage: python -I model_access.py [--probe]")
    worker(probe_only=sys.argv[1:] == ["--probe"])
