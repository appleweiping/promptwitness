"""Controller-verified, code-only staging of the pinned official BFCL scorer.

The receipt is an inventory, not a security signature. The cooperating trusted
controller verifies exact Git blob bytes before handing a stage to a worker.
No Git object database, data, answers, environment file or responses are copied.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

from reproduce import bfcl_native
from reproduce.strict_scoring import ScoringError

RECEIPT = "bfcl-stage.json"


def code_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        len(parts) > 1
        and parts[0] == "bfcl_eval"
        and path.endswith(".py")
        and not {"data", "eval", "result", "score", "__pycache__", ".."}.intersection(parts)
    )


def source_blobs(source: Path) -> dict[str, bytes]:
    """Read only code blobs from the verified fixed revision, not bundled data."""
    bfcl_native.verify_bfcl_source(source)
    prefix = bfcl_native.PACKAGE_RELATIVE.parent.as_posix() + "/"
    paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "ls-tree",
            "-r",
            "--name-only",
            bfcl_native.BFCL_REVISION,
            "--",
            bfcl_native.PACKAGE_RELATIVE.as_posix(),
        ],
        text=True,
    ).splitlines()
    paths = [path for path in paths if path.startswith(prefix) and code_path(path[len(prefix) :])]
    if not paths:
        raise ScoringError("no pinned BFCL code paths")
    payload = subprocess.check_output(
        ["git", "-C", str(source), "archive", bfcl_native.BFCL_REVISION, "--", *paths]
    )
    blobs = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name not in paths:
                raise ScoringError("unexpected non-code archive member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ScoringError("missing code archive member")
            blobs[member.name[len(prefix) :]] = stream.read()
    if len(blobs) != len(paths):
        raise ScoringError("incomplete BFCL code archive")
    return blobs


def stage_bfcl(source: Path, stage: Path) -> dict[str, Any]:
    blobs = source_blobs(source)
    stage.mkdir(parents=True, exist_ok=False)
    for name, content in sorted(blobs.items()):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    receipt = {
        "format": "promptwitness.bfcl-code-stage/v1",
        "revision": bfcl_native.BFCL_REVISION,
        "files": sorted(blobs),
        "verification": "exact fixed-revision Git blobs; trusted controller",
        "contains_data_or_git_objects": False,
    }
    with (stage / RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    verify_staged_bfcl(source, stage)
    return receipt


def staged_inventory(stage: Path) -> dict[str, Any]:
    receipt = json.loads((stage / RECEIPT).read_text(encoding="utf-8"))
    files = receipt.get("files")
    if (
        receipt.get("format") != "promptwitness.bfcl-code-stage/v1"
        or receipt.get("revision") != bfcl_native.BFCL_REVISION
        or not isinstance(files, list)
        or not files
        or any(not isinstance(name, str) or not code_path(name) for name in files)
        or len(files) != len(set(files))
    ):
        raise ScoringError("invalid BFCL code-only inventory")
    actual = set()
    for path in stage.rglob("*"):
        if path.is_symlink():
            raise ScoringError("staged runtime cannot contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(stage).as_posix())
    if actual != {*files, RECEIPT}:
        raise ScoringError("BFCL stage contains missing or unregistered files")
    return cast(dict[str, Any], receipt)


def verify_staged_bfcl(source: Path, stage: Path) -> None:
    """Controller only: compare every staged byte against real pinned Git code."""
    receipt = staged_inventory(stage)
    blobs = source_blobs(source)
    if set(receipt["files"]) != set(blobs) or any(
        (stage / name).read_bytes() != content for name, content in blobs.items()
    ):
        raise ScoringError("staged BFCL code differs from the pinned source")


def load_staged_bfcl(
    stage: Path, work_root: Path, model: str
) -> tuple[bfcl_native.BFCLChecker, dict[str, Any]]:
    """Worker binding after controller byte verification; no source-access bypass."""
    staged_inventory(stage)
    checker, metadata = bfcl_native.bind_bfcl_package(stage / "bfcl_eval", work_root, model)
    metadata["source_binding"] = "controller-verified code-only stage; no worker Git access"
    return checker, metadata
