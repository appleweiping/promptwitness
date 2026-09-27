"""Trusted-controller staging of native text scorer code and NLTK resources.

Only literal scorer paths and four declared resource trees are copied. No
checkout, benchmark dataset, annotations or responses are granted to workers.
The inventory is not a signature; the controller checks original bytes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path, PurePosixPath

from reproduce import check_strict_scorers as native
from reproduce.strict_scoring import ScoringError

HOTPOT_REVISION = "3635853403a8735609ee997664e1528f4480762a"
HOTPOT_URL = (
    "https://raw.githubusercontent.com/hotpotqa/hotpot/"
    + HOTPOT_REVISION
    + "/hotpot_evaluate_v1.py"
)
CODE = {
    "IFBench": (
        "evaluation_lib.py",
        "ifbench/__init__.py",
        "ifbench/instructions.py",
        "ifbench/classic_instructions.py",
        "ifbench/instructions_registry.py",
        "ifbench/instructions_util.py",
    ),
    "open-instruct-verifiers": (
        "open_instruct/__init__.py",
        "open_instruct/IFEvalG/instructions.py",
        "open_instruct/IFEvalG/instructions_registry.py",
        "open_instruct/IFEvalG/instructions_util.py",
    ),
}
CODE_FILES = (
    *(f"{directory}/{name}" for directory, names in CODE.items() for name in names),
    "hotpot-scorer/hotpot_evaluate_v1.py",
)
RESOURCE_DIRS = (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger_eng",
)
REVISIONS = {**native.SOURCE_REVISIONS, "HotpotQA": HOTPOT_REVISION}
RECEIPT = "text-scorer-stage.json"
EMPTY_IMPORT_DIR = "IFBench/ifbench/.nltk_data"


def source_blobs(source: Path) -> dict[str, bytes]:
    native.verify_native_sources(source)
    blobs = {}
    for directory, paths in CODE.items():
        revision = native.SOURCE_REVISIONS[directory]
        for name in paths:
            original = subprocess.check_output(
                ["git", "-C", str(source / directory), "show", f"{revision}:{name}"]
            )
            if (source / directory / name).read_bytes() != original:
                raise ScoringError("native text scorer working code differs from pinned Git blob")
            blobs[f"{directory}/{name}"] = original
    # Hotpot's scorer was downloaded alone, not cloned with its bundled data.
    # The fixed official HTTPS URL is not supplied by the operator or response.
    with urllib.request.urlopen(HOTPOT_URL, timeout=30) as stream:
        original = stream.read()
    name = "hotpot-scorer/hotpot_evaluate_v1.py"
    if (source / name).read_bytes() != original:
        raise ScoringError("Hotpot scorer differs from the pinned official file")
    blobs[name] = original
    return blobs


def resource_paths(source: Path) -> list[str]:
    names = []
    for resource in RESOURCE_DIRS:
        root = source / "nltk-data" / resource
        if not root.is_dir() or root.is_symlink():
            raise ScoringError("missing declared NLTK resource directory")
        files = []
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ScoringError("NLTK resources must be regular staged files")
            if path.is_file():
                files.append(path.relative_to(source).as_posix())
        if not files:
            raise ScoringError("empty declared NLTK resource directory")
        names.extend(files)
    return sorted(names)


def resource_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        not PurePosixPath(name).is_absolute()
        and ".." not in parts
        and any(name.startswith("nltk-data/" + root + "/") for root in RESOURCE_DIRS)
    )


def staged_inventory(stage: Path) -> dict:
    receipt = json.loads((stage / RECEIPT).read_text(encoding="utf-8"))
    resources = receipt.get("resource_files")
    if (
        receipt.get("format") != "promptwitness.text-scorer-stage/v1"
        or receipt.get("revisions") != REVISIONS
        or receipt.get("code_files") != sorted(CODE_FILES)
        or not isinstance(resources, list)
        or not resources
        or any(not isinstance(name, str) or not resource_path(name) for name in resources)
        or len(resources) != len(set(resources))
    ):
        raise ScoringError("invalid native text scorer inventory")
    actual = set()
    for path in stage.rglob("*"):
        if path.is_symlink():
            raise ScoringError("text scorer stage cannot contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(stage).as_posix())
    if actual != {*CODE_FILES, *resources, RECEIPT}:
        raise ScoringError("text scorer stage contains missing or unregistered files")
    if not (stage / EMPTY_IMPORT_DIR).is_dir():
        raise ScoringError("missing precreated native import directory")
    return receipt


def verify_stage(source: Path, stage: Path) -> None:
    receipt = staged_inventory(stage)
    blobs = source_blobs(source)
    if any((stage / name).read_bytes() != content for name, content in blobs.items()):
        raise ScoringError("staged text scorer differs from pinned original code")
    resources = resource_paths(source)
    if resources != receipt["resource_files"] or any(
        (source / name).read_bytes() != (stage / name).read_bytes() for name in resources
    ):
        raise ScoringError("staged NLTK resources differ from the prepared resources")


def stage_native(source: Path, stage: Path) -> dict:
    blobs = source_blobs(source)
    resources = resource_paths(source)
    stage.mkdir(parents=True, exist_ok=False)
    for name in [*blobs, *resources]:
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(blobs[name] if name in blobs else (source / name).read_bytes())
    # Original IFBench creates this directory on import. Precreate it, without
    # granting writes to code or modifying that original import implementation.
    (stage / EMPTY_IMPORT_DIR).mkdir()
    receipt = {
        "format": "promptwitness.text-scorer-stage/v1",
        "revisions": REVISIONS,
        "code_files": sorted(CODE_FILES),
        "resource_files": resources,
        "verification": "exact pinned original code and prepared resource bytes; controller",
        "contains_benchmark_data_or_git_objects": False,
    }
    with (stage / RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    verify_stage(source, stage)
    return receipt


def load_staged(stage: Path):
    staged_inventory(stage)
    return native.bind_native(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    report = stage_native(args.source, args.stage)
    print(
        json.dumps(
            {
                "revisions": REVISIONS,
                "code_files": len(report["code_files"]),
                "resource_files": len(report["resource_files"]),
            }
        )
    )
