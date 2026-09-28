"""Pinned BBH CSV splits for the structured-gradient development pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASKS = ("logical_deduction_three_objects", "date_understanding")
SOURCE_COMMIT = "42a22d9211e55528e8894c89ca7d13ae817366d2"
SPLIT_SEED = 20260928
_ANSWER = re.compile(r"^\([A-Z]\)$")


@dataclass(frozen=True, slots=True)
class BBHRow:
    row_id: str
    question: str
    answer: str


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bbh_rows(path: Path, task: str) -> tuple[BBHRow, ...]:
    if task not in TASKS:
        raise ValueError("unsupported pilot task")
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 250:
        raise ValueError("expected 250 rows in the pinned BBH task")
    result: list[BBHRow] = []
    for index, row in enumerate(rows):
        question = row.get("goal", "")
        answer = row.get("final_target", "")
        if not question or not _ANSWER.fullmatch(answer):
            raise ValueError(f"invalid question or answer at row {index}")
        result.append(BBHRow(f"{task}:{index:03d}", question, answer))
    return tuple(result)


def make_manifest(data_dir: Path) -> dict[str, Any]:
    """Select disjoint 32 fit, 32 validation and 100 holdout IDs before runs."""
    tasks: dict[str, Any] = {}
    for task in TASKS:
        path = data_dir / f"{task}.json"
        rows = read_bbh_rows(path, task)
        ordered = sorted(
            range(len(rows)),
            key=lambda index: hashlib.sha256(
                f"{SPLIT_SEED}:{task}:{index}".encode("ascii")
            ).digest(),
        )
        tasks[task] = {
            "sha256": _digest(path),
            "rows": len(rows),
            "fit_ids": [rows[index].row_id for index in ordered[:32]],
            "validation_ids": [rows[index].row_id for index in ordered[32:64]],
            "holdout_ids": [rows[index].row_id for index in ordered[64:164]],
        }
    return {
        "format": "promptwitness.structured-gradient-splits/v1",
        "source_repository": "https://github.com/psunlpgroup/GreaTer",
        "source_commit": SOURCE_COMMIT,
        "split_seed": SPLIT_SEED,
        "tasks": tasks,
    }


def load_split(data_dir: Path, manifest_path: Path, task: str, split: str) -> tuple[BBHRow, ...]:
    if task not in TASKS or split not in ("fit", "validation", "holdout"):
        raise ValueError("unknown task or split")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "promptwitness.structured-gradient-splits/v1":
        raise ValueError("wrong split manifest format")
    path = data_dir / f"{task}.json"
    task_manifest = manifest["tasks"][task]
    if _digest(path) != task_manifest["sha256"]:
        raise ValueError("BBH task bytes differ from pinned manifest")
    rows = {row.row_id: row for row in read_bbh_rows(path, task)}
    ids = task_manifest[f"{split}_ids"]
    if len(ids) != len(set(ids)) or any(row_id not in rows for row_id in ids):
        raise ValueError("split contains duplicate or unknown IDs")
    all_ids = (
        task_manifest["fit_ids"] + task_manifest["validation_ids"] + task_manifest["holdout_ids"]
    )
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split sets overlap")
    return tuple(rows[row_id] for row_id in ids)
