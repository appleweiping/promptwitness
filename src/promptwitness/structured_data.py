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


def write_split_shards(
    data_dir: Path, manifest_path: Path, shard_dir: Path, holdout_dir: Path
) -> None:
    """Prepare private, disjoint label shards before launching any optimizer."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "promptwitness.structured-gradient-splits/v1":
        raise ValueError("wrong split manifest format")
    shard_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        source = data_dir / f"{task}.json"
        task_manifest = manifest["tasks"][task]
        if _digest(source) != task_manifest["sha256"]:
            raise ValueError("BBH task bytes differ from pinned manifest")
        rows = {row.row_id: row for row in read_bbh_rows(source, task)}
        all_ids = [
            row_id
            for split in ("fit", "validation", "holdout")
            for row_id in task_manifest[f"{split}_ids"]
        ]
        if len(all_ids) != len(set(all_ids)) or any(row_id not in rows for row_id in all_ids):
            raise ValueError("split contains duplicate or unknown IDs")
        for split in ("fit", "validation", "holdout"):
            destination_dir = holdout_dir if split == "holdout" else shard_dir
            destination = destination_dir / f"{task}.{split}.jsonl"
            lines = [
                json.dumps(
                    {
                        "row_id": row_id,
                        "question": rows[row_id].question,
                        "answer": rows[row_id].answer,
                    },
                    ensure_ascii=False,
                )
                for row_id in task_manifest[f"{split}_ids"]
            ]
            destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for destination_dir, splits in (
        (shard_dir, ("fit", "validation")),
        (holdout_dir, ("holdout",)),
    ):
        hashes = {
            f"{task}.{split}.jsonl": _digest(destination_dir / f"{task}.{split}.jsonl")
            for task in TASKS
            for split in splits
        }
        (destination_dir / "shard_manifest.json").write_text(
            json.dumps({"source_commit": SOURCE_COMMIT, "sha256": hashes}, indent=2) + "\n",
            encoding="utf-8",
        )


def load_split(shard_dir: Path, manifest_path: Path, task: str, split: str) -> tuple[BBHRow, ...]:
    if task not in TASKS or split not in ("fit", "validation", "holdout"):
        raise ValueError("unknown task or split")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "promptwitness.structured-gradient-splits/v1":
        raise ValueError("wrong split manifest format")
    task_manifest = manifest["tasks"][task]
    ids = task_manifest[f"{split}_ids"]
    path = shard_dir / f"{task}.{split}.jsonl"
    shard_manifest = json.loads((shard_dir / "shard_manifest.json").read_text(encoding="utf-8"))
    if shard_manifest.get("source_commit") != manifest["source_commit"]:
        raise ValueError("private shard source commit differs from pinned manifest")
    if shard_manifest["sha256"].get(path.name) != _digest(path):
        raise ValueError("private shard bytes differ from prepared digest")
    rows = [BBHRow(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    if [row.row_id for row in rows] != ids or len(ids) != len(set(ids)):
        raise ValueError("private shard does not match pinned split IDs")
    if any(not row.question or not _ANSWER.fullmatch(row.answer) for row in rows):
        raise ValueError("private shard has an invalid question or answer")
    return tuple(rows)
