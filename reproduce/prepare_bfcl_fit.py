"""Trusted auditor preparation: fixed fit IDs only, original dataset annotation.

Nonfit JSONL bodies are never decoded. All original lines necessarily pass
through the auditor's byte stream; this is disclosed, not called an air gap.
Development/optimizer/predictor workers must never receive the source checkout.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from reproduce.bfcl_native import BFCL_REVISION, PACKAGE_RELATIVE, TASK_MODELS, verify_bfcl_source
from reproduce.process_access import LEAVES
from reproduce.strict_scoring import BFCL_CATEGORIES, ScoringError

FIRST_ID = re.compile(r'^\s*\{\s*"id"\s*:\s*("(?:[^"\\]|\\.)*")')
ARCHIVE_ID = re.compile(r'(?<!\\)"id"\s*:\s*("(?:[^"\\]|\\.)*")')
MODEL_REVISIONS = {
    TASK_MODELS[0]: "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    TASK_MODELS[1]: "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
}


def selected_lines(
    lines: Iterable[str],
    selected: set[str],
    *,
    archive: bool = False,
    forbidden: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Read only identifier tokens before decoding a selected whole body."""
    for line in lines:
        if not line.strip():
            continue
        matches = ARCHIVE_ID.findall(line) if archive else FIRST_ID.findall(line)
        if len(matches) != 1:
            raise ScoringError("JSONL identifier encoding differs from the audited source")
        unit = json.loads(matches[0])
        if forbidden and unit in forbidden:
            raise ScoringError("archived response belongs to final-derived membership")
        if unit not in selected:
            continue
        row = json.loads(line)
        if row.get("id") != unit:
            raise ScoringError("JSONL identifier is not the selected top-level ID")
        yield row


def git_rows(source: Path, relative: str, selected: set[str]) -> list[dict[str, Any]]:
    """Auditor only: stream selected IDs from a declared original data object."""
    with subprocess.Popen(
        ["git", "-C", str(source), "show", f"{BFCL_REVISION}:{relative}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ) as process:
        if process.stdout is None:
            raise ScoringError("missing original annotation stream")
        rows = list(selected_lines(process.stdout, selected))
        _, errors = process.communicate()
        if process.returncode:
            raise ScoringError(f"original BFCL object could not be read: {errors.strip()}")
    return rows


def unique_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ScoringError("duplicate selected BFCL ID")
        result[row["id"]] = row
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(
    source: Path, pools_path: Path, training_root: Path, archive_paths: list[Path], store: Path
) -> dict[str, Any]:
    verify_bfcl_source(source)
    proposal = json.loads(pools_path.read_text(encoding="utf-8"))
    if proposal["status"] != "METADATA_PROPOSAL_NOT_FORMAL_FREEZE":
        raise ScoringError("unexpected scientific membership state")
    pools = proposal["bfcl"]["pools"]
    seen: set[str] = set()
    owners: dict[str, str] = {}
    for pool, members in pools.items():
        for row in members:
            if row["id"] in seen:
                raise ScoringError("duplicate cross-pool BFCL ID")
            seen.add(row["id"])
            for group in row["groups"]:
                if owners.setdefault(group, pool) != pool:
                    raise ScoringError("BFCL groups overlap pools")
    fit = unique_rows(pools["fit"])
    if not fit or any(row["category"] not in BFCL_CATEGORIES for row in fit.values()):
        raise ScoringError("missing or unsupported fit membership")
    inputs, gold = {}, {}
    for category in sorted(BFCL_CATEGORIES):
        ids = {unit for unit, row in fit.items() if row["category"] == category}
        if not ids:
            continue
        prefix = f"{PACKAGE_RELATIVE.as_posix()}/data"
        raw = unique_rows(git_rows(source, f"{prefix}/BFCL_v4_{category}.json", ids))
        if set(raw) != ids:
            raise ScoringError("missing original fit input")
        for unit, row in raw.items():
            if sorted({item["name"] for item in row["function"]}) != fit[unit]["groups"]:
                raise ScoringError("original fit functions differ from fixed membership metadata")
            inputs[unit] = {
                "id": unit,
                "category": category,
                "function": row["function"],
                "question": row["question"],
            }
        if category == "irrelevance":
            annotations = {unit: {"id": unit, "ground_truth": None} for unit in ids}
        else:
            annotations = unique_rows(
                git_rows(source, f"{prefix}/possible_answer/BFCL_v4_{category}.json", ids)
            )
            if set(annotations) != ids or any(
                not isinstance(row.get("ground_truth"), list) or not row["ground_truth"]
                for row in annotations.values()
            ):
                raise ScoringError("missing original fit ground truth")
        gold.update(annotations)
    # Compare the prior cost probe's actual input corpus, not a newer question.
    original = {}
    for path in [
        training_root / "bfcl-simple-python.jsonl",
        *sorted((training_root / "bfcl-offline").glob("*.json")),
    ]:
        with path.open(encoding="utf-8") as stream:
            original.update(unique_rows(selected_lines(stream, set(fit))))
    if set(original) != set(inputs) or any(
        original[unit]["function"] != row["function"]
        or original[unit]["question"] != row["question"]
        for unit, row in inputs.items()
    ):
        raise ScoringError("pinned scorer dataset differs from original fit cost inputs")
    archived: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_REVISIONS}
    selected = {"bfcl:" + unit for unit in fit}
    forbidden = {"bfcl:" + row["id"] for row in pools["final_derived"]}
    for archive_path in archive_paths:
        with archive_path.open(encoding="utf-8") as stream:
            for row in selected_lines(stream, selected, archive=True, forbidden=forbidden):
                if (
                    row["family"] != "bfcl"
                    or row["role"] != "task"
                    or row.get("profile", "base") != "base"
                ):
                    continue
                model = row["model"]
                if (
                    model not in MODEL_REVISIONS
                    or row["revision"] != MODEL_REVISIONS[model]
                    or row["scoring_executed"] is not False
                ):
                    raise ScoringError(
                        "archive identity differs from predeclared main-model fit audit"
                    )
                archived[model].append(
                    {
                        "id": row["id"][5:],
                        "model": model,
                        "revision": row["revision"],
                        "status": row["status"],
                        "output": row["output"],
                        "original_scoring_executed": False,
                    }
                )
    records = {model: unique_rows(rows) for model, rows in archived.items()}
    if any(not rows for rows in records.values()):
        raise ScoringError("missing prior main-model fit responses; not a zero-result audit")
    store.mkdir(parents=True, exist_ok=False)
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    write_jsonl(store / "fit/inputs/bfcl.jsonl", (inputs[unit] for unit in sorted(inputs)))
    write_jsonl(store / "fit/gold/bfcl.jsonl", (gold[unit] for unit in sorted(gold)))
    for index, model in enumerate(MODEL_REVISIONS):
        write_jsonl(
            store / f"fit/records/bfcl-{index}.jsonl",
            (records[model][unit] for unit in sorted(records[model])),
        )
    report = {
        "format": "promptwitness.bfcl-fit-preparation/v1",
        "membership": "proposed-v2 unchanged; NOT_FORMAL_FREEZE",
        "source_revision": BFCL_REVISION,
        "fit_inputs": len(inputs),
        "fit_annotation_records": len(gold),
        "prior_fit_responses": {model: len(rows) for model, rows in records.items()},
        "category_counts": dict(sorted(Counter(row["category"] for row in fit.values()).items())),
        "selection_rule": (
            "all preexisting base/task main-model responses intersecting fixed fit IDs; "
            "no score selection"
        ),
        "original_fit_input_equality": True,
        "access_disclosure": (
            "auditor streamed declared offline category JSONL bytes; only selected fit "
            "bodies decoded; nonfit ID tokens inspected; no nonfit gold/input bodies decoded"
        ),
        "nonfit_store_contents_prepared": False,
        "new_model_calls": 0,
        "not_predictor_training_or_pilot": True,
    }
    with (store / "preparation.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("pools", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("qwen_archive", type=Path)
    parser.add_argument("olmo_archive", type=Path)
    parser.add_argument("store", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.pools,
                args.training_root,
                [args.qwen_archive, args.olmo_archive],
                args.store,
            ),
            ensure_ascii=False,
        )
    )
