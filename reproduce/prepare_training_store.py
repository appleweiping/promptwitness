"""Auditor preparation of the complete proposed training pools, never final data.

All three families retain fit/search/selection IDs and groups. This prepares the
data needed by the real pipeline, not its reference scores, freeze or acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduce import prepare_text_fit as text
from reproduce.bfcl_native import BFCL_REVISION, PACKAGE_RELATIVE, verify_bfcl_source
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, git_rows, unique_rows, write_jsonl
from reproduce.process_access import LEAVES
from reproduce.strict_scoring import BFCL_CATEGORIES, ScoringError

POOLS = ("fit", "search", "selection")
FAMILIES = ("bfcl", *text.FAMILIES)


def memberships(proposal: dict) -> dict[str, dict[str, dict]]:
    # Reuse the current text validation, including the exact three-pool schema.
    text.memberships(proposal)
    result = {}
    for family in FAMILIES:
        pools = proposal[family]["pools"]
        expected = {*POOLS, "final_derived"} if family == "bfcl" else set(POOLS)
        if set(pools) != expected:
            raise ScoringError("unexpected training pool names")
        seen, owners = set(), {}
        for pool, members in pools.items():
            for row in members:
                unit = row["id"]
                if not isinstance(unit, str) or not unit or unit in seen:
                    raise ScoringError("duplicate or invalid cross-pool ID")
                seen.add(unit)
                if not row["groups"]:
                    raise ScoringError("missing membership groups")
                for group in row["groups"]:
                    if owners.setdefault(group, pool) != pool:
                        raise ScoringError("source groups overlap pools")
        result[family] = {pool: unique_rows(pools[pool]) for pool in POOLS}
        if any(not rows for rows in result[family].values()):
            raise ScoringError("all three training pools must be nonempty")
        if family == "bfcl" and any(
            row["category"] not in BFCL_CATEGORIES
            for rows in result[family].values()
            for row in rows.values()
        ):
            raise ScoringError("unsupported BFCL training category")
    return result


def bfcl_rows(source: Path, members: dict[str, dict]) -> tuple[dict, dict]:
    """Declared Git objects streamed; only selected training bodies decoded."""
    inputs, gold = {}, {}
    for category in sorted(BFCL_CATEGORIES):
        selected = {unit for unit, row in members.items() if row["category"] == category}
        if not selected:
            continue
        prefix = f"{PACKAGE_RELATIVE.as_posix()}/data"
        raw = unique_rows(git_rows(source, f"{prefix}/BFCL_v4_{category}.json", selected))
        if set(raw) != selected:
            raise ScoringError("missing original BFCL training input")
        for unit, row in raw.items():
            if sorted({item["name"] for item in row["function"]}) != members[unit]["groups"]:
                raise ScoringError("original BFCL groups differ from proposed membership")
            inputs[unit] = {
                "id": unit,
                "category": category,
                "function": row["function"],
                "question": row["question"],
            }
        annotations = (
            {unit: {"id": unit, "ground_truth": None} for unit in selected}
            if category == "irrelevance"
            else unique_rows(
                git_rows(source, f"{prefix}/possible_answer/BFCL_v4_{category}.json", selected)
            )
        )
        if set(annotations) != selected or (
            category != "irrelevance"
            and any(
                not isinstance(row.get("ground_truth"), list) or not row["ground_truth"]
                for row in annotations.values()
            )
        ):
            raise ScoringError("missing original BFCL training annotation")
        gold.update(annotations)
    return inputs, gold


def text_rows(source: Path, family: str, members: dict) -> tuple[dict, dict]:
    raw = text.selected_parquet(source, family, set(members))
    key = "id" if family == "hotpotqa" else "key"
    raw_by_id = unique_rows({**row, "id": row[key]} for row in raw)
    if set(raw_by_id) != set(members):
        raise ScoringError("missing original text training input or annotation")
    pairs = {unit: text.convert(family, row, members[unit]) for unit, row in raw_by_id.items()}
    return (
        {unit: pair[0] for unit, pair in pairs.items()},
        {unit: pair[1] for unit, pair in pairs.items()},
    )


def read_rows(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return unique_rows(json.loads(line) for line in stream if line.strip())


def fit_records(fit_store: Path, family: str, inputs: dict, gold: dict) -> dict:
    # Only the earlier filtered fit store is opened, never a full response archive.
    if read_rows(fit_store / f"fit/inputs/{family}.jsonl") != inputs or (
        read_rows(fit_store / f"fit/gold/{family}.jsonl") != gold
    ):
        raise ScoringError("training store differs from previously qualified fit data")
    result = {}
    for index, (model, revision) in enumerate(MODEL_REVISIONS.items()):
        rows = read_rows(fit_store / f"fit/records/{family}-{index}.jsonl")
        if (
            not rows
            or not set(rows) <= set(inputs)
            or any(
                row["model"] != model
                or row["revision"] != revision
                or row["original_scoring_executed"] is not False
                for row in rows.values()
            )
        ):
            raise ScoringError("invalid previously filtered fit response identity")
        result[str(index)] = rows
    return result


def prepare(
    source: Path, bfcl_source: Path, pools_path: Path, bfcl_fit: Path, text_fit: Path, store: Path
) -> dict:
    proposal = json.loads(pools_path.read_text(encoding="utf-8"))
    members = memberships(proposal)  # Validate before opening benchmark data.
    verify_bfcl_source(bfcl_source)
    inputs, gold, records = {}, {}, {}
    for family in FAMILIES:
        all_members = {unit: row for pool in POOLS for unit, row in members[family][pool].items()}
        all_inputs, all_gold = (
            bfcl_rows(bfcl_source, all_members)
            if family == "bfcl"
            else text_rows(source, family, all_members)
        )
        inputs[family] = {
            pool: {unit: all_inputs[unit] for unit in members[family][pool]} for pool in POOLS
        }
        gold[family] = {
            pool: {unit: all_gold[unit] for unit in members[family][pool]} for pool in POOLS
        }
        records[family] = fit_records(
            bfcl_fit if family == "bfcl" else text_fit,
            family,
            inputs[family]["fit"],
            gold[family]["fit"],
        )
    # Exclusive output is created only after semantic checks. Partial I/O failures
    # remain visible; a receipt is written last, never used as permission to score.
    store.mkdir(parents=True, exist_ok=False)
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    for family in FAMILIES:
        for pool in POOLS:
            write_jsonl(store / f"{pool}/inputs/{family}.jsonl", inputs[family][pool].values())
            write_jsonl(store / f"{pool}/gold/{family}.jsonl", gold[family][pool].values())
        for index, rows in records[family].items():
            write_jsonl(store / f"fit/records/{family}-{index}.jsonl", rows.values())
    report = {
        "format": "promptwitness.training-store-preparation/v1",
        "status": "PREPARED_NOT_SCIENTIFIC_FREEZE",
        "membership": "proposed-v2 unchanged; NOT_FORMAL_FREEZE",
        "counts": {
            family: {pool: len(rows) for pool, rows in pools.items()}
            for family, pools in members.items()
        },
        "fit_response_counts": {
            family: {index: len(rows) for index, rows in models.items()}
            for family, models in records.items()
        },
        "BFCL_source_revision": BFCL_REVISION,
        "previous_fit_inputs_and_gold_equal": True,
        "empty_leaves": ["search/reference", "search/parent", "final/inputs", "final/gold"],
        "reference_scores": "NOT_GENERATED",
        "candidate_or_configuration_freeze": "NOT_GENERATED",
        "selection_outputs_or_scores": "NOT_GENERATED",
        "final_content_read": False,
        "access_disclosure": "Training ID tokens streamed; only selected training bodies decoded. "
        "Parquet physical row-group bytes may contain nontraining rows. "
        "BFCL final-derived ID/group metadata checked, bodies never decoded. "
        "Not an air gap; historical test-preview disclosure retained.",
        "new_model_calls": 0,
        "new_allocated_GPU_hours": 0,
        "not_M1_or_Pilot": True,
    }
    with (store / "preparation.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "bfcl_source", "pools", "bfcl_fit", "text_fit", "store"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source, args.bfcl_source, args.pools, args.bfcl_fit, args.text_fit, args.store
            )
        )
    )
