"""Auditor-only preparation of fixed HotpotQA/IFTrain fit inputs and real gold.

Parquet filters select fit IDs before Python row decoding. Physical row groups
can contain nonfit bytes; this is logical selection, not an air gap. Archived
JSONL ID tokens are inspected before only fit response bodies are decoded.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, selected_lines, unique_rows, write_jsonl
from reproduce.prepare_preflight_requests import SEEDS
from reproduce.process_access import LEAVES
from reproduce.strict_scoring import ScoringError

FAMILIES = ("hotpotqa", "instruction_following")


def memberships(proposal: dict) -> dict[str, dict]:
    if proposal["status"] != "METADATA_PROPOSAL_NOT_FORMAL_FREEZE":
        raise ScoringError("unexpected scientific membership state")
    result = {}
    for family in FAMILIES:
        seen, owners = set(), {}
        pools = proposal[family]["pools"]
        if set(pools) != {"fit", "search", "selection"}:
            raise ScoringError("unexpected text training pool names")
        for pool, members in pools.items():
            for row in members:
                if row["id"] in seen:
                    raise ScoringError("duplicate cross-pool text ID")
                seen.add(row["id"])
                for group in row["groups"]:
                    if owners.setdefault(group, pool) != pool:
                        raise ScoringError("text groups overlap pools")
        result[family] = unique_rows(pools["fit"])
        if not result[family]:
            raise ScoringError("missing text fit membership")
    return result


def selected_parquet(source: Path, family: str, ids: set[str]) -> list[dict]:
    import pyarrow.parquet as parquet

    if family == "hotpotqa":
        paths = ("hotpot-train-0.parquet", "hotpot-train-1.parquet")
        key, columns = "id", ["id", "question", "context", "answer"]
    else:
        paths = ("if-train.parquet",)
        key, columns = "key", ["key", "messages", "ground_truth"]
    return [
        row
        for name in paths
        for row in parquet.read_table(
            source / name, columns=columns, filters=[(key, "in", sorted(ids))]
        ).to_pylist()
    ]


def convert(family: str, raw: dict, member: dict) -> tuple[dict, dict]:
    unit = member["id"]
    if family == "hotpotqa":
        context = raw["context"]
        groups = sorted(set(context["title"]))
        paragraphs = "\n\n".join(
            f"{title}: {''.join(sentences)}"
            for title, sentences in zip(context["title"], context["sentences"], strict=True)
        )
        messages = [
            {"role": "user", "content": f"Context:\n{paragraphs}\n\nQuestion: {raw['question']}"}
        ]
        gold = {"id": unit, "answer": raw["answer"]}
    else:
        annotation = ast.literal_eval(raw["ground_truth"])
        if not isinstance(annotation, list) or len(annotation) != 1:
            raise ScoringError("unexpected IFTrain annotation envelope")
        ids, arguments = annotation[0]["instruction_id"], annotation[0]["kwargs"]
        if not ids or len(ids) != len(arguments):
            raise ScoringError("IFTrain annotation constraints missing or misaligned")
        groups = sorted(set(ids))
        messages = raw["messages"]
        if not messages or any(message["role"] not in {"user", "system"} for message in messages):
            raise ScoringError("training input contains a completion or unsupported message")
        gold = {"id": unit, "instruction_id": ids, "kwargs": arguments}
    if groups != member["groups"]:
        raise ScoringError("original text fit groups differ from fixed metadata")
    return (
        {
            "id": unit,
            "family": family,
            "messages": [{"role": "system", "content": SEEDS[family]}, *messages],
        },
        gold,
    )


def prepare(source: Path, pools_path: Path, archive_paths: list[Path], store: Path) -> dict:
    proposal = json.loads(pools_path.read_text(encoding="utf-8"))
    fit = memberships(proposal)
    inputs, gold = {}, {}
    for family in FAMILIES:
        raw = selected_parquet(source, family, set(fit[family]))
        key = "id" if family == "hotpotqa" else "key"
        by_id = unique_rows({**row, "id": row[key]} for row in raw)
        if set(by_id) != set(fit[family]):
            raise ScoringError("missing original text fit input or annotation")
        pairs = {unit: convert(family, row, fit[family][unit]) for unit, row in by_id.items()}
        inputs[family] = {unit: pair[0] for unit, pair in pairs.items()}
        gold[family] = {unit: pair[1] for unit, pair in pairs.items()}
    selected = {f"{family}:{unit}" for family in FAMILIES for unit in fit[family]}
    # These two training proposals have only fit/search/selection. Their final
    # corpora are separate official datasets, not BFCL's final_derived pool.
    # Do not open final metadata/content to build an artificial exclusion list.
    records = {family: {model: [] for model in MODEL_REVISIONS} for family in FAMILIES}
    for path in archive_paths:
        with path.open(encoding="utf-8") as stream:
            for row in selected_lines(stream, selected, archive=True):
                if row["role"] != "task" or row.get("profile", "base") != "base":
                    continue
                family, unit = row["id"].split(":", 1)
                model = row["model"]
                if (
                    row["family"] != family
                    or model not in MODEL_REVISIONS
                    or row["revision"] != MODEL_REVISIONS[model]
                    or row["scoring_executed"] is not False
                ):
                    raise ScoringError("changed main-model text fit archive identity")
                records[family][model].append(
                    {
                        "id": unit,
                        "model": model,
                        "revision": row["revision"],
                        "status": row["status"],
                        "output": row["output"],
                        "original_scoring_executed": False,
                    }
                )
    records = {
        family: {model: unique_rows(rows) for model, rows in models.items()}
        for family, models in records.items()
    }
    if any(not rows for models in records.values() for rows in models.values()):
        raise ScoringError("missing prior main-model text fit responses")
    # Compare archived response inputs against the actual preflight corpus;
    # never require that all fixed fit members were included in the small probe.
    with (source / "preflight-context-requests.jsonl").open(encoding="utf-8") as stream:
        original = unique_rows(selected_lines(stream, selected, archive=True))
    for family in FAMILIES:
        used = set().union(*(set(rows) for rows in records[family].values()))
        if any(
            f"{family}:{unit}" not in original
            or original[f"{family}:{unit}"]["messages"] != inputs[family][unit]["messages"]
            for unit in used
        ):
            raise ScoringError("text fit input differs from original archived cost request")
    store.mkdir(parents=True, exist_ok=False)
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    for family in FAMILIES:
        write_jsonl(store / f"fit/inputs/{family}.jsonl", inputs[family].values())
        write_jsonl(store / f"fit/gold/{family}.jsonl", gold[family].values())
        for index, model in enumerate(MODEL_REVISIONS):
            write_jsonl(
                store / f"fit/records/{family}-{index}.jsonl", records[family][model].values()
            )
    report = {
        "format": "promptwitness.text-fit-preparation/v1",
        "membership": "proposed-v2 unchanged; NOT_FORMAL_FREEZE",
        "fit_inputs": {family: len(rows) for family, rows in inputs.items()},
        "prior_fit_responses": {
            family: {model: len(rows) for model, rows in models.items()}
            for family, models in records.items()
        },
        "original_archived_input_equality": True,
        "selection_rule": "all existing base/task main-model fit responses, not score selection",
        "access_disclosure": (
            "Parquet row-group bytes may include nonfit rows; only filtered fit rows decoded; "
            "archive/nonfit ID tokens and bytes streamed, only fit bodies decoded; not an air gap"
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
    parser.add_argument("qwen_archive", type=Path)
    parser.add_argument("olmo_archive", type=Path)
    parser.add_argument("store", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source, args.pools, [args.qwen_archive, args.olmo_archive], args.store)
        )
    )
