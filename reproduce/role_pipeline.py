"""Real scorer/predictor applications for the restricted research role pipes.

These messages never supply annotations. Scorers open only their granted gold
leaf; predictors have no response field and cannot open gold. The trusted
controller owns provenance, stage order and routing, not a hostile local user.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from promptwitness.incremental.features import ParentTrace, extract_features
from promptwitness.incremental.identity import ExecutionIdentity
from promptwitness.incremental.predictor import TransitionPredictor
from promptwitness.parser import parse_prompt
from reproduce.bfcl_stage import load_staged_bfcl
from reproduce.check_strict_scorers import construction_rng
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, unique_rows
from reproduce.strict_scoring import (
    ScoringError,
    completed_text,
    score_bfcl,
    score_hotpotqa,
    score_iftrain,
)
from reproduce.text_scorer_stage import load_staged

FORMAT = "promptwitness.role-message/v1"
FAMILIES = ("bfcl", "hotpotqa", "instruction_following")
COMMON = {"format", "kind", "family", "model", "execution", "candidate"}
RESPONSE_FIELDS = {
    "id",
    "status",
    "output",
    "replicate",
    "input_tokens",
    "output_tokens",
    "allocated_seconds",
}


def fields(value: dict, expected: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ScoringError("unknown, missing or forbidden role message fields")


def validate(message: dict, role: str, stage: str) -> None:
    if role == "predictor" and stage == "search":
        fields(
            message,
            COMMON | {"parent", "predictor", "structured", "reference", "parent_observations"},
        )
        if message["kind"] != "predict":
            raise ScoringError("predictor requires a prediction message")
        parse_prompt(message["parent"])
        TransitionPredictor.from_dict(message["predictor"])
        if type(message["structured"]) is not bool:
            raise ScoringError("explicit feature mode required")
        if not isinstance(message["parent_observations"], dict):
            raise ScoringError("observed parent trace mapping required")
        for trace in message["parent_observations"].values():
            fields(trace, {"correct", "input_tokens", "output_tokens", "allocated_seconds"})
            ParentTrace(**trace)
    elif (role, stage) in {("search_scorer", "search"), ("selection_scorer", "selection")}:
        fields(message, COMMON | {"responses"})
        if message["kind"] != "score" or not isinstance(message["responses"], list):
            raise ScoringError("scorer requires actual response records")
        records = unique_rows(message["responses"])
        if not records:
            raise ScoringError("empty score request")
        for row in records.values():
            fields(row, RESPONSE_FIELDS)
            completed_text(row["output"], row["status"])
            if not isinstance(row["replicate"], str) or not row["replicate"]:
                raise ScoringError("recorded random unit required")
            ParentTrace(
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                allocated_seconds=row["allocated_seconds"],
            )
    else:
        raise ScoringError("application role/stage not supported")
    if message["format"] != FORMAT or message["family"] not in FAMILIES:
        raise ScoringError("unsupported role format or task")
    model = message["model"]
    identity = ExecutionIdentity(**message["execution"])
    if model not in MODEL_REVISIONS or identity.model_revision != MODEL_REVISIONS[model]:
        raise ScoringError("model differs from the frozen task revision")
    parse_prompt(message["candidate"])


def rows(store: Path, leaf: str, family: str) -> dict:
    with (store / leaf / (family + ".jsonl")).open(encoding="utf-8") as stream:
        return unique_rows(json.loads(line) for line in stream if line.strip())


def score(message: dict, store: Path, runtimes: dict[str, Path], scratch: Path, stage: str) -> dict:
    family = message["family"]
    inputs, gold = rows(store, stage + "/inputs", family), rows(store, stage + "/gold", family)
    if not inputs or set(inputs) != set(gold):
        raise ScoringError("incomplete input/annotation population")
    records = unique_rows(message["responses"])
    if not set(records) <= set(inputs):
        raise ScoringError("response outside granted population")
    if family == "bfcl":
        checker, _ = load_staged_bfcl(runtimes["bfcl"], scratch / "official", message["model"])
    else:
        registry, _, hotpot_em = load_staged(runtimes["text"])
    observations = {}
    for unit, row in records.items():
        annotation = gold[unit]
        with construction_rng():
            if family == "bfcl":
                value = score_bfcl(
                    row["output"],
                    row["status"],
                    inputs[unit]["category"],
                    inputs[unit]["function"],
                    annotation["ground_truth"],
                    checker,
                )
            elif family == "hotpotqa":
                value = score_hotpotqa(
                    row["output"], row["status"], annotation["answer"], hotpot_em
                )
            else:
                value = score_iftrain(
                    row["output"],
                    row["status"],
                    annotation["instruction_id"],
                    annotation["kwargs"],
                    registry,
                )
        observations[unit] = {
            "score": value.value,
            "scorer_profile": value.profile,
            "replicate": row["replicate"],
            "trace": asdict(
                ParentTrace(
                    value.value, row["input_tokens"], row["output_tokens"], row["allocated_seconds"]
                )
            ),
        }
    return {"observations": observations}


def predict(message: dict, store: Path) -> dict:
    inputs = rows(store, "search/inputs", message["family"])
    reference = message["reference"]
    if (
        not inputs
        or not isinstance(reference, dict)
        or set(reference) != set(inputs)
        or any(type(v) is not int or v not in (0, 1) for v in reference.values())
    ):
        raise ScoringError("complete fixed reference required before prediction")
    parent_rows = message["parent_observations"]
    if not set(parent_rows) <= set(inputs):
        raise ScoringError("parent observation outside search population")
    parent, candidate = parse_prompt(message["parent"]), parse_prompt(message["candidate"])
    if parent == candidate:
        raise ScoringError("current candidate cannot supply its own parent trace")
    predictor = TransitionPredictor.from_dict(message["predictor"])
    predictions = {}
    for unit, row in inputs.items():
        # BFCL question structure includes every original turn and function;
        # text records retain all original messages/contexts/constraints.
        text = json.dumps(row, ensure_ascii=False, sort_keys=True)
        trace = ParentTrace(**parent_rows[unit]) if unit in parent_rows else None
        features = extract_features(
            parent, candidate, text, trace=trace, structured=message["structured"]
        )
        predictions[unit] = asdict(predictor.predict(features))
    return {"predictions": predictions, "predictor_status": predictor.status}


def application(store: Path, runtimes: dict[str, Path], scratch: Path, message: dict) -> dict:
    role, stage = os.environ.get("PW_ACCESS_ROLE"), os.environ.get("PW_ACCESS_STAGE")
    validate(message, role, stage)
    result = (
        predict(message, store)
        if role == "predictor"
        else score(message, store, runtimes, scratch, stage)
    )
    return {
        "format": "promptwitness.role-result/v1",
        "kind": message["kind"],
        "family": message["family"],
        "model": message["model"],
        "execution": message["execution"],
        "candidate": message["candidate"],
        "role": role,
        "stage": stage,
        "worker_pid": os.getpid(),
        "landlock_abi": int(os.environ["PW_LANDLOCK_ABI"]),
        **result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "bfcl_runtime", "text_runtime", "scratch"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            application(
                args.store,
                {"bfcl": args.bfcl_runtime, "text": args.text_runtime},
                args.scratch,
                json.load(sys.stdin),
            ),
            allow_nan=False,
        )
    )
