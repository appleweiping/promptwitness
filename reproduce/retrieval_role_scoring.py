"""Restricted CIR scorer entrypoint: rankings in, scorer-only gold read.

No image/model execution, dataset acquisition, final admission, or native
benchmark parity is implied. Search can score a nonempty subset; fit and
selection require a complete supplied population. Final is deliberately not
exposed until a separate scientific admission gate exists.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from reproduce.process_access import AccessBoundaryError, launch_role
from reproduce.retrieval_scoring import RetrievalGold, score_ranking
from reproduce.retrieval_split_audit import RetrievalInputIdentity

FORMAT = "promptwitness.retrieval-role-score/v1"
DATASETS = frozenset({"cirr", "fashioniq"})
STAGES = frozenset({"fit", "search", "selection"})


def _rows(path: Path, expected: frozenset[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != expected:
                raise ValueError("retrieval row has forbidden or missing fields")
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in result:
                raise ValueError("retrieval rows need unique nonempty IDs")
            result[identifier] = row
    if not result:
        raise ValueError("empty retrieval population")
    return result


def _gallery(path: Path, dataset: str) -> dict[str, tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = {"cirr"} if dataset == "cirr" else {"dress", "shirt", "toptee"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("exact retrieval gallery categories required")
    result = {}
    for category, values in raw.items():
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("ordered unique gallery IDs required")
        result[category] = tuple(values)
    return result


def _rankings(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("nonempty rank mapping required")
    result = {}
    for identifier, ranking in value.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(ranking, list)
            or not ranking
            or any(not isinstance(item, str) or not item for item in ranking)
        ):
            raise ValueError("rankings require query IDs and ordered candidate IDs")
        result[identifier] = tuple(ranking)
    return result


def score_worker(store: Path, message: dict[str, Any]) -> dict[str, Any]:
    role = os.environ.get("PW_ACCESS_ROLE")
    stage = os.environ.get("PW_ACCESS_STAGE")
    if not isinstance(message, dict) or set(message) != {"format", "dataset", "stage", "rankings"}:
        raise ValueError("forbidden or missing retrieval scorer message field")
    dataset = message["dataset"]
    if (
        message["format"] != FORMAT
        or dataset not in DATASETS
        or stage not in STAGES
        or message["stage"] != stage
        or role != f"retrieval_{stage}_scorer"
    ):
        raise AccessBoundaryError("retrieval scorer role/stage mismatch")
    inputs = _rows(
        store / stage / "inputs" / f"{dataset}.jsonl",
        frozenset({"id", "reference_id", "modification", "category"}),
    )
    gold_fields = {"id", "target_id", "subset"} if dataset == "cirr" else {"id", "target_id"}
    golds = _rows(store / stage / "gold" / f"{dataset}.jsonl", frozenset(gold_fields))
    if set(inputs) != set(golds):
        raise ValueError("incomplete scorer-only annotation population")
    galleries = _gallery(store / stage / "inputs" / f"{dataset}-gallery.json", dataset)
    rankings = _rankings(message["rankings"])
    if not set(rankings) <= set(inputs) or (stage != "search" and set(rankings) != set(inputs)):
        raise ValueError("scored queries differ from the allowed stage population")
    observations = {}
    for identifier, ranking in rankings.items():
        row = inputs[identifier]
        if not isinstance(row["modification"], str) or not row["modification"]:
            raise ValueError("nonempty modification input required")
        identity = RetrievalInputIdentity(identifier, row["reference_id"], dataset, row["category"])
        annotation = golds[identifier]
        subset = annotation["subset"] if dataset == "cirr" else ()
        if dataset == "cirr" and not isinstance(subset, list):
            raise ValueError("CIRR subset requires ordered IDs")
        gold = RetrievalGold(
            query_id=identity.query_id,
            dataset=dataset,
            reference_id=identity.reference_id,
            target_id=annotation["target_id"],
            category=identity.category,
            subset=tuple(subset),
        )
        pool = galleries[identity.category if dataset == "fashioniq" else "cirr"]
        result = score_ranking(gold, candidate_ids=pool, ranking=ranking)
        observations[identifier] = {
            "primary_hit": result.primary_hit,
            "recalls": dict(result.recalls),
            "subset_recalls": dict(result.subset_recalls),
        }
    return {
        "format": FORMAT,
        "dataset": dataset,
        "stage": stage,
        "role": role,
        "worker_pid": os.getpid(),
        "landlock_abi": int(os.environ["PW_LANDLOCK_ABI"]),
        "observations": observations,
    }


def score_rankings_restricted(
    store: Path,
    scratch: Path,
    dataset: str,
    stage: str,
    rankings: dict[str, tuple[str, ...]],
    *,
    timeout: float = 900,
) -> dict[str, Any]:
    """Use only the CIR scorer role; failure is never imputed as score zero."""
    if dataset not in DATASETS or stage not in STAGES:
        raise AccessBoundaryError("unsupported retrieval dataset or unadmitted stage")
    if not rankings or any(
        not isinstance(identifier, str)
        or not identifier
        or not isinstance(ranking, tuple)
        or not ranking
        for identifier, ranking in rankings.items()
    ):
        raise ValueError("nonempty immutable rank mapping required")
    message = {
        "format": FORMAT,
        "dataset": dataset,
        "stage": stage,
        "rankings": {identifier: list(ranking) for identifier, ranking in rankings.items()},
    }
    role = f"retrieval_{stage}_scorer"
    root = store.resolve(strict=True)
    result = launch_role(
        role,
        stage,
        root,
        scratch,
        Path(__file__),
        [str(root)],
        timeout=timeout,
        message=message,
    )
    if result.returncode:
        raise ValueError(f"restricted retrieval scorer failed with exit {result.returncode}")
    report = json.loads(result.stdout)
    if (
        not isinstance(report, dict)
        or report.get("format") != FORMAT
        or report.get("dataset") != dataset
        or report.get("stage") != stage
        or report.get("role") != role
        or type(report.get("landlock_abi")) is not int
        or report["landlock_abi"] < 1
        or type(report.get("worker_pid")) is not int
        or report["worker_pid"] <= 0
        or report["worker_pid"] == os.getpid()
        or not isinstance(report.get("observations"), dict)
        or set(report["observations"]) != set(rankings)
    ):
        raise ValueError("restricted retrieval scorer returned invalid identity or coverage")
    for observation in report["observations"].values():
        if (
            not isinstance(observation, dict)
            or set(observation) != {"primary_hit", "recalls", "subset_recalls"}
            or type(observation["primary_hit"]) is not int
            or observation["primary_hit"] not in (0, 1)
        ):
            raise ValueError("restricted retrieval scorer returned invalid observations")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: retrieval_role_scoring.py STORE")
    print(json.dumps(score_worker(Path(sys.argv[1]), json.load(sys.stdin)), allow_nan=False))
