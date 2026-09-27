"""Fixed training-side task wires and original typed proposer cost fixtures.

One input-only fit unit per family, chosen before responses by fixed hash order.
The two original proposer fixtures may contain previously disclosed fit gold;
neither is a full native proposer workflow. No search/selection/final body is
opened, no correctness filtering or new task constraint is introduced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, unique_rows, write_jsonl
from reproduce.prepare_preflight_requests import SEEDS
from reproduce.torch_runtime import TASK_CAPS

PROPOSER_SOURCE_SHA = "59c619982e613b0d6e2200253eed071c75646a74a830f64a0e2a5b715e7a50d3"


def prepare(store: Path, original_requests: Path) -> list[dict]:
    raw = original_requests.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROPOSER_SOURCE_SHA:
        raise ValueError("immutable original native context cost fixture changed")
    original = unique_rows(json.loads(line) for line in raw.decode("utf-8").splitlines())
    tasks = []
    for family in TASK_CAPS:
        with (store / "fit/inputs" / (family + ".jsonl")).open(encoding="utf-8") as stream:
            population = unique_rows(json.loads(line) for line in stream if line.strip())
        if len(population) != 512:
            raise ValueError("original prepared full 512-unit fit population required")
        unit = min(
            population,
            key=lambda u: hashlib.sha256(("backendtransport-v1:" + u).encode()).hexdigest(),
        )
        tasks.append(
            {
                "id": "task:" + family + ":" + unit,
                "purpose": "fit",
                "family": family,
                "unit": population[unit],
                "candidate": {
                    "schema_version": 1,
                    "id": "original-seed-" + family,
                    "messages": [{"id": "instruction", "role": "system", "content": SEEDS[family]}],
                },
            }
        )
    proposers = []
    for family in ("hotpotqa", "instruction_following"):
        row = original["typed-proposer:" + family]
        if row["role"] != "proposer" or row["max_new_tokens"] != 2048:
            raise ValueError("original typed proposer fixture interface differs")
        proposers.append({"id": row["id"], "purpose": "proposer", "original": row})
    return [
        {"model": model, **copy.deepcopy(row)}
        for model in MODEL_REVISIONS
        for row in (*tasks, *proposers)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "original_requests", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    records = prepare(args.store, args.original_requests)
    # The canonical JSONL writer expects ID uniqueness only within a model.
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "requests": len(records),
                "requests_per_model": 5,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "final_body_accessed": False,
                "scoring_executed": False,
            }
        )
    )
