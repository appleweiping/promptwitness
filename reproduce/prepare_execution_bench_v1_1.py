"""Freeze 32 input-only calibration requests; no scientific score selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def prepare(source: Path) -> list[dict]:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != (
        "59c619982e613b0d6e2200253eed071c75646a74a830f64a0e2a5b715e7a50d3"
    ):
        raise ValueError("the frozen v1 training-side request source changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    proposer = next(row for row in rows if row["id"] == "typed-proposer:hotpotqa")
    text = proposer["messages"][-1]["content"]
    demos = json.loads(
        text.split("[[ ## task_demos ## ]]\n", 1)[1]
        .split("[[ ## previous_instructions ## ]]", 1)[0]
        .strip()
    )
    if len(demos) != 8 or [m["role"] for m in demos] != ["user", "assistant"] * 4:
        raise ValueError("four original fit-only Hotpot demonstrations required")
    selected = []
    for family, profile in (
        ("bfcl", "base"),
        ("hotpotqa", "base"),
        ("instruction_following", "base"),
        ("instruction_following", "four_valid_demo_cost_context"),
    ):
        group = sorted(
            (
                row
                for row in rows
                if row["family"] == family
                and row["profile"] == profile
                and row.get("repeat_of") is None
            ),
            key=lambda row: (sum(len(m["content"]) for m in row["messages"]), row["id"]),
        )
        if len(group) != 16:
            raise ValueError("frozen coverage changed")
        # Fixed length quantiles, including both extremes. Not output filtering.
        for index in (0, 2, 4, 6, 9, 11, 13, 15):
            row = copy.deepcopy(group[index])
            if family == "hotpotqa":
                row["messages"] = [row["messages"][0], *demos, row["messages"][-1]]
                row["profile"] = "four_fit_gold_demo_cost_context"
            row["id"] = "execution-v1.1:" + row["id"]
            selected.append(row)
    if len(selected) != 32 or len({row["id"] for row in selected}) != 32:
        raise ValueError("32 unique complete training-side requests required")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        for row in prepare(args.source):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps({"requests": 32, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})
    )
