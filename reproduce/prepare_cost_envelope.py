"""Prepare cost-only inputs with preserved tasks and an eight-demo stress proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reproduce.prepare_preflight_requests import SEEDS


def prepare(source: Path, pools_path: Path) -> list[dict]:
    import pyarrow.parquet as parquet

    pools = json.loads(pools_path.read_text(encoding="utf-8"))
    bfcl = {}
    for path in [
        source / "bfcl-simple-python.jsonl",
        *sorted((source / "bfcl-offline").glob("*.json")),
    ]:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            bfcl[row["id"]] = row
    hotpot = {
        row["id"]: row
        for name in ("hotpot-train-0.parquet", "hotpot-train-1.parquet")
        for row in parquet.read_table(source / name).to_pylist()
    }
    instruction = {
        row["key"]: row for row in parquet.read_table(source / "if-train.parquet").to_pylist()
    }
    original = {
        row["id"]: row
        for line in (source / "preflight-context-requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if (row := json.loads(line))["role"] == "task"
    }
    recorded = {
        row["id"]: row
        for line in (source / "infra1/qwen-infra1.jsonl").read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))["role"] == "task"
    }
    demos = {}
    for family, ids in pools["proposer_context_training_inputs_reserved_for_fit"].items():
        sequence = []
        for item_id in ids:
            key = family + ":" + item_id
            request, response = original[key], recorded[key]
            if response["status"] != "completed" or response["scoring_executed"] is not False:
                raise ValueError("cost demonstration was scored or not completed")
            sequence.extend(
                [
                    {
                        "role": "user",
                        "content": "\n".join(
                            message["content"] for message in request["messages"][1:]
                        ),
                    },
                    {"role": "assistant", "content": response["output"]},
                ]
            )
        demos[family] = sequence * 2
    requests = []
    for family, table, cap in (
        ("bfcl", bfcl, 1024),
        ("hotpotqa", hotpot, 64),
        ("instruction_following", instruction, 4096),
    ):
        available = pools[family]["pools"]["fit"]
        ordered = sorted(
            available,
            key=lambda row: hashlib.sha256(("cost-envelope-v1:" + row["id"]).encode()).hexdigest(),
        )
        if family == "bfcl":
            selected = []
            for category, size in (
                ("simple_python", 6),
                ("multiple", 6),
                ("parallel", 6),
                ("parallel_multiple", 7),
                ("irrelevance", 7),
            ):
                selected.extend([row for row in ordered if row["category"] == category][:size])
        else:
            selected = ordered[:32]
        if len(selected) != 32:
            raise ValueError("training-side cost coverage unavailable")
        for index, metadata in enumerate(selected):
            row = table[metadata["id"]]
            if family == "bfcl":
                messages = [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": row["question"], "functions": row["function"]},
                            ensure_ascii=False,
                        ),
                    }
                ]
            elif family == "hotpotqa":
                context = "\n\n".join(
                    f"{title}: {''.join(sentences)}"
                    for title, sentences in zip(
                        row["context"]["title"], row["context"]["sentences"], strict=True
                    )
                )
                messages = [
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {row['question']}",
                    }
                ]
            else:
                messages = row["messages"]
                if any(message["role"] not in ("user", "system") for message in messages):
                    raise ValueError("training input contains completion")
            profile = "eight_demo_stress_proxy" if index % 2 else "base"
            requests.append(
                {
                    "id": f"{family}:{metadata['id']}",
                    "family": family,
                    "role": "task",
                    "profile": profile,
                    "messages": [
                        {"role": "system", "content": SEEDS[family]},
                        *(demos[family] if index % 2 else []),
                        *messages,
                    ],
                    "max_new_tokens": cap,
                }
            )
    for index, family in enumerate(("bfcl", "hotpotqa", "instruction_following", "hotpotqa")):
        requests.append(
            {
                "id": f"envelope-proposer:{index}",
                "family": family,
                "role": "proposer",
                "profile": "serial_training_context_proxy",
                "messages": [
                    {
                        "role": "user",
                        "content": "Rewrite the task instruction without removing any required "
                        "input, constraint or output interface. Return only the instruction.\n"
                        + SEEDS[family]
                        + "\nPreviously recorded unscored training demonstrations "
                        "(cost proxy only):\n" + json.dumps(demos[family], ensure_ascii=False),
                    }
                ],
                "max_new_tokens": 2048,
            }
        )
    return requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("pools", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = prepare(args.source, args.pools)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "requests_per_model": len(rows),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "proxies_not_native_baseline": True,
                "scores_accessed": False,
            }
        )
    )
