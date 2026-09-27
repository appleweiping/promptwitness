"""Create training-side requests for a cost probe, without reading final-test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = {
    "bfcl": (
        "Select the required functions. "
        "Return only a JSON array of objects with name and arguments. "
        "Return [] if no function is appropriate."
    ),
    "hotpotqa": (
        "Answer the question using the supplied context. "
        "Return only the short answer, without explanation."
    ),
    "instruction_following": (
        "Follow every instruction in the user's request precisely. "
        "Return only the requested response."
    ),
}


def select(rows: list[dict], key: str) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(str(row[key]).encode()).hexdigest())[:32]


def prepare(source: Path) -> list[dict]:
    import pyarrow.parquet as parquet

    bfcl = [
        json.loads(line) for line in (source / "bfcl-simple-python.jsonl").read_text().splitlines()
    ]
    hotpot = (
        parquet.read_table(source / "hotpot-train-0.parquet").to_pylist()
        + parquet.read_table(source / "hotpot-train-1.parquet").to_pylist()
    )
    if len(hotpot) != 90447:
        raise ValueError("unexpected HotpotQA training population size")
    instruction = parquet.read_table(source / "if-train.parquet").to_pylist()
    requests = []
    for family, rows, key, limit in (
        ("bfcl", bfcl, "id", 256),
        ("hotpotqa", hotpot, "id", 64),
        ("instruction_following", instruction, "key", 512),
    ):
        for row in select(rows, key):
            if family == "bfcl":
                content = json.dumps(
                    {"question": row["question"], "functions": row["function"]}, ensure_ascii=False
                )
                messages = [{"role": "user", "content": content}]
            elif family == "hotpotqa":
                paragraphs = "\n\n".join(
                    f"{title}: {''.join(sentences)}"
                    for title, sentences in zip(
                        row["context"]["title"], row["context"]["sentences"], strict=True
                    )
                )
                messages = [
                    {
                        "role": "user",
                        "content": f"Context:\n{paragraphs}\n\nQuestion: {row['question']}",
                    }
                ]
            else:
                messages = row["messages"]
                if any(message["role"] not in ("user", "system") for message in messages):
                    raise ValueError(
                        "training record contains a completion; cannot use as a probe input"
                    )
            requests.append(
                {
                    "id": f"{family}:{row[key]}",
                    "family": family,
                    "role": "task",
                    "messages": [{"role": "system", "content": SEEDS[family]}, *messages],
                    "max_new_tokens": limit,
                }
            )
    for index in range(4):
        family = tuple(SEEDS)[index % 3]
        examples = [row["messages"] for row in requests if row["family"] == family][:4]
        requests.append(
            {
                "id": f"proposer:{index}",
                "family": family,
                "role": "proposer",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Rewrite this task instruction to be clear and precise, "
                            "without changing its output interface. "
                            "Return only the rewritten instruction.\n"
                        )
                        + SEEDS[family]
                        + "\nRepresentative training inputs (no reference answers):\n"
                        + json.dumps(examples, ensure_ascii=False),
                    }
                ],
                "max_new_tokens": 512,
            }
        )
    return requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = prepare(args.source)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "requests_per_model": len(rows),
                "models": 2,
                "total_planned": 2 * len(rows),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "split": "training_side_probe_excluded_from_final",
            }
        )
    )
