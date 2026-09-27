"""Audit legal training-side group availability, without printing questions or labels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


class Components:
    """Small union-find for shared group identifiers (not a statistical model)."""

    def __init__(self) -> None:
        self.parents: dict[str, str] = {}

    def root(self, key: str) -> str:
        self.parents.setdefault(key, key)
        while self.parents[key] != key:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def join(self, left: str, right: str) -> None:
        a, b = self.root(left), self.root(right)
        self.parents[max(a, b)] = min(a, b)


def component_counts(rows: list[tuple[str, set[str]]]) -> dict:
    components = Components()
    for item_id, groups in rows:
        for group in groups:
            components.join("row:" + item_id, "group:" + group)
    sizes = Counter(components.root("row:" + item_id) for item_id, _ in rows)
    return {
        "rows": len(rows),
        "components": len(sizes),
        "largest_component_rows": sorted(sizes.values(), reverse=True)[:10],
    }


def audit(source: Path) -> dict:
    import pyarrow.parquet as parquet

    bfcl = []
    categories = {}
    files = [source / "bfcl-simple-python.jsonl", *sorted((source / "bfcl-offline").glob("*.json"))]
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        categories[path.name] = len(rows)
        bfcl.extend((row["id"], {function["name"] for function in row["function"]}) for row in rows)
    hotpot = []
    for name in ("hotpot-train-0.parquet", "hotpot-train-1.parquet"):
        rows = parquet.read_table(source / name, columns=["id", "context.title"]).to_pylist()
        for row in rows:
            titles = row["context"]["title"] if "context" in row else row["title"]
            if not isinstance(titles, list) or not all(isinstance(title, str) for title in titles):
                raise ValueError("unknown context-title projection")
            hotpot.append((row["id"], set(titles)))
    instruction = parquet.read_table(
        source / "if-train.parquet", columns=["key", "ground_truth"]
    ).to_pylist()
    constraint_types = Counter()
    instruction_groups = []
    for row in instruction:
        annotation = ast.literal_eval(row["ground_truth"])
        if not isinstance(annotation, list) or len(annotation) != 1:
            raise ValueError("unknown training constraint annotation")
        value = annotation[0]["instruction_id"]
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) for item in value)
        ):
            raise ValueError("unknown constraint-family metadata; cannot assert separation")
        constraint_types.update(value)
        instruction_groups.append((row["key"], set(value)))
    return {
        "format": "promptwitness.delta.training-group-audit/v1",
        "scores_or_final_test_accessed": False,
        "bfcl_categories": categories,
        "bfcl_exact_function_groups": component_counts(bfcl),
        "hotpot_all_context_document_groups": component_counts(hotpot),
        "if_training_constraint_groups": component_counts(instruction_groups),
        "if_training_constraint_counts": dict(sorted(constraint_types.items())),
        "source_file_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
        "warning": (
            "Whole-corpus components are diagnostics, not a frozen split; no model labels used. "
            "Instruction identifiers are read from training-side verifier annotations, "
            "not the generic constraint_type='multi' descriptor."
        ),
        "supersedes": (
            "training-groups-v1.json: constraint_type was a wrapper descriptor, "
            "not a family identifier; its IF group results are invalid."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--inspect-constraint-metadata", action="store_true")
    args = parser.parse_args()
    if args.inspect_constraint_metadata:
        import pyarrow.parquet as parquet

        table = parquet.read_table(
            args.source / "if-train.parquet",
            columns=["constraint_type", "constraint", "ground_truth"],
        )
        row = table.slice(0, 1).to_pylist()[0]
        summary = {key: {"type": type(value).__name__} for key, value in row.items()}
        for key, value in row.items():
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    summary[key]["descriptor"] = value[:120]
                else:
                    summary[key]["json_type"] = type(parsed).__name__
                    if isinstance(parsed, dict):
                        summary[key]["keys"] = list(parsed)
                        summary[key]["instruction_ids"] = parsed.get("instruction_id_list")
            elif isinstance(value, dict):
                summary[key]["keys"] = list(value)
        print(json.dumps(summary, indent=2))
        return
    if args.output is None:
        parser.error("output is required unless inspecting training metadata")
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
