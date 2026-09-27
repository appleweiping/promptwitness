"""Check training-only verifier IDs/arguments; never score model completions."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


def audit(source: Path, upstream: Path, pools_path: Path) -> dict:
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    sys.path.insert(0, str(upstream))
    from open_instruct.IFEvalG import instructions_registry

    plan = json.loads(pools_path.read_text(encoding="utf-8"))
    selected = {
        row["id"] for rows in plan["instruction_following"]["pools"].values() for row in rows
    }
    table = parquet.read_table(source / "if-train.parquet", columns=["key", "ground_truth"])
    table = table.filter(arrow.compute.is_in(table["key"], value_set=arrow.array(sorted(selected))))
    if table.num_rows != 1024:
        raise ValueError("training pool annotation coverage changed")
    checks = Counter()
    unsupported = []
    random_default_rows = []
    for row in table.to_pylist():
        annotation = ast.literal_eval(row["ground_truth"])[0]
        ids, kwargs = annotation["instruction_id"], annotation["kwargs"]
        if len(ids) != len(kwargs):
            raise ValueError("instruction/argument alignment broken")
        for identifier, arguments in zip(ids, kwargs, strict=True):
            checks[identifier] += 1
            if identifier not in instructions_registry.INSTRUCTION_DICT:
                unsupported.append(
                    {"unit": row["key"], "instruction_id": identifier, "reason": "unknown verifier"}
                )
                continue
            arguments = {
                key: value for key, value in (arguments or {}).items() if value is not None
            }
            states = []
            for seed in (11, 71):
                random.seed(seed)
                checker = instructions_registry.INSTRUCTION_DICT[identifier](identifier)
                try:
                    description = checker.build_description(**copy.deepcopy(arguments))
                except (TypeError, ValueError, KeyError) as error:
                    unsupported.append(
                        {
                            "unit": row["key"],
                            "instruction_id": identifier,
                            "reason": type(error).__name__,
                        }
                    )
                    break
                states.append((description, copy.deepcopy(checker.__dict__)))
            if len(states) == 2 and states[0] != states[1]:
                random_default_rows.append(
                    {
                        "unit": row["key"],
                        "instruction_id": identifier,
                        "declared_arguments": arguments,
                        "states_by_seed": [state[1] for state in states],
                    }
                )
    return {
        "format": "promptwitness.delta.training-verifier-audit/v1",
        "status": "PASSED_METADATA_ARGUMENT_CHECK"
        if not unsupported and not random_default_rows
        else "UNRESOLVED_VERIFIER_SEMANTICS",
        "source_commit": "99b1ee970490a2d0d5664663eb0b1acc410b944c",
        "units": table.num_rows,
        "constraint_instances": sum(checks.values()),
        "instruction_registry_keys": len(instructions_registry.INSTRUCTION_DICT),
        "covered_training_keys": dict(sorted(checks.items())),
        "unsupported": unsupported,
        "random_default_constraint_instances": random_default_rows,
        "model_responses_or_final_data_scored": False,
        "annotation_sha256": hashlib.sha256((source / "if-train.parquet").read_bytes()).hexdigest(),
        "pool_proposal_sha256": hashlib.sha256(pools_path.read_bytes()).hexdigest(),
        "limitations": [
            "Build-description checks do not establish semantic validity "
            "of every verifier or instruction.",
            "Exact row annotations are preserved. Training reward averaging is not the binary "
            "strict prompt metric; all required checks must pass.",
            "IFBench uses its separately pinned strict scorer for final testing; "
            "no IFBench prompt or response is read here.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("pools", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.source, args.upstream, args.pools)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "units",
                    "constraint_instances",
                    "instruction_registry_keys",
                    "unsupported",
                    "random_default_constraint_instances",
                )
            },
            indent=2,
        )
    )
