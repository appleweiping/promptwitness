"""Plan outcome-blind, group-isolated derived pools from training metadata only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from reproduce.audit_training_groups import Components

TARGETS = {"fit": 512, "search": 256, "selection": 256}


def stable_key(value: str) -> str:
    return hashlib.sha256(("delta-pools-v1:" + value).encode()).hexdigest()


def invalid_paragraph_index(identifier: str, arguments: dict | None) -> bool:
    """Reject inconsistent metadata, without looking at any generated response."""
    if identifier != "length_constraints:nth_paragraph_first_word":
        return False
    arguments = arguments or {}
    count, nth = arguments.get("num_paragraphs"), arguments.get("nth_paragraph")
    return type(count) is not int or type(nth) is not int or not 1 <= nth <= count


def verify(pools: dict[str, list[dict]]) -> dict:
    owners: dict[str, str] = {}
    ids: set[str] = set()
    for pool, rows in pools.items():
        for row in rows:
            if row["id"] in ids:
                raise ValueError("duplicate pool unit")
            ids.add(row["id"])
            for group in row["groups"]:
                previous = owners.setdefault(group, pool)
                if previous != pool:
                    raise ValueError("cross-pool group leakage")
    return {
        "sizes": {pool: len(rows) for pool, rows in pools.items()},
        "shared_groups_across_pools": 0,
        "unique_group_count": len(owners),
        "category_counts": {
            pool: dict(sorted(Counter(row.get("category", "unspecified") for row in rows).items()))
            for pool, rows in pools.items()
        },
    }


def function_pools(rows: list[dict], exposed: set[str], reserved_fit: set[str]) -> dict:
    """Whole exact-function components, including unused members, have one owner."""
    components = Components()
    for row in rows:
        components.root("row:" + row["id"])
        for group in row["groups"]:
            components.join("row:" + row["id"], "group:" + group)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[components.root("row:" + row["id"])].append(row)
    pools: dict[str, list[dict]] = {name: [] for name in (*TARGETS, "final_derived")}
    pending = []
    excluded_exposed = 0
    for key, members in sorted(
        groups.items(), key=lambda pair: (-len(pair[1]), stable_key(pair[0]))
    ):
        if any(row["id"] in reserved_fit for row in members):
            pools["fit"].extend(sorted(members, key=lambda row: stable_key(row["id"])))
        else:
            pending.append((key, members))
    # Metadata-driven assignment, not a search over task/model performance.
    for _, members in pending:
        available = [pool for pool, target in TARGETS.items() if len(pools[pool]) < target]
        if available:
            pool = max(available, key=lambda name: (TARGETS[name] - len(pools[name]), name))
            pools[pool].extend(sorted(members, key=lambda row: stable_key(row["id"])))
        elif not any(row["id"] in exposed for row in members):
            pools["final_derived"].extend(sorted(members, key=lambda row: stable_key(row["id"])))
        else:
            excluded_exposed += len(members)
    # Unused members of an assigned component cannot enter any other pool.
    for pool, target in TARGETS.items():
        if pool == "fit":
            pools[pool].sort(key=lambda row: (row["id"] not in reserved_fit, stable_key(row["id"])))
        pools[pool] = pools[pool][:target]
    if not reserved_fit <= {row["id"] for row in pools["fit"]}:
        raise ValueError("could not reserve function proposer inputs")
    pools["final_derived"] = pools["final_derived"][:256]
    return {
        "pools": pools,
        "audit": verify(pools),
        "excluded_exposed_component_rows_after_training_targets": excluded_exposed,
        "all_context_preserved": True,
        "warning": (
            "BFCL-derived exact-function grouping, not a full official leaderboard split. "
            "Exact names are a proxy, not a verified semantic tool-family taxonomy. "
            "Final size may be below 256 because the selected five categories have only "
            "1240 rows and components are indivisible across pools."
        ),
    }


def document_pools(rows: list[dict], reserved_fit: set[str]) -> dict:
    """Select complete rows and exclude bridges; do not remove distractor documents."""
    pools: dict[str, list[dict]] = {name: [] for name in TARGETS}
    owners: dict[str, str] = {}
    chosen: set[str] = set()
    ordered = sorted(rows, key=lambda row: (row["id"] not in reserved_fit, stable_key(row["id"])))
    conflicts = 0
    for row in ordered:
        if all(len(pools[pool]) >= target for pool, target in TARGETS.items()):
            break
        available = [pool for pool, target in TARGETS.items() if len(pools[pool]) < target]
        preferred = (
            "fit"
            if row["id"] in reserved_fit
            else max(available, key=lambda name: (TARGETS[name] - len(pools[name]), name))
        )
        possible = [
            pool
            for pool in (preferred, *available)
            if pool in available and all(owners.get(group, pool) == pool for group in row["groups"])
        ]
        if not possible:
            conflicts += 1
            continue
        pool = possible[0]
        pools[pool].append(row)
        chosen.add(row["id"])
        for group in row["groups"]:
            owners[group] = pool
    if not reserved_fit <= {row["id"] for row in pools["fit"]}:
        raise ValueError("could not reserve previously exposed proposer inputs for fit")
    return {
        "pools": pools,
        "audit": verify(pools),
        "conflicting_rows_skipped_before_targets": conflicts,
        "unused_rows": len(rows) - len(chosen),
        "all_context_preserved": True,
        "warning": (
            "Metadata-selected document-disjoint training subset; excluded bridge rows create "
            "selection bias. Not a random split of the giant whole-corpus component. "
            "Dev final pool remains unselected/unaccessed here."
        ),
    }


def constraint_pools(rows: list[dict], reserved_fit: set[str]) -> dict:
    """Assign verifier IDs before outcomes and exclude cross-partition conjunctions."""
    all_groups = sorted({group for row in rows for group in row["groups"]}, key=stable_key)
    forced = {group for row in rows if row["id"] in reserved_fit for group in row["groups"]}
    owners = {group: "fit" for group in forced}
    remainder = [group for group in all_groups if group not in forced]
    # Fixed rank proportions; not revised in response to accuracy or pool counts.
    for index, group in enumerate(remainder):
        fraction = (index + 0.5) / len(remainder)
        owners[group] = "fit" if fraction < 0.5 else "search" if fraction < 0.75 else "selection"
    pools: dict[str, list[dict]] = {name: [] for name in TARGETS}
    eligible = Counter()
    conflicts = 0
    for row in sorted(
        rows, key=lambda item: (item["id"] not in reserved_fit, stable_key(item["id"]))
    ):
        destinations = {owners[group] for group in row["groups"]}
        if len(destinations) != 1:
            conflicts += 1
            continue
        pool = destinations.pop()
        eligible[pool] += 1
        if len(pools[pool]) < TARGETS[pool]:
            pools[pool].append(row)
    if not reserved_fit <= {row["id"] for row in pools["fit"]}:
        raise ValueError("could not reserve instruction proposer inputs")
    return {
        "pools": pools,
        "audit": verify(pools),
        "eligible_training_rows": dict(sorted(eligible.items())),
        "cross_partition_conjunctions_excluded": conflicts,
        "constraint_group_assignment": dict(sorted(owners.items())),
        "all_original_constraints_preserved": True,
        "warning": (
            "Training verifier-ID-disjoint derived subset. Related semantic families may share "
            "prefixes. IFBench includes official training-like and OOD constraints; this does not "
            "assert all training/final verifier IDs are disjoint. "
            "Final test remains inaccessible to this script."
        ),
    }


def plan(source: Path) -> dict:
    import pyarrow.parquet as parquet

    bfcl = []
    for path in [
        source / "bfcl-simple-python.jsonl",
        *sorted((source / "bfcl-offline").glob("*.json")),
    ]:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            bfcl.append(
                {
                    "id": row["id"],
                    "category": row["id"].rsplit("_", 1)[0],
                    "groups": sorted({item["name"] for item in row["function"]}),
                }
            )
    hotpot = []
    for name in ("hotpot-train-0.parquet", "hotpot-train-1.parquet"):
        for row in parquet.read_table(source / name, columns=["id", "context.title"]).to_pylist():
            titles = row["context"]["title"] if "context" in row else row["title"]
            hotpot.append({"id": row["id"], "groups": sorted(set(titles))})
    instruction = []
    invalid_instruction_metadata = []
    for row in parquet.read_table(
        source / "if-train.parquet", columns=["key", "ground_truth"]
    ).to_pylist():
        annotation = ast.literal_eval(row["ground_truth"])
        invalid = False
        if len(annotation[0]["instruction_id"]) != len(annotation[0]["kwargs"]):
            raise ValueError("instruction/argument metadata misaligned")
        for identifier, arguments in zip(
            annotation[0]["instruction_id"], annotation[0]["kwargs"], strict=True
        ):
            if invalid_paragraph_index(identifier, arguments):
                invalid_instruction_metadata.append(
                    {
                        "id": row["key"],
                        "instruction_id": identifier,
                        "arguments": arguments or {},
                        "reason": (
                            "Declared paragraph index outside the declared paragraph count; "
                            "upstream silently replaces it with a random index."
                        ),
                    }
                )
                invalid = True
        if invalid:
            continue
        instruction.append(
            {"id": row["key"], "groups": sorted(set(annotation[0]["instruction_id"]))}
        )
    probes = [
        json.loads(line)
        for line in (source / "preflight-context-requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    exposed: dict[str, list[str]] = defaultdict(list)
    for row in probes:
        if row["role"] == "task":
            exposed[row["family"]].append(row["id"].split(":", 1)[1])
    result = {
        "format": "promptwitness.delta.training-pool-plan/v2",
        "status": "METADATA_PROPOSAL_NOT_FORMAL_FREEZE",
        "scores_accessed": False,
        "targets": TARGETS,
        "proposer_context_training_inputs_reserved_for_fit": {
            family: ids[:4] for family, ids in exposed.items()
        },
        "bfcl": function_pools(bfcl, set(exposed["bfcl"]), set(exposed["bfcl"][:4])),
        "hotpotqa": document_pools(hotpot, set(exposed["hotpotqa"][:4])),
        "instruction_following": constraint_pools(
            instruction, set(exposed["instruction_following"][:4])
        ),
        "instruction_metadata_exclusions": invalid_instruction_metadata,
        "metadata_exclusion_policy": (
            "All training rows with an invalid declared paragraph index are excluded BEFORE "
            "formal freeze, irrespective of model outputs. No prompt/requirement is rewritten "
            "or relaxed, and no final-test row is borrowed. Other unsupported semantics fail "
            "the readiness audit. This is a disclosed derived subset, not untouched IF-RLVR."
        ),
        "supersedes": (
            "training-pools-proposed-v1.json: scorer-construction audit identified four "
            "selected instances whose out-of-range paragraph index was randomized. "
            "The general metadata predicate is applied across the entire training corpus."
        ),
        "final_other_families": (
            "Official HotpotQA dev / IFBench sealed auditor-only artifacts; "
            "not read or selected by this script."
        ),
    }
    result["training_targets_satisfied"] = all(
        result[family]["audit"]["sizes"].get(pool) == target
        for family in ("bfcl", "hotpotqa", "instruction_following")
        for pool, target in TARGETS.items()
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = plan(args.source)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "training_targets_satisfied": result["training_targets_satisfied"],
                **{
                    family: result[family]["audit"]
                    for family in ("bfcl", "hotpotqa", "instruction_following")
                },
            },
            indent=2,
        )
    )
