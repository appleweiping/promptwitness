"""Compare authored-ranking metrics with pinned SEARLE validation function bodies.

No benchmark records, model weights or upstream code are committed. The caller
supplies a private copy of SEARLE's pinned validate.py. Only its two metric
function definitions are compiled; their prediction generators are replaced
with authored, tie-free CPU features. This is not official server parity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, cast

from reproduce.retrieval_scoring import RetrievalGold, summarize_rankings

FORMAT = "promptwitness.searle-metric-parity/v1"
SOURCE_COMMIT = "a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37"
SOURCE_SHA256 = "83eef3dd2448e82ba9ef59708df8dfa3a88a8f17ec1d4c0f152676dc5dde28fc"
FUNCTIONS = frozenset({"fiq_compute_val_metrics", "cirr_compute_val_metrics"})


def _verified_functions(source: Path) -> dict[str, Any]:
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError("pinned SEARLE validate.py source SHA-256 mismatch")
    parsed = ast.parse(raw.decode("utf-8"), filename=str(source))
    selected = {
        node.name: node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS
    }
    if set(selected) != FUNCTIONS or any(
        not isinstance(node, ast.FunctionDef) for node in selected.values()
    ):
        raise ValueError("pinned SEARLE metric functions are missing")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *selected.values()], type_ignores=[])
    )
    import numpy as np
    import torch
    import torch.nn.functional as functional

    namespace: dict[str, Any] = {
        "torch": torch,
        "np": np,
        "F": functional,
        "device": torch.device("cpu"),
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def _features(torch: Any, prefix: str) -> tuple[tuple[str, ...], Any, Any]:
    names = tuple(f"authored-{prefix}-{index:02d}" for index in range(55))
    scores = [0.9 - index * 0.015 for index in range(len(names))]
    index = torch.tensor(
        [(score, math.sqrt(1 - score * score)) for score in scores], dtype=torch.float32
    )
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    return names, index, query


def _compare(label: str, observed: float, expected: float) -> dict[str, float]:
    if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-4):
        raise AssertionError(f"{label} diverged from pinned SEARLE metric body")
    return {"upstream_percent": observed, "ours_percent": expected}


def check(source: Path, output: Path) -> dict[str, object]:
    """Run tie-free, authored CIRR/FashionIQ metric-body differentials once."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicit empty CUDA_VISIBLE_DEVICES required")
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("private parity output must be outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, object] = {
        "format": FORMAT,
        "status": "RUNNING",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "authored_data_only": True,
        "source_metric_function_bodies_executed": False,
        "official_server_parity_established": False,
        "official_full_gallery_parity_established": False,
        "ties_qualified": False,
        "model_forward_calls": 0,
        "scientific_result": False,
    }
    try:
        namespace = _verified_functions(source)
        torch = namespace["torch"]
        cirr_names, cirr_index, query = _features(torch, "cirr")
        specs = (
            (0, 1, (0, 1, 2, 3, 4, 5)),
            (0, 5, (0, 4, 5, 10, 20, 30)),
            (54, 10, (0, 1, 10, 20, 30, 54)),
            (0, 50, (0, 1, 10, 50, 51, 52)),
            (54, 50, (0, 1, 10, 50, 53, 54)),
        )
        cirr_golds = [
            RetrievalGold(
                f"authored-cirr-{number}",
                "cirr",
                cirr_names[reference],
                cirr_names[target],
                subset=tuple(cirr_names[index] for index in group),
            )
            for number, (reference, target, group) in enumerate(specs)
        ]
        namespace["cirr_generate_val_predictions"] = lambda *_: (
            query.repeat(len(specs), 1),
            [gold.reference_id for gold in cirr_golds],
            [gold.target_id for gold in cirr_golds],
            [list(gold.subset) for gold in cirr_golds],
        )
        cirr_upstream = namespace["cirr_compute_val_metrics"](
            None, None, cirr_index, list(cirr_names), [], None
        )
        cirr_ours = summarize_rankings(
            cirr_golds,
            {gold.query_id: cirr_names for gold in cirr_golds},
            {"cirr": cirr_names},
        )
        cirr_micro = cast(dict[str, float], cirr_ours["query_micro_recall"])
        cirr_subset = cast(dict[str, float], cirr_ours["subset_recall"])
        cirr_checks = {
            key: _compare(
                f"cirr R@{key}",
                float(cirr_upstream[f"cirr_recall_at{key}"]),
                100 * cirr_micro[key],
            )
            for key in ("1", "5", "10", "50")
        }
        subset_checks = {
            key: _compare(
                f"cirr subset R@{key}",
                float(cirr_upstream[f"cirr_group_recall_at{key}"]),
                100 * cirr_subset[key],
            )
            for key in ("1", "2", "3")
        }

        fashion_specs = {"dress": (9,), "shirt": (10, 50), "toptee": (9, 10, 49)}
        fashion_golds: list[RetrievalGold] = []
        fashion_rankings: dict[str, tuple[str, ...]] = {}
        fashion_pools: dict[str, tuple[str, ...]] = {}
        fashion_upstream: dict[str, dict[str, float]] = {}
        for category, targets in fashion_specs.items():
            names, index, category_query = _features(torch, category)
            fashion_pools[category] = names
            golds = [
                RetrievalGold(
                    f"authored-{category}-{number}",
                    "fashioniq",
                    names[0],
                    names[target],
                    category=category,
                )
                for number, target in enumerate(targets)
            ]
            fashion_golds.extend(golds)
            fashion_rankings.update({gold.query_id: names for gold in golds})
            namespace["fiq_generate_val_predictions"] = lambda *_, q=category_query, rows=golds: (
                q.repeat(len(rows), 1),
                [gold.target_id for gold in rows],
            )
            fashion_upstream[category] = namespace["fiq_compute_val_metrics"](
                None, None, index, list(names), [], None
            )
        fashion_ours = summarize_rankings(fashion_golds, fashion_rankings, fashion_pools)
        fashion_categories = cast(dict[str, dict[str, float]], fashion_ours["category_recalls"])
        fashion_macro = cast(dict[str, float], fashion_ours["category_macro_recall"])
        fashion_micro = cast(dict[str, float], fashion_ours["query_micro_recall"])
        fashion_checks = {
            category: {
                key: _compare(
                    f"fashioniq {category} R@{key}",
                    float(values[f"fiq_recall_at{key}"]),
                    100 * fashion_categories[category][key],
                )
                for key in ("10", "50")
            }
            for category, values in fashion_upstream.items()
        }
        for key in ("10", "50"):
            macro = sum(float(row[f"fiq_recall_at{key}"]) for row in fashion_upstream.values()) / 3
            micro = sum(
                float(fashion_upstream[category][f"fiq_recall_at{key}"]) * len(targets)
                for category, targets in fashion_specs.items()
            ) / len(fashion_golds)
            _compare(f"fashioniq macro R@{key}", macro, 100 * fashion_macro[key])
            _compare(f"fashioniq micro R@{key}", micro, 100 * fashion_micro[key])
        report["source_metric_function_bodies_executed"] = True
        report.update(
            status="PASS_PINNED_SEARLE_METRIC_BODY_AUTHORED_NO_TIES",
            cirr_queries=len(cirr_golds),
            cirr_gallery_size=len(cirr_names),
            fashioniq_queries=len(fashion_golds),
            fashioniq_gallery_size_per_category=55,
            cirr_recall=cirr_checks,
            cirr_subset_recall=subset_checks,
            fashioniq_category_recall=fashion_checks,
            fashioniq_macro_recall=fashion_macro,
            fashioniq_micro_recall=fashion_micro,
        )
        return report
    except BaseException as error:
        report.update(status="FAILED_RETAINED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["checker_wall_seconds"] = time.monotonic() - started
        (output / "qualification.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pinned_source", type=Path)
    parser.add_argument("new_private_output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.pinned_source, args.new_private_output), allow_nan=False))


if __name__ == "__main__":
    main()
