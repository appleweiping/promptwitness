"""Selected published metric kernels on authored features, never image results.

The caller supplies trusted SEARLE validate.py at the revision below. Its two
original metric function ASTs execute without importing its data/model loader.
Only generation is replaced with authored feature tensors. No upstream code is
vendored or relicensed; the private original source retains CC BY-NC4.0.
This is not the original whole CLI or official benchmark/encoder parity.
"""

from __future__ import annotations
import __future__

import argparse
import ast
import json
import math
import os
import time
from pathlib import Path

from reproduce.retrieval_scoring import RetrievalGold, rank_cosine, summarize_rankings

SOURCE_REVISION = "a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37"
FUNCTIONS = ("cirr_compute_val_metrics", "fiq_compute_val_metrics")


def load_metric_kernels(source: str, namespace: dict) -> dict:
    """Compile only original named functions, including their bodies/decorators.

    This executes trusted external research code, not a sandbox. No top-level
    imports, main, datasets, prediction generators or model loading are executed.
    Future annotations avoid requiring CLIP/Dataset merely for type names.
    """
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    if len(selected) != 2 or {n.name for n in selected} != set(FUNCTIONS):
        raise ValueError("exactly the two expected published metric functions required")
    module = ast.Module(body=selected, type_ignores=[])
    scope = dict(namespace)
    exec(
        compile(
            module, "trusted-SEARLE-validate.py", "exec", flags=__future__.annotations.compiler_flag
        ),
        scope,
    )
    return {name: scope[name] for name in FUNCTIONS}


def compare_metrics(native: dict, expected: dict) -> int:
    if set(native) != set(expected):
        raise ValueError("native metric keys differ")
    for key in expected:
        actual = native[key]
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or not math.isclose(actual, expected[key], rel_tol=0, abs_tol=1e-5)
        ):
            raise ValueError(f"native percentage differs at {key}")
    return len(expected)


def check(source_path: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("set CUDA_VISIBLE_DEVICES empty before this CPU-only invocation")
    # Lazy imports: this standalone diagnostic does not add Torch to the package.
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    namespace = {"torch": torch, "np": np, "F": F, "device": "cpu"}
    names = tuple(f"authored-s{i:02}" for i in range(72))
    gallery = torch.eye(len(names), device="cpu", dtype=torch.float32)
    query = F.normalize(torch.arange(72, 0, -1, device="cpu", dtype=torch.float32), dim=0)
    vectors = {name: tuple(row.tolist()) for name, row in zip(names, gallery, strict=True)}
    ranking = rank_cosine(tuple(query.tolist()), vectors)
    if ranking != names:
        raise ValueError("strict-order authored geometry changed")

    def native(golds, *, tied=False):
        features = (
            torch.ones((len(golds), 72), device="cpu") if tied else query.repeat(len(golds), 1)
        )
        features = F.normalize(features, dim=-1)
        bindings = dict(namespace)
        bindings["cirr_generate_val_predictions"] = lambda *_: (
            features,
            [g.reference_id for g in golds],
            [g.target_id for g in golds],
            [list(g.subset) for g in golds],
        )
        bindings["fiq_generate_val_predictions"] = lambda *_: (
            features,
            [g.target_id for g in golds],
        )
        kernel = load_metric_kernels(source_path.read_text(encoding="utf-8"), bindings)
        return kernel[FUNCTIONS[0 if golds[0].dataset == "cirr" else 1]](
            None, None, gallery, list(names), [], None
        )

    positions = (1, 5, 10, 50, 51)
    distractors = ((2, 3, 4, 6), (1, 2, 6, 7), (1, 2, 3, 11), (1, 51, 52, 53), (1, 2, 3, 4))
    cirr = tuple(
        RetrievalGold(
            f"authored-cirr-{i}",
            "cirr",
            names[0],
            names[p],
            subset=(names[0], names[p], *(names[j] for j in others)),
        )
        for i, (p, others) in enumerate(zip(positions, distractors, strict=True))
    )
    independent = summarize_rankings(cirr, {g.query_id: ranking for g in cirr}, {"cirr": names})
    expected = {f"cirr_recall_at{k}": v * 100 for k, v in independent["query_micro_recall"].items()}
    expected.update(
        {f"cirr_group_recall_at{k}": v * 100 for k, v in independent["subset_recall"].items()}
    )
    comparisons = compare_metrics(native(cirr), expected)

    fashion = tuple(
        RetrievalGold(
            f"authored-{category}-{i}", "fashioniq", names[0], names[p], category=category
        )
        for category, ranks in (("dress", (9, 10)), ("shirt", (50,)), ("toptee", (1, 9, 49)))
        for i, p in enumerate(ranks)
    )
    independent = summarize_rankings(
        fashion,
        {g.query_id: ranking for g in fashion},
        {category: names for category in ("dress", "shirt", "toptee")},
    )
    native_categories = {}
    for category in ("dress", "shirt", "toptee"):
        native_categories[category] = native(tuple(g for g in fashion if g.category == category))
        comparisons += compare_metrics(
            native_categories[category],
            {
                f"fiq_recall_at{k}": v * 100
                for k, v in independent["category_recalls"][category].items()
            },
        )
    # Independent metric function per category; no claim to execute upstream main.
    comparisons += compare_metrics(
        {
            f"fiq_recall_at{k}": sum(row[f"fiq_recall_at{k}"] for row in native_categories.values())
            / 3
            for k in (10, 50)
        },
        {f"fiq_recall_at{k}": v * 100 for k, v in independent["category_macro_recall"].items()},
    )
    if independent["category_macro_recall"] == independent["query_micro_recall"]:
        raise ValueError("authored unequal-category macro/micro contrast lost")

    # Deliberately observe tied native ranking rather than pretending CPU lexical
    # policy equals Torch's default argsort. Coincidental agreement is not proof.
    tie_gold = (cirr[0],)
    tied_vectors = {name: (1.0,) for name in reversed(names)}
    tied_rank = rank_cosine((1.0,), tied_vectors)
    tied_independent = summarize_rankings(
        tie_gold, {tie_gold[0].query_id: tied_rank}, {"cirr": names}
    )
    tied_native = native(tie_gold, tied=True)
    result = {
        "format": "promptwitness.published-retrieval-metric-check/v1",
        "status": "PASS_STRICT_ORDER_AUTHORED_ONLY",
        "source_revision": SOURCE_REVISION,
        "source_functions": list(FUNCTIONS),
        "metric_bodies_modified": False,
        "prediction_generation": "REPLACED_WITH_AUTHORED_CPU_FEATURES",
        "strict_order_queries": len(cirr) + len(fashion),
        "strict_order_metric_comparisons": comparisons,
        "native_CPU_metric_function_calls": 5,
        "native_CPU_tie_probe": tied_native,
        "independent_lexical_tie_probe": tied_independent["query_micro_recall"],
        "tie_policy_qualified": False,
        "CPU_wall_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "new_real_model_calls": 0,
        "new_GPU_allocations": 0,
        "image_or_text_encoder_executed": False,
        "benchmark_records_accessed": False,
        "whole_native_CLI_executed": False,
        "official_benchmark_parity_established": False,
        "scientific_admission": False,
    }
    (output / "qualification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.source, args.output), allow_nan=False))


if __name__ == "__main__":
    main()
