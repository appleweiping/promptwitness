"""Authored CPU ranking parity including ties, never image experiment results."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from reproduce.check_retrieval_native_metrics import (
    FUNCTIONS,
    SOURCE_REVISION,
    compare_metrics,
    load_metric_kernels,
)
from reproduce.check_retrieval_scoring import check_case
from reproduce.retrieval_cpu_ranking import rank_float32_cpu
from reproduce.retrieval_scoring import RetrievalGold, rank_cosine, score_ranking


def check(source_path: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("empty CUDA visibility required before this CPU invocation")
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = source_path.read_text(encoding="utf-8")
    names = tuple(f"authored-s{i:02}" for i in range(72))
    gallery = torch.eye(72, dtype=torch.float32, device="cpu")
    strict = F.normalize(torch.arange(72, 0, -1, dtype=torch.float32), dim=0)
    tied = F.normalize(torch.ones(72, dtype=torch.float32), dim=0)
    nearby = torch.ones(72, dtype=torch.float32)
    nearby[1] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    nearby = F.normalize(nearby, dim=0)
    dots = nearby.unsqueeze(0) @ gallery.T
    distances = 1 - dots
    rounded_near_tie = bool(dots[0, 0] != dots[0, 1] and distances[0, 0] == distances[0, 1])
    if not rounded_near_tie:
        raise ValueError("authored float32 distance rounding witness changed")

    scenarios = (
        ("strict", strict, gallery, names),
        ("all_tie", tied, gallery, names),
        ("near_tie", nearby, gallery, names),
        ("reverse_gallery_tie", tied, gallery.flip(0), tuple(reversed(names))),
    )
    comparisons = 0
    ranks_checked = 0
    case_results = []
    lexical_counterexample = False
    for case, query, index, ids in scenarios:
        ranking = rank_float32_cpu(query, index, ids)
        native_indices = []

        class TorchWitness:
            """Capture the real operator result, returning it without modification."""

            def __init__(self, sink):
                self.sink = sink

            def __getattr__(self, name):
                return getattr(torch, name)

            def argsort(self, *args, **kwargs):
                result = torch.argsort(*args, **kwargs)
                self.sink.append(tuple(result[0].tolist()))
                return result

        golds = (
            RetrievalGold(case + "-cirr", "cirr", names[0], names[1], subset=names[:6]),
            RetrievalGold(case + "-fiq", "fashioniq", names[0], names[1], category="dress"),
        )
        scores = {}
        for gold in golds:
            bindings = {"torch": TorchWitness(native_indices), "np": np, "F": F, "device": "cpu"}
            bindings["cirr_generate_val_predictions"] = lambda *_, query=query, gold=gold: (
                query.unsqueeze(0),
                [gold.reference_id],
                [gold.target_id],
                [list(gold.subset)],
            )
            bindings["fiq_generate_val_predictions"] = lambda *_, query=query, gold=gold: (
                query.unsqueeze(0),
                [gold.target_id],
            )
            native = load_metric_kernels(source, bindings)[
                FUNCTIONS[0 if gold.dataset == "cirr" else 1]
            ](None, None, index, list(ids), [], None)
            captured = tuple(ids[i] for i in native_indices[-1])
            if captured != ranking:
                raise ValueError("full native rank differs, not just cutoff metrics")
            ranks_checked += 1
            score = score_ranking(gold, candidate_ids=ids, ranking=ranking)
            prefix = "cirr" if gold.dataset == "cirr" else "fiq"
            expected = {f"{prefix}_recall_at{k}": value * 100 for k, value in score.recalls}
            expected.update(
                {f"cirr_group_recall_at{k}": value * 100 for k, value in score.subset_recalls}
            )
            comparisons += compare_metrics(native, expected)
            scores[gold.dataset] = native
        if len(native_indices) != 2:
            raise ValueError("native ranking witness call count differs")
        if case == "all_tie":
            scalar = rank_cosine(
                tuple(query.tolist()), {x: row.tolist() for x, row in zip(ids, index, strict=True)}
            )
            lexical_counterexample = scalar != ranking
        case_results.append({"case": case, "full_rank_matches": 2, "native_metrics": scores})
    if not lexical_counterexample:
        raise ValueError("original lexical/native tie counterexample lost")

    invalid = (
        (strict.double(), gallery),
        (strict.unsqueeze(0), gallery),
        (strict, gallery[:1]),
        (torch.zeros_like(strict), gallery),
        (strict * 2, gallery),
        (strict, torch.zeros_like(gallery)),
        (strict * float("nan"), gallery),
        (strict, gallery * float("inf")),
    )
    for bad_query, bad_gallery in invalid:
        try:
            rank_float32_cpu(bad_query, bad_gallery, names)
        except ValueError:
            continue
        raise ValueError("malformed features did not fail closed")

    def actual_rank(vector, pool):
        unit = F.normalize(torch.tensor(vector, dtype=torch.float32), dim=0)
        rows = torch.tensor(tuple(pool.values()), dtype=torch.float32)
        return rank_float32_cpu(unit, rows, tuple(pool))

    gates = [
        check_case(output, case, rank_backend=actual_rank)
        for case in ("eligible", "rejected", "failed")
    ]
    result = {
        "format": "promptwitness.retrieval-cpu-ranking-check/v1",
        "status": "PASS_AUTHORED_SINGLE_QUERY_CPU_ONLY",
        "source_revision": SOURCE_REVISION,
        "metric_bodies_modified": False,
        "native_operator_witness_returns_original_unchanged": True,
        "prediction_generation": "AUTHORED_FEATURES_NO_ENCODER",
        "native_metric_calls": ranks_checked,
        "full_rank_comparisons": ranks_checked,
        "metric_comparisons": comparisons,
        "scenarios": case_results,
        "float32_dot_distinct_distance_equal": rounded_near_tie,
        "lexical_counterexample_retained": lexical_counterexample,
        "invalid_features_rejected": len(invalid),
        "gate_cases": gates,
        "query_matmul_batch": 1,
        "gallery_row_order_preserved": True,
        "CPU_wall_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "real_model_calls": 0,
        "new_GPU_allocations": 0,
        "benchmark_or_encoder_executed": False,
        "GPU_or_multiquery_batch_parity": False,
        "official_benchmark_parity": False,
        "real_M1_or_Pilot": False,
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
