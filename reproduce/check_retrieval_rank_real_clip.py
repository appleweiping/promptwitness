"""Qualify direct and supplied-description CLIP ranking on authored images.

This is not a CIRR result: all images, query text and labels are generated here.
The checkpoint must be an already obtained first-party ViT-L/14 file. The new
output directory must share a filesystem with it so a private hard link can be
placed under the ranker's input-only leaf without copying or redistributing it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from reproduce.process_access import LEAVES
from reproduce.retrieval_clip_cpu import require_cpu, verify_checkpoint
from reproduce.retrieval_rank_role import RestrictedRankerSession
from reproduce.retrieval_role_scoring import score_rankings_restricted
from reproduce.retrieval_work_ledger import RetrievalWorkLedger

FORMAT = "promptwitness.restricted-real-clip-qualification/v1"


def check(checkpoint: Path, output: Path) -> dict[str, object]:
    """Run one cold index, two direct queries and one supplied description."""
    require_cpu()
    checkout = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(checkout):
        raise ValueError("private qualification output must be outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    outer: RetrievalWorkLedger | None = None
    result: dict[str, object] = {
        "format": FORMAT,
        "status": "RUNNING",
        "benchmark_images_or_annotations": False,
        "scientific_result": False,
        "model": "OpenAI CLIP ViT-L/14",
        "device": "cpu",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "fusion_weight": 0.5,
        "fusion_weight_scientifically_frozen": False,
        "official_evaluator_parity_established": False,
        "whole_process_cpu_allocation_measured": False,
        "supplied_description_from_real_generator": False,
    }
    try:
        weight = checkpoint.resolve(strict=True)
        result["checkpoint_sha256"] = verify_checkpoint(weight)
        from PIL import Image, ImageDraw

        store = output / "store"
        for leaf in LEAVES:
            (store / leaf).mkdir(parents=True)
        inputs = store / "search/inputs"
        os.link(weight, inputs / "ViT-L-14.pt")
        image_specs = (
            ("authored-red", "red", (301, 225), "RGB"),
            ("authored-blue", "blue", (225, 301), "RGBA"),
        )
        image_paths: dict[str, str] = {}
        for image_id, color, size, mode in image_specs:
            image = Image.new(mode, size, "white")
            ImageDraw.Draw(image).rectangle((70, 70, 150, 150), fill=color)
            filename = f"{image_id}.png"
            image.save(inputs / filename)
            image_paths[image_id] = filename
        Image.new("L", (240, 240), 128).save(inputs / "authored-grey.png")
        image_paths["authored-grey"] = "authored-grey.png"
        pool = tuple(image_paths)
        queries = (
            ("authored-q-red", "authored-red", "make the square blue", "authored-blue"),
            ("authored-q-blue", "authored-blue", "make the square red", "authored-red"),
        )
        (inputs / "cirr.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": query_id,
                        "reference_id": reference_id,
                        "modification": modification,
                        "category": "",
                    }
                )
                + "\n"
                for query_id, reference_id, modification, _ in queries
            ),
            encoding="utf-8",
        )
        (inputs / "cirr-gallery.json").write_text(json.dumps({"cirr": pool}), encoding="utf-8")
        (store / "search/gold/cirr.jsonl").write_text(
            "".join(
                json.dumps({"id": query_id, "target_id": target_id, "subset": pool}) + "\n"
                for query_id, _, _, target_id in queries
            ),
            encoding="utf-8",
        )
        rank_scratch = output / "ranker-scratch"
        score_scratch = output / "scorer-scratch"
        description_score_scratch = output / "description-scorer-scratch"
        rank_scratch.mkdir()
        score_scratch.mkdir()
        description_score_scratch.mkdir()
        outer = RetrievalWorkLedger(output / "outer-work.sqlite")
        with RestrictedRankerSession(
            store=store,
            scratch=rank_scratch,
            dataset="cirr",
            image_paths=image_paths,
            checkpoint="ViT-L-14.pt",
            image_weight=0.5,
            attempt_prefix="authored-real-clip",
        ) as session:
            if session.ready["query_count"] != len(queries) or session.ready["gallery_sizes"] != {
                "cirr": len(pool)
            }:
                raise ValueError("restricted ranker loaded the wrong authored population")
            ranker_pid = session.ready["pid"]
            result["ranker_pid"] = ranker_pid
            rankings: dict[str, tuple[str, ...]] = {}
            for query_id, *_ in queries:

                def rank_one(identifier: str = query_id) -> tuple[str, ...]:
                    return session.rank_one(identifier)

                rankings[query_id] = outer.run(
                    f"outer:rank:{query_id}",
                    "search",
                    "rank_callback",
                    rank_one,
                )
                result["full_rankings"] = {
                    identifier: list(ranking) for identifier, ranking in rankings.items()
                }
            description_ranking = outer.run(
                "outer:rank-description:authored-red",
                "search",
                "rank_callback",
                lambda: session.rank_description(
                    "authored-q-red",
                    "authored-description-red",
                    "a blue square on a white background",
                ),
            )
            result["supplied_description_full_ranking"] = list(description_ranking)
        score = outer.run(
            "outer:score:authored-pair",
            "search",
            "restricted_score",
            lambda: score_rankings_restricted(store, score_scratch, "cirr", "search", rankings),
        )
        result["scorer_pid"] = score["worker_pid"]
        result["observations"] = score["observations"]
        description_score = outer.run(
            "outer:score-description:authored-red",
            "search",
            "restricted_score",
            lambda: score_rankings_restricted(
                store,
                description_score_scratch,
                "cirr",
                "search",
                {"authored-q-red": description_ranking},
            ),
        )
        result["description_scorer_pid"] = description_score["worker_pid"]
        result["supplied_description_observation"] = description_score["observations"][
            "authored-q-red"
        ]
        inner = RetrievalWorkLedger(rank_scratch / "ranker-work.sqlite")
        try:
            inner_summary = inner.summary()
        finally:
            inner.close()
        if (
            inner_summary["attempts"] != 10
            or inner_summary["completed"] != 10
            or inner_summary["unresolved"] != 0
            or inner_summary["known_forward_calls"] != 6
            or score["worker_pid"] == ranker_pid
            or description_score["worker_pid"] == ranker_pid
        ):
            raise ValueError("restricted real-CLIP work receipts or role identity are incomplete")
        result.update(
            status="PASS_AUTHORED_REAL_CLIP_DESCRIPTION_RESTRICTED_SEARCH_AND_SCORER",
            gallery_size=len(pool),
            query_count=len(queries),
            inner_operation_ledger=inner_summary,
        )
        return result
    except BaseException as error:
        result.update(status="FAILED_RETAINED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if outer is not None:
            result["outer_operation_ledger"] = outer.summary()
            outer.close()
        result["checker_wall_seconds"] = time.monotonic() - started
        (output / "qualification.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("new_output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint, args.new_output), allow_nan=False))


if __name__ == "__main__":
    main()
