"""Input-only, full-gallery direct CLIP baseline for composed image retrieval.

The caller supplies image paths from an admitted source. This component reads
query inputs and ordered gallery IDs, never target/subset annotations. It does
not qualify the image license, source split, fusion weight, or final access.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from reproduce.retrieval_clip_cpu import ClipCPUEncoder, require_cpu
from reproduce.retrieval_composed_cpu import rank_direct_composed
from reproduce.retrieval_split_audit import RetrievalInputIdentity
from reproduce.retrieval_work_ledger import MeteredClipEncoder, RetrievalWorkLedger


def require_complete_ranking(ranking: object, pool: tuple[str, ...]) -> tuple[str, ...]:
    """Reject missing, extra, duplicate or wrong-category IDs before scoring."""
    if (
        not isinstance(pool, tuple)
        or not pool
        or not isinstance(ranking, tuple)
        or len(ranking) != len(pool)
        or any(not isinstance(item, str) or not item for item in ranking)
        or len(set(ranking)) != len(ranking)
        or set(ranking) != set(pool)
    ):
        raise ValueError("ranker must return the complete category gallery")
    return ranking


def load_input_only(
    input_dir: Path, dataset: str
) -> tuple[dict[str, tuple[str, str, str]], dict[str, tuple[str, ...]]]:
    """Read the scorer's input-side schema without opening its gold leaf."""
    if dataset not in {"cirr", "fashioniq"}:
        raise ValueError("only CIRR/FashionIQ input schemas are supported")
    queries: dict[str, tuple[str, str, str]] = {}
    with (input_dir / f"{dataset}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {
                "id",
                "reference_id",
                "modification",
                "category",
            }:
                raise ValueError("query row must contain input-only fields")
            identity = RetrievalInputIdentity(
                row["id"], row["reference_id"], dataset, row["category"]
            )
            if (
                identity.query_id in queries
                or not isinstance(row["modification"], str)
                or not row["modification"].strip()
            ):
                raise ValueError("unique query ID and nonempty modification required")
            category = identity.category if dataset == "fashioniq" else "cirr"
            queries[identity.query_id] = (identity.reference_id, row["modification"], category)
    if not queries:
        raise ValueError("nonempty query population required")
    raw = json.loads((input_dir / f"{dataset}-gallery.json").read_text(encoding="utf-8"))
    keys = {"cirr"} if dataset == "cirr" else {"dress", "shirt", "toptee"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("exact gallery categories required")
    galleries: dict[str, tuple[str, ...]] = {}
    for category, values in raw.items():
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("ordered unique complete gallery IDs required")
        galleries[category] = tuple(values)
    for reference_id, _, category in queries.values():
        if reference_id not in galleries[category]:
            raise ValueError("reference must belong to its complete gallery")
    return queries, galleries


class DirectClipRanker:
    """Build one input-only image index for a fit, search or selection population.

    Image load and encoding are nested under a durable image-encode receipt;
    encoder cold start and each text forward have their own receipts. The
    audit bridge owns the outer rank callback and restricted scorer receipts.
    Final ranking is not implemented by this component.
    The caller must also account for whole-process CPU allocation and sources.
    """

    def __init__(
        self,
        *,
        input_dir: Path,
        dataset: str,
        image_paths: Mapping[str, Path],
        checkpoint: Path,
        image_weight: float,
        ledger: RetrievalWorkLedger,
        attempt_prefix: str,
        stage: str = "search",
    ) -> None:
        require_cpu()
        if stage not in {"fit", "search", "selection"}:
            raise ValueError("only fit, search or selection input-only ranking is supported")
        if not attempt_prefix or not isinstance(attempt_prefix, str):
            raise ValueError("nonempty attempt prefix required")
        if (
            not isinstance(image_weight, float)
            or not math.isfinite(image_weight)
            or not 0.0 <= image_weight <= 1.0
        ):
            raise ValueError("finite explicit image weight in [0,1] required")
        queries, galleries = load_input_only(input_dir, dataset)
        image_ids = tuple(dict.fromkeys(item for pool in galleries.values() for item in pool))
        if set(image_paths) != set(image_ids) or any(
            not isinstance(path, Path) or not path.is_file() for path in image_paths.values()
        ):
            raise ValueError("every gallery ID needs exactly one existing image path")

        import torch
        from PIL import Image

        self.queries = queries
        self.galleries = galleries
        self.image_weight = image_weight
        self.ledger = ledger
        self.attempt_prefix = attempt_prefix
        self.stage = stage
        encoder = ledger.run(
            f"{attempt_prefix}:encoder-load",
            "shared",
            "encoder_load",
            lambda: ClipCPUEncoder(checkpoint),
        )
        self.checkpoint_sha256 = encoder.checkpoint_sha256
        self.encoder = MeteredClipEncoder(encoder, ledger, stage)
        features = {}

        def image_forward_count() -> int:
            value = encoder.costs["image_forward_calls"]
            if type(value) is not int or value < 0:
                raise ValueError("encoder image-forward counter must be nonnegative integer")
            return value

        for image_id in image_ids:

            def encode_path(path: Path = image_paths[image_id]):
                with Image.open(path) as image:
                    return encoder.encode_image(image)

            features[image_id] = ledger.run(
                f"{attempt_prefix}:image:{image_id}",
                "shared",
                "image_encode",
                encode_path,
                forward_counter=image_forward_count,
            )
        self.index = {
            category: torch.stack([features[image_id] for image_id in pool])
            for category, pool in galleries.items()
        }

    def rank_one(self, query_id: str) -> tuple[str, ...]:
        """Return all category candidates; scorer applies CIRR reference exclusion."""
        if query_id not in self.queries:
            raise ValueError("query is outside the input-only population")

        def rank():
            reference_id, modification, category = self.queries[query_id]
            text = self.encoder.encode_text(f"{self.attempt_prefix}:text:{query_id}", modification)
            return require_complete_ranking(
                rank_direct_composed(
                    self.index[category],
                    self.galleries[category],
                    reference_id,
                    text,
                    self.image_weight,
                ),
                self.galleries[category],
            )

        return self.ledger.run(
            f"{self.attempt_prefix}:query:{query_id}", self.stage, "rank_callback", rank
        )

    def rank_description(
        self, query_id: str, request_id: str, target_description: str
    ) -> tuple[str, ...]:
        """Rank a supplied target description against the same frozen image index.

        The original reference image/category still comes only from input data.
        A distinct request ID charges each supplied text and ranking once;
        repeating an ID is a failed replay, not a cache hit or free model call.
        This method does not generate the description or authenticate its source.
        """
        if query_id not in self.queries:
            raise ValueError("query is outside the input-only population")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("nonempty description request ID required")
        if not isinstance(target_description, str) or not target_description.strip():
            raise ValueError("nonempty target description required")

        def rank() -> tuple[str, ...]:
            reference_id, _, category = self.queries[query_id]
            text = self.encoder.encode_text(
                f"{self.attempt_prefix}:description:{request_id}:text", target_description
            )
            return require_complete_ranking(
                rank_direct_composed(
                    self.index[category],
                    self.galleries[category],
                    reference_id,
                    text,
                    self.image_weight,
                ),
                self.galleries[category],
            )

        return self.ledger.run(
            f"{self.attempt_prefix}:description:{request_id}:rank",
            self.stage,
            "rank_callback",
            rank,
        )
