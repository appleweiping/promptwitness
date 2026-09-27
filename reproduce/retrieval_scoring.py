"""Independent full-ranking single-target CIR scoring for the ICMR design.

No encoder, dataset loader, native scorer parity, access sandbox or model calls
are provided by this module. Gold records belong in the scorer, never in a
query-rewrite prompt. Conventions were inspected in SEARLE validate.py at
a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37; no upstream code is vendored.
FashionIQ keeps the reference; CIRR removes it before global/subset ranking.
Only query-micro primary hit is binary; category macro is not certified by the
existing binary finite-population gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _ids(values: tuple[str, ...], name: str) -> None:
    if (
        not isinstance(values, tuple)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} requires nonempty unique immutable IDs")


def _normalized(vector: Sequence[float]) -> tuple[float, ...]:
    if not vector or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in vector
    ):
        raise ValueError("finite nonempty numeric vector required")
    # Scaling first avoids overflow/underflow when individually finite vectors
    # have a norm outside the representable range. No zero-vector imputation.
    scale = max(abs(x) for x in vector)
    if scale == 0:
        raise ValueError("zero vector has no cosine direction")
    scaled = tuple(x / scale for x in vector)
    norm = math.sqrt(math.fsum(x * x for x in scaled))
    return tuple(x / norm for x in scaled)


def rank_cosine(
    query: Sequence[float], candidates: Mapping[str, Sequence[float]]
) -> tuple[str, ...]:
    """Exact CPU diagnostic, full pool, descending cosine then lexical ID ties.

    This tie policy is explicit, not claimed identical to torch.argsort or
    FAISS. Real GPU/index execution must separately establish its fixed policy
    and numeric parity. This deliberately does not implement ANN or encoders.
    """
    _ids(tuple(candidates), "candidate pool")
    unit_query = _normalized(query)
    scores = []
    for identifier, vector in candidates.items():
        unit = _normalized(vector)
        if len(unit) != len(unit_query):
            raise ValueError("query/candidate embedding dimensions differ")
        scores.append((identifier, math.fsum(x * y for x, y in zip(unit_query, unit, strict=True))))
    return tuple(identifier for identifier, _ in sorted(scores, key=lambda x: (-x[1], x[0])))


def compose_vectors(
    reference: Sequence[float], text: Sequence[float], *, image_weight: float
) -> tuple[float, ...]:
    """Simple normalized fusion diagnostic, not a claimed strong CIR baseline.

    image_weight is frozen before outcomes. Zero or one are modality ablations,
    not the default scientific setting. Cancellation remains an explicit error.
    """
    if (
        isinstance(image_weight, bool)
        or not isinstance(image_weight, (int, float))
        or not math.isfinite(image_weight)
        or not 0 <= image_weight <= 1
    ):
        raise ValueError("finite image_weight in [0,1] required")
    image, words = _normalized(reference), _normalized(text)
    if len(image) != len(words):
        raise ValueError("image/text embedding dimensions differ")
    return _normalized(
        tuple(image_weight * x + (1 - image_weight) * y for x, y in zip(image, words, strict=True))
    )


@dataclass(frozen=True, slots=True)
class RetrievalGold:
    """Scorer-only target and CIRR subset; no multi-positive shortcut."""

    query_id: str
    dataset: str
    reference_id: str
    target_id: str
    category: str = ""
    subset: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.query_id, self.reference_id, self.target_id)
        ):
            raise ValueError("query/reference/target string IDs required")
        if self.reference_id == self.target_id:
            raise ValueError("reference and target must differ")
        if self.dataset == "cirr":
            _ids(self.subset, "CIRR subset")
            if self.category or not {self.reference_id, self.target_id} <= set(self.subset):
                raise ValueError("CIRR requires reference and target in its unlabelled subset")
        elif self.dataset == "fashioniq":
            if self.category not in ("dress", "shirt", "toptee") or self.subset != ():
                raise ValueError("FashionIQ requires a known category and no CIRR subset")
        else:
            raise ValueError("only single-target CIRR/FashionIQ are supported")


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    primary_hit: int
    recalls: tuple[tuple[int, int], ...]
    subset_recalls: tuple[tuple[int, int], ...]


def score_ranking(
    gold: RetrievalGold, *, candidate_ids: tuple[str, ...], ranking: tuple[str, ...]
) -> RetrievalScore:
    """Validate a complete actual ranking, then score against target identity.

    Missing targets, duplicate IDs, incomplete pools or malformed results are
    failures, never observed zeros. Scores are fractions, not percentages.
    """
    _ids(candidate_ids, "candidate pool")
    _ids(ranking, "ranking")
    pool = set(candidate_ids)
    if set(ranking) != pool:
        raise ValueError("ranking must cover the exact complete candidate pool")
    if not {gold.reference_id, gold.target_id} <= pool:
        raise ValueError("reference and target must exist in the candidate pool")
    if gold.dataset == "cirr":
        if not set(gold.subset) <= pool:
            raise ValueError("CIRR subset has candidates outside the pool")
        eligible = tuple(x for x in ranking if x != gold.reference_id)
        subset_ranking = tuple(x for x in eligible if x in set(gold.subset))
        subset_rank = subset_ranking.index(gold.target_id) + 1
        subset_recalls = tuple((k, int(subset_rank <= k)) for k in (1, 2, 3))
        cutoffs: tuple[int, ...] = (1, 5, 10, 50)
        primary_k = 5
    else:
        eligible = ranking  # FIQ convention: do not inherit CIRR exclusion.
        subset_recalls = ()
        cutoffs = (10, 50)
        primary_k = 10
    target_rank = eligible.index(gold.target_id) + 1
    return RetrievalScore(
        int(target_rank <= primary_k),
        tuple((k, int(target_rank <= k)) for k in cutoffs),
        subset_recalls,
    )


def summarize_rankings(
    golds: Sequence[RetrievalGold],
    rankings: Mapping[str, tuple[str, ...]],
    pools: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    """Query micro and, separately, equal-category FashionIQ macro.

    CIRR uses pools['cirr']; FashionIQ requires all three category pools and at
    least one scored query in each. Incomplete coverage is not a macro report.
    """
    if not golds:
        raise ValueError("nonempty gold collection required")
    _ids(tuple(g.query_id for g in golds), "query collection")
    datasets = {g.dataset for g in golds}
    if len(datasets) != 1 or set(rankings) != {g.query_id for g in golds}:
        raise ValueError("one dataset and exactly all query rankings required")
    dataset = golds[0].dataset
    expected_pools = {"cirr"} if dataset == "cirr" else {"dress", "shirt", "toptee"}
    if set(pools) != expected_pools or (
        dataset == "fashioniq" and {g.category for g in golds} != expected_pools
    ):
        raise ValueError("exact complete dataset/category coverage required")
    scores = [
        score_ranking(
            g,
            candidate_ids=pools[g.category if dataset == "fashioniq" else "cirr"],
            ranking=rankings[g.query_id],
        )
        for g in golds
    ]

    def means(indices: list[int], attribute: str) -> dict[str, float]:
        rows = [dict(getattr(scores[i], attribute)) for i in indices]
        return {str(k): sum(row[k] for row in rows) / len(rows) for k in rows[0]}

    micro = means(list(range(len(scores))), "recalls")
    categories = {
        category: means([i for i, g in enumerate(golds) if g.category == category], "recalls")
        for category in sorted(expected_pools)
        if dataset == "fashioniq"
    }
    macro = (
        {k: sum(row[k] for row in categories.values()) / 3 for k in micro} if categories else None
    )
    return {
        "dataset": dataset,
        "queries": len(scores),
        "metric_scale": "fraction",
        "query_micro_recall": micro,
        "category_macro_recall": macro,
        "category_recalls": categories,
        "subset_recall": means(list(range(len(scores))), "subset_recalls"),
        "primary_query_micro_hit": sum(score.primary_hit for score in scores) / len(scores),
        "missing_or_failed_scores_imputed": False,
        "official_native_parity_established": False,
    }
