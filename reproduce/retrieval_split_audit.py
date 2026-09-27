"""Input-only train-pool grouping audit for composed image retrieval.

This checks a proposed fit/search/selection assignment before benchmark rows
are made available to an optimizer. It cannot discover near duplicates: the
group mapping must come from a separately qualified image audit. It never
accepts targets, subsets, captions, or final-pool membership, and it does not
enforce a filesystem/process boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

POOLS = ("fit", "search", "selection")
FASHION_CATEGORIES = frozenset({"dress", "shirt", "toptee"})


@dataclass(frozen=True, slots=True)
class RetrievalInputIdentity:
    query_id: str
    reference_id: str
    dataset: str
    category: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("nonempty query ID required")
        if not isinstance(self.reference_id, str) or not self.reference_id:
            raise ValueError("nonempty reference ID required")
        if self.dataset == "cirr":
            if self.category:
                raise ValueError("CIRR input has no FashionIQ category")
        elif self.dataset == "fashioniq":
            if self.category not in FASHION_CATEGORIES:
                raise ValueError("FashionIQ input requires a known category")
        else:
            raise ValueError("only CIRR/FashionIQ input identities are supported")


def audit_training_groups(
    inputs: Sequence[RetrievalInputIdentity],
    assignments: Mapping[str, tuple[str, ...]],
    reference_groups: Mapping[str, str],
) -> dict[str, object]:
    """Reject missing/overlapping training identities and reference groups.

    Each reference_group is a caller-supplied canonical ID shared by exact or
    near-duplicate images. A unique ID per image is not proof duplicates were
    detected. The return value is an input-only audit, never M1 admission.
    """
    if not inputs or any(not isinstance(row, RetrievalInputIdentity) for row in inputs):
        raise ValueError("nonempty typed retrieval inputs required")
    datasets = {row.dataset for row in inputs}
    query_ids = [row.query_id for row in inputs]
    if len(datasets) != 1 or len(set(query_ids)) != len(query_ids):
        raise ValueError("one dataset with unique query IDs required")
    dataset = next(iter(datasets))
    references = {row.reference_id for row in inputs}
    if set(reference_groups) != references or any(
        not isinstance(group, str) or not group for group in reference_groups.values()
    ):
        raise ValueError("exactly all reference IDs need nonempty supplied groups")
    if set(assignments) != set(POOLS):
        raise ValueError("exact fit/search/selection assignments required")
    owners: dict[str, str] = {}
    counts: dict[str, int] = {}
    for pool in POOLS:
        members = assignments[pool]
        if (
            not isinstance(members, tuple)
            or not members
            or any(not isinstance(member, str) or not member for member in members)
        ):
            raise ValueError("every pool needs nonempty immutable query IDs")
        counts[pool] = len(members)
        for member in members:
            if member in owners:
                raise ValueError("query ID appears in multiple pools or twice")
            owners[member] = pool
    if set(owners) != set(query_ids):
        raise ValueError("assignments must cover exactly the supplied training queries")
    group_owners: dict[str, str] = {}
    for row in inputs:
        group = reference_groups[row.reference_id]
        pool = owners[row.query_id]
        if group in group_owners and group_owners[group] != pool:
            raise ValueError("reference or near-duplicate group crosses training pools")
        group_owners[group] = pool
    return {
        "dataset": dataset,
        "status": "INPUT_METADATA_DISJOINT_ONLY",
        "query_counts": counts,
        "reference_group_count": len(group_owners),
        "target_or_subset_labels_read": False,
        "near_duplicate_detection_qualified": False,
        "final_overlap_or_seal_qualified": False,
        "process_access_qualified": False,
    }
