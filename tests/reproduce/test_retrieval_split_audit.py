"""Authored input metadata only: no benchmark rows or images."""

from __future__ import annotations

import pytest

from reproduce.retrieval_split_audit import RetrievalInputIdentity, audit_training_groups


def fixture():
    rows = (
        RetrievalInputIdentity("f1", "a", "fashioniq", "dress"),
        RetrievalInputIdentity("f2", "a-copy", "fashioniq", "dress"),
        RetrievalInputIdentity("s1", "b", "fashioniq", "shirt"),
        RetrievalInputIdentity("v1", "c", "fashioniq", "toptee"),
    )
    pools = {"fit": ("f1", "f2"), "search": ("s1",), "selection": ("v1",)}
    groups = {"a": "duplicate-a", "a-copy": "duplicate-a", "b": "b", "c": "c"}
    return rows, pools, groups


def test_accepts_disjoint_caller_supplied_reference_groups_without_gold():
    rows, pools, groups = fixture()
    report = audit_training_groups(rows, pools, groups)
    assert report["query_counts"] == {"fit": 2, "search": 1, "selection": 1}
    assert report["reference_group_count"] == 3
    assert report["status"] == "INPUT_METADATA_DISJOINT_ONLY"
    assert report["target_or_subset_labels_read"] is False
    assert report["near_duplicate_detection_qualified"] is False
    assert report["final_overlap_or_seal_qualified"] is False
    assert report["process_access_qualified"] is False


def test_same_reference_or_duplicate_group_cannot_cross_pools():
    rows, pools, groups = fixture()
    pools["fit"] = ("f1",)
    pools["search"] = ("f2", "s1")
    with pytest.raises(ValueError, match="crosses"):
        audit_training_groups(rows, pools, groups)
    groups["a-copy"] = "different"
    assert audit_training_groups(rows, pools, groups)["status"] == "INPUT_METADATA_DISJOINT_ONLY"


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "empty", "wrong_type"])
def test_bad_query_assignments_fail_closed(kind):
    rows, pools, groups = fixture()
    if kind == "missing":
        pools["fit"] = ("f1",)
    elif kind == "extra":
        pools["fit"] = ("f1", "f2", "unseen")
    elif kind == "duplicate":
        pools["search"] = ("f1", "s1")
    elif kind == "empty":
        pools["selection"] = ()
    else:
        pools["selection"] = ["v1"]
    with pytest.raises(ValueError):
        audit_training_groups(rows, pools, groups)


def test_reference_group_mapping_must_cover_exact_inputs():
    rows, pools, groups = fixture()
    del groups["a-copy"]
    with pytest.raises(ValueError, match="all reference"):
        audit_training_groups(rows, pools, groups)
    groups["a-copy"] = "duplicate-a"
    groups["unused"] = "unused"
    with pytest.raises(ValueError, match="all reference"):
        audit_training_groups(rows, pools, groups)


def test_duplicate_query_id_and_dataset_mixture_fail():
    rows, pools, groups = fixture()
    with pytest.raises(ValueError, match="unique query"):
        audit_training_groups((*rows, rows[0]), pools, groups)
    with pytest.raises(ValueError, match="one dataset"):
        audit_training_groups(
            (*rows[:-1], RetrievalInputIdentity("v1", "c", "cirr")), pools, groups
        )


@pytest.mark.parametrize(
    "args",
    [
        ("", "a", "cirr", ""),
        ("q", "", "cirr", ""),
        ("q", "a", "cirr", "dress"),
        ("q", "a", "fashioniq", "unknown"),
        ("q", "a", "circo", ""),
    ],
)
def test_input_identity_rejects_invalid_metadata(args):
    with pytest.raises(ValueError):
        RetrievalInputIdentity(*args)
