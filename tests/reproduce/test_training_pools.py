"""Synthetic metadata fixtures check isolation; they are not experiment results."""

import pytest

from reproduce.plan_training_pools import (
    constraint_pools,
    document_pools,
    function_pools,
    invalid_paragraph_index,
    verify,
)


@pytest.mark.parametrize("count,nth", [(5, 6), (10, 11), (2, 0), (0, 1), (True, 1), (3, None)])
def test_invalid_paragraph_metadata_rejected(count, nth):
    assert invalid_paragraph_index(
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": count, "nth_paragraph": nth},
    )


def test_valid_metadata_is_not_relaxed_or_mutated():
    arguments = {"num_paragraphs": 5, "nth_paragraph": 5, "first_word": "example"}
    original = dict(arguments)
    assert not invalid_paragraph_index("length_constraints:nth_paragraph_first_word", arguments)
    assert arguments == original
    assert not invalid_paragraph_index("other:instruction", None)
    assert invalid_paragraph_index("length_constraints:nth_paragraph_first_word", None)


def test_cross_pool_groups_rejected():
    with pytest.raises(ValueError, match="leakage"):
        verify({"fit": [{"id": "a", "groups": ["x"]}], "search": [{"id": "b", "groups": ["x"]}]})


def test_duplicate_unit_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        verify({"fit": [{"id": "a", "groups": ["x"]}], "search": [{"id": "a", "groups": ["y"]}]})


def test_complete_function_components_have_one_owner():
    rows = [
        {"id": str(index), "groups": ["shared" if index < 520 else str(index)]}
        for index in range(1300)
    ]
    result = function_pools(rows, {"0"}, {"0"})
    assert result["audit"]["shared_groups_across_pools"] == 0
    assert any(row["id"] == "0" for row in result["pools"]["fit"])
    assert all(
        "shared" not in row["groups"]
        for pool in ("search", "selection", "final_derived")
        for row in result["pools"][pool]
    )
    assert result["audit"]["sizes"]["fit"] == 512


def test_all_documents_preserved_and_bridge_excluded():
    rows = [{"id": "reserved", "groups": ["x", "y"]}]
    rows += [{"id": str(index), "groups": [f"d{index}", f"e{index}"]} for index in range(1400)]
    result = document_pools(rows, {"reserved"})
    assert result["audit"]["sizes"] == {"fit": 512, "search": 256, "selection": 256}
    assert result["all_context_preserved"]
    by_id = {row["id"]: row for pool in result["pools"].values() for row in pool}
    assert by_id["reserved"]["groups"] == ["x", "y"]


def test_instruction_conjunctions_not_rewritten():
    rows = [{"id": str(index), "groups": [f"g{index % 54}"]} for index in range(1800)]
    rows += [{"id": "bridge", "groups": [f"g{index}" for index in range(54)]}]
    result = constraint_pools(rows, {"0"})
    assert result["cross_partition_conjunctions_excluded"] == 1
    assert result["all_original_constraints_preserved"]
    assert all(row["id"] != "bridge" for pool in result["pools"].values() for row in pool)
    assert result["audit"]["shared_groups_across_pools"] == 0
