"""Authored message/transport tests, not actual model or online qualification."""

import copy

import pytest

from reproduce.prepare_bfcl_fit import MODEL_REVISIONS
from reproduce.prepare_preflight_requests import SEEDS
from reproduce.torch_runtime import (
    RETRIEVAL_DESCRIPTION_CAP,
    TASK_CAPS,
    generation_cap,
    task_wire,
)


def request(family="hotpotqa"):
    unit = {
        "bfcl": {
            "id": "u",
            "category": "multiple",
            "question": [[{"role": "user", "content": "all turns"}]],
            "function": [{"name": "f", "parameters": {"type": "object"}}],
        },
        "hotpotqa": {
            "id": "u",
            "family": "hotpotqa",
            "messages": [
                {"role": "system", "content": SEEDS["hotpotqa"]},
                {"role": "user", "content": "Context:\none: ab\n\ntwo: c\n\nQuestion: q"},
            ],
        },
        "instruction_following": {
            "id": "u",
            "family": "instruction_following",
            "messages": [
                {"role": "system", "content": SEEDS["instruction_following"]},
                {"role": "user", "content": "every original constraint"},
            ],
        },
    }[family]
    return {
        "family": family,
        "model": next(iter(MODEL_REVISIONS)),
        "execution": {},
        "configuration": {"purpose": "AUTHORED"},
        "unit": unit,
        "replicate": "authored",
        "candidate": {
            "schema_version": 1,
            "id": "p",
            "messages": [{"role": "system", "id": "instruction", "content": "literal instruction"}],
        },
    }


@pytest.mark.parametrize("family", TASK_CAPS)
def test_all_original_input_and_prefix_retained(family):
    original = request(family)
    unchanged = copy.deepcopy(original)
    wire = task_wire(original)
    assert original == unchanged
    assert wire["max_new_tokens"] == TASK_CAPS[family]
    assert wire["messages"][0]["content"] == "literal instruction"
    assert len(wire["messages"]) == 2
    if family == "hotpotqa":
        assert wire["messages"][-1]["content"] == "Context:\none: ab\n\ntwo: c\n\nQuestion: q"
    elif family == "bfcl":
        assert "all turns" in wire["messages"][-1]["content"]
        assert '"functions"' in wire["messages"][-1]["content"]
    else:
        assert wire["messages"][-1] == original["unit"]["messages"][-1]


def test_template_and_demonstrations_are_not_dropped():
    r = request()
    r["candidate"]["messages"] += [
        {"role": "user", "content": "fit demo input"},
        {"role": "assistant", "content": "fit demo response"},
        {"role": "user", "content": "Use the actual input: {{ task_input }}"},
    ]
    wire = task_wire(r)
    assert len(wire["messages"]) == 4
    assert wire["messages"][2]["content"] == "fit demo response"
    assert "one: ab" in wire["messages"][-1]["content"]
    assert "{{" not in wire["messages"][-1]["content"]


@pytest.mark.parametrize("family", ["hotpotqa", "instruction_following"])
def test_actual_normalized_store_shape_replaces_seed_preserves_original_input(family):
    r = request(family)
    r["unit"] = {
        "id": "u",
        "family": family,
        "messages": [
            {"role": "system", "content": SEEDS[family]},
            {"role": "user", "content": "all original context and constraints"},
        ],
    }
    wire = task_wire(r)
    assert wire["messages"] == [
        {"role": "system", "content": "literal instruction"},
        {"role": "user", "content": "all original context and constraints"},
    ]


@pytest.mark.parametrize(
    "change", ["gold", "extra", "tools", "parts", "missing_variable", "malformed"]
)
def test_unsupported_or_annotation_messages_not_silently_flattened(change):
    r = request()
    if change == "gold":
        r["unit"]["answer"] = "a"
    elif change == "extra":
        r["reference"] = {"u": 1}
    elif change == "tools":
        r["candidate"]["tools"] = [{"name": "tool", "parameters": {}}]
    elif change == "parts":
        r["candidate"]["messages"][0]["content"] = [
            {"type": "input_image", "image_url": "AUTHORED"}
        ]
    else:
        r["candidate"]["messages"][0]["content"] = (
            "{{ answer }}" if change == "missing_variable" else "{{ malformed"
        )
    with pytest.raises((ValueError, KeyError)):
        task_wire(r)


def test_description_family_is_task_only_and_keeps_legacy_family_caps():
    assert generation_cap({"family": "cir_description", "role": "task"}) == (
        RETRIEVAL_DESCRIPTION_CAP
    )
    for family, cap in TASK_CAPS.items():
        assert generation_cap({"family": family, "role": "task"}) == cap
    with pytest.raises(ValueError, match="task request only"):
        generation_cap({"family": "cir_description", "role": "proposer"})
    with pytest.raises(ValueError, match="unknown"):
        generation_cap({"family": "unknown", "role": "task"})
