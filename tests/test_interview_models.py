"""Independent source-citation and closed interview-contract oracles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from promptwitness.interview_models import (
    AssessedCoverage,
    BoundEvidence,
    CriterionAssessment,
    CriterionLink,
    EvidenceRef,
    InterviewBudgets,
    InterviewContractError,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    MemoryRecord,
    ParticipantAnswer,
    RetrievalPolicy,
    answer_catalog,
    assessed_coverage,
    contract_digest,
    contract_json,
    load_interview_json,
)
from promptwitness.models import Message


def plan() -> InterviewPlan:
    return InterviewPlan(
        "career",
        "1",
        "Understand the participant's work and learning.",
        (
            InterviewTopic(
                "work",
                "Current work",
                (
                    InterviewCriterion("role", "Describe your role."),
                    InterviewCriterion("challenge", "Describe a challenge."),
                ),
            ),
            InterviewTopic(
                "learning",
                "Learning",
                (InterviewCriterion("skill", "Explain a skill you learned."),),
            ),
        ),
    )


def answer() -> ParticipantAnswer:
    return ParticipantAnswer("interview", "participant", "answer", "question", "A😀e\u0301\r\nZ", 3)


def evidence() -> BoundEvidence:
    return BoundEvidence.from_answer(answer(), 1, 4)


def memory() -> MemoryRecord:
    return MemoryRecord(
        "participant",
        (evidence(),),
        (CriterionLink("work", "role"),),
        "A model-proposed interpretation.",
    )


def test_plan_roundtrip_order_identity_and_immutable_generation() -> None:
    original = plan()
    assert original.required_criteria == (
        ("work", "role"),
        ("work", "challenge"),
        ("learning", "skill"),
    )
    assert InterviewPlan.from_dict(original.to_dict()) == original
    assert InterviewPlan.from_json(contract_json(original.to_dict()).encode("utf-8")) == original
    independent = json.dumps(
        original.to_dict(),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert original.digest == hashlib.sha256(independent.encode("utf-8")).hexdigest()
    assert replace(original, topics=tuple(reversed(original.topics))).digest != original.digest
    mutable = {"max_tokens": 100, "temperature": 0.0}
    constructed = replace(original, analyst_generation=mutable)
    mutable["max_tokens"] = 999
    assert constructed.analyst_generation["max_tokens"] == 100
    with pytest.raises(TypeError):
        constructed.analyst_generation["max_tokens"] = 2  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        constructed.version = "changed"  # type: ignore[misc]
    wire = constructed.to_dict()
    wire["topics"][0]["criteria"][0]["description"] = "changed"
    assert constructed.topics[0].criteria[0].description == "Describe your role."


@pytest.mark.parametrize(
    "path", [(), ("topics", 0), ("topics", 0, "criteria", 0), ("budgets",), ("retrieval",)]
)
@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_plan_nested_objects_have_closed_fields(path: tuple[Any, ...], mode: str) -> None:
    value = plan().to_dict()
    nested = value
    for key in path:
        nested = nested[key]
    if mode == "missing":
        del nested[next(iter(nested))]
    else:
        nested["endpoint"] = "not configurable here"
    with pytest.raises(InterviewContractError):
        InterviewPlan.from_dict(value)


@pytest.mark.parametrize(
    "change",
    [
        {"plan_id": ""},
        {"version": True},
        {"purpose": "\ud800"},
        {"topics": ()},
        {"topics": ("bad",)},
        {"budgets": {}},
        {"retrieval": {}},
        {"analyst_prompt_version": ""},
        {"analyst_generation": {"max_tokens": True}},
        {"analyst_generation": {"max_tokens": 0}},
        {"analyst_generation": {"max_tokens": 1000001}},
        {"analyst_generation": {"max_tokens": 1, "endpoint": "unsafe"}},
        {"analyst_generation": {"max_tokens": 1, "temperature": float("nan")}},
        {"questioner_generation": {"max_tokens": 1, "seed": 2**63}},
        {"questioner_generation": {"max_tokens": 1, "seed": True}},
    ],
)
def test_invalid_public_plan_construction(change: dict[str, Any]) -> None:
    with pytest.raises(InterviewContractError):
        replace(plan(), **change)


def test_plan_rejects_duplicate_ids_and_inconsistent_budget() -> None:
    original = plan()
    with pytest.raises(InterviewContractError, match="unique"):
        replace(original, topics=(original.topics[0], original.topics[0]))
    repeated = InterviewTopic("other", "Other", (InterviewCriterion("role", "Same ID elsewhere"),))
    with pytest.raises(InterviewContractError, match="unique"):
        replace(original, topics=(original.topics[0], repeated))
    with pytest.raises(InterviewContractError, match="unique"):
        InterviewTopic(
            "t", "Topic", (InterviewCriterion("c", "first"), InterviewCriterion("c", "second"))
        )
    with pytest.raises(InterviewContractError, match="retrieval budget"):
        replace(original, budgets=InterviewBudgets(context_bytes=100))
    value = original.to_dict()
    value["format"] = "future"
    with pytest.raises(InterviewContractError):
        InterviewPlan.from_dict(value)
    value = original.to_dict()
    value["retrieval"]["tokenizer"] = "unknown"
    with pytest.raises(InterviewContractError):
        InterviewPlan.from_dict(value)


@pytest.mark.parametrize(
    "field_name",
    [
        "participant_turns",
        "provider_calls",
        "question_bytes",
        "answer_bytes",
        "context_bytes",
        "followups_per_topic",
        "emergent_topics",
    ],
)
@pytest.mark.parametrize("bad", [True, -1, 1.0, "1", 2**63])
def test_budget_integer_types_and_bounds(field_name: str, bad: Any) -> None:
    with pytest.raises(InterviewContractError):
        InterviewBudgets(**{field_name: bad})


@pytest.mark.parametrize(
    "change",
    [
        {"top_k": True},
        {"top_k": -1},
        {"top_k": 1001},
        {"context_bytes": False},
        {"context_bytes": 0},
        {"k1": True},
        {"k1": 0},
        {"k1": float("inf")},
        {"k1": "1.2"},
        {"b": True},
        {"b": -0.1},
        {"b": 1.1},
        {"include_model_summaries": 1},
    ],
)
def test_retrieval_policy_rejects_nonportable_or_unbounded_settings(change: dict[str, Any]) -> None:
    with pytest.raises(InterviewContractError):
        RetrievalPolicy(**change)
    assert RetrievalPolicy.from_dict(RetrievalPolicy(top_k=0, b=0).to_dict()).top_k == 0


def test_answer_codepoint_identity_preserves_emoji_combining_marks_and_crlf() -> None:
    original = answer()
    assert len(original.text) == 7
    assert len(original.text.encode("utf-8")) == 11
    assert original.text_digest == hashlib.sha256("A😀e\u0301\r\nZ".encode("utf-8")).hexdigest()
    assert ParticipantAnswer.from_json(contract_json(original.to_dict())) == original
    assert replace(original, text="A😀é\r\nZ").text_digest != original.text_digest
    assert replace(original, participant_id="different").text_digest == original.text_digest
    assert replace(original, participant_id="different").digest != original.digest
    empty = replace(original, text="")
    assert ParticipantAnswer.from_dict(empty.to_dict()).text == ""
    with pytest.raises(InterviewContractError):
        EvidenceRef.from_answer(empty, 0, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"revision": True},
        {"revision": 0},
        {"revision": 1.0},
        {"text": "\udfff"},
        {"answer_id": ""},
        {"question_id": None},
    ],
)
def test_answer_constructor_strictness(change: dict[str, Any]) -> None:
    with pytest.raises(InterviewContractError):
        replace(answer(), **change)


@pytest.mark.parametrize(
    "change",
    [
        {"format": "future"},
        {"source_kind": "assistant"},
        {"source_kind": "tool"},
        {"text_digest": "0" * 64},
        {"extra": 1},
    ],
)
def test_answer_deserialization_requires_participant_source_and_valid_digest(
    change: dict[str, Any],
) -> None:
    with pytest.raises(InterviewContractError):
        ParticipantAnswer.from_dict(answer().to_dict() | change)


def test_exact_reference_binding_and_content_id_are_independent_of_bytes() -> None:
    source = answer()
    reference = EvidenceRef.from_answer(source, 1, 4)
    assert reference.quote == "😀e\u0301"
    assert reference.end - reference.start == 3
    assert len(reference.quote.encode("utf-8")) == 7
    reference.validate(source, participant_id="participant", max_revision=3)
    assert EvidenceRef.from_dict(reference.to_dict()) == reference
    value = reference.to_dict()
    identifier = value.pop("id")
    assert (
        identifier
        == "ev-"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    assert BoundEvidence(reference, source).evidence_id == identifier
    with pytest.raises(InterviewContractError):
        reference.validate(source, participant_id="someone else")
    with pytest.raises(InterviewContractError, match="newer"):
        reference.validate(source, max_revision=2)
    with pytest.raises(InterviewContractError):
        reference.validate(Message("assistant", source.text))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "start,end", [(True, 4), (1, True), (-1, 2), (1.0, 4), (1, 8), (4, 3), (3, 3)]
)
def test_reference_intervals_reject_empty_invalid_or_boolean_offsets(start: Any, end: Any) -> None:
    with pytest.raises(InterviewContractError):
        EvidenceRef.from_answer(answer(), start, end)


@pytest.mark.parametrize(
    "change",
    [
        {"answer_id": "another"},
        {"answer_digest": "0" * 64},
        {"answer_revision": 4},
        {"participant_id": "other"},
        {"interview_id": "other"},
        {"quote": "abc"},
        {"start": 2, "end": 5},
        {"start": 1, "end": 3, "quote": "😀é"},
    ],
)
def test_reference_cannot_bind_wrong_source_or_changed_exact_quote(change: dict[str, Any]) -> None:
    altered = replace(evidence().reference, **change)
    with pytest.raises(InterviewContractError):
        BoundEvidence(altered, answer())


def test_ambiguous_answer_catalog_uses_composite_identity_not_answer_id_only() -> None:
    original = answer()
    other = replace(original, interview_id="different", text="different answer")
    catalog = answer_catalog((original, other))
    assert catalog[("interview", "answer")] == original
    assert catalog[("different", "answer")] == other
    assert MemoryRecord.from_dict(memory().to_dict(), answers=(other, original)) == memory()
    for duplicate in (
        original,
        replace(original, text="changed"),
        replace(original, participant_id="other"),
    ):
        with pytest.raises(InterviewContractError, match="ambiguous"):
            answer_catalog((original, duplicate))


def test_memory_is_content_addressed_source_bound_and_summary_is_not_a_quote() -> None:
    original = memory()
    assert (
        MemoryRecord.from_json(contract_json(original.to_dict()), answers=(answer(),)) == original
    )
    payload = original.to_dict()
    identity = payload.pop("id")
    assert (
        identity
        == "mem-"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    assert payload["summary"] == {"kind": "model_proposed", "text": original.model_summary}
    assert payload["evidence"][0]["quote"] == "😀e\u0301"
    corrected = replace(
        original, model_summary="Reviewed correction", supersedes=original.memory_id
    )
    assert corrected.memory_id != original.memory_id
    assert corrected.supersedes == original.memory_id
    assert original.model_summary == "A model-proposed interpretation."
    assert replace(original, model_summary="").memory_id != original.memory_id
    with pytest.raises(InterviewContractError):
        MemoryRecord.from_dict(original.to_dict(), answers=())
    with pytest.raises(InterviewContractError):
        replace(original, participant_id="other")


@pytest.mark.parametrize(
    "change",
    [
        {"evidence": ()},
        {"evidence": ("unbound",)},
        {"links": ()},
        {"links": ({},)},
        {"supersedes": "invented"},
        {"model_summary": "\ud800"},
    ],
)
def test_memory_rejects_unbound_or_invalid_fields(change: dict[str, Any]) -> None:
    with pytest.raises(InterviewContractError):
        replace(memory(), **change)


def test_duplicate_memory_links_references_and_corrupt_serialization() -> None:
    original = memory()
    with pytest.raises(InterviewContractError, match="duplicate evidence"):
        replace(original, evidence=(evidence(), evidence()))
    with pytest.raises(InterviewContractError):
        replace(original, links=(original.links[0], original.links[0]))
    for change in (
        {"id": "mem-" + "0" * 64},
        {"supersedes": original.memory_id},
        {"extra": True},
        {"summary": {"kind": "fact", "text": "claim"}},
    ):
        with pytest.raises(InterviewContractError):
            MemoryRecord.from_dict(original.to_dict() | change, answers=(answer(),))
    wire = original.to_dict()
    wire["evidence"][0]["quote"] = "bad"
    with pytest.raises(InterviewContractError):
        MemoryRecord.from_dict(wire, answers=(answer(),))


def assessment(
    criterion_id: str, status: str = "covered", *, topic_id: str = "work"
) -> CriterionAssessment:
    return CriterionAssessment(
        "participant",
        "interview",
        CriterionLink(topic_id, criterion_id),
        status,
        () if status == "unanswered" else (evidence(),),
        "Model assessment, not factual verification.",
    )


def test_assessed_coverage_keeps_original_and_accepted_emergent_denominators_separate() -> None:
    original = plan()
    emergent = InterviewTopic(
        "unexpected", "An explicitly accepted topic", (InterviewCriterion("new", "New criterion"),)
    )
    rows = (
        assessment("role"),
        assessment("challenge", "partial"),
        assessment("new", topic_id="unexpected"),
    )
    report = assessed_coverage(original, rows, accepted_emergent=(emergent,)).to_dict()
    assert report["required"] == {
        "total": 3,
        "covered": 1,
        "partial": 1,
        "unanswered": 1,
        "assessed_coverage": 1 / 3,
    }
    assert report["emergent"] == {
        "total": 1,
        "covered": 1,
        "partial": 0,
        "unanswered": 0,
        "assessed_coverage": 1.0,
    }
    assert assessed_coverage(original, ()).to_dict()["required"]["assessed_coverage"] == 0
    assert assessed_coverage(original, ()).to_dict()["emergent"]["assessed_coverage"] is None
    assert original.required_criteria == (
        ("work", "role"),
        ("work", "challenge"),
        ("learning", "skill"),
    )
    # A memory's existence never contributes an implicit covered decision.
    assert memory().links[0] == CriterionLink("work", "role")
    assert assessed_coverage(original, ()).required_covered == 0
    assert CriterionAssessment.from_dict(rows[0].to_dict(), answers=(answer(),)) == rows[0]


@pytest.mark.parametrize("status", ["covered", "partial"])
def test_assessed_positive_status_requires_existing_scoped_evidence(status: str) -> None:
    with pytest.raises(InterviewContractError):
        replace(assessment("role", status), evidence=())
    with pytest.raises(InterviewContractError):
        replace(assessment("role", status), evidence=(evidence().reference,))
    with pytest.raises(InterviewContractError):
        replace(assessment("role", status), interview_id="different")
    with pytest.raises(InterviewContractError):
        replace(assessment("role", status), participant_id="different")
    with pytest.raises(InterviewContractError):
        replace(assessment("role", status), evidence=(evidence(), evidence()))


def test_coverage_rejects_duplicates_unknown_pending_or_cross_interview_decisions() -> None:
    row = assessment("role")
    with pytest.raises(InterviewContractError, match="duplicate criterion"):
        assessed_coverage(plan(), (row, row))
    with pytest.raises(InterviewContractError, match="unknown or unaccepted"):
        assessed_coverage(plan(), (assessment("new", topic_id="pending"),))
    with pytest.raises(InterviewContractError, match="cannot merge"):
        assessed_coverage(
            plan(),
            (
                row,
                CriterionAssessment(
                    "other", "other-interview", CriterionLink("work", "challenge"), "unanswered"
                ),
            ),
        )
    with pytest.raises(InterviewContractError):
        assessed_coverage(plan(), ("covered",))  # type: ignore[arg-type]
    with pytest.raises(InterviewContractError):
        assessed_coverage(plan(), (), accepted_emergent=(plan().topics[0],))
    with pytest.raises(InterviewContractError):
        AssessedCoverage(0, 0, 0, 0, 0, 0)
    with pytest.raises(InterviewContractError):
        AssessedCoverage(1, 1, 1, 0, 0, 0)
    with pytest.raises(InterviewContractError):
        replace(row, status="deferred")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":1,"\\u0078":2}',
        b"NaN",
        b"1e999",
        b'"\\ud800"',
        b"\xff",
        b"[" * 1200,
        str(2**63),
        None,
    ],
)
def test_json_loader_rejects_ambiguous_nonportable_values(raw: Any) -> None:
    with pytest.raises(InterviewContractError):
        load_interview_json(raw)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), 2**63, {1: "not a key"}, b"bytes", object(), "\ud800"]
)
def test_canonical_contract_json_checks_python_values(value: Any) -> None:
    with pytest.raises(InterviewContractError):
        contract_json(value)


def test_contract_depth_and_byte_limits_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    nested: Any = "x"
    for _ in range(34):
        nested = [nested]
    with pytest.raises(InterviewContractError, match="nesting"):
        contract_json(nested)
    monkeypatch.setattr("promptwitness.interview_models.MAX_CONTRACT_BYTES", 8)
    with pytest.raises(InterviewContractError):
        contract_json({"long": "value"})
    with pytest.raises(InterviewContractError):
        load_interview_json(b'{"long":"value"}')
    assert load_interview_json(b'{"x":1}') == {"x": 1}
    assert contract_digest(True) != contract_digest(1)


@pytest.mark.parametrize("kind", ["plan", "answer", "evidence", "memory"])
def test_complete_export_bytes_bound_constructor_not_only_identity_body(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = {
        "plan": plan(),
        "answer": answer(),
        "evidence": evidence().reference,
        "memory": memory(),
    }
    value = objects[kind]
    exported = value.to_dict()
    full_bytes = len(contract_json(exported).encode("utf-8"))
    monkeypatch.setattr("promptwitness.interview_models.MAX_CONTRACT_BYTES", full_bytes)
    assert contract_json(replace(value).to_dict()) == contract_json(exported)
    monkeypatch.setattr("promptwitness.interview_models.MAX_CONTRACT_BYTES", full_bytes - 1)
    with pytest.raises(InterviewContractError, match="byte limit"):
        replace(value)
