"""Independent adversarial source-consistency and retrieval arithmetic checks."""

import math
from dataclasses import replace

import pytest

import promptwitness.interview_models as contracts
from promptwitness.interview_memory import (
    BM25MemoryIndex,
    MemoryEntry,
    MemoryHit,
    MemorySnapshot,
    SourceHead,
)
from promptwitness.interview_models import (
    BoundEvidence,
    CriterionAssessment,
    CriterionLink,
    InterviewContractError,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    MemoryRecord,
    ParticipantAnswer,
    RetrievalPolicy,
    assessed_coverage,
    contract_json,
)

LINK = CriterionLink("topic", "criterion")


def answer(text="one two", *, interview="session", participant="participant"):
    return ParticipantAnswer(interview, participant, "answer", "question", text, 1)


def evidence(source):
    return BoundEvidence.from_answer(source, 0, len(source.text))


@pytest.mark.parametrize("kind", ["memory", "assessment"])
@pytest.mark.parametrize("change", [{"text": "changed"}, {"question_id": "other"}, {"revision": 2}])
def test_evidence_bundle_rejects_conflicting_composite_source_identity(kind, change):
    source = answer()
    conflicting = replace(source, **change)
    references = (evidence(source), evidence(conflicting))
    with pytest.raises(InterviewContractError):
        if kind == "memory":
            MemoryRecord("participant", references, (LINK,))
        else:
            CriterionAssessment("participant", "session", LINK, "covered", references)


def test_consistent_distinct_quotes_of_one_answer_remain_roundtrippable():
    source = answer()
    quotes = (BoundEvidence.from_answer(source, 0, 3), BoundEvidence.from_answer(source, 4, 7))
    record = MemoryRecord("participant", quotes, (LINK,))
    assessment = CriterionAssessment("participant", "session", LINK, "covered", quotes)
    assert MemoryRecord.from_dict(record.to_dict(), answers=(source,)) == record
    assert CriterionAssessment.from_dict(assessment.to_dict(), answers=(source,)) == assessment


def test_coverage_cannot_merge_conflicting_source_answers_across_criteria():
    plan = InterviewPlan(
        "review",
        "1",
        "Check coherent evidence.",
        (
            InterviewTopic(
                "topic", "Topic", (InterviewCriterion("a", "A"), InterviewCriterion("b", "B"))
            ),
        ),
    )
    rows = (
        CriterionAssessment(
            "participant", "session", CriterionLink("topic", "a"), "covered", (evidence(answer()),)
        ),
        CriterionAssessment(
            "participant",
            "session",
            CriterionLink("topic", "b"),
            "covered",
            (evidence(answer("different")),),
        ),
    )
    with pytest.raises(InterviewContractError):
        assessed_coverage(plan, rows)


@pytest.mark.parametrize("field", ["k1", "b", "score"])
def test_huge_numeric_inputs_are_contract_errors_not_float_overflows(field):
    with pytest.raises(InterviewContractError):
        if field == "score":
            MemoryHit("mem-" + "0" * 64, 10**400)
        else:
            RetrievalPolicy(**{field: 10**400})


def test_assessment_enforces_total_canonical_bytes_not_only_each_quote(monkeypatch):
    source = answer("a" * 1800)
    quotes = tuple(BoundEvidence.from_answer(source, start, start + 1000) for start in range(4))
    # Scale only the global contract limit; each independently bound source and
    # reference remains within it, but the combined assessment exceeds the cap.
    monkeypatch.setattr(contracts, "MAX_CONTRACT_BYTES", 4096)
    with pytest.raises(InterviewContractError):
        CriterionAssessment("participant", "session", LINK, "covered", quotes)


def entry(text, *, interview, participant="participant"):
    source = answer(text, interview=interview, participant=participant)
    record = MemoryRecord(participant, (evidence(source),), (LINK,))
    return MemoryEntry(record, interview, 2)


def scoped_snapshot():
    first = entry("apple apple banana", interview="one")
    second = entry("banana", interview="two")
    excluded = entry("apple " * 30, interview="other", participant="other")
    heads = (SourceHead("one", 2, "1" * 64), SourceHead("two", 2, "2" * 64))
    snapshot = MemorySnapshot.build("participant", heads, (first, second, excluded))
    assert snapshot.digest == MemorySnapshot.build("participant", heads, (first, second)).digest
    return snapshot


def test_scoped_bm25_matches_independent_two_document_formula():
    snapshot = scoped_snapshot()
    restored = MemorySnapshot.from_json(contract_json(snapshot.to_dict()))
    assert restored.digest == snapshot.digest
    assert {(item.interview_id, item.answer_id) for item in restored.answers} == {
        ("one", "answer"),
        ("two", "answer"),
    }
    index = BM25MemoryIndex(snapshot, RetrievalPolicy(include_model_summaries=False))
    result = index.search("apple apple")
    # Corpus lengths 3/1, average 2, tf 2, df 1, N 2, k1=1.2, b=.75.
    expected = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5)) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    assert len(result.hits) == 1
    assert result.hits[0].score == pytest.approx(expected, rel=1e-15)
    assert result.hits == index.search("apple").hits


def test_full_utf8_envelope_budget_includes_prefix_suffix_and_canonical_metadata():
    snapshot = scoped_snapshot()
    prefix, suffix = "😀\n", "e\u0301"
    initial = BM25MemoryIndex(snapshot).search(
        "apple", context_prefix=prefix, context_suffix=suffix
    )
    exact = len(initial.context.encode("utf-8"))
    assert initial.context_bytes == exact
    assert len(initial.context) < exact
    for difference in (0, -1):
        policy = RetrievalPolicy(context_bytes=exact + difference)
        result = BM25MemoryIndex(snapshot, policy).search(
            "apple", context_prefix=prefix, context_suffix=suffix
        )
        assert result.context_bytes <= exact + difference
        assert len(result.selected) == int(difference == 0)
        assert result.hits[0].omission == (None if difference == 0 else "context_bytes")
