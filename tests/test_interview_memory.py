"""Independent BM25 arithmetic, immutable snapshot and UTF-8 budget oracles."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from promptwitness.interview_memory import (
    BM25MemoryIndex,
    MemoryEntry,
    MemoryHit,
    MemoryRetrieval,
    MemorySnapshot,
    SourceHead,
    memory_tokens,
)
from promptwitness.interview_models import (
    BoundEvidence,
    CriterionLink,
    InterviewContractError,
    MemoryRecord,
    ParticipantAnswer,
    RetrievalPolicy,
    contract_digest,
    contract_json,
)


def entry(
    text: str = "red red blue",
    *,
    identifier: str = "a",
    participant: str = "person",
    interview: str = "prior",
    revision: int = 4,
    answer_revision: int = 3,
    summary: str = "",
) -> MemoryEntry:
    answer = ParticipantAnswer(
        interview, participant, identifier, "question", text, answer_revision
    )
    memory = MemoryRecord(
        participant,
        (BoundEvidence.from_answer(answer, 0, len(text)),),
        (CriterionLink("topic", "criterion"),),
        summary,
    )
    return MemoryEntry(memory, interview, revision)


def snapshot(entries: tuple[MemoryEntry, ...] | None = None) -> MemorySnapshot:
    if entries is None:
        entries = (
            entry(),
            entry("blue green", identifier="b"),
            entry("red green green green", identifier="c"),
        )
    return MemorySnapshot.build("person", (SourceHead("prior", 6, "a" * 64),), entries)


def context_oracle(
    source: MemorySnapshot, selected: tuple[MemoryEntry, ...], prefix: str = "", suffix: str = ""
) -> str:
    value = {
        "format": "promptwitness.interview-memory-context/v1",
        "memories": [item.to_dict() for item in selected],
        "snapshot_digest": source.digest,
    }
    return (
        prefix
        + json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        + suffix
    )


def test_independent_three_document_bm25_calculation_and_query_term_deduplication() -> None:
    source = snapshot()
    result = BM25MemoryIndex(source).search("red blue red")
    # Independent hand arithmetic: N=3, lengths=(3,2,4), avgdl=3, df(red)=df(blue)=2.
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = {
        "red red blue": idf * (2 * 2.2 / (2 + 1.2) + 1 * 2.2 / (1 + 1.2)),
        "blue green": idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / 3)),
        "red green green green": idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 4 / 3)),
    }
    texts = {
        item.memory.memory_id: item.memory.evidence[0].reference.quote for item in source.entries
    }
    assert [texts[hit.memory_id] for hit in result.hits] == [
        "red red blue",
        "blue green",
        "red green green green",
    ]
    for hit in result.hits:
        assert hit.score == pytest.approx(expected[texts[hit.memory_id]], rel=1e-14)
    once = BM25MemoryIndex(source).search("BLUE red")
    assert result.hits == once.hits
    assert result.context == context_oracle(source, result.selected)
    assert result.context_bytes == len(result.context.encode("utf-8"))
    report = result.to_dict()
    assert (report["considered"], report["matched"], report["selected"]) == (3, 3, 3)
    assert (
        report["omitted_no_overlap"]
        == report["omitted_top_k"]
        == report["omitted_context_bytes"]
        == 0
    )


def test_zero_token_document_contributes_to_scoped_corpus_statistics() -> None:
    rows = (entry("red", identifier="a"), entry("😀...", identifier="b"))
    result = BM25MemoryIndex(snapshot(rows)).search("red")
    # N=2 and avgdl=(1+0)/2. The empty lexical row is not discarded from N.
    expected = math.log(1 + 1.5 / 1.5) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 1 / 0.5))
    assert len(result.hits) == 1 and result.hits[0].score == pytest.approx(expected)
    assert result.to_dict()["omitted_no_overlap"] == 1


def test_same_score_ties_use_content_id_order_not_catalog_input_order() -> None:
    rows = (entry("red blue", identifier="a"), entry("red blue", identifier="b"))
    normal = BM25MemoryIndex(snapshot(rows)).search("red")
    reversed_result = BM25MemoryIndex(snapshot(tuple(reversed(rows)))).search("red")
    assert normal.hits == reversed_result.hits
    assert normal.hits[0].score == normal.hits[1].score
    assert normal.memory_ids == tuple(sorted(item.memory.memory_id for item in rows))
    assert snapshot(rows).digest == snapshot(tuple(reversed(rows))).digest


def test_fixed_unicode_tokenizer_casefold_without_normalizing_original_source() -> None:
    assert memory_tokens("Straße STRASSE Café e\u0301 _ A_B \uff11\uff12") == (
        "strasse",
        "strasse",
        "café",
        "e",
        "a",
        "b",
        "\uff11\uff12",
    )
    rows = (entry("Straße e\u0301", identifier="a"), entry("é", identifier="b"))
    source = snapshot(rows)
    assert BM25MemoryIndex(source).search("STRASSE").memory_ids == (rows[0].memory.memory_id,)
    assert BM25MemoryIndex(source).search("é").memory_ids == (rows[1].memory.memory_id,)
    assert rows[0].memory.evidence[0].reference.quote == "Straße e\u0301"


def test_other_participants_interviews_and_future_entries_do_not_change_scope_statistics() -> None:
    kept = entry("red blue")
    own = snapshot((kept,))
    other_person = entry(
        "red " * 100, identifier="b", participant="other-person", interview="unrelated"
    )
    other_interview = entry("red " * 200, identifier="c", interview="not-selected")
    future = entry("red " * 300, identifier="d", revision=7)
    mixed = snapshot((other_person, kept, future, other_interview))
    assert mixed.digest == own.digest
    assert mixed.to_dict() == own.to_dict()
    assert (
        BM25MemoryIndex(mixed).search("red").to_dict()
        == BM25MemoryIndex(own).search("red").to_dict()
    )
    assert not any(answer.participant_id == "other-person" for answer in mixed.answers)
    assert BM25MemoryIndex(mixed).search("red").to_dict()["considered"] == 1


def test_unselected_cross_interview_evidence_is_not_partially_stripped() -> None:
    first = entry()
    other = entry("past evidence", identifier="b", interview="second")
    cross = MemoryEntry(
        replace(first.memory, evidence=first.memory.evidence + other.memory.evidence), "prior", 4
    )
    assert snapshot((cross,)).entries == ()
    selected = MemorySnapshot.build(
        "person", (SourceHead("prior", 6, "a" * 64), SourceHead("second", 3, "b" * 64)), (cross,)
    )
    assert len(selected.entries) == 1 and len(selected.answers) == 2
    with pytest.raises(InterviewContractError, match="newer"):
        MemorySnapshot.build(
            "person",
            (SourceHead("prior", 6, "a" * 64), SourceHead("second", 2, "b" * 64)),
            (cross,),
        )
    with pytest.raises(InterviewContractError, match="precede"):
        replace(first, revision=2)


def test_snapshot_is_frozen_self_contained_and_roundtrips_without_a_live_catalog() -> None:
    rows = [entry(), entry("blue", identifier="b")]
    heads = [SourceHead("prior", 6, "a" * 64)]
    source = MemorySnapshot.build("person", heads, rows)
    index = BM25MemoryIndex(source)
    before = index.search("blue").to_dict()
    rows.clear()
    heads.clear()
    assert index.search("blue").to_dict() == before
    encoded = contract_json(source.to_dict())
    restored = MemorySnapshot.from_json(encoded.encode("utf-8"))
    assert restored.to_dict() == source.to_dict()
    assert BM25MemoryIndex(restored).search("blue").to_dict() == before
    with pytest.raises(FrozenInstanceError):
        source.participant_id = "different"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        index.snapshot = snapshot(())  # type: ignore[misc]
    with pytest.raises(AttributeError):
        index.policy = RetrievalPolicy(top_k=0)  # type: ignore[misc]
    changed = source.to_dict()
    changed["answers"][0]["text"] = "changed externally"
    assert source.answers[0].text != "changed externally"


def test_composite_answer_ids_across_interviews_never_shadow_sources() -> None:
    rows = (
        entry("first source", identifier="shared"),
        entry("second source", identifier="shared", interview="second"),
    )
    heads = (SourceHead("prior", 6, "a" * 64), SourceHead("second", 6, "b" * 64))
    source = MemorySnapshot.build("person", heads, rows)
    assert [(answer.interview_id, answer.answer_id) for answer in source.answers] == [
        ("prior", "shared"),
        ("second", "shared"),
    ]
    assert MemorySnapshot.from_dict(source.to_dict()).answers == source.answers
    ambiguous = entry("altered source", identifier="shared")
    with pytest.raises(InterviewContractError, match="ambiguous"):
        MemorySnapshot.build("person", heads, (rows[0], ambiguous))


def test_corrections_have_distinct_ids_and_do_not_rewrite_pinned_history() -> None:
    original = entry("red original")
    correction = MemoryEntry(
        replace(
            original.memory,
            model_summary="Corrected interpretation",
            supersedes=original.memory.memory_id,
        ),
        "prior",
        7,
    )
    pinned = snapshot((original, correction))
    assert pinned.entries == (original,)
    later = MemorySnapshot.build(
        "person", (SourceHead("prior", 7, "b" * 64),), (original, correction)
    )
    assert len(later.entries) == 2
    assert pinned.digest != later.digest
    assert original.memory.model_summary == ""
    # This foundation keeps both records visible; it does not infer which is true.
    assert BM25MemoryIndex(later).search("red").to_dict()["considered"] == 2


@pytest.mark.parametrize(
    "change",
    [
        {"revision": True},
        {"revision": 0},
        {"revision": 1.0},
        {"digest": "A" * 64},
        {"interview_id": ""},
    ],
)
def test_source_heads_reject_invalid_exact_types_and_digests(change: dict[str, Any]) -> None:
    with pytest.raises(InterviewContractError):
        replace(SourceHead("prior", 6, "a" * 64), **change)


def test_snapshot_scope_duplicate_and_type_checks() -> None:
    one = entry()
    head = SourceHead("prior", 6, "a" * 64)
    for callback in (
        lambda: MemorySnapshot("person", (head, head), (one,)),
        lambda: MemorySnapshot("person", (head,), (one, one)),
        lambda: MemorySnapshot("other", (head,), (one,)),
        lambda: MemorySnapshot("person", (), (one,)),
        lambda: MemorySnapshot("person", (replace(head, revision=3),), (one,)),
        lambda: MemorySnapshot("person", ("bad",), (one,)),
        lambda: MemorySnapshot("person", (head,), ("bad",)),
        lambda: MemorySnapshot.build("person", (head, head), (one,)),
        lambda: MemorySnapshot.build("person", (head,), (one, one)),
        lambda: MemorySnapshot.build("person", (head,), ("bad",)),
        lambda: MemoryEntry(None, "prior", 1),
    ):
        with pytest.raises(InterviewContractError):
            callback()


@pytest.mark.parametrize(
    "mode",
    [
        "digest",
        "quote",
        "head",
        "duplicate_answer",
        "extra_answer",
        "participant",
        "extra",
        "source_kind",
    ],
)
def test_snapshot_serialization_rejects_corruption_and_unused_sources(mode: str) -> None:
    data = snapshot((entry(),)).to_dict()
    if mode == "digest":
        data["digest"] = "0" * 64
    elif mode == "quote":
        data["entries"][0]["memory"]["evidence"][0]["quote"] = "changed"
    elif mode == "head":
        data["heads"][0]["revision"] = 3
    elif mode == "duplicate_answer":
        data["answers"].append(data["answers"][0])
    elif mode == "extra_answer":
        data["answers"].append(
            entry("unused", identifier="unused").memory.evidence[0].answer.to_dict()
        )
    elif mode == "participant":
        data["participant_id"] = "other"
    elif mode == "source_kind":
        data["answers"][0]["source_kind"] = "assistant"
    else:
        data["extra"] = True
    if mode != "digest":
        data["digest"] = contract_digest(
            {key: value for key, value in data.items() if key != "digest"}
        )
    with pytest.raises(InterviewContractError):
        MemorySnapshot.from_dict(data)


def test_model_summary_retrieval_is_explicitly_labelled_and_optional() -> None:
    only = entry("red quote", summary="unique summary assertion")
    source = snapshot((only,))
    included = BM25MemoryIndex(source).search("unique")
    assert included.memory_ids == (only.memory.memory_id,)
    assert '"kind":"model_proposed"' in included.context
    excluded = BM25MemoryIndex(source, RetrievalPolicy(include_model_summaries=False)).search(
        "unique"
    )
    assert excluded.memory_ids == () and excluded.to_dict()["omitted_no_overlap"] == 1
    quotes = BM25MemoryIndex(source, RetrievalPolicy(include_model_summaries=False)).search("red")
    assert quotes.memory_ids == included.memory_ids


@pytest.mark.parametrize("query", ["", "  ", "😀...", "nooverlap"])
def test_empty_and_no_overlap_queries_return_no_arbitrary_memory(query: str) -> None:
    result = BM25MemoryIndex(snapshot()).search(query)
    assert result.memory_ids == () and result.hits == ()
    assert result.to_dict()["omitted_no_overlap"] == 3
    assert json.loads(result.context)["memories"] == []
    empty = MemorySnapshot.build("person", (), ())
    assert BM25MemoryIndex(empty).search(query).to_dict()["considered"] == 0
    no_terms = snapshot((entry("😀..."),))
    assert BM25MemoryIndex(no_terms).search("red").hits == ()


def test_utf8_budget_counts_prefix_suffix_and_envelope_at_exact_boundary() -> None:
    one = entry("red 😀")
    source = snapshot((one,))
    prefix, suffix = "Instructions 😀\r\n", "\nAgenda e\u0301"
    rendered = context_oracle(source, (one,), prefix, suffix)
    exact = len(rendered.encode("utf-8"))
    fit = BM25MemoryIndex(source, RetrievalPolicy(context_bytes=exact)).search(
        "red", context_prefix=prefix, context_suffix=suffix
    )
    assert fit.context == rendered and fit.context_bytes == exact
    assert fit.memory_ids == (one.memory.memory_id,)
    missed = BM25MemoryIndex(source, RetrievalPolicy(context_bytes=exact - 1)).search(
        "red", context_prefix=prefix, context_suffix=suffix
    )
    assert missed.memory_ids == ()
    assert missed.hits[0].omission == "context_bytes"
    assert missed.context == context_oracle(source, (), prefix, suffix)
    assert one.memory.memory_id not in missed.context
    baseline = len(context_oracle(source, (), prefix, suffix).encode("utf-8"))
    with pytest.raises(InterviewContractError, match="fixed context"):
        BM25MemoryIndex(source, RetrievalPolicy(context_bytes=baseline - 1)).search(
            "", context_prefix=prefix, context_suffix=suffix
        )


def test_byte_omission_backfills_smaller_ranked_memory_without_misreporting_context_ids() -> None:
    large = entry("red " * 100, identifier="a")
    small = entry("red", identifier="b")
    source = snapshot((large, small))
    budget = len(context_oracle(source, (small,)).encode("utf-8"))
    result = BM25MemoryIndex(source, RetrievalPolicy(top_k=1, context_bytes=budget, b=0)).search(
        "red"
    )
    assert [hit.memory_id for hit in result.hits] == [
        large.memory.memory_id,
        small.memory.memory_id,
    ]
    assert [hit.omission for hit in result.hits] == ["context_bytes", None]
    assert result.memory_ids == (small.memory.memory_id,)
    assert large.memory.memory_id not in result.context
    report = result.to_dict()
    assert report["omitted_context_bytes"] == 1 and report["omitted_top_k"] == 0
    assert result.context_bytes == budget


def test_top_k_omission_is_distinct_from_byte_omission_and_zero_k_is_supported() -> None:
    source = snapshot()
    result = BM25MemoryIndex(source, RetrievalPolicy(top_k=1)).search("red blue")
    assert len(result.selected) == 1
    assert result.to_dict()["omitted_top_k"] == 2
    assert result.to_dict()["omitted_context_bytes"] == 0
    zero = BM25MemoryIndex(source, RetrievalPolicy(top_k=0)).search("red blue")
    assert zero.memory_ids == () and zero.to_dict()["omitted_top_k"] == 3
    assert all(hit.omission == "top_k" for hit in zero.hits)


@pytest.mark.parametrize("mode", ["selected", "order", "duplicate", "scope", "budget", "hit_type"])
def test_public_retrieval_result_rejects_inconsistent_reported_context(mode: str) -> None:
    result = BM25MemoryIndex(snapshot()).search("red blue")
    with pytest.raises(InterviewContractError):
        if mode == "selected":
            replace(result, selected=())
        elif mode == "order":
            replace(result, hits=tuple(reversed(result.hits)))
        elif mode == "duplicate":
            replace(result, hits=(result.hits[0], result.hits[0]))
        elif mode == "scope":
            replace(result, snapshot=snapshot(()))
        elif mode == "budget":
            replace(result, policy=RetrievalPolicy(context_bytes=1))
        else:
            replace(result, hits=("bad",))


@pytest.mark.parametrize("score", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_hit_scores_are_exact_finite_positive_numbers(score: Any) -> None:
    with pytest.raises(InterviewContractError):
        MemoryHit("mem-" + "a" * 64, score)


def test_retrieval_input_types_and_unicode_bounds() -> None:
    with pytest.raises(InterviewContractError):
        BM25MemoryIndex(None)  # type: ignore[arg-type]
    with pytest.raises(InterviewContractError):
        BM25MemoryIndex(snapshot(), {})  # type: ignore[arg-type]
    index = BM25MemoryIndex(snapshot())
    for query in (None, "\ud800", "x" * (1024 * 1024 + 1)):
        with pytest.raises(InterviewContractError):
            index.search(query)  # type: ignore[arg-type]
    with pytest.raises(InterviewContractError):
        index.search("red", context_suffix="\ud800")
    with pytest.raises(InterviewContractError):
        MemoryHit("invalid", 1)
    with pytest.raises(InterviewContractError):
        MemoryHit("mem-" + "a" * 64, 1, "unknown")
    with pytest.raises(InterviewContractError):
        MemoryRetrieval(None, RetrievalPolicy(), "", (), ())  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["entry", "snapshot", "retrieval"])
def test_complete_export_envelopes_fit_the_generic_contract_cap(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = entry()
    source = snapshot((row,))
    result = BM25MemoryIndex(source).search("red")
    value = {"entry": row, "snapshot": source, "retrieval": result}[kind]
    exported = value.to_dict()
    full_bytes = len(contract_json(exported).encode("utf-8"))
    monkeypatch.setattr("promptwitness.interview_models.MAX_CONTRACT_BYTES", full_bytes)
    assert contract_json(replace(value).to_dict()) == contract_json(exported)
    monkeypatch.setattr("promptwitness.interview_models.MAX_CONTRACT_BYTES", full_bytes - 1)
    with pytest.raises(InterviewContractError, match="byte limit"):
        replace(value)
