"""No-paid stage contract fixtures with independent JSON/source/byte oracles."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
from interview_fixtures import completion_envelope

from promptwitness._interview_prompts import SCHEMAS, TEMPLATES
from promptwitness.interview_memory import MemoryEntry, MemorySnapshot, SourceHead
from promptwitness.interview_models import (
    BoundEvidence,
    CriterionAssessment,
    CriterionLink,
    InterviewBudgets,
    InterviewContractError,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    MemoryRecord,
    ParticipantAnswer,
    RetrievalPolicy,
    contract_digest,
    contract_json,
)
from promptwitness.interview_stages import (
    RENDERER_VERSION,
    SCHEMA_HASHES,
    TEMPLATE_HASHES,
    AnalysisResult,
    CompletionRecord,
    EmergentProposal,
    InterviewQuestion,
    InterviewStageContext,
    InterviewStageRequest,
    QuestionResult,
    StageTarget,
    build_analysis_request,
    build_question_request,
    parse_analysis_completion,
    parse_question_completion,
)
from promptwitness.providers import OpenAICompatibleProvider, prepare_chat_body
from promptwitness.sessions import OpenAISessionProvider


def identity() -> dict[str, Any]:
    return {
        "provider": "fixture",
        "transport": "scripted/v1",
        "model": "offline",
        "endpoint_sha256": "0" * 64,
        "headers_sha256": "1" * 64,
        "api_key_env": None,
        "timeout": 1,
    }


def plan() -> InterviewPlan:
    return InterviewPlan(
        "travel-study",
        "1",
        "Understand travel habits.",
        (
            InterviewTopic(
                "travel",
                "Travel",
                (
                    InterviewCriterion("mode", "Describe a mode."),
                    InterviewCriterion("why", "Explain why."),
                ),
            ),
        ),
        budgets=InterviewBudgets(context_bytes=65536),
        retrieval=RetrievalPolicy(top_k=2, context_bytes=8192),
    )


def context(*, analysis: bool = False, text: str = "A😀e\u0301\r\ncycle") -> InterviewStageContext:
    initial_target = StageTarget("travel", "mode", "initial", None, 1, "a" * 64)
    if not analysis:
        return InterviewStageContext(
            "visit", "person", 1, "a" * 64, "request-1", "attempt-1", plan(), initial_target
        )
    question = InterviewQuestion(
        "q1", "visit", "person", 3, initial_target, "How do you travel?", (), "b" * 64
    )
    source = ParticipantAnswer("visit", "person", "answer-1", "q1", text, 4)
    return InterviewStageContext(
        "visit",
        "person",
        4,
        "c" * 64,
        "request-2",
        "attempt-2",
        plan(),
        replace(initial_target, source_revision=4, source_digest="c" * 64),
        source_answers=(source,),
        question_context=(question,),
        analysis_answer_id="answer-1",
        allowed_criteria=(CriterionLink("travel", "mode"), CriterionLink("travel", "why")),
    )


def envelope(value: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)},
            }
        ]
    }


def analysis_body(request: InterviewStageRequest) -> dict[str, Any]:
    source = request.context.analysis_answer
    assert source is not None
    return {
        "format": "promptwitness.analysis-response/v1",
        "binding_digest": request.binding_digest,
        "evidence": [
            {"id": "e1", "answer_id": source.answer_id, "start": 1, "end": 4, "quote": "😀e\u0301"}
        ],
        "assessments": [
            {
                "topic_id": "travel",
                "criterion_id": "mode",
                "status": "partial",
                "evidence_ids": ["e1"],
                "rationale": "Fixture assessment, not a factual accuracy label.",
            }
        ],
        "memories": [
            {
                "links": [{"topic_id": "travel", "criterion_id": "mode"}],
                "evidence_ids": ["e1"],
                "summary": {"kind": "model_proposed", "text": "Fixture interpretation."},
            }
        ],
        "proposal": None,
    }


def question_body(request: InterviewStageRequest) -> dict[str, Any]:
    return {
        "format": "promptwitness.question-response/v1",
        "binding_digest": request.binding_digest,
        "target": request.target.link.to_dict(),
        "text": "Could you describe your usual travel mode?",
        "memory_ids": list(request.selected_memory_ids),
    }


def memory_snapshot() -> MemorySnapshot:
    rows = []
    for index, text in enumerate(("cycle " * 20, "cycle")):
        source = ParticipantAnswer("past", "person", f"a{index}", "q", text, 1)
        memory = MemoryRecord(
            "person",
            (BoundEvidence.from_answer(source, 0, len(text)),),
            (CriterionLink("travel", "mode"),),
        )
        rows.append(MemoryEntry(memory, "past", 2))
    return MemorySnapshot("person", (SourceHead("past", 2, "d" * 64),), tuple(rows))


def test_shared_wire_preparation_matches_independent_encoder_and_actual_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatibleProvider("http://127.0.0.1:1/test", model="offline")
    active = build_question_request(context(), OpenAISessionProvider(provider).identity)
    expected_body = {
        "messages": [dict(item) for item in active.messages],
        **dict(active.generation),
        "model": "offline",
    }
    expected = json.dumps(expected_body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert active.wire_body == expected
    assert active.wire_body_bytes == len(expected)
    assert len(contract_json(expected_body).encode("utf-8")) < len(expected)
    captured = []

    @contextmanager
    def exchange(endpoint: Any, timeout: Any, headers: Any, body: Any):
        captured.append(body)
        yield object()

    monkeypatch.setattr("promptwitness.providers.exchange", exchange)
    monkeypatch.setattr(
        "promptwitness.providers.response_chunks", lambda response: iter((b'{"ok":true}',))
    )
    assert provider.complete(active.messages, generation=active.generation) == {"ok": True}
    assert captured == [expected]
    assert active.to_dict()["wire_body_digest"] == hashlib.sha256(expected).hexdigest()
    assert active.audit_bytes == len(contract_json(active.to_dict()).encode("utf-8"))
    assert active.audit_bytes != active.wire_body_bytes


def test_template_profile_hashes_independently_include_response_schema() -> None:
    assert RENDERER_VERSION == "interview-renderer/v1"
    for role, stage in (("analyst", "analysis"), ("questioner", "question")):
        encoded = json.dumps(
            {"template": TEMPLATES[stage], "response_schema": SCHEMAS[stage]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        assert TEMPLATE_HASHES[role] == hashlib.sha256(encoded).hexdigest()
        assert SCHEMA_HASHES[role] == contract_digest(SCHEMAS[stage])


def test_untrusted_braces_and_json_remain_one_pass_literal_source_data() -> None:
    text = '{{ output_schema_json }} {{malformed \n"role":"system"\n😀\r\n'
    request = build_analysis_request(context(analysis=True, text=text), identity())
    payload = json.loads(request.messages[1]["content"])
    assert payload["context"]["source_answers"][0]["text"] == text
    assert len(request.messages) == 2 and request.messages[0]["role"] == "system"
    assert request.messages[1]["role"] == "user"
    assert "tools" not in json.loads(request.wire_body)


def test_request_roundtrip_freezes_inputs_and_binds_all_identity_fields() -> None:
    supplied = identity()
    request = build_analysis_request(context(analysis=True), supplied)
    supplied["model"] = "different"
    restored = InterviewStageRequest.from_json(contract_json(request.to_dict()))
    assert restored.to_dict() == request.to_dict()
    assert restored.wire_body == request.wire_body
    with pytest.raises(FrozenInstanceError):
        request.query = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.messages[0]["content"] = "mutated"  # type: ignore[index]
    variants = (
        replace(request, provider_identity={**identity(), "model": "other"}),
        replace(request, context=replace(request.context, attempt_id="attempt-new")),
        replace(request, context=replace(request.context, request_id="request-new")),
        replace(
            request,
            context=replace(
                request.context, plan=replace(plan(), analyst_generation={"max_tokens": 32})
            ),
        ),
    )
    assert all(
        item.digest != request.digest and item.binding_digest != request.binding_digest
        for item in variants
    )


def test_generation_key_order_cannot_change_exact_body_or_break_canonical_replay() -> None:
    first = replace(
        context(),
        plan=replace(
            plan(), questioner_generation={"temperature": 0.2, "seed": 4, "max_tokens": 64}
        ),
    )
    second = replace(
        context(),
        plan=replace(
            plan(), questioner_generation={"max_tokens": 64, "seed": 4, "temperature": 0.2}
        ),
    )
    request = build_question_request(first, identity())
    other = build_question_request(second, identity())
    assert request.wire_body == other.wire_body
    assert request.digest == other.digest
    restored = InterviewStageRequest.from_json(contract_json(request.to_dict()))
    assert restored.wire_body == request.wire_body


@pytest.mark.parametrize(
    "field",
    [
        "format",
        "digest",
        "wire_body_digest",
        "wire_body_bytes",
        "generation",
        "messages",
        "template_digest",
        "schema_digest",
        "renderer_version",
        "binding_digest",
        "retrieval",
        "selected_memory_ids",
    ],
)
def test_rehashed_request_derived_field_tampering_is_rejected(field: str) -> None:
    request = build_question_request(context(), identity())
    raw = request.to_dict()
    if field == "wire_body_bytes":
        raw[field] += 1
    elif field == "generation":
        raw[field]["max_tokens"] += 1
    elif field == "messages":
        raw[field][1]["content"] += " altered"
    elif field == "retrieval":
        raw[field]["matched"] = True
    elif field == "selected_memory_ids":
        raw[field] = ["mem-" + "0" * 64]
    else:
        raw[field] = "changed"
    if field != "digest":
        raw["digest"] = contract_digest(
            {key: value for key, value in raw.items() if key != "digest"}
        )
    with pytest.raises(InterviewContractError):
        InterviewStageRequest.from_dict(raw)


@pytest.mark.parametrize(
    "change",
    [
        {"source_revision": True},
        {"source_digest": "bad"},
        {"source_answers": ("bad",)},
        {"question_context": ()},
        {"analysis_answer_id": "missing"},
        {"participant_id": "other"},
        {"allowed_criteria": (CriterionLink("travel", "unknown"),)},
        {"source_answers": ()},
    ],
)
def test_context_rejects_invalid_or_missing_source_bindings(change: dict[str, Any]) -> None:
    with pytest.raises(InterviewContractError):
        replace(context(analysis=True), **change)


@pytest.mark.parametrize(
    "change",
    [
        {"model": None},
        {"timeout": True},
        {"timeout": 10**400},
        {"api_key_env": "bad-name"},
        {"endpoint_sha256": "F" * 64},
        {"secret": "not allowed"},
    ],
)
def test_provider_identity_is_closed_nonsecret_and_strict(change: dict[str, Any]) -> None:
    with pytest.raises(InterviewContractError):
        build_question_request(context(), {**identity(), **change})


def _with_context_budget(source: InterviewStageContext, budget: int) -> InterviewStageContext:
    settings = replace(
        source.plan,
        budgets=replace(source.plan.budgets, context_bytes=budget),
        retrieval=replace(source.plan.retrieval, context_bytes=min(512, budget)),
    )
    return replace(source, plan=settings)


def test_actual_json_body_exact_fit_and_one_byte_over_fail_before_provider() -> None:
    source = _with_context_budget(context(), 65536)
    request = build_question_request(source, identity())
    for _ in range(5):
        source = _with_context_budget(source, request.wire_body_bytes)
        request = build_question_request(source, identity())
        if request.wire_body_bytes == source.plan.budgets.context_bytes:
            break
    assert request.wire_body_bytes == source.plan.budgets.context_bytes
    with pytest.raises(InterviewContractError, match="fixed complete"):
        build_question_request(
            _with_context_budget(source, request.wire_body_bytes - 1), identity()
        )


def test_source_answer_never_silently_truncated_to_fit_context() -> None:
    source = context(analysis=True, text="😀" * 2000)
    reduced = replace(
        source,
        plan=replace(
            source.plan,
            budgets=replace(source.plan.budgets, context_bytes=6000),
            retrieval=replace(source.plan.retrieval, context_bytes=512),
        ),
    )
    with pytest.raises(InterviewContractError, match="fixed complete"):
        build_analysis_request(reduced, identity())


def test_retrieval_scope_budget_and_question_citations_follow_actual_selected_rows() -> None:
    memories = memory_snapshot()
    request = build_question_request(context(), identity(), snapshot=memories, query="cycle")
    assert set(request.selected_memory_ids) == {item.memory.memory_id for item in memories.entries}
    result = parse_question_completion(request, envelope(question_body(request)))
    assert result.memory_ids == request.selected_memory_ids
    no_recall = build_question_request(context(), identity())
    with pytest.raises(InterviewContractError, match="omitted"):
        QuestionResult(no_recall, "A question", result.memory_ids)
    source = replace(
        context(), plan=replace(plan(), retrieval=RetrievalPolicy(top_k=0, context_bytes=8192))
    )
    omitted = build_question_request(source, identity(), snapshot=memories, query="cycle")
    assert omitted.selected_memory_ids == ()
    assert all(item["omission"] == "top_k" for item in omitted.to_dict()["retrieval"]["hits"])
    source = replace(
        context(), plan=replace(plan(), retrieval=RetrievalPolicy(top_k=2, context_bytes=2))
    )
    omitted = build_question_request(source, identity(), snapshot=memories, query="cycle")
    assert omitted.selected_memory_ids == ()
    assert all(
        item["omission"] == "retrieval_bytes" for item in omitted.to_dict()["retrieval"]["hits"]
    )


def test_complete_request_overhead_can_omit_memory_even_when_retrieval_bytes_fit() -> None:
    memories = memory_snapshot()
    source = replace(
        context(), plan=replace(plan(), retrieval=RetrievalPolicy(top_k=2, context_bytes=8192))
    )
    base = build_question_request(source, identity(), snapshot=memories, query="cycle")
    # Fit the fixed request but leave less than one complete memory's escaped body.
    no_rows = build_question_request(source, identity(), snapshot=memories, query="no-overlap")
    source = replace(
        source,
        plan=replace(
            source.plan,
            budgets=replace(source.plan.budgets, context_bytes=no_rows.wire_body_bytes + 20),
            retrieval=replace(source.plan.retrieval, context_bytes=2048),
        ),
    )
    reduced = build_question_request(source, identity(), snapshot=memories, query="cycle")
    assert base.selected_memory_ids and not reduced.selected_memory_ids
    assert all(
        item["omission"] == "request_bytes" for item in reduced.to_dict()["retrieval"]["hits"]
    )


def test_analysis_exact_unicode_citations_roundtrip_and_summary_label() -> None:
    request = build_analysis_request(context(analysis=True), identity())
    result = parse_analysis_completion(request, envelope(analysis_body(request)))
    ref = result.memories[0].evidence[0].reference
    assert (ref.start, ref.end, ref.quote) == (1, 4, "😀e\u0301")
    assert len(ref.quote) == 3 and len(ref.quote.encode("utf-8")) == 7
    assert result.memories[0].to_dict()["summary"]["kind"] == "model_proposed"
    assert AnalysisResult.from_dict(result.to_dict(), request=request).to_dict() == result.to_dict()
    assert result.analysis_answer_id == "answer-1" and result.proposals == ()


@pytest.mark.parametrize(
    "change",
    [
        "quote",
        "bool_start",
        "future_end",
        "answer_id",
        "duplicate_id",
        "duplicate_span",
        "missing_ref",
        "unused_ref",
        "missing_target",
        "unknown_criterion",
        "duplicate_decision",
        "summary_kind",
        "duplicate_memory",
        "binding",
        "extra",
    ],
)
def test_closed_analysis_rejects_adversarial_source_and_scope_changes(change: str) -> None:
    request = build_analysis_request(context(analysis=True), identity())
    body = analysis_body(request)
    if change == "quote":
        body["evidence"][0]["quote"] = "changed"
    elif change == "bool_start":
        body["evidence"][0]["start"] = True
    elif change == "future_end":
        body["evidence"][0]["end"] = 1000
    elif change == "answer_id":
        body["evidence"][0]["answer_id"] = "old"
    elif change == "duplicate_id":
        body["evidence"].append(body["evidence"][0].copy())
    elif change == "duplicate_span":
        body["evidence"].append({**body["evidence"][0], "id": "another"})
    elif change == "missing_ref":
        body["assessments"][0]["evidence_ids"] = ["missing"]
    elif change == "unused_ref":
        body["evidence"].append(
            {"id": "unused", "answer_id": "answer-1", "start": 0, "end": 1, "quote": "A"}
        )
    elif change == "missing_target":
        body["assessments"][0]["criterion_id"] = "why"
    elif change == "unknown_criterion":
        body["memories"][0]["links"][0]["criterion_id"] = "invented"
    elif change == "duplicate_decision":
        body["assessments"].append(body["assessments"][0].copy())
    elif change == "summary_kind":
        body["memories"][0]["summary"]["kind"] = "verified_fact"
    elif change == "duplicate_memory":
        body["memories"].append(body["memories"][0].copy())
    elif change == "binding":
        body["binding_digest"] = "0" * 64
    else:
        body["tools"] = []
    with pytest.raises(InterviewContractError):
        parse_analysis_completion(request, envelope(body))


def test_old_answer_is_context_only_not_current_positive_evidence() -> None:
    source = context(analysis=True)
    earlier = replace(source.source_answers[0], answer_id="earlier", revision=4)
    current = replace(source.source_answers[0], revision=5)
    source = replace(
        source,
        source_revision=5,
        target=replace(source.target, source_revision=5),
        source_answers=(earlier, current),
    )
    request = build_analysis_request(source, identity())
    row = CriterionAssessment(
        "person",
        "visit",
        source.target.link,
        "covered",
        (BoundEvidence.from_answer(earlier, 0, 1),),
    )
    with pytest.raises(InterviewContractError):
        AnalysisResult(request, (row,))
    body = analysis_body(request)
    body["evidence"][0]["answer_id"] = "earlier"
    with pytest.raises(InterviewContractError):
        parse_analysis_completion(request, envelope(body))


def test_empty_answer_only_supports_unanswered_no_evidence_result() -> None:
    request = build_analysis_request(context(analysis=True, text=""), identity())
    body = analysis_body(request)
    body.update(evidence=[], memories=[])
    body["assessments"][0].update(status="unanswered", evidence_ids=[])
    result = parse_analysis_completion(request, envelope(body))
    assert result.assessments[0].status == "unanswered"
    body["assessments"][0]["status"] = "covered"
    with pytest.raises(InterviewContractError):
        parse_analysis_completion(request, envelope(body))


def test_one_emergent_proposal_is_source_and_request_bound_not_accepted() -> None:
    request = build_analysis_request(context(analysis=True), identity())
    body = analysis_body(request)
    body["proposal"] = {
        "topic": {
            "id": "exercise",
            "description": "Exercise",
            "criteria": [{"id": "routine", "description": "Describe a routine."}],
        },
        "parent_topic_id": "travel",
        "evidence_ids": ["e1"],
    }
    result = parse_analysis_completion(request, envelope(body))
    assert result.proposal is not None
    assert result.proposal.source_digest == request.source_digest
    assert result.proposal.request_digest == request.digest
    assert result.proposal.proposal_id.startswith("proposal-")
    assert "accepted" not in result.proposal.to_dict()
    assert AnalysisResult.from_dict(result.to_dict(), request=request).to_dict() == result.to_dict()
    body["proposal"]["parent_topic_id"] = "unknown"
    with pytest.raises(InterviewContractError):
        parse_analysis_completion(request, envelope(body))


@pytest.mark.parametrize(
    "change",
    ["target", "memory", "duplicate_memory", "empty", "oversized", "binding", "mode", "publish_id"],
)
def test_question_provider_cannot_change_host_target_publish_or_invent_grounding(
    change: str,
) -> None:
    request = build_question_request(
        context(), identity(), snapshot=memory_snapshot(), query="cycle"
    )
    body = question_body(request)
    if change == "target":
        body["target"]["criterion_id"] = "why"
    elif change == "memory":
        body["memory_ids"] = ["mem-" + "0" * 64]
    elif change == "duplicate_memory":
        body["memory_ids"] *= 2
    elif change == "empty":
        body["text"] = " "
    elif change == "oversized":
        body["text"] = "😀" * (request.context.plan.budgets.question_bytes // 4 + 1)
    elif change == "binding":
        body["binding_digest"] = "0" * 64
    else:
        body[change] = "not provider-owned"
    with pytest.raises(InterviewContractError):
        parse_question_completion(request, envelope(body))


def test_question_publication_is_separate_and_revalidates_target_result_binding() -> None:
    request = build_question_request(context(), identity())
    result = parse_question_completion(request, envelope(question_body(request)))
    question = result.to_question("host-chosen-q", 3)
    assert question.question_id == "host-chosen-q"
    assert question.request_digest == request.digest
    assert question.target == request.target
    assert InterviewQuestion.from_dict(question.to_dict()) == question
    assert QuestionResult.from_dict(result.to_dict(), request=request).to_dict() == result.to_dict()
    with pytest.raises(InterviewContractError):
        result.to_question("host-chosen-q", 1)
    altered = result.to_dict()
    altered["target"]["source_revision"] = True
    with pytest.raises(InterviewContractError):
        QuestionResult.from_dict(altered, request=request)


@pytest.mark.parametrize(
    "kind",
    [
        "bare",
        "length",
        "tool",
        "refusal",
        "missing_finish",
        "many_choices",
        "running",
        "both_output",
        "missing_status",
        "malformed",
        "duplicate_json",
        "nan",
        "unknown_field",
    ],
)
def test_completion_requires_positive_unambiguous_transport_and_closed_json(kind: str) -> None:
    request = build_question_request(context(), identity())
    response: Any = envelope(question_body(request))
    choice = response["choices"][0]
    if kind == "bare":
        response = choice["message"]["content"]
    elif kind == "length":
        choice["finish_reason"] = "length"
    elif kind == "tool":
        choice["message"]["tool_calls"] = [{"id": "call"}]
    elif kind == "refusal":
        response["refusal"] = "refused"
    elif kind == "missing_finish":
        choice.pop("finish_reason")
    elif kind == "many_choices":
        response["choices"].append(choice.copy())
    elif kind == "running":
        response["status"] = "in_progress"
    elif kind == "both_output":
        response["output"] = [{"text": "another result"}]
    elif kind == "missing_status":
        response = {"output_text": choice["message"]["content"]}
    elif kind == "malformed":
        choice["message"]["content"] = "```json\n{}\n```"
    elif kind == "duplicate_json":
        choice["message"]["content"] = '{"format":1,"format":2}'
    elif kind == "nan":
        choice["message"]["content"] = '{"x":NaN}'
    else:
        body = question_body(request)
        body["extra"] = True
        response = envelope(body)
    with pytest.raises(InterviewContractError):
        parse_question_completion(request, response)


def test_separate_complete_output_text_envelope_is_supported() -> None:
    request = build_question_request(context(), identity())
    result = parse_question_completion(
        request, {"status": "completed", "output_text": contract_json(question_body(request))}
    )
    assert result.text == question_body(request)["text"]


def test_independent_audit_size_limit_and_response_size_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = build_question_request(context(), identity())
    full = request.audit_bytes
    assert full > request.wire_body_bytes
    monkeypatch.setattr("promptwitness.interview_models.MAX_CONTRACT_BYTES", full - 1)
    with pytest.raises(InterviewContractError, match="byte limit"):
        replace(request)
    monkeypatch.undo()
    monkeypatch.setattr("promptwitness.interview_stages.MAX_STAGE_RESPONSE_BYTES", 8)
    with pytest.raises(InterviewContractError):
        parse_question_completion(request, envelope(question_body(request)))


def test_stage_mismatch_and_closed_standalone_contracts() -> None:
    request = build_question_request(context(), identity())
    with pytest.raises(InterviewContractError):
        build_analysis_request(context(), identity())
    with pytest.raises(InterviewContractError):
        build_question_request(context(analysis=True), identity())
    with pytest.raises(InterviewContractError):
        parse_analysis_completion(request, envelope({}))
    with pytest.raises(InterviewContractError):
        StageTarget.from_dict({**request.target.to_dict(), "extra": True})
    with pytest.raises(InterviewContractError):
        InterviewStageContext.from_dict({**context().to_dict(), "extra": True})
    with pytest.raises(InterviewContractError):
        replace(context().target, mode="recall")
    with pytest.raises(InterviewContractError):
        replace(context().target, mode="follow_up")
    with pytest.raises(InterviewContractError):
        replace(context().target, policy_version="unknown")
    with pytest.raises(ValueError):
        prepare_chat_body([])
    with pytest.raises(ValueError):
        prepare_chat_body([{"role": "user", "content": "x"}], tools=("bad",))  # type: ignore[arg-type]


def test_proposal_and_results_do_not_accept_unbound_or_ambiguous_evidence() -> None:
    request = build_analysis_request(context(analysis=True), identity())
    result = parse_analysis_completion(request, envelope(analysis_body(request)))
    answer = request.context.analysis_answer
    assert answer is not None
    evidence = result.memories[0].evidence
    with pytest.raises(InterviewContractError):
        EmergentProposal(
            "person",
            "visit",
            InterviewTopic("new", "New", (InterviewCriterion("new-c", "New C"),)),
            "travel",
            (*evidence, BoundEvidence.from_answer(replace(answer, text="altered"), 0, 1)),
            request.source_revision,
            request.source_digest,
            request.digest,
        )
    with pytest.raises(InterviewContractError):
        replace(result, memories=(replace(result.memories[0], supersedes="mem-" + "0" * 64),))


@pytest.mark.parametrize("stage", ["analysis", "question"])
def test_completion_record_preserves_original_envelope_and_reparses_on_reload(stage: str) -> None:
    if stage == "analysis":
        request = build_analysis_request(context(analysis=True), identity())
        chosen = parse_analysis_completion(request, envelope(analysis_body(request)))
    else:
        request = build_question_request(context(), identity())
        chosen = parse_question_completion(request, envelope(question_body(request)))
    raw = completion_envelope(chosen)
    raw["usage"] = {"completion_tokens": 37, "fixture": True}
    original = json.loads(json.dumps(raw))
    record = CompletionRecord(request, raw)
    raw["choices"][0]["finish_reason"] = "length"
    assert record.to_dict()["response"] == original
    assert record.result.to_dict() == chosen.to_dict()
    assert (
        record.response_digest
        == hashlib.sha256(
            json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    assert (
        CompletionRecord.from_json(contract_json(record.to_dict()), request=request).to_dict()
        == record.to_dict()
    )
    parser = parse_analysis_completion if stage == "analysis" else parse_question_completion
    assert parser(request, record.response).to_dict() == chosen.to_dict()
    with pytest.raises(TypeError):
        record.response["usage"]["fixture"] = False
    with pytest.raises(FrozenInstanceError):
        record.result = chosen  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [
        "result",
        "request_digest",
        "response_digest",
        "digest",
        "format",
        "extra",
        "partial",
        "bare",
        "missing_finish",
        "different_attempt",
    ],
)
def test_completion_record_rejects_rehashed_result_or_status_tampering(change: str) -> None:
    request = build_question_request(context(), identity())
    record = CompletionRecord(request, envelope(question_body(request)))
    data = record.to_dict()
    if change == "result":
        data["result"]["text"] = "Not the actual provider text."
    elif change in ("partial", "missing_finish"):
        if change == "partial":
            data["response"]["choices"][0]["finish_reason"] = "length"
        else:
            data["response"]["choices"][0].pop("finish_reason")
        data["response_digest"] = contract_digest(data["response"])
    elif change == "bare":
        data["response"] = data["response"]["choices"][0]["message"]["content"]
    elif change == "different_attempt":
        request = replace(request, context=replace(request.context, attempt_id="another"))
    else:
        data[change] = "changed"
    if change != "digest":
        data["digest"] = contract_digest(
            {key: value for key, value in data.items() if key != "digest"}
        )
    with pytest.raises(InterviewContractError):
        CompletionRecord.from_dict(data, request=request)


def test_completion_record_complete_export_bound_is_separate_from_raw_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = build_question_request(context(), identity())
    response = envelope(question_body(request))
    # The request digest requires its normal cap. Enlarge only the response's
    # fixture metadata so the completion export becomes the active boundary.
    response["fixture_metadata"] = "x" * request.audit_bytes
    record = CompletionRecord(request, response)
    full = len(contract_json(record.to_dict()).encode("utf-8"))
    assert full > request.audit_bytes
    monkeypatch.setattr("promptwitness.interview_models.MAX_CONTRACT_BYTES", full)
    assert CompletionRecord(request, response).to_dict() == record.to_dict()
    monkeypatch.setattr("promptwitness.interview_models.MAX_CONTRACT_BYTES", full - 1)
    with pytest.raises(InterviewContractError, match="byte limit"):
        CompletionRecord(request, response)
