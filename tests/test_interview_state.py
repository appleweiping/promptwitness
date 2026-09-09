"""Hand-computed pure interview trajectories and adversarial replay boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
from interview_fixtures import completion_envelope

from promptwitness import interview_memory, interview_models
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
    contract_json,
)
from promptwitness.interview_stages import (
    RENDERER_VERSION,
    TEMPLATE_HASHES,
    AnalysisResult,
    EmergentProposal,
    InterviewStageContext,
    InterviewStageRequest,
    QuestionResult,
    build_analysis_request,
    build_question_request,
)
from promptwitness.interview_state import (
    SCHEDULER_VERSION,
    InterviewCommand,
    InterviewEvent,
    InterviewSkip,
    InterviewState,
    InterviewTransition,
    NextAction,
    ProposalDecision,
    StageReservation,
    apply_event,
    current_memory_snapshot,
    interview_report,
    make_event,
    replay_interview,
    schedule,
)


def identity() -> dict[str, Any]:
    return {
        "provider": "local-scripted-contract-fixture",
        "transport": "fixture/v1",
        "model": "authored-response",
        "endpoint_sha256": "a" * 64,
        "headers_sha256": "b" * 64,
        "api_key_env": None,
        "timeout": 1,
    }


def plan(**budgets: int) -> InterviewPlan:
    return InterviewPlan(
        "learning",
        "1",
        "Ask about work and learning.",
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
                "learning", "Learning", (InterviewCriterion("skill", "Describe a skill."),)
            ),
        ),
        InterviewBudgets(**budgets),
    )


def one_plan(**budgets: int) -> InterviewPlan:
    return replace(
        plan(**budgets),
        topics=(
            InterviewTopic(
                "work", "Current work", (InterviewCriterion("role", "Describe your role."),)
            ),
        ),
    )


def creation(
    original: InterviewPlan | None = None, snapshot: MemorySnapshot | None = None
) -> dict[str, Any]:
    return {
        "participant_id": "person",
        "plan": (original or plan()).to_dict(),
        "provider_identities": {"analyst": identity(), "questioner": identity()},
        "template_hashes": dict(TEMPLATE_HASHES),
        "renderer_version": RENDERER_VERSION,
        "scheduler_version": SCHEDULER_VERSION,
        "memory_snapshot": (snapshot or MemorySnapshot("person", (), ())).to_dict(),
    }


class Trace:
    def __init__(
        self, original: InterviewPlan | None = None, snapshot: MemorySnapshot | None = None
    ) -> None:
        self.events: list[InterviewEvent] = []
        self.state: InterviewState
        self.add("created", creation(original, snapshot))

    def command(self, kind: str, payload: Any, /, **changes: Any) -> InterviewCommand:
        values = {
            "interview_id": "session",
            "command_id": f"command-{len(self.events) + 1}",
            "expected_revision": self.state.revision if self.events else 0,
            "expected_digest": self.state.digest if self.events else None,
            "kind": kind,
            "payload": payload,
        }
        return InterviewCommand(**{**values, **changes})

    def add(self, kind: str, payload: Any) -> InterviewEvent:
        state = self.state if self.events else None
        event = make_event(state, self.command(kind, payload))
        transition = apply_event(state, event)
        self.state = transition.state
        self.events.append(event)
        return event

    def request(
        self, *, retry: bool = False, all_criteria: bool = False, **changes: Any
    ) -> InterviewStageRequest:
        state = self.state
        action = schedule(replace(state, pending=None)) if retry else schedule(state)
        assert action.target is not None
        context = InterviewStageContext(
            state.interview_id,
            state.participant_id,
            state.revision,
            state.digest,
            f"request-{len(self.events)}",
            f"attempt-{len(self.events)}",
            state.plan,
            action.target,
            state.accepted_emergent,
            state.answers,
            state.questions,
            action.answer_id,
            tuple(
                CriterionLink(topic.topic_id, criterion.criterion_id)
                for topic in (*state.plan.topics, *state.accepted_emergent)
                for criterion in topic.criteria
            )
            if all_criteria
            else (),
        )
        context = replace(context, **changes)
        if action.kind == "analysis":
            return build_analysis_request(context, identity())
        return build_question_request(
            context, identity(), snapshot=current_memory_snapshot(state), query="work fixture"
        )

    def reserve(self, *, all_criteria: bool = False) -> InterviewStageRequest:
        request = self.request(all_criteria=all_criteria)
        self.add(request.stage + "_reserved", {"request": request.to_dict()})
        return request

    @staticmethod
    def completion(request: InterviewStageRequest, **changes: Any) -> dict[str, Any]:
        if "result" in changes and "completion" not in changes:
            raw = changes["result"]
            if raw:
                result = (
                    AnalysisResult.from_dict(raw, request=request)
                    if request.stage == "analysis"
                    else QuestionResult.from_dict(raw, request=request)
                )
                changes["completion"] = completion_envelope(result)
            else:
                # Deliberate invalid-result fixture for wrong-stage rejection.
                changes["completion"] = {}
        return {
            "request_id": request.request_id,
            "attempt_id": request.attempt_id,
            "request_digest": request.digest,
            **changes,
        }

    def question(self, text: str | None = None) -> None:
        number = len(self.state.questions) + 1
        request = self.reserve()
        result = QuestionResult(
            request, text or f"Question {number}: describe work?", request.selected_memory_ids
        )
        self.add(
            "question_committed",
            self.completion(request, question_id=f"q{number}", result=result.to_dict()),
        )

    def answer(self, text: str = "work fixture 😀e\u0301\r\n") -> None:
        self.add(
            "answer_committed",
            {
                "question_id": self.state.questions[-1].question_id,
                "answer_id": f"a{len(self.state.answers) + 1}",
                "text": text,
            },
        )

    def analysis(
        self,
        status: str = "covered",
        *,
        memory: bool = False,
        topic: InterviewTopic | None = None,
        all_criteria: bool = False,
        extra: dict[CriterionLink, str] | None = None,
    ) -> None:
        request = self.reserve(all_criteria=all_criteria)
        result = analysis_result(request, status, memory=memory, topic=topic, extra=extra)
        self.add("analysis_committed", self.completion(request, result=result.to_dict()))


def analysis_result(
    request: InterviewStageRequest,
    status: str = "covered",
    *,
    memory: bool = False,
    topic: InterviewTopic | None = None,
    extra: dict[CriterionLink, str] | None = None,
) -> AnalysisResult:
    answer = request.context.analysis_answer
    assert answer is not None
    evidence = (BoundEvidence.from_answer(answer, 0, len(answer.text)),) if answer.text else ()
    statuses = {request.target.link: status, **(extra or {})}
    assessments = tuple(
        CriterionAssessment(
            "person",
            "session",
            link,
            value,
            evidence if value != "unanswered" else (),
            "Model-proposed, not truth.",
        )
        for link, value in statuses.items()
    )
    memories = (
        (MemoryRecord("person", evidence, (request.target.link,), "work fixture summary"),)
        if memory
        else ()
    )
    proposal = (
        EmergentProposal(
            "person",
            "session",
            topic,
            request.target.topic_id,
            evidence,
            request.source_revision,
            request.source_digest,
            request.digest,
        )
        if topic
        else None
    )
    return AnalysisResult(request, assessments, memories, proposal)


def extra_topic() -> InterviewTopic:
    return InterviewTopic(
        "future", "Future interests", (InterviewCriterion("interest", "Describe an interest."),)
    )


def decide(trace: Trace, decision: str) -> None:
    proposal = next(item for item in trace.state.proposals if item.status == "pending")
    trace.add(
        "proposal_decided", {"proposal_id": proposal.proposal.proposal_id, "decision": decision}
    )


def test_hand_calculated_trace_partial_followup_skip_emergent_and_gaps() -> None:
    trace = Trace(plan(participant_turns=5, followups_per_topic=2))
    assert schedule(trace.state).target.criterion_id == "role"  # type: ignore[union-attr]
    trace.question("  What is your rôle?\r\n")
    trace.answer()
    assert trace.state.answers[-1].revision == 4
    trace.analysis("partial", memory=True)
    assert trace.state.revision == 6
    assert trace.state.memories[0].revision == 6
    assert schedule(trace.state).target.mode == "follow_up"  # type: ignore[union-attr]
    trace.question()
    assert trace.state.questions[-1].memory_ids == (trace.state.memories[0].memory.memory_id,)
    trace.answer("More work fixture details.")
    trace.analysis("partial", topic=extra_topic())
    assert trace.state.revision == 11
    assert schedule(trace.state).kind == "await_review"
    assert interview_report(trace.state)["assessed_coverage"]["emergent"]["total"] == 0
    decide(trace, "accepted")
    trace.question()
    trace.add("question_skipped", {"question_id": "q3", "scope": "criterion"})
    assert trace.state.revision == 15
    assert schedule(trace.state).target.criterion_id == "skill"  # type: ignore[union-attr]
    trace.question()
    trace.answer()
    trace.analysis()
    assert trace.state.revision == 20
    assert schedule(trace.state).target.topic_id == "future"  # type: ignore[union-attr]
    trace.question()
    trace.answer()
    assert trace.state.participant_turns == 5
    assert schedule(trace.state).kind == "analysis"
    trace.analysis()
    assert trace.state.revision == 25
    assert schedule(trace.state).reason == "unresolved"
    trace.add("finished", {"reason": "unresolved"})
    report = interview_report(trace.state)
    assert trace.state.revision == 26
    assert (report["questions"], report["answers"], report["participant_turns"]) == (5, 4, 5)
    assert (
        report["stage_reservations"],
        report["stage_completions"],
        report["provider_call_count"],
    ) == (9, 9, None)
    assert report["assessed_coverage"]["required"] == {
        "total": 3,
        "covered": 1,
        "partial": 1,
        "unanswered": 1,
        "assessed_coverage": 1 / 3,
    }
    assert report["assessed_coverage"]["emergent"] == {
        "total": 1,
        "covered": 1,
        "partial": 0,
        "unanswered": 0,
        "assessed_coverage": 1.0,
    }
    assert [
        (row["criterion_id"], row["skipped"], row["deferred"]) for row in report["criteria"]
    ] == [
        ("role", True, True),
        ("challenge", False, True),
        ("skill", False, False),
        ("interest", False, False),
    ]
    assert report["reason"] == "unresolved"
    assert replay_interview(trace.events, trace.state.head).to_dict() == trace.state.to_dict()


@pytest.mark.parametrize("scope,next_criterion", [("criterion", "challenge"), ("topic", "skill")])
def test_explicit_skip_scope_is_one_turn_and_preserves_denominator(
    scope: str, next_criterion: str
) -> None:
    trace = Trace()
    trace.question()
    trace.add("question_skipped", {"question_id": "q1", "scope": scope})
    assert trace.state.participant_turns == 1 and not trace.state.answers
    assert schedule(trace.state).target.criterion_id == next_criterion  # type: ignore[union-attr]
    assert interview_report(trace.state)["assessed_coverage"]["required"]["total"] == 3


def test_skip_string_is_real_answer_with_exact_unicode_and_digest() -> None:
    trace = Trace(one_plan())
    trace.question()
    trace.answer("skip")
    assert trace.state.answers[0].text == "skip" and not trace.state.skips
    assert schedule(trace.state).kind == "analysis"
    trace.analysis(memory=True)
    bound = trace.state.memories[0].memory.evidence[0]
    assert bound.reference.quote == "skip" and bound.answer.revision == 4


@pytest.mark.parametrize(
    "provider_calls,followups,reason,analyzed",
    [
        (1, 1, "budget_exhausted", False),
        (2, 1, "budget_exhausted", True),
        (2, 0, "unresolved", True),
    ],
)
def test_final_answer_analysis_precedes_turn_budget(
    provider_calls: int, followups: int, reason: str, analyzed: bool
) -> None:
    trace = Trace(
        one_plan(participant_turns=1, provider_calls=provider_calls, followups_per_topic=followups)
    )
    trace.question()
    trace.answer()
    if analyzed:
        assert schedule(trace.state).kind == "analysis"
        trace.analysis("partial")
    assert schedule(trace.state).reason == reason
    assert interview_report(trace.state)["unanalyzed_answer_ids"] == ([] if analyzed else ["a1"])


@pytest.mark.parametrize("decision", ["accepted", "rejected"])
def test_required_completion_waits_for_review_and_eligible_emergent(decision: str) -> None:
    trace = Trace(one_plan())
    trace.question()
    trace.answer()
    trace.analysis(topic=extra_topic())
    assert schedule(trace.state).kind == "await_review"
    with pytest.raises(InterviewContractError):
        trace.add("finished", {"reason": "agenda_completed"})
    decide(trace, decision)
    if decision == "accepted":
        assert schedule(trace.state).kind == "question"
        assert schedule(trace.state).target.mode == "emergent"  # type: ignore[union-attr]
        trace.question()
        trace.answer()
        trace.analysis()
    assert schedule(trace.state).reason == "agenda_completed"
    trace.add("finished", {"reason": "agenda_completed"})
    assert interview_report(trace.state)["assessed_coverage"]["emergent"]["total"] == int(
        decision == "accepted"
    )


@pytest.mark.parametrize("skip,reason", [(False, "budget_exhausted"), (True, "agenda_completed")])
def test_exhausted_emergent_is_not_complete_unless_explicitly_skipped(
    skip: bool, reason: str
) -> None:
    trace = Trace(one_plan(followups_per_topic=0))
    trace.question()
    trace.answer()
    trace.analysis(topic=extra_topic())
    decide(trace, "accepted")
    trace.question()
    if skip:
        trace.add("question_skipped", {"question_id": "q2", "scope": "topic"})
    else:
        trace.answer()
        trace.analysis("partial")
    assert schedule(trace.state).reason == reason


def test_stronger_assessment_survives_later_weaker_current_answer_proposal() -> None:
    trace = Trace()
    trace.question()
    trace.answer()
    trace.analysis("covered")
    first = trace.state.assessments[0]
    trace.question()
    trace.answer("Different answer for challenge.")
    trace.analysis(
        "partial", all_criteria=True, extra={CriterionLink("work", "role"): "unanswered"}
    )
    role = next(item for item in trace.state.assessments if item.link.criterion_id == "role")
    assert role.to_dict() == first.to_dict()
    proposed = trace.events[-1].payload["result"]["assessments"]
    assert (
        next(item for item in proposed if item["link"]["criterion_id"] == "role")["status"]
        == "unanswered"
    )


@pytest.mark.parametrize("failed", [False, True])
def test_explicit_retry_replaces_reservation_and_fences_late_old_result(failed: bool) -> None:
    trace = Trace(one_plan(provider_calls=3))
    old = trace.reserve()
    assert old.source_revision == 1 and trace.state.pending.revision == 2  # type: ignore[union-attr]
    if failed:
        trace.add("stage_failed", trace.completion(old, error_code="interrupted"))
        assert schedule(trace.state).kind == "await_retry"
        with pytest.raises(InterviewContractError):
            trace.add(
                "question_committed",
                trace.completion(
                    old, result=QuestionResult(old, "Late question?").to_dict(), question_id="late"
                ),
            )
    current = trace.request(retry=True)
    assert current.source_revision == trace.state.revision
    trace.add("retry_authorized", {"request": current.to_dict()})
    assert trace.state.stage_reservations == 2 and schedule(trace.state).kind == "await_completion"
    saved = trace.state.to_dict()
    with pytest.raises(InterviewContractError, match=r"stale|different"):
        trace.add(
            "question_committed",
            trace.completion(
                old, result=QuestionResult(old, "Late question?").to_dict(), question_id="late"
            ),
        )
    assert trace.state.to_dict() == saved
    trace.add(
        "question_committed",
        trace.completion(
            current, result=QuestionResult(current, "Valid question?").to_dict(), question_id="q1"
        ),
    )
    assert len(trace.state.questions) == 1 and trace.state.stage_completions == 1
    trace.answer()
    trace.analysis()
    assert trace.state.stage_reservations == 3
    assert replay_interview(trace.events).to_dict() == trace.state.to_dict()


def test_retry_neither_refunds_budget_nor_reuses_request_or_attempt_id() -> None:
    trace = Trace(one_plan(provider_calls=2))
    old = trace.reserve()
    for changes in ({"attempt_id": old.attempt_id}, {"request_id": old.request_id}):
        request = trace.request(retry=True, **changes)
        with pytest.raises(InterviewContractError, match="reused"):
            trace.add("retry_authorized", {"request": request.to_dict()})
    trace.add("retry_authorized", {"request": trace.request(retry=True).to_dict()})
    with pytest.raises(InterviewContractError):
        trace.add("retry_authorized", {"request": old.to_dict()})
    assert trace.state.stage_reservations == 2 and not trace.state.questions


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("answer_committed", {"question_id": "q1", "answer_id": "a1", "text": "x"}),
        ("question_skipped", {"question_id": "q1", "scope": "topic"}),
        ("proposal_decided", {"proposal_id": "unknown", "decision": "accepted"}),
    ],
)
def test_active_reservation_disallows_interleaved_domain_input(kind: str, payload: Any) -> None:
    trace = Trace()
    trace.reserve()
    before = trace.state.to_dict()
    with pytest.raises(InterviewContractError):
        trace.add(kind, payload)
    assert trace.state.to_dict() == before


@pytest.mark.parametrize(
    "field,bad", [("request_id", "other"), ("attempt_id", "other"), ("request_digest", "0" * 64)]
)
def test_completion_requires_exact_three_part_request_identity(field: str, bad: Any) -> None:
    trace = Trace()
    request = trace.reserve()
    payload = trace.completion(
        request, result=QuestionResult(request, "Q?").to_dict(), question_id="q1"
    )
    payload[field] = bad
    with pytest.raises(InterviewContractError):
        trace.add("question_committed", payload)


@pytest.mark.parametrize(
    "change",
    [
        {"expected_revision": 0, "expected_digest": None},
        {"expected_digest": "0" * 64},
        {"interview_id": "other"},
    ],
)
def test_exact_state_head_cas_rejects_stale_or_wrong_interview(change: dict[str, Any]) -> None:
    trace = Trace()
    command = trace.command("finished", {"reason": "participant_stopped"}, **change)
    with pytest.raises(InterviewContractError):
        make_event(trace.state, command)


def test_created_cannot_be_repeated_and_commands_cannot_reappear() -> None:
    trace = Trace()
    with pytest.raises(InterviewContractError):
        trace.add("created", creation())
    reused = trace.command("finished", {"reason": "participant_stopped"}, command_id="command-1")
    with pytest.raises(InterviewContractError, match="command ID"):
        make_event(trace.state, reused)


@pytest.mark.parametrize("kind", ["finished", "question_reserved", "created"])
def test_finished_state_fences_all_late_events(kind: str) -> None:
    trace = Trace()
    request = trace.request()
    trace.add("finished", {"reason": "participant_stopped"})
    payload = (
        {"reason": "participant_stopped"}
        if kind == "finished"
        else ({"request": request.to_dict()} if kind == "question_reserved" else creation())
    )
    with pytest.raises(InterviewContractError, match="finished"):
        trace.add(kind, payload)


def test_failed_finish_requires_recorded_failure_and_retains_uncertain_request() -> None:
    trace = Trace()
    with pytest.raises(InterviewContractError):
        trace.add("finished", {"reason": "failed"})
    request = trace.reserve()
    with pytest.raises(InterviewContractError):
        trace.add("stage_failed", trace.completion(request, error_code="secret-api-key"))
    trace.add("stage_failed", trace.completion(request, error_code="provider_error"))
    trace.add("finished", {"reason": "failed"})
    report = interview_report(trace.state)
    assert report["reason"] == "failed" and report["pending"]["status"] == "failed"
    assert report["failures"][0]["error_code"] == "provider_error"


@pytest.mark.parametrize("text", [" STRASSE \n what? ", "straße\twhat?"])
def test_question_dedup_uses_casefold_and_collapsed_whitespace_not_rewriting(text: str) -> None:
    trace = Trace()
    trace.question("Straße  what?")
    trace.answer()
    trace.analysis("partial")
    request = trace.reserve()
    before = trace.state.to_dict()
    with pytest.raises(InterviewContractError, match="already published"):
        trace.add(
            "question_committed",
            trace.completion(
                request, question_id="q2", result=QuestionResult(request, text).to_dict()
            ),
        )
    assert trace.state.questions[0].text == "Straße  what?" and trace.state.to_dict() == before


def test_all_analysis_records_validate_before_any_mutation() -> None:
    trace = Trace()
    trace.question()
    trace.answer()
    request = trace.reserve()
    typed_result = analysis_result(request, "covered", memory=True, topic=extra_topic())
    result = typed_result.to_dict()
    result["memories"][0]["evidence"][0]["quote"] = "x" * len(trace.state.answers[-1].text)
    before = trace.state.to_dict()
    with pytest.raises(InterviewContractError):
        trace.add(
            "analysis_committed",
            trace.completion(request, result=result, completion=completion_envelope(typed_result)),
        )
    assert trace.state.to_dict() == before
    assert not trace.state.memories and not trace.state.assessments and not trace.state.proposals


@pytest.mark.parametrize("field", ["text", "revision", "participant_id"])
def test_request_cannot_substitute_self_consistent_answer_source(field: str) -> None:
    trace = Trace()
    trace.question()
    trace.answer()
    request = trace.request()
    answer = trace.state.answers[-1]
    bad = replace(
        answer, **{field: {"text": "fabricated", "revision": 3, "participant_id": "other"}[field]}
    )
    with pytest.raises(InterviewContractError):
        changed = replace(request.context, source_answers=(bad,))
        forged = build_analysis_request(changed, identity())
        trace.add("analysis_reserved", {"request": forged.to_dict()})


def test_request_rejects_wrong_agenda_provider_target_and_question() -> None:
    trace = Trace()
    request = trace.request()
    altered = [
        replace(request, provider_identity={**identity(), "model": "different"}),
        replace(
            request,
            context=replace(request.context, plan=replace(trace.state.plan, purpose="different")),
        ),
        replace(
            request,
            context=replace(
                request.context,
                target=replace(request.target, criterion_id="challenge"),
                allowed_criteria=(),
            ),
        ),
    ]
    for changed in altered:
        with pytest.raises(InterviewContractError):
            trace.add("question_reserved", {"request": changed.to_dict()})
    trace.question()
    trace.answer()
    request = trace.request()
    fake = replace(trace.state.questions[0], text="fabricated interviewer question")
    with pytest.raises(InterviewContractError, match="published question"):
        trace.add(
            "analysis_reserved",
            {
                "request": replace(
                    request, context=replace(request.context, question_context=(fake,))
                ).to_dict()
            },
        )


def prior_snapshot() -> MemorySnapshot:
    answer = ParticipantAnswer("prior", "person", "prior-a", "prior-q", "work fixture", 4)
    memory = MemoryRecord(
        "person",
        (BoundEvidence.from_answer(answer, 0, len(answer.text)),),
        (CriterionLink("work", "role"),),
        "labelled summary",
    )
    return MemorySnapshot(
        "person", (SourceHead("prior", 6, "c" * 64),), (MemoryEntry(memory, "prior", 6),)
    )


def test_recall_cannot_inject_foreign_scope_or_change_pinned_source_head() -> None:
    trace = Trace(snapshot=prior_snapshot())
    assert interview_report(trace.state)["assessed_coverage"]["required"]["covered"] == 0
    request = trace.request()
    assert request.selected_memory_ids
    changed_head = replace(prior_snapshot(), heads=(SourceHead("prior", 7, "d" * 64),))
    with pytest.raises(InterviewContractError, match="recall"):
        trace.add(
            "question_reserved", {"request": replace(request, snapshot=changed_head).to_dict()}
        )
    empty = Trace()
    with pytest.raises(InterviewContractError, match="recall"):
        empty.add(
            "question_reserved",
            {"request": replace(empty.request(), snapshot=prior_snapshot()).to_dict()},
        )
    no_recall = replace(request, snapshot=None, query="")
    trace.add("question_reserved", {"request": no_recall.to_dict()})
    assert not trace.state.pending.request.selected_memory_ids  # type: ignore[union-attr]


def test_normalized_duplicate_proposal_rejects_whole_analysis() -> None:
    trace = Trace()
    trace.question()
    trace.answer()
    request = trace.reserve()
    topic = InterviewTopic("new", " CURRENT\nWORK ", (InterviewCriterion("new-c", "new"),))
    result = analysis_result(request, topic=topic, memory=True)
    with pytest.raises(InterviewContractError, match="duplicates"):
        trace.add("analysis_committed", trace.completion(request, result=result.to_dict()))
    assert not trace.state.proposals and not trace.state.memories


def test_closed_serialization_hashes_and_immutability() -> None:
    data = creation()
    command = InterviewCommand("session", "start", 0, None, "created", data)
    data["provider_identities"]["analyst"]["model"] = "mutated"
    assert command.payload["provider_identities"]["analyst"]["model"] == "authored-response"
    with pytest.raises(TypeError):
        command.payload["new"] = True  # type: ignore[index]
    event = make_event(None, command)
    state = apply_event(None, event).state
    with pytest.raises(FrozenInstanceError):
        state.revision = 2  # type: ignore[misc]
    raw = event.to_dict()
    digest = raw.pop("digest")
    encoded = json.dumps(
        raw, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == digest == state.digest
    assert InterviewCommand.from_dict(command.to_dict()).digest == command.digest
    assert InterviewEvent.from_dict(event.to_dict()).digest == event.digest
    view = state.to_dict()
    view["plan"]["topics"].clear()
    assert state.plan.topics


@pytest.mark.parametrize(
    "change",
    [
        {"expected_revision": True},
        {"expected_digest": "x"},
        {"kind": "arbitrary"},
        {"payload": {"reason": "participant_stopped", "extra": 1}},
        {"payload": {"reason": float("nan")}},
        {"payload": {"reason": object()}},
    ],
)
def test_commands_reject_nonportable_or_unknown_fields(change: dict[str, Any]) -> None:
    trace = Trace()
    with pytest.raises(InterviewContractError):
        trace.command("finished", {"reason": "participant_stopped"}, **change)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scheduler_version", "future"),
        ("renderer_version", "future"),
        ("template_hashes", {"analyst": "0" * 64, "questioner": "0" * 64}),
        ("provider_identities", {"analyst": {"secret": "raw"}, "questioner": identity()}),
        ("participant_id", "another-person"),
    ],
)
def test_creation_pins_real_versions_identity_and_memory_participant(
    field: str, value: Any
) -> None:
    data = creation()
    data[field] = value
    command = InterviewCommand("session", "start", 0, None, "created", data)
    with pytest.raises(InterviewContractError):
        make_event(None, command)


def test_complete_envelope_byte_limit_includes_overhead(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"question_id": "q1", "answer_id": "a1", "text": "x" * 900}
    assert len(contract_json(payload).encode()) < 1000
    monkeypatch.setattr(interview_models, "MAX_CONTRACT_BYTES", 1000)
    with pytest.raises(InterviewContractError, match="byte limit"):
        InterviewCommand("session", "answer", 1, "a" * 64, "answer_committed", payload)


@pytest.mark.parametrize(
    "mutation",
    ["content", "digest", "command_digest", "reordered", "truncated", "duplicate", "extra"],
)
def test_replay_detects_corruption_order_and_truncation_with_trusted_head(mutation: str) -> None:
    trace = Trace()
    trace.question()
    trace.answer()
    trace.analysis()
    events = [event.to_dict() for event in trace.events]
    if mutation == "content":
        events[0]["payload"]["plan"]["purpose"] = "changed"
    elif mutation in ("digest", "command_digest"):
        events[1][mutation] = "0" * 64
    elif mutation == "reordered":
        events[1], events[2] = events[2], events[1]
    elif mutation == "truncated":
        events.pop()
    elif mutation == "duplicate":
        events.append(events[-1])
    else:
        events[0]["extra"] = True
    with pytest.raises(InterviewContractError):
        replay_interview(events, expected_head=trace.state.head.to_dict())
    assert replay_interview(trace.events[:-1]).revision == trace.state.revision - 1


def test_empty_or_noncreation_history_invalid() -> None:
    with pytest.raises(InterviewContractError):
        replay_interview([])
    command = InterviewCommand(
        "session", "stop", 0, None, "finished", {"reason": "participant_stopped"}
    )
    with pytest.raises(InterviewContractError):
        make_event(None, command)


def test_reducer_has_no_io_clock_random_or_provider_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure reducer attempted an external effect")

    for name in (
        "builtins.open",
        "socket.socket",
        "sqlite3.connect",
        "time.time",
        "time.monotonic",
        "random.random",
        "uuid.uuid4",
    ):
        monkeypatch.setattr(name, forbidden)
    trace = Trace(one_plan())
    trace.question()
    trace.answer()
    trace.analysis(memory=True)
    trace.add("finished", {"reason": "agenda_completed"})
    assert replay_interview(trace.events, trace.state.head).to_dict() == trace.state.to_dict()


@pytest.mark.parametrize(
    "factory,args",
    [
        (InterviewSkip, ("q", "t", "c", "invalid", 1)),
        (NextAction, ("unknown",)),
        (ProposalDecision, ("not-a-proposal",)),
        (StageReservation, ("not-a-request", 1, "a" * 64)),
    ],
)
def test_invalid_public_helper_values_are_rejected(factory: Any, args: tuple[Any, ...]) -> None:
    with pytest.raises(InterviewContractError):
        factory(*args)


def test_analysis_retry_fences_old_memory_and_all_commits_use_replacement_head() -> None:
    trace = Trace(one_plan())
    trace.question()
    trace.answer()
    old = trace.reserve()
    old_result = analysis_result(old, memory=True)
    trace.add("stage_failed", trace.completion(old, error_code="response_incomplete"))
    current = trace.request(retry=True)
    assert current.context.analysis_answer_id == old.context.analysis_answer_id == "a1"
    assert current.target.link == old.target.link
    assert current.source_digest != old.source_digest
    trace.add("retry_authorized", {"request": current.to_dict()})
    with pytest.raises(InterviewContractError):
        trace.add("analysis_committed", trace.completion(old, result=old_result.to_dict()))
    assert not trace.state.memories and not trace.state.assessments
    command = trace.command(
        "analysis_committed",
        trace.completion(current, result=analysis_result(current, memory=True).to_dict()),
    )
    event = make_event(trace.state, command)
    transition = apply_event(trace.state, event)
    assert transition.memory_additions == transition.state.memories
    assert transition.memory_additions[0].revision == event.sequence
    assert transition.state.analyzed_answer_ids == ("a1",)
    assert transition.state.stage_reservations == 3 and transition.state.stage_completions == 2


def test_proposal_review_rejects_unknown_decision_and_repeated_decision() -> None:
    trace = Trace(one_plan())
    trace.question()
    trace.answer()
    trace.analysis(topic=extra_topic())
    proposal = trace.state.proposals[0]
    for payload in (
        {"proposal_id": proposal.proposal.proposal_id, "decision": "auto"},
        {"proposal_id": "unknown", "decision": "accepted"},
    ):
        with pytest.raises(InterviewContractError):
            trace.add("proposal_decided", payload)
    decide(trace, "rejected")
    with pytest.raises(InterviewContractError):
        trace.add(
            "proposal_decided",
            {"proposal_id": proposal.proposal.proposal_id, "decision": "accepted"},
        )
    assert not trace.state.accepted_emergent


def test_reserved_event_kind_and_completion_kind_cannot_switch_roles() -> None:
    trace = Trace()
    request = trace.request()
    with pytest.raises(InterviewContractError):
        trace.add("analysis_reserved", {"request": request.to_dict()})
    request = trace.reserve()
    with pytest.raises(InterviewContractError):
        trace.add("analysis_committed", trace.completion(request, result={}))
    assert trace.state.pending.request.stage == "question"  # type: ignore[union-attr]


def test_answer_and_question_ids_cannot_be_reused() -> None:
    trace = Trace()
    trace.question()
    trace.answer()
    trace.analysis("partial")
    request = trace.reserve()
    with pytest.raises(InterviewContractError, match="already published"):
        trace.add(
            "question_committed",
            trace.completion(
                request,
                question_id="q1",
                result=QuestionResult(request, "Different question?").to_dict(),
            ),
        )
    trace.add(
        "question_committed",
        trace.completion(
            request,
            question_id="q2",
            result=QuestionResult(request, "Different question?").to_dict(),
        ),
    )
    with pytest.raises(InterviewContractError, match="answer ID"):
        trace.add("answer_committed", {"question_id": "q2", "answer_id": "a1", "text": "different"})


def test_public_helper_errors_remain_controlled_and_results_immutable() -> None:
    trace = Trace()
    request = trace.request()
    for callback in (
        lambda: schedule(None),  # type: ignore[arg-type]
        lambda: current_memory_snapshot(None),  # type: ignore[arg-type]
        lambda: interview_report(None),  # type: ignore[arg-type]
        lambda: make_event(trace.state, None),  # type: ignore[arg-type]
        lambda: apply_event(trace.state, None),  # type: ignore[arg-type]
        lambda: NextAction("question", target="not typed"),  # type: ignore[arg-type]
        lambda: NextAction("await_review", proposal_ids=("p", "p")),
        lambda: NextAction("finish", reason=[]),  # type: ignore[arg-type]
        lambda: StageReservation(request, 2, "a" * 64, "failed", []),  # type: ignore[arg-type]
        lambda: InterviewTransition(None, NextAction("finish")),  # type: ignore[arg-type]
        lambda: InterviewTransition(trace.state, schedule(trace.state), ("not a memory",)),  # type: ignore[arg-type]
    ):
        with pytest.raises(InterviewContractError):
            callback()
    action = NextAction("await_review", proposal_ids=["p"])  # type: ignore[arg-type]
    assert action.proposal_ids == ("p",)
    result = InterviewTransition(trace.state, schedule(trace.state), [])  # type: ignore[arg-type]
    assert result.memory_additions == ()


def test_report_enforces_its_own_complete_envelope_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = Trace()
    size = len(contract_json(interview_report(trace.state)).encode())
    # Plan/head constituents individually fit this boundary; their final report
    # envelope does not. No provider request or stage rendering occurs here.
    monkeypatch.setattr(interview_models, "MAX_CONTRACT_BYTES", size - 1)
    with pytest.raises(InterviewContractError, match="byte limit"):
        interview_report(trace.state)


@pytest.mark.parametrize("stage", ["question", "analysis"])
@pytest.mark.parametrize(
    "mutation",
    [
        "length",
        "missing_finish",
        "tool_call",
        "changed_result",
        "missing_completion",
        "duplicate_json_keys",
    ],
)
def test_replayable_raw_completion_is_required_and_must_exactly_produce_result(
    stage: str, mutation: str
) -> None:
    trace = Trace(one_plan())
    if stage == "analysis":
        trace.question()
        trace.answer()
    request = trace.reserve()
    result = (
        analysis_result(request, memory=True)
        if stage == "analysis"
        else QuestionResult(request, "Valid question?")
    )
    payload = trace.completion(request, result=result.to_dict())
    if stage == "question":
        payload["question_id"] = "q1"
    completion = payload["completion"]
    choice = completion["choices"][0]
    if mutation == "length":
        choice["finish_reason"] = "length"
    elif mutation == "missing_finish":
        choice.pop("finish_reason")
    elif mutation == "tool_call":
        choice["message"]["tool_calls"] = [
            {"id": "tool", "function": {"name": "unwanted", "arguments": "{}"}}
        ]
    elif mutation == "changed_result":
        if stage == "question":
            payload["result"]["text"] = "Different published question?"
        else:
            payload["result"]["assessments"][0]["rationale"] = "Unrecorded rationale"
    elif mutation == "missing_completion":
        payload.pop("completion")
    else:
        text = choice["message"]["content"]
        choice["message"]["content"] = '{"format":"wrong",' + text[1:]
    before = trace.state.to_dict()
    with pytest.raises(InterviewContractError):
        trace.add(stage + "_committed", payload)
    assert trace.state.to_dict() == before
    assert not trace.state.memories and trace.state.pending is not None


def test_provider_raw_metadata_remains_private_in_event_and_replay_validates_it() -> None:
    trace = Trace(one_plan())
    request = trace.reserve()
    result = QuestionResult(request, "Question?")
    response = completion_envelope(result)
    response["vendor_private_metadata"] = {
        "marker": "authored-private-metadata-fixture",
        "flag": True,
        "count": 1,
    }
    event = trace.add(
        "question_committed",
        trace.completion(request, result=result.to_dict(), completion=response, question_id="q1"),
    )
    assert event.to_dict()["payload"]["completion"] == response
    response["vendor_private_metadata"]["flag"] = False
    assert event.payload["completion"]["vendor_private_metadata"]["flag"] is True
    assert replay_interview(trace.events, trace.state.head).to_dict() == trace.state.to_dict()
    # An attacker can recompute content and command hashes; semantic replay must
    # still reject an internally rehashed incomplete final provider response.
    payload = event.to_dict()["payload"]
    payload["completion"]["choices"][0]["finish_reason"] = "length"
    command = InterviewCommand(
        event.interview_id,
        event.command_id,
        event.sequence - 1,
        event.previous_digest,
        event.kind,
        payload,
    )
    forged = InterviewEvent(
        command.interview_id,
        command.command_id,
        event.sequence,
        command.expected_digest,
        command.digest,
        command.kind,
        command.payload,
    )
    with pytest.raises(InterviewContractError):
        replay_interview([*trace.events[:-1], forged])


def test_creation_reserves_one_of_128_memory_head_slots_for_current_interview() -> None:
    heads = tuple(SourceHead(f"prior-{index}", 1, "a" * 64) for index in range(128))
    snapshot = MemorySnapshot("person", heads, ())
    with pytest.raises(InterviewContractError, match="127"):
        Trace(snapshot=snapshot)
    trace = Trace(one_plan(), replace(snapshot, heads=heads[:-1]))
    trace.question()
    trace.answer()
    trace.analysis(memory=True)
    assert len(current_memory_snapshot(trace.state).heads) == 128
    assert current_memory_snapshot(trace.state).heads[-1].interview_id == "session"


@pytest.mark.parametrize("field", ["analyst_prompt_version", "questioner_prompt_version"])
def test_creation_rejects_a_plan_the_current_renderer_cannot_execute(field: str) -> None:
    future = replace(plan(), **{field: "future/v2"})
    assert InterviewPlan.from_dict(future.to_dict()).digest == future.digest
    with pytest.raises(InterviewContractError, match="prompt versions"):
        Trace(future)


def test_combined_recall_capacity_failure_cannot_partially_commit_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_array = interview_memory._array

    def small_entries(
        value: Any, name: str, *, minimum: int = 0, maximum: int = 10000
    ) -> tuple[Any, ...]:
        return original_array(
            value, name, minimum=minimum, maximum=2 if name == "memory entries" else maximum
        )

    # Keep all other transport/schema budgets unchanged; use the same existing
    # bounded-array validator with a tiny catalog for an inexpensive boundary.
    monkeypatch.setattr(interview_memory, "_array", small_entries)
    trace = Trace(snapshot=prior_snapshot())
    trace.question()
    trace.answer()
    trace.analysis("partial", memory=True)
    assert len(current_memory_snapshot(trace.state).entries) == 2
    trace.question()
    trace.answer("A second independent work fixture answer.")
    request = trace.reserve()
    result = analysis_result(request, memory=True)
    before = trace.state.to_dict()
    with pytest.raises(InterviewContractError, match="memory entries"):
        trace.add("analysis_committed", trace.completion(request, result=result.to_dict()))
    assert trace.state.to_dict() == before
    assert len(trace.state.memories) == 1
    assert trace.state.assessments[0].status == "partial"
    assert trace.state.pending is not None
