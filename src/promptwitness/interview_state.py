"""Pure, immutable interview event reduction and deterministic scheduler/v1.

No provider calls, clocks, IDs, database writes or randomness live here. An
external journal authenticates heads and implements historical command replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .interview_memory import MemoryEntry, MemorySnapshot, SourceHead
from .interview_models import (
    CriterionAssessment,
    CriterionLink,
    InterviewContractError,
    InterviewPlan,
    InterviewTopic,
    ParticipantAnswer,
    _array,
    _closed,
    _identifier,
    _integer,
    _sha,
    _string,
    _versioned,
    assessed_coverage,
    contract_digest,
    contract_json,
)
from .interview_stages import (
    RENDERER_VERSION,
    TEMPLATE_HASHES,
    AnalysisResult,
    EmergentProposal,
    InterviewQuestion,
    InterviewStageRequest,
    QuestionResult,
    StageTarget,
    _provider_identity,
    parse_analysis_completion,
    parse_question_completion,
)
from .models import _freeze_json, _thaw_json

SCHEDULER_VERSION = "scheduler/v1"
_COMPLETION = {"request_id", "attempt_id", "request_digest"}
_FIELDS = {
    "created": {
        "participant_id",
        "plan",
        "provider_identities",
        "template_hashes",
        "renderer_version",
        "scheduler_version",
        "memory_snapshot",
    },
    "question_reserved": {"request"},
    "analysis_reserved": {"request"},
    "question_committed": _COMPLETION | {"question_id", "result", "completion"},
    "analysis_committed": _COMPLETION | {"result", "completion"},
    "answer_committed": {"question_id", "answer_id", "text"},
    "question_skipped": {"question_id", "scope"},
    "stage_failed": _COMPLETION | {"error_code"},
    "retry_authorized": {"request"},
    "proposal_decided": {"proposal_id", "decision"},
    "finished": {"reason"},
}
_FAILURES = {
    "provider_error",
    "invalid_response",
    "deadline_exceeded",
    "interrupted",
    "response_incomplete",
}
_FINISH = {"agenda_completed", "budget_exhausted", "unresolved", "participant_stopped", "failed"}


def _snapshot(value: Any) -> Any:
    contract_json(value)
    return _freeze_json(value, "interview state")


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _payload(kind: Any, value: Any) -> Mapping[str, Any]:
    if not isinstance(kind, str) or kind not in _FIELDS:
        raise InterviewContractError("unsupported interview event kind")
    return _closed(value, _FIELDS[kind], "interview event payload")


@dataclass(frozen=True, slots=True)
class InterviewCommand:
    interview_id: str
    command_id: str
    expected_revision: int
    expected_digest: str | None
    kind: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.interview_id, "interview ID")
        _identifier(self.command_id, "command ID")
        _integer(self.expected_revision, "expected revision")
        if self.expected_revision:
            _sha(self.expected_digest, "expected head digest")
        elif self.expected_digest is not None:
            raise InterviewContractError("creation requires a null previous digest")
        object.__setattr__(self, "payload", _snapshot(_payload(self.kind, self.payload)))
        contract_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-command/v1",
            "interview_id": self.interview_id,
            "command_id": self.command_id,
            "expected_revision": self.expected_revision,
            "expected_digest": self.expected_digest,
            "kind": self.kind,
            "payload": _thaw_json(self.payload),
        }

    @property
    def digest(self) -> str:
        return contract_digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> InterviewCommand:
        data = _versioned(value, "interview-command", set(cls.__dataclass_fields__))
        return cls(*(data[name] for name in cls.__dataclass_fields__))


@dataclass(frozen=True, slots=True)
class InterviewEvent:
    interview_id: str
    command_id: str
    sequence: int
    previous_digest: str | None
    command_digest: str
    kind: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _integer(self.sequence, "event sequence", minimum=1, maximum=100_000)
        command = InterviewCommand(
            self.interview_id,
            self.command_id,
            self.sequence - 1,
            self.previous_digest,
            self.kind,
            self.payload,
        )
        if command.digest != self.command_digest:
            raise InterviewContractError("event does not match its bound command digest")
        object.__setattr__(self, "payload", command.payload)
        contract_json(self.to_dict())

    def _body(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-event/v1",
            "interview_id": self.interview_id,
            "command_id": self.command_id,
            "sequence": self.sequence,
            "previous_digest": self.previous_digest,
            "command_digest": self.command_digest,
            "kind": self.kind,
            "payload": _thaw_json(self.payload),
        }

    @property
    def digest(self) -> str:
        return contract_digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value: Any) -> InterviewEvent:
        data = _versioned(value, "interview-event", set(cls.__dataclass_fields__) | {"digest"})
        result = cls(*(data[name] for name in cls.__dataclass_fields__))
        if result.digest != data["digest"]:
            raise InterviewContractError("event digest does not match its content")
        return result


@dataclass(frozen=True, slots=True)
class InterviewSkip:
    question_id: str
    topic_id: str
    criterion_id: str
    scope: str
    revision: int

    def __post_init__(self) -> None:
        for name in ("question_id", "topic_id", "criterion_id"):
            _identifier(getattr(self, name), name)
        if self.scope not in ("criterion", "topic"):
            raise InterviewContractError("skip scope must be criterion or topic")
        _integer(self.revision, "skip revision", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ProposalDecision:
    proposal: EmergentProposal
    status: str = "pending"
    decision_revision: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, EmergentProposal) or self.status not in (
            "pending",
            "accepted",
            "rejected",
        ):
            raise InterviewContractError("invalid emergent proposal decision")
        if self.status == "pending":
            if self.decision_revision is not None:
                raise InterviewContractError("pending proposal has no decision revision")
        else:
            _integer(self.decision_revision, "proposal decision revision", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "status": self.status,
            "decision_revision": self.decision_revision,
        }


@dataclass(frozen=True, slots=True)
class StageReservation:
    request: InterviewStageRequest
    revision: int
    digest: str
    status: str = "pending"
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, InterviewStageRequest):
            raise InterviewContractError("reservation requires a validated stage request")
        _integer(self.revision, "reservation revision", minimum=1)
        _sha(self.digest, "reservation digest")
        if self.status not in ("pending", "failed") or (
            self.error_code is not None
            if self.status == "pending"
            else not isinstance(self.error_code, str) or self.error_code not in _FAILURES
        ):
            raise InterviewContractError("invalid reservation status/error code")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "revision": self.revision,
            "digest": self.digest,
            "status": self.status,
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class InterviewState:
    """A derived view, not an independently trusted serialized import format.

    Recover with replay_interview, never by trusting a cached state dictionary.
    """

    interview_id: str
    participant_id: str
    plan: InterviewPlan
    provider_identities: Mapping[str, Any]
    template_hashes: Mapping[str, Any]
    renderer_version: str
    scheduler_version: str
    memory_snapshot: MemorySnapshot
    revision: int
    digest: str
    questions: tuple[InterviewQuestion, ...] = ()
    answers: tuple[ParticipantAnswer, ...] = ()
    skips: tuple[InterviewSkip, ...] = ()
    assessments: tuple[CriterionAssessment, ...] = ()
    proposals: tuple[ProposalDecision, ...] = ()
    memories: tuple[MemoryEntry, ...] = ()
    analyzed_answer_ids: tuple[str, ...] = ()
    pending: StageReservation | None = None
    stage_reservations: int = 0
    stage_completions: int = 0
    failures: tuple[Mapping[str, Any], ...] = ()
    attempts: Mapping[str, str] = field(default_factory=dict)
    request_ids: tuple[str, ...] = ()
    command_digests: Mapping[str, str] = field(default_factory=dict)
    finished_reason: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.interview_id, "interview ID")
        _identifier(self.participant_id, "participant ID")
        _integer(self.revision, "state revision", minimum=1, maximum=100_000)
        _sha(self.digest, "state head digest")
        if not isinstance(self.plan, InterviewPlan) or not isinstance(
            self.memory_snapshot, MemorySnapshot
        ):
            raise InterviewContractError("state requires a plan and original memory snapshot")
        if self.memory_snapshot.participant_id != self.participant_id or any(
            head.interview_id == self.interview_id for head in self.memory_snapshot.heads
        ):
            raise InterviewContractError("creation memory must be scoped to other prior interviews")
        if len(self.memory_snapshot.heads) > 127:
            raise InterviewContractError(
                "creation allows at most 127 prior heads, reserving one for this interview"
            )
        identities = _closed(self.provider_identities, {"analyst", "questioner"}, "provider roles")
        if any(
            not isinstance(identity, Mapping) or not identity for identity in identities.values()
        ):
            raise InterviewContractError("each provider role requires a nonempty identity object")
        for identity in identities.values():
            _provider_identity(identity)
        if (
            self.plan.analyst_prompt_version != "analysis/v1"
            or self.plan.questioner_prompt_version != "question/v1"
        ):
            raise InterviewContractError(
                "creation requires supported analyst and questioner prompt versions"
            )
        if contract_digest(self.template_hashes) != contract_digest(TEMPLATE_HASHES) or (
            self.renderer_version != RENDERER_VERSION or self.scheduler_version != SCHEDULER_VERSION
        ):
            raise InterviewContractError(
                "creation must pin the actual template and policy versions"
            )
        for name in ("provider_identities", "template_hashes", "attempts", "command_digests"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))
        for name, kind in (
            ("questions", InterviewQuestion),
            ("answers", ParticipantAnswer),
            ("skips", InterviewSkip),
            ("assessments", CriterionAssessment),
            ("proposals", ProposalDecision),
            ("memories", MemoryEntry),
        ):
            values = _array(getattr(self, name), name)
            if any(not isinstance(item, kind) for item in values):
                raise InterviewContractError("state collections require validated typed values")
            object.__setattr__(self, name, values)
        for name in ("analyzed_answer_ids", "request_ids"):
            values = _array(getattr(self, name), name)
            for item in values:
                _identifier(item, name)
            if len(set(values)) != len(values):
                raise InterviewContractError("state cannot repeat identities")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self, "failures", tuple(_snapshot(item) for item in _array(self.failures, "failures"))
        )
        for name in ("stage_reservations", "stage_completions"):
            _integer(getattr(self, name), name, maximum=self.plan.budgets.provider_calls)
        if (
            self.stage_completions > self.stage_reservations
            or self.participant_turns > self.plan.budgets.participant_turns
        ):
            raise InterviewContractError("interview counts exceed their budgets")
        if self.pending is not None and not isinstance(self.pending, StageReservation):
            raise InterviewContractError("invalid pending stage reservation")
        if self.finished_reason is not None and (
            not isinstance(self.finished_reason, str) or self.finished_reason not in _FINISH
        ):
            raise InterviewContractError("invalid interview termination reason")
        contract_json(self.to_dict())

    @property
    def participant_turns(self) -> int:
        return len(self.answers) + len(self.skips)

    @property
    def accepted_emergent(self) -> tuple[InterviewTopic, ...]:
        accepted = [item for item in self.proposals if item.status == "accepted"]
        accepted.sort(key=lambda item: item.decision_revision or 0)
        return tuple(item.proposal.topic for item in accepted)

    @property
    def head(self) -> SourceHead:
        return SourceHead(self.interview_id, self.revision, self.digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-state/v1",
            "interview_id": self.interview_id,
            "participant_id": self.participant_id,
            "plan": self.plan.to_dict(),
            "provider_identities": _thaw_json(self.provider_identities),
            "template_hashes": _thaw_json(self.template_hashes),
            "renderer_version": self.renderer_version,
            "scheduler_version": self.scheduler_version,
            "memory_snapshot": self.memory_snapshot.to_dict(),
            "revision": self.revision,
            "digest": self.digest,
            **{
                name: [item.to_dict() for item in getattr(self, name)]
                for name in (
                    "questions",
                    "answers",
                    "skips",
                    "assessments",
                    "proposals",
                    "memories",
                )
            },
            "analyzed_answer_ids": list(self.analyzed_answer_ids),
            "pending": self.pending.to_dict() if self.pending else None,
            "participant_turns": self.participant_turns,
            "stage_reservations": self.stage_reservations,
            "stage_completions": self.stage_completions,
            "failures": [_thaw_json(item) for item in self.failures],
            "attempts": _thaw_json(self.attempts),
            "request_ids": list(self.request_ids),
            "command_digests": _thaw_json(self.command_digests),
            "finished_reason": self.finished_reason,
        }


@dataclass(frozen=True, slots=True)
class NextAction:
    kind: str
    target: StageTarget | None = None
    reason: str | None = None
    question_id: str | None = None
    answer_id: str | None = None
    proposal_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in (
            "question",
            "analysis",
            "await_answer",
            "await_review",
            "await_retry",
            "await_completion",
            "finish",
            "finished",
        ):
            raise InterviewContractError("invalid next action")
        if self.target is not None and not isinstance(self.target, StageTarget):
            raise InterviewContractError("invalid scheduled target")
        proposals = _array(self.proposal_ids, "pending proposal IDs")
        for item in proposals:
            _identifier(item, "proposal ID")
        if len(set(proposals)) != len(proposals):
            raise InterviewContractError("next action cannot repeat proposal IDs")
        object.__setattr__(self, "proposal_ids", proposals)
        for name in ("question_id", "answer_id"):
            if getattr(self, name) is not None:
                _identifier(getattr(self, name), name)
        if self.reason is not None and (
            not isinstance(self.reason, str) or self.reason not in _FINISH
        ):
            raise InterviewContractError("invalid next-action reason")
        contract_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target.to_dict() if self.target else None,
            "reason": self.reason,
            "question_id": self.question_id,
            "answer_id": self.answer_id,
            "proposal_ids": list(self.proposal_ids),
        }


@dataclass(frozen=True, slots=True)
class InterviewTransition:
    state: InterviewState
    action: NextAction
    memory_additions: tuple[MemoryEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, InterviewState) or not isinstance(self.action, NextAction):
            raise InterviewContractError("transition requires typed state and action")
        entries = _array(self.memory_additions, "memory additions", maximum=128)
        if any(not isinstance(entry, MemoryEntry) for entry in entries):
            raise InterviewContractError("transition requires typed memory additions")
        object.__setattr__(self, "memory_additions", entries)


def current_memory_snapshot(state: InterviewState) -> MemorySnapshot:
    """The immutable recall catalog at this exact pre-reservation head."""
    if not isinstance(state, InterviewState):
        raise InterviewContractError("memory snapshot requires an InterviewState")
    return MemorySnapshot(
        state.participant_id,
        (*state.memory_snapshot.heads, state.head)
        if state.memories
        else state.memory_snapshot.heads,
        (*state.memory_snapshot.entries, *state.memories),
    )


def _waiting_question(state: InterviewState) -> InterviewQuestion | None:
    if not state.questions:
        return None
    question = state.questions[-1]
    handled = {answer.question_id for answer in state.answers} | {
        skip.question_id for skip in state.skips
    }
    return question if question.question_id not in handled else None


def _skipped(state: InterviewState, link: CriterionLink) -> bool:
    return any(
        item.topic_id == link.topic_id
        and (item.scope == "topic" or item.criterion_id == link.criterion_id)
        for item in state.skips
    )


def _question_counts(state: InterviewState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in state.questions:
        topic = question.target.topic_id
        counts[topic] = counts.get(topic, 0) + 1
    return counts


def _target(state: InterviewState, link: CriterionLink, *, analysis: bool = False) -> StageTarget:
    previous = state.questions[-1] if state.questions else None
    same_topic = previous is not None and previous.target.topic_id == link.topic_id
    if analysis:
        assert previous is not None
        mode, parent = previous.target.mode, previous.target.parent_question_id
    elif same_topic and previous is not None:
        mode, parent = "follow_up", previous.question_id
    elif link.topic_id in {topic.topic_id for topic in state.accepted_emergent}:
        mode, parent = "emergent", None
    else:
        mode, parent = ("initial" if previous is None else "transition"), None
    return StageTarget(
        link.topic_id,
        link.criterion_id,
        mode,
        parent,
        state.revision,
        state.digest,
        SCHEDULER_VERSION,
    )


def schedule(state: InterviewState) -> NextAction:
    if not isinstance(state, InterviewState):
        raise InterviewContractError("scheduler requires an InterviewState")
    if state.finished_reason:
        return NextAction("finished", reason=state.finished_reason)
    if state.pending:
        return NextAction(
            "await_retry" if state.pending.status == "failed" else "await_completion",
            target=state.pending.request.target,
        )
    waiting = _waiting_question(state)
    if waiting:
        return NextAction("await_answer", question_id=waiting.question_id)
    unanswered_analysis = [
        answer for answer in state.answers if answer.answer_id not in state.analyzed_answer_ids
    ]
    if unanswered_analysis:
        answer = unanswered_analysis[-1]
        question = state.questions[-1]
        if state.stage_reservations >= state.plan.budgets.provider_calls:
            return NextAction("finish", reason="budget_exhausted", answer_id=answer.answer_id)
        return NextAction(
            "analysis",
            target=_target(
                state,
                CriterionLink(question.target.topic_id, question.target.criterion_id),
                analysis=True,
            ),
            answer_id=answer.answer_id,
        )
    pending = tuple(
        item.proposal.proposal_id for item in state.proposals if item.status == "pending"
    )
    if pending:
        return NextAction("await_review", proposal_ids=pending)
    covered = {item.link for item in state.assessments if item.status == "covered"}
    topics = (*state.plan.topics, *state.accepted_emergent)
    links = [
        CriterionLink(topic.topic_id, item.criterion_id)
        for topic in topics
        for item in topic.criteria
    ]
    if state.questions:
        current = state.questions[-1].target
        preferred = CriterionLink(current.topic_id, current.criterion_id)
        links = [
            preferred,
            *(link for link in links if link.topic_id == current.topic_id and link != preferred),
            *(link for link in links if link.topic_id != current.topic_id),
        ]
    counts = _question_counts(state)
    eligible = [
        link
        for link in links
        if link not in covered
        and not _skipped(state, link)
        and counts.get(link.topic_id, 0) < 1 + state.plan.budgets.followups_per_topic
    ]
    if not eligible:
        required = {CriterionLink(*pair) for pair in state.plan.required_criteria}
        if not required <= covered:
            return NextAction("finish", reason="unresolved")
        deferred_emergent = any(
            link not in required and link not in covered and not _skipped(state, link)
            for link in links
        )
        return NextAction(
            "finish", reason="budget_exhausted" if deferred_emergent else "agenda_completed"
        )
    if (
        state.participant_turns >= state.plan.budgets.participant_turns
        or state.stage_reservations >= state.plan.budgets.provider_calls
    ):
        return NextAction("finish", reason="budget_exhausted")
    return NextAction("question", target=_target(state, eligible[0]))


def make_event(state: InterviewState | None, command: InterviewCommand) -> InterviewEvent:
    if not isinstance(command, InterviewCommand):
        raise InterviewContractError("event creation requires an InterviewCommand")
    event = InterviewEvent(
        command.interview_id,
        command.command_id,
        command.expected_revision + 1,
        command.expected_digest,
        command.digest,
        command.kind,
        command.payload,
    )
    apply_event(state, event)
    return event


def _creation(event: InterviewEvent) -> InterviewState:
    data = event.payload
    return InterviewState(
        event.interview_id,
        data["participant_id"],
        InterviewPlan.from_dict(data["plan"]),
        data["provider_identities"],
        data["template_hashes"],
        data["renderer_version"],
        data["scheduler_version"],
        MemorySnapshot.from_dict(data["memory_snapshot"]),
        event.sequence,
        event.digest,
        command_digests={event.command_id: event.command_digest},
    )


def _matching_reservation(state: InterviewState, data: Mapping[str, Any]) -> StageReservation:
    reservation = state.pending
    if (
        reservation is None
        or reservation.status != "pending"
        or (reservation.revision != state.revision or reservation.digest != state.digest)
    ):
        raise InterviewContractError("stage completion requires the exact active reservation head")
    request = reservation.request
    if (data["request_id"], data["attempt_id"], data["request_digest"]) != (
        request.request_id,
        request.attempt_id,
        request.digest,
    ):
        raise InterviewContractError("stage result belongs to a stale or different attempt")
    return reservation


def _request(state: InterviewState, raw: Any, *, retry: bool) -> InterviewStageRequest:
    request = InterviewStageRequest.from_dict(raw)
    context = request.context
    action = schedule(replace(state, pending=None)) if retry else schedule(state)
    if action.kind != request.stage or action.target is None:
        raise InterviewContractError("requested stage is not the scheduled next action")
    if contract_digest(request.target.to_dict()) != contract_digest(action.target.to_dict()):
        raise InterviewContractError("stage target does not match the current scheduled target")
    if (
        context.interview_id != state.interview_id
        or context.participant_id != state.participant_id
        or context.source_revision != state.revision
        or context.source_digest != state.digest
    ):
        raise InterviewContractError("request source identity differs from the current state head")
    if context.plan.digest != state.plan.digest or contract_digest(
        [topic.to_dict() for topic in context.accepted_emergent]
    ) != contract_digest([topic.to_dict() for topic in state.accepted_emergent]):
        raise InterviewContractError("request agenda differs from the pinned interview agenda")
    role = "analyst" if request.stage == "analysis" else "questioner"
    if contract_digest(request.provider_identity) != contract_digest(
        state.provider_identities[role]
    ):
        raise InterviewContractError("request provider differs from the pinned role identity")
    if context.analysis_answer_id != action.answer_id:
        raise InterviewContractError("request analysis answer is not the pending real answer")
    answers = {answer.answer_id: answer.digest for answer in state.answers}
    if any(answers.get(answer.answer_id) != answer.digest for answer in context.source_answers):
        raise InterviewContractError("request includes a fabricated or changed participant answer")
    questions = {
        question.question_id: contract_digest(question.to_dict()) for question in state.questions
    }
    if any(
        questions.get(question.question_id) != contract_digest(question.to_dict())
        for question in context.question_context
    ):
        raise InterviewContractError("request includes a fabricated or changed published question")
    if request.request_id in state.request_ids or request.attempt_id in state.attempts:
        raise InterviewContractError("request and attempt IDs must never be reused")
    if (
        request.snapshot is not None
        and request.snapshot.digest != current_memory_snapshot(state).digest
    ):
        raise InterviewContractError(
            "request recall differs from the actual scoped memory snapshot"
        )
    if retry and (state.pending is None or request.stage != state.pending.request.stage):
        raise InterviewContractError("retry must replace the same pending stage")
    return request


def apply_event(state: InterviewState | None, event: InterviewEvent) -> InterviewTransition:
    if not isinstance(event, InterviewEvent):
        raise InterviewContractError("reducer requires a validated InterviewEvent")
    if state is None:
        if event.sequence != 1 or event.kind != "created":
            raise InterviewContractError("interview history must begin with creation")
        result = _creation(event)
        return InterviewTransition(result, schedule(result))
    if not isinstance(state, InterviewState) or (
        event.interview_id,
        event.sequence - 1,
        event.previous_digest,
    ) != (state.interview_id, state.revision, state.digest):
        raise InterviewContractError("event does not extend this exact interview head")
    if event.command_id in state.command_digests:
        raise InterviewContractError("command ID already exists; retrieve its historical result")
    if state.finished_reason is not None or event.kind == "created":
        raise InterviewContractError("cannot change a finished or recreate an existing interview")
    data = event.payload
    changes: dict[str, Any] = {}
    additions: tuple[MemoryEntry, ...] = ()
    if event.kind in ("question_reserved", "analysis_reserved", "retry_authorized"):
        retry = event.kind == "retry_authorized"
        if retry and state.pending is None:
            raise InterviewContractError(
                "retry requires an existing uncertain or failed reservation"
            )
        request = _request(state, data["request"], retry=retry)
        if event.kind != "retry_authorized" and event.kind != request.stage + "_reserved":
            raise InterviewContractError("reservation kind differs from its request stage")
        changes.update(
            pending=StageReservation(request, event.sequence, event.digest),
            stage_reservations=state.stage_reservations + 1,
            attempts={**state.attempts, request.attempt_id: request.digest},
            request_ids=(*state.request_ids, request.request_id),
        )
    elif event.kind in ("question_committed", "analysis_committed", "stage_failed"):
        reservation = _matching_reservation(state, data)
        request = reservation.request
        if event.kind == "stage_failed":
            if not isinstance(data["error_code"], str) or data["error_code"] not in _FAILURES:
                raise InterviewContractError(
                    "stage failure requires a supported redacted error code"
                )
            changes.update(
                pending=replace(reservation, status="failed", error_code=data["error_code"]),
                failures=(
                    *state.failures,
                    {
                        "revision": event.sequence,
                        "request_id": request.request_id,
                        "attempt_id": request.attempt_id,
                        "error_code": data["error_code"],
                    },
                ),
            )
        else:
            if event.kind != request.stage + "_committed":
                raise InterviewContractError("completion kind differs from its reserved stage")
            changes.update(pending=None, stage_completions=state.stage_completions + 1)
            if request.stage == "question":
                parsed_question = parse_question_completion(request, _thaw_json(data["completion"]))
                if contract_json(parsed_question.to_dict()) != contract_json(data["result"]):
                    raise InterviewContractError(
                        "question result differs from its stored provider completion"
                    )
                result_question = QuestionResult.from_dict(data["result"], request=request)
                _identifier(data["question_id"], "question ID")
                question = result_question.to_question(data["question_id"], event.sequence)
                if any(
                    item.question_id == question.question_id
                    or _normalized(item.text) == _normalized(question.text)
                    for item in state.questions
                ):
                    raise InterviewContractError(
                        "question ID or normalized question text was already published"
                    )
                changes["questions"] = (*state.questions, question)
            else:
                parsed_analysis = parse_analysis_completion(request, _thaw_json(data["completion"]))
                if contract_json(parsed_analysis.to_dict()) != contract_json(data["result"]):
                    raise InterviewContractError(
                        "analysis result differs from its stored provider completion"
                    )
                result_analysis = AnalysisResult.from_dict(data["result"], request=request)
                additions = _analysis_changes(state, event, result_analysis, changes)
    elif event.kind in ("answer_committed", "question_skipped"):
        waiting = _waiting_question(state)
        if (
            state.pending is not None
            or waiting is None
            or data["question_id"] != waiting.question_id
        ):
            raise InterviewContractError(
                "participant input must address the exact pending question"
            )
        if event.kind == "answer_committed":
            _string(
                data["text"],
                "participant answer",
                limit=state.plan.budgets.answer_bytes,
                empty=True,
            )
            answer = ParticipantAnswer(
                state.interview_id,
                state.participant_id,
                data["answer_id"],
                waiting.question_id,
                data["text"],
                event.sequence,
            )
            if any(item.answer_id == answer.answer_id for item in state.answers):
                raise InterviewContractError("answer ID was already committed")
            changes["answers"] = (*state.answers, answer)
        else:
            changes["skips"] = (
                *state.skips,
                InterviewSkip(
                    waiting.question_id,
                    waiting.target.topic_id,
                    waiting.target.criterion_id,
                    data["scope"],
                    event.sequence,
                ),
            )
    elif event.kind == "proposal_decided":
        if schedule(state).kind != "await_review":
            raise InterviewContractError("proposal decisions require the review boundary")
        if data["decision"] not in ("accepted", "rejected"):
            raise InterviewContractError("proposal decision must be accepted or rejected")
        found = [
            item
            for item in state.proposals
            if item.proposal.proposal_id == data["proposal_id"] and item.status == "pending"
        ]
        if not found:
            raise InterviewContractError("proposal is not pending in this interview")
        if (
            data["decision"] == "accepted"
            and len(state.accepted_emergent) >= state.plan.budgets.emergent_topics
        ):
            raise InterviewContractError("accepted emergent topics exceed the original budget")
        changes["proposals"] = tuple(
            replace(item, status=data["decision"], decision_revision=event.sequence)
            if item is found[0]
            else item
            for item in state.proposals
        )
    elif event.kind == "finished":
        reason = data["reason"]
        if not isinstance(reason, str) or reason not in _FINISH:
            raise InterviewContractError("unsupported termination reason")
        action = schedule(state)
        if reason == "failed":
            if state.pending is None or state.pending.status != "failed":
                raise InterviewContractError("failed termination requires a recorded stage failure")
        elif reason != "participant_stopped" and (
            action.kind != "finish" or reason != action.reason
        ):
            raise InterviewContractError(
                "termination reason does not match the deterministic scheduler"
            )
        changes["finished_reason"] = reason
    result = replace(
        state,
        revision=event.sequence,
        digest=event.digest,
        command_digests={**state.command_digests, event.command_id: event.command_digest},
        **changes,
    )
    if additions:
        # Validate the complete next recall catalog before returning any domain
        # additions. A journal must not commit a memory that makes recall fail.
        current_memory_snapshot(result)
    return InterviewTransition(result, schedule(result), additions)


def _analysis_changes(
    state: InterviewState, event: InterviewEvent, result: AnalysisResult, changes: dict[str, Any]
) -> tuple[MemoryEntry, ...]:
    # Parsing above binds evidence to the pinned request. Reparse every record
    # against the authoritative answers again, never an externally bound object.
    assessments = tuple(
        CriterionAssessment.from_dict(item.to_dict(), answers=state.answers)
        for item in result.assessments
    )
    latest = {item.link: item for item in state.assessments}
    ranks = {"unanswered": 0, "partial": 1, "covered": 2}
    for item in assessments:
        previous = latest.get(item.link)
        if previous is None or ranks[item.status] >= ranks[previous.status]:
            latest[item.link] = item
    additions = tuple(
        MemoryEntry.from_dict(
            {
                "interview_id": state.interview_id,
                "revision": event.sequence,
                "memory": item.to_dict(),
            },
            answers=state.answers,
        )
        for item in result.memories
    )
    existing = {
        entry.memory.memory_id for entry in (*state.memory_snapshot.entries, *state.memories)
    }
    for entry in additions:
        if entry.memory.memory_id in existing:
            raise InterviewContractError("analysis cannot overwrite an existing memory identity")
        if entry.memory.supersedes is not None and entry.memory.supersedes not in existing:
            raise InterviewContractError("memory correction refers to an unknown prior memory")
        existing.add(entry.memory.memory_id)
    proposals = tuple(ProposalDecision(item) for item in result.proposals)
    prior_topics = (*state.plan.topics, *(item.proposal.topic for item in state.proposals))
    used_topics = {topic.topic_id for topic in prior_topics}
    used_criteria = {item.criterion_id for topic in prior_topics for item in topic.criteria}
    descriptions = {_normalized(topic.description) for topic in prior_topics}
    proposal_ids = {item.proposal.proposal_id for item in state.proposals}
    for proposal in proposals:
        topic = proposal.proposal.topic
        if (
            proposal.proposal.proposal_id in proposal_ids
            or topic.topic_id in used_topics
            or any(item.criterion_id in used_criteria for item in topic.criteria)
            or _normalized(topic.description) in descriptions
        ):
            raise InterviewContractError(
                "emergent proposal duplicates an existing topic, criterion or description"
            )
    changes.update(
        assessments=tuple(latest[key] for key in sorted(latest)),
        memories=(*state.memories, *additions),
        proposals=(*state.proposals, *proposals),
        analyzed_answer_ids=(*state.analyzed_answer_ids, result.analysis_answer_id),
    )
    return additions


def replay_interview(
    events: Sequence[InterviewEvent | Mapping[str, Any]],
    expected_head: SourceHead | Mapping[str, Any] | None = None,
) -> InterviewState:
    values = _array(events, "interview events", minimum=1, maximum=100_000)
    state: InterviewState | None = None
    for value in values:
        event = value if isinstance(value, InterviewEvent) else InterviewEvent.from_dict(value)
        state = apply_event(state, event).state
    assert state is not None
    if expected_head is not None:
        head = (
            expected_head
            if isinstance(expected_head, SourceHead)
            else SourceHead.from_dict(expected_head)
        )
        if state.head != head:
            raise InterviewContractError(
                "replayed interview does not match the trusted expected head"
            )
    return state


def interview_report(state: InterviewState) -> dict[str, Any]:
    if not isinstance(state, InterviewState):
        raise InterviewContractError("report requires an InterviewState")
    coverage = assessed_coverage(
        state.plan, state.assessments, accepted_emergent=state.accepted_emergent
    )
    latest = {item.link: item for item in state.assessments}
    counts = _question_counts(state)
    criteria = []
    for topic in (*state.plan.topics, *state.accepted_emergent):
        exhausted = counts.get(topic.topic_id, 0) >= 1 + state.plan.budgets.followups_per_topic
        for criterion in topic.criteria:
            link = CriterionLink(topic.topic_id, criterion.criterion_id)
            assessment = latest.get(link)
            criteria.append(
                {
                    **link.to_dict(),
                    "required": topic in state.plan.topics,
                    "status": assessment.status if assessment else "unanswered",
                    "skipped": _skipped(state, link),
                    "deferred": exhausted
                    and (assessment is None or assessment.status != "covered"),
                    "assessment": assessment.to_dict() if assessment else None,
                }
            )
    result = {
        "format": "promptwitness.interview-report/v1",
        "head": state.head.to_dict(),
        "participant_id": state.participant_id,
        "plan_digest": state.plan.digest,
        "status": "finished" if state.finished_reason else "active",
        "reason": state.finished_reason,
        "next_action": schedule(state).to_dict(),
        "assessed_coverage": coverage.to_dict(),
        "criteria": criteria,
        "questions": len(state.questions),
        "answers": len(state.answers),
        "skips": [skip.to_dict() for skip in state.skips],
        "participant_turns": state.participant_turns,
        "stage_reservations": state.stage_reservations,
        "stage_completions": state.stage_completions,
        "provider_call_count": None,
        "failures": [_thaw_json(item) for item in state.failures],
        "proposals": [item.to_dict() for item in state.proposals],
        "memory_ids": [entry.memory.memory_id for entry in state.memories],
        "unanalyzed_answer_ids": [
            answer.answer_id
            for answer in state.answers
            if answer.answer_id not in state.analyzed_answer_ids
        ],
        "pending": state.pending.to_dict() if state.pending else None,
    }
    contract_json(result)
    return result
