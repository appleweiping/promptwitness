"""Pure, bounded interview stage preparation and source-bound completion parsing.

No journal, state mutation, credential lookup, provider invocation or retry occurs
here. Hashes bind supplied data; they do not authenticate its real-world owner.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ._interview_prompts import RENDERER_VERSION, SCHEMA_HASHES, SCHEMAS, TEMPLATE_HASHES, TEMPLATES
from .interview_memory import BM25MemoryIndex, MemoryEntry, MemorySnapshot
from .interview_models import (
    MAX_CONTRACT_BYTES,
    BoundEvidence,
    CriterionAssessment,
    CriterionLink,
    InterviewContractError,
    InterviewPlan,
    InterviewTopic,
    MemoryRecord,
    ParticipantAnswer,
    _array,
    _bound,
    _closed,
    _identifier,
    _integer,
    _sha,
    _string,
    _versioned,
    answer_catalog,
    assessed_coverage,
    bind_evidence,
    contract_digest,
    contract_json,
    load_interview_json,
)
from .invocations import validate_tool_arguments
from .matrix import Scenario, render_matrix
from .models import ToolSpec, _freeze_json, _thaw_json
from .parser import parse_prompt
from .providers import prepare_chat_body
from .task_scores import response_text
from .validation import ValidationPolicy, validate_prompt

MAX_STAGE_RESPONSE_BYTES = 1024 * 1024

__all__ = [
    "MAX_STAGE_RESPONSE_BYTES",
    "RENDERER_VERSION",
    "SCHEMA_HASHES",
    "TEMPLATE_HASHES",
    "AnalysisResult",
    "CompletionRecord",
    "EmergentProposal",
    "InterviewQuestion",
    "InterviewStageContext",
    "InterviewStageRequest",
    "QuestionResult",
    "StageTarget",
    "build_analysis_request",
    "build_question_request",
    "parse_analysis_completion",
    "parse_question_completion",
]


def _same(left: Any, right: Any, name: str) -> None:
    if contract_json(left) != contract_json(right):
        raise InterviewContractError(f"{name} does not match its bound content")


def _ids(values: Any, name: str, *, maximum: int = 128) -> tuple[str, ...]:
    items = _array(values, name, maximum=maximum)
    for item in items:
        _identifier(item, name)
    if len(set(items)) != len(items):
        raise InterviewContractError(f"duplicate {name}")
    return items


def _memory_ids(values: Any) -> tuple[str, ...]:
    result = _ids(values, "memory IDs", maximum=1000)
    if any(re.fullmatch(r"mem-[0-9a-f]{64}", value) is None for value in result):
        raise InterviewContractError("invalid content-derived memory ID")
    return result


def _provider_identity(value: Any) -> Mapping[str, Any]:
    data = _closed(
        value,
        {
            "provider",
            "transport",
            "model",
            "endpoint_sha256",
            "headers_sha256",
            "api_key_env",
            "timeout",
        },
        "stage provider identity",
    )
    for name in ("provider", "transport", "model"):
        _identifier(data[name], name)
    for name in ("endpoint_sha256", "headers_sha256"):
        _sha(data[name], name)
    if data["api_key_env"] is not None and (
        not isinstance(data["api_key_env"], str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", data["api_key_env"]) is None
    ):
        raise InterviewContractError("invalid provider environment-variable name")
    if type(data["timeout"]) not in (int, float) or not 0 < data["timeout"] <= 300:
        raise InterviewContractError("provider timeout must be finite and bounded")
    result: Mapping[str, Any] = _freeze_json(
        load_interview_json(contract_json(data)), "provider identity"
    )
    return result


@dataclass(frozen=True, slots=True)
class StageTarget:
    topic_id: str
    criterion_id: str
    mode: str
    parent_question_id: str | None
    source_revision: int
    source_digest: str
    policy_version: str = "scheduler/v1"

    def __post_init__(self) -> None:
        _identifier(self.topic_id, "topic ID")
        _identifier(self.criterion_id, "criterion ID")
        if self.mode not in ("initial", "follow_up", "transition", "emergent"):
            raise InterviewContractError("invalid host-selected question mode")
        if self.parent_question_id is not None:
            _identifier(self.parent_question_id, "parent question ID")
        if self.mode == "follow_up" and self.parent_question_id is None:
            raise InterviewContractError("follow-up requires a parent question")
        _integer(self.source_revision, "target source revision", minimum=1)
        _sha(self.source_digest, "target source digest")
        if self.policy_version != "scheduler/v1":
            raise InterviewContractError("unsupported scheduling policy")

    @property
    def link(self) -> CriterionLink:
        return CriterionLink(self.topic_id, self.criterion_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.stage-target/v1",
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }

    @classmethod
    def from_dict(cls, value: Any) -> StageTarget:
        data = _versioned(value, "stage-target", set(cls.__dataclass_fields__))
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class InterviewQuestion:
    question_id: str
    interview_id: str
    participant_id: str
    revision: int
    target: StageTarget
    text: str
    memory_ids: tuple[str, ...]
    request_digest: str

    def __post_init__(self) -> None:
        for name in ("question_id", "interview_id", "participant_id"):
            _identifier(getattr(self, name), name)
        _integer(self.revision, "published question revision", minimum=1)
        if not isinstance(self.target, StageTarget) or self.revision <= self.target.source_revision:
            raise InterviewContractError("question must follow its target source revision")
        _string(self.text, "question text")
        object.__setattr__(self, "memory_ids", _memory_ids(self.memory_ids))
        _sha(self.request_digest, "question request digest")
        contract_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-question/v1",
            "question_id": self.question_id,
            "interview_id": self.interview_id,
            "participant_id": self.participant_id,
            "revision": self.revision,
            "target": self.target.to_dict(),
            "text": self.text,
            "memory_ids": list(self.memory_ids),
            "request_digest": self.request_digest,
        }

    @classmethod
    def from_dict(cls, value: Any) -> InterviewQuestion:
        data = _versioned(value, "interview-question", set(cls.__dataclass_fields__))
        return cls(
            **{
                **{name: data[name] for name in cls.__dataclass_fields__},
                "target": StageTarget.from_dict(data["target"]),
            }
        )

    @classmethod
    def from_result(
        cls, result: QuestionResult, question_id: str, revision: int
    ) -> InterviewQuestion:
        if not isinstance(result, QuestionResult):
            raise InterviewContractError("question publication requires a validated result")
        return cls(
            question_id,
            result.request.context.interview_id,
            result.request.context.participant_id,
            revision,
            result.target,
            result.text,
            result.memory_ids,
            result.request.digest,
        )


@dataclass(frozen=True, slots=True)
class InterviewStageContext:
    interview_id: str
    participant_id: str
    source_revision: int
    source_digest: str
    request_id: str
    attempt_id: str
    plan: InterviewPlan
    target: StageTarget
    accepted_emergent: tuple[InterviewTopic, ...] = ()
    source_answers: tuple[ParticipantAnswer, ...] = ()
    question_context: tuple[InterviewQuestion, ...] = ()
    analysis_answer_id: str | None = None
    allowed_criteria: tuple[CriterionLink, ...] = ()

    def __post_init__(self) -> None:
        for name in ("interview_id", "participant_id", "request_id", "attempt_id"):
            _identifier(getattr(self, name), name)
        _integer(self.source_revision, "stage source revision", minimum=1)
        _sha(self.source_digest, "stage source digest")
        if not isinstance(self.plan, InterviewPlan) or not isinstance(self.target, StageTarget):
            raise InterviewContractError("stage context requires a typed plan and target")
        if (self.target.source_revision, self.target.source_digest) != (
            self.source_revision,
            self.source_digest,
        ):
            raise InterviewContractError("target belongs to a different source head")
        topics = _array(self.accepted_emergent, "accepted topics", maximum=32)
        assessed_coverage(self.plan, (), accepted_emergent=topics)
        object.__setattr__(self, "accepted_emergent", topics)
        valid = {
            CriterionLink(topic.topic_id, criterion.criterion_id)
            for topic in (*self.plan.topics, *topics)
            for criterion in topic.criteria
        }
        allowed = _array(self.allowed_criteria, "allowed criteria", maximum=128)
        if not allowed:
            allowed = (self.target.link,)
        if (
            any(not isinstance(item, CriterionLink) or item not in valid for item in allowed)
            or len(set(allowed)) != len(allowed)
            or self.target.link not in allowed
        ):
            raise InterviewContractError(
                "allowed criteria must be unique, known and include target"
            )
        object.__setattr__(self, "allowed_criteria", allowed)
        answers = _array(self.source_answers, "stage source answers", maximum=128)
        answer_catalog(answers)
        questions = _array(self.question_context, "question context", maximum=128)
        if any(not isinstance(item, InterviewQuestion) for item in questions) or len(
            {item.question_id for item in questions}
        ) != len(questions):
            raise InterviewContractError("question context must contain unique typed questions")
        by_question = {item.question_id: item for item in questions}
        for item in (*answers, *questions):
            if (
                item.interview_id != self.interview_id
                or item.participant_id != self.participant_id
                or item.revision > self.source_revision
            ):
                raise InterviewContractError("stage transcript is outside participant/source head")
        for answer in answers:
            _string(answer.text, "source answer", limit=self.plan.budgets.answer_bytes, empty=True)
            question = by_question.get(answer.question_id)
            if question is None or question.revision >= answer.revision:
                raise InterviewContractError(
                    "answer needs its earlier published question in context"
                )
        if (
            self.target.parent_question_id is not None
            and self.target.parent_question_id not in by_question
        ):
            raise InterviewContractError("target parent question is missing from context")
        if self.analysis_answer_id is not None:
            _identifier(self.analysis_answer_id, "analysis answer ID")
            if not any(item.answer_id == self.analysis_answer_id for item in answers):
                raise InterviewContractError(
                    "current analysis answer is missing from source catalog"
                )
            current = next(item for item in answers if item.answer_id == self.analysis_answer_id)
            if by_question[current.question_id].target.link != self.target.link:
                raise InterviewContractError("analysis target differs from the answered question")
        object.__setattr__(self, "source_answers", answers)
        object.__setattr__(self, "question_context", questions)
        contract_json(self.to_dict())

    @property
    def analysis_answer(self) -> ParticipantAnswer | None:
        return next(
            (item for item in self.source_answers if item.answer_id == self.analysis_answer_id),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-stage-context/v1",
            **{
                name: getattr(self, name)
                for name in (
                    "interview_id",
                    "participant_id",
                    "source_revision",
                    "source_digest",
                    "request_id",
                    "attempt_id",
                    "analysis_answer_id",
                )
            },
            "plan": self.plan.to_dict(),
            "target": self.target.to_dict(),
            "accepted_emergent": [item.to_dict() for item in self.accepted_emergent],
            "source_answers": [item.to_dict() for item in self.source_answers],
            "question_context": [item.to_dict() for item in self.question_context],
            "allowed_criteria": [item.to_dict() for item in self.allowed_criteria],
        }

    @classmethod
    def from_dict(cls, value: Any) -> InterviewStageContext:
        data = _versioned(value, "interview-stage-context", set(cls.__dataclass_fields__))
        body = {name: data[name] for name in cls.__dataclass_fields__}
        body.update(
            plan=InterviewPlan.from_dict(data["plan"]), target=StageTarget.from_dict(data["target"])
        )
        for name, factory in (
            ("accepted_emergent", InterviewTopic.from_dict),
            ("source_answers", ParticipantAnswer.from_dict),
            ("question_context", InterviewQuestion.from_dict),
            ("allowed_criteria", CriterionLink.from_dict),
        ):
            body[name] = tuple(factory(item) for item in _array(data[name], name, maximum=128))
        return cls(**body)


@dataclass(frozen=True, slots=True)
class InterviewStageRequest:
    stage: str
    context: InterviewStageContext
    provider_identity: Mapping[str, Any]
    snapshot: MemorySnapshot | None = None
    query: str = ""
    _messages: tuple[Mapping[str, str], ...] = field(init=False, repr=False)
    _wire_body: bytes = field(init=False, repr=False)
    _retrieval: Mapping[str, Any] = field(init=False, repr=False)
    _binding_digest: str = field(init=False, repr=False)
    _selected: tuple[MemoryEntry, ...] = field(init=False, repr=False)
    _generation: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.stage not in ("analysis", "question") or not isinstance(
            self.context, InterviewStageContext
        ):
            raise InterviewContractError("invalid stage request context")
        if self.stage == "analysis" and (
            self.context.analysis_answer is None or self.snapshot is not None or self.query
        ):
            raise InterviewContractError(
                "analysis requires a current answer and cannot import recall"
            )
        if self.stage == "question" and self.context.analysis_answer_id is not None:
            raise InterviewContractError("question stage cannot declare a current analysis answer")
        object.__setattr__(self, "provider_identity", _provider_identity(self.provider_identity))
        original_generation = (
            self.context.plan.analyst_generation
            if self.stage == "analysis"
            else self.context.plan.questioner_generation
        )
        object.__setattr__(
            self,
            "_generation",
            _freeze_json(
                load_interview_json(contract_json(original_generation)), "stage generation"
            ),
        )
        if self.snapshot is not None and (
            not isinstance(self.snapshot, MemorySnapshot)
            or self.snapshot.participant_id != self.context.participant_id
        ):
            raise InterviewContractError("retrieval snapshot belongs to another participant")
        if self.snapshot is not None:
            for head in self.snapshot.heads:
                if head.interview_id == self.context.interview_id and (
                    head.revision,
                    head.digest,
                ) != (self.source_revision, self.source_digest):
                    raise InterviewContractError(
                        "current interview recall head differs from request source"
                    )
        _string(self.query, "retrieval query", empty=True)
        if (
            self.context.plan.analyst_prompt_version
            if self.stage == "analysis"
            else self.context.plan.questioner_prompt_version
        ) != ("analysis/v1" if self.stage == "analysis" else "question/v1"):
            raise InterviewContractError("unsupported pinned interview template version")
        document = parse_prompt(TEMPLATES[self.stage])
        policy = ValidationPolicy(
            allowed_roles=frozenset({"system", "user"}), require_system_first=True
        )
        if not validate_prompt(document, policy).valid or document.tools:
            raise InterviewContractError("interview template is invalid")
        selected: list[MemoryEntry] = []
        messages, encoded, binding = self._render(document, selected)
        if len(encoded) > self.context.plan.budgets.context_bytes:
            raise InterviewContractError(
                "fixed complete provider JSON body exceeds request byte budget"
            )
        hits = []
        count = 0
        if self.snapshot is not None:
            count = len(self.snapshot.entries)
            ranking_policy = replace(
                self.context.plan.retrieval, top_k=0, context_bytes=MAX_CONTRACT_BYTES
            )
            ranking = BM25MemoryIndex(self.snapshot, ranking_policy).search(self.query)
            available = {item.memory.memory_id: item for item in self.snapshot.entries}
            for hit in ranking.hits:
                candidate = [*selected, available[hit.memory_id]]
                reason = None
                if len(selected) >= self.context.plan.retrieval.top_k:
                    reason = "top_k"
                elif (
                    len(contract_json([item.to_dict() for item in candidate]).encode("utf-8"))
                    > self.context.plan.retrieval.context_bytes
                ):
                    reason = "retrieval_bytes"
                else:
                    proposed_messages, proposed_body, proposed_binding = self._render(
                        document, candidate
                    )
                    if len(proposed_body) > self.context.plan.budgets.context_bytes:
                        reason = "request_bytes"
                    else:
                        selected = candidate
                        messages, encoded, binding = (
                            proposed_messages,
                            proposed_body,
                            proposed_binding,
                        )
                hits.append({"memory_id": hit.memory_id, "score": hit.score, "omission": reason})
        object.__setattr__(self, "_messages", _freeze_json(messages, "stage messages"))
        object.__setattr__(self, "_wire_body", encoded)
        object.__setattr__(self, "_binding_digest", binding)
        object.__setattr__(self, "_selected", tuple(selected))
        object.__setattr__(
            self,
            "_retrieval",
            _freeze_json(
                {
                    "considered": count,
                    "matched": len(hits),
                    "omitted_no_overlap": count - len(hits),
                    "hits": hits,
                },
                "stage retrieval audit",
            ),
        )
        contract_json(self.to_dict())

    @property
    def role(self) -> str:
        return "analyst" if self.stage == "analysis" else "questioner"

    @property
    def generation(self) -> Mapping[str, Any]:
        return self._generation

    def _render(
        self, document: Any, selected: Sequence[MemoryEntry]
    ) -> tuple[list[dict[str, str]], bytes, str]:
        binding = contract_digest(
            {
                "stage": self.stage,
                "context_digest": contract_digest(self.context.to_dict()),
                "provider_identity": _thaw_json(self.provider_identity),
                "generation": _thaw_json(self.generation),
                "template_digest": TEMPLATE_HASHES[self.role],
                "schema_digest": SCHEMA_HASHES[self.role],
                "renderer_version": RENDERER_VERSION,
                "snapshot_digest": self.snapshot.digest if self.snapshot else None,
                "query_digest": hashlib.sha256(self.query.encode("utf-8")).hexdigest(),
                "memory_ids": [item.memory.memory_id for item in selected],
            }
        )
        payload = {
            "format": "promptwitness.interview-stage-input/v1",
            "binding_digest": binding,
            "context": self.context.to_dict(),
            "memories": [item.to_dict() for item in selected],
        }
        row = render_matrix(
            document,
            (
                Scenario(
                    self.context.request_id,
                    {
                        "payload_json": contract_json(payload),
                        "output_schema_json": contract_json(SCHEMAS[self.stage]),
                    },
                ),
            ),
            strict=True,
        )[0]
        messages = [{"role": item.role, "content": item.content} for item in row.messages]
        encoded = prepare_chat_body(
            messages, generation=_thaw_json(self.generation), model=self.provider_identity["model"]
        )
        return messages, encoded, binding

    @property
    def messages(self) -> tuple[Mapping[str, str], ...]:
        return self._messages

    @property
    def wire_body(self) -> bytes:
        return self._wire_body

    @property
    def wire_body_bytes(self) -> int:
        return len(self._wire_body)

    @property
    def binding_digest(self) -> str:
        return self._binding_digest

    @property
    def selected_memory_ids(self) -> tuple[str, ...]:
        return tuple(item.memory.memory_id for item in self._selected)

    @property
    def target(self) -> StageTarget:
        return self.context.target

    @property
    def request_id(self) -> str:
        return self.context.request_id

    @property
    def attempt_id(self) -> str:
        return self.context.attempt_id

    @property
    def source_revision(self) -> int:
        return self.context.source_revision

    @property
    def source_digest(self) -> str:
        return self.context.source_digest

    @property
    def plan_digest(self) -> str:
        return self.context.plan.digest

    @property
    def snapshot_digest(self) -> str | None:
        return self.snapshot.digest if self.snapshot else None

    def _body(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-stage-request/v1",
            "stage": self.stage,
            "context": self.context.to_dict(),
            "provider_identity": _thaw_json(self.provider_identity),
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
            "query": self.query,
            "generation": _thaw_json(self.generation),
            "renderer_version": RENDERER_VERSION,
            "template_digest": TEMPLATE_HASHES[self.role],
            "schema_digest": SCHEMA_HASHES[self.role],
            "binding_digest": self.binding_digest,
            "messages": _thaw_json(self.messages),
            "wire_body_bytes": self.wire_body_bytes,
            "wire_body_digest": hashlib.sha256(self.wire_body).hexdigest(),
            "selected_memory_ids": list(self.selected_memory_ids),
            "retrieval": _thaw_json(self._retrieval),
        }

    @property
    def digest(self) -> str:
        return contract_digest(self._body())

    @property
    def audit_bytes(self) -> int:
        return len(contract_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value: Any) -> InterviewStageRequest:
        if not isinstance(value, Mapping):
            raise InterviewContractError("stage request must be an object")
        required = {
            "format",
            "stage",
            "context",
            "provider_identity",
            "snapshot",
            "query",
            "generation",
            "renderer_version",
            "template_digest",
            "schema_digest",
            "binding_digest",
            "messages",
            "wire_body_bytes",
            "wire_body_digest",
            "selected_memory_ids",
            "retrieval",
            "digest",
        }
        data = _closed(value, required, "stage request")
        result = cls(
            data["stage"],
            InterviewStageContext.from_dict(data["context"]),
            data["provider_identity"],
            MemorySnapshot.from_dict(data["snapshot"]) if data["snapshot"] is not None else None,
            data["query"],
        )
        _same(data, result.to_dict(), "stage request")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> InterviewStageRequest:
        return cls.from_dict(load_interview_json(value))


def build_analysis_request(
    context: InterviewStageContext, provider_identity: Mapping[str, Any]
) -> InterviewStageRequest:
    return InterviewStageRequest("analysis", context, provider_identity)


def build_question_request(
    context: InterviewStageContext,
    provider_identity: Mapping[str, Any],
    *,
    snapshot: MemorySnapshot | None = None,
    query: str = "",
) -> InterviewStageRequest:
    return InterviewStageRequest("question", context, provider_identity, snapshot, query)


def _request(value: Any, stage: str) -> InterviewStageRequest:
    if not isinstance(value, InterviewStageRequest) or value.stage != stage:
        raise InterviewContractError(f"result requires a bound {stage} request")
    return value


def _current_evidence(request: InterviewStageRequest, values: Sequence[BoundEvidence]) -> None:
    answer = request.context.analysis_answer
    if answer is None:
        raise InterviewContractError("analysis request has no current participant answer")
    for value in values:
        if not isinstance(value, BoundEvidence):
            raise InterviewContractError("analysis requires validated source evidence")
        value.reference.validate(
            answer,
            participant_id=request.context.participant_id,
            max_revision=request.source_revision,
        )


@dataclass(frozen=True, slots=True)
class EmergentProposal:
    participant_id: str
    interview_id: str
    topic: InterviewTopic
    parent_topic_id: str
    evidence: tuple[BoundEvidence, ...]
    source_revision: int
    source_digest: str
    request_digest: str

    def __post_init__(self) -> None:
        for name in ("participant_id", "interview_id", "parent_topic_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.topic, InterviewTopic):
            raise InterviewContractError("proposal requires an InterviewTopic")
        _integer(self.source_revision, "proposal source revision", minimum=1)
        _sha(self.source_digest, "proposal source digest")
        _sha(self.request_digest, "proposal request digest")
        evidence = _bound(self.evidence, self.participant_id, interview_id=self.interview_id)
        for item in evidence:
            if not isinstance(item, BoundEvidence):
                raise InterviewContractError("proposal requires source-bound evidence")
            item.reference.validate(
                item.answer, participant_id=self.participant_id, max_revision=self.source_revision
            )
            if item.answer.interview_id != self.interview_id:
                raise InterviewContractError("proposal evidence belongs to another interview")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise InterviewContractError("duplicate proposal evidence")
        object.__setattr__(
            self, "evidence", tuple(sorted(evidence, key=lambda item: item.evidence_id))
        )
        contract_json(self.to_dict())

    def _body(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.emergent-proposal/v1",
            "participant_id": self.participant_id,
            "interview_id": self.interview_id,
            "topic": self.topic.to_dict(),
            "parent_topic_id": self.parent_topic_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "source_revision": self.source_revision,
            "source_digest": self.source_digest,
            "request_digest": self.request_digest,
        }

    @property
    def proposal_id(self) -> str:
        return "proposal-" + contract_digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "id": self.proposal_id}

    @classmethod
    def from_dict(cls, value: Any, *, answers: Sequence[ParticipantAnswer]) -> EmergentProposal:
        data = _versioned(value, "emergent-proposal", set(cls.__dataclass_fields__) | {"id"})
        result = cls(
            **{
                **{name: data[name] for name in cls.__dataclass_fields__},
                "topic": InterviewTopic.from_dict(data["topic"]),
                "evidence": bind_evidence(data["evidence"], answers),
            }
        )
        _same(data, result.to_dict(), "emergent proposal")
        return result


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    request: InterviewStageRequest
    assessments: tuple[CriterionAssessment, ...]
    memories: tuple[MemoryRecord, ...] = ()
    proposal: EmergentProposal | None = None

    @property
    def proposals(self) -> tuple[EmergentProposal, ...]:
        return (self.proposal,) if self.proposal is not None else ()

    @property
    def analysis_answer_id(self) -> str:
        answer_id = self.request.context.analysis_answer_id
        if answer_id is None:
            raise InterviewContractError("analysis result has no bound current answer")
        return answer_id

    def __post_init__(self) -> None:
        request = _request(self.request, "analysis")
        rows = _array(self.assessments, "assessments", minimum=1, maximum=128)
        memories = _array(self.memories, "new memories", maximum=128)
        allowed = set(request.context.allowed_criteria)
        for row in rows:
            if (
                not isinstance(row, CriterionAssessment)
                or row.link not in allowed
                or (row.participant_id, row.interview_id)
                != (request.context.participant_id, request.context.interview_id)
            ):
                raise InterviewContractError("assessment is outside the request scope")
            _current_evidence(request, row.evidence)
        if len({row.link for row in rows}) != len(rows) or request.target.link not in {
            row.link for row in rows
        }:
            raise InterviewContractError(
                "decisions must be unique and include the requested target"
            )
        for memory in memories:
            if (
                not isinstance(memory, MemoryRecord)
                or memory.participant_id != request.context.participant_id
                or any(link not in allowed for link in memory.links)
                or memory.supersedes is not None
            ):
                raise InterviewContractError(
                    "new memory is outside request scope or attempts correction"
                )
            _current_evidence(request, memory.evidence)
        if len({item.memory_id for item in memories}) != len(memories):
            raise InterviewContractError("duplicate new memory identity")
        if self.proposal is not None:
            proposal = self.proposal
            if not isinstance(proposal, EmergentProposal):
                raise InterviewContractError("analysis supports at most one typed proposal")
            _current_evidence(request, proposal.evidence)
            _same(
                (
                    proposal.participant_id,
                    proposal.interview_id,
                    proposal.source_revision,
                    proposal.source_digest,
                    proposal.request_digest,
                ),
                (
                    request.context.participant_id,
                    request.context.interview_id,
                    request.source_revision,
                    request.source_digest,
                    request.digest,
                ),
                "proposal source request",
            )
            topics = (*request.context.plan.topics, *request.context.accepted_emergent)
            if proposal.parent_topic_id not in {topic.topic_id for topic in topics}:
                raise InterviewContractError("proposal parent topic is outside the request")
            assessed_coverage(
                request.context.plan,
                (),
                accepted_emergent=(*request.context.accepted_emergent, proposal.topic),
            )
        object.__setattr__(self, "assessments", rows)
        object.__setattr__(self, "memories", memories)
        contract_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.analysis-result/v1",
            "request_digest": self.request.digest,
            "assessments": [item.to_dict() for item in self.assessments],
            "memories": [item.to_dict() for item in self.memories],
            "proposal": self.proposal.to_dict() if self.proposal else None,
        }

    @classmethod
    def from_dict(cls, value: Any, *, request: InterviewStageRequest) -> AnalysisResult:
        _request(request, "analysis")
        data = _versioned(
            value, "analysis-result", {"request_digest", "assessments", "memories", "proposal"}
        )
        answers = request.context.source_answers
        result = cls(
            request,
            tuple(
                CriterionAssessment.from_dict(item, answers=answers)
                for item in _array(data["assessments"], "assessments", maximum=128)
            ),
            tuple(
                MemoryRecord.from_dict(item, answers=answers)
                for item in _array(data["memories"], "memories", maximum=128)
            ),
            EmergentProposal.from_dict(data["proposal"], answers=answers)
            if data["proposal"] is not None
            else None,
        )
        _same(data, result.to_dict(), "analysis result")
        return result


@dataclass(frozen=True, slots=True)
class QuestionResult:
    request: InterviewStageRequest
    text: str
    memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        request = _request(self.request, "question")
        _string(self.text, "question text", limit=request.context.plan.budgets.question_bytes)
        ids = _memory_ids(self.memory_ids)
        if any(item not in request.selected_memory_ids for item in ids):
            raise InterviewContractError("question cites a memory omitted from its actual request")
        object.__setattr__(self, "memory_ids", ids)
        contract_json(self.to_dict())

    @property
    def target(self) -> StageTarget:
        return self.request.target

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.question-result/v1",
            "request_digest": self.request.digest,
            "target": self.target.to_dict(),
            "text": self.text,
            "memory_ids": list(self.memory_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, *, request: InterviewStageRequest) -> QuestionResult:
        _request(request, "question")
        data = _versioned(
            value, "question-result", {"request_digest", "target", "text", "memory_ids"}
        )
        result = cls(request, data["text"], data["memory_ids"])
        _same(data, result.to_dict(), "question result")
        return result

    def to_question(self, question_id: str, revision: int) -> InterviewQuestion:
        return InterviewQuestion.from_result(self, question_id, revision)


def _completion(request: InterviewStageRequest, response: Any, stage: str) -> Mapping[str, Any]:
    _request(request, stage)
    # Bare strings have no completion status. Scripted adapters must provide the
    # same completed envelope as a live adapter, never bypass this boundary.
    if not isinstance(response, Mapping):
        raise InterviewContractError("stage requires a positively completed provider envelope")
    response = load_interview_json(contract_json(response))
    if response.get("refusal") or (response.get("choices") and response.get("output")):
        raise InterviewContractError("stage response is refused or has ambiguous output branches")
    try:
        text = response_text(response)
    except ValueError:
        raise InterviewContractError("stage response is not one complete text result") from None
    _string(text, "stage response", limit=MAX_STAGE_RESPONSE_BYTES)
    raw = load_interview_json(text)
    schema = SCHEMAS[stage]
    tool = ToolSpec(
        "interview_response",
        "Local structured-response validation; never exposed as a tool.",
        schema["properties"],
        tuple(schema["required"]),
    )
    if not isinstance(raw, Mapping) or not validate_tool_arguments(tool, raw).valid:
        raise InterviewContractError("stage response does not match its closed schema")
    if raw["binding_digest"] != request.binding_digest:
        raise InterviewContractError("stage response belongs to another request binding")
    return raw


def parse_analysis_completion(request: InterviewStageRequest, response: Any) -> AnalysisResult:
    raw = _completion(request, response, "analysis")
    source = request.context.analysis_answer
    if source is None:
        raise InterviewContractError("analysis source answer is missing")
    evidence: dict[str, BoundEvidence] = {}
    for item in raw["evidence"]:
        key = _identifier(item["id"], "local evidence ID")
        if key in evidence or item["answer_id"] != source.answer_id:
            raise InterviewContractError("evidence ID is duplicate or cites a non-current answer")
        bound = BoundEvidence.from_answer(source, item["start"], item["end"])
        if bound.reference.quote != item["quote"]:
            raise InterviewContractError("proposed quote differs from the exact source span")
        evidence[key] = bound
    if len({item.evidence_id for item in evidence.values()}) != len(evidence):
        raise InterviewContractError("duplicate source spans under different local evidence IDs")
    used: set[str] = set()

    def references(values: Any) -> tuple[BoundEvidence, ...]:
        ids = _ids(values, "local evidence references")
        if any(item not in evidence for item in ids):
            raise InterviewContractError("unknown local evidence reference")
        used.update(ids)
        return tuple(evidence[item] for item in ids)

    rows = tuple(
        CriterionAssessment(
            request.context.participant_id,
            request.context.interview_id,
            CriterionLink(item["topic_id"], item["criterion_id"]),
            item["status"],
            references(item["evidence_ids"]),
            item["rationale"],
        )
        for item in raw["assessments"]
    )
    memories = tuple(
        MemoryRecord(
            request.context.participant_id,
            references(item["evidence_ids"]),
            tuple(CriterionLink.from_dict(link) for link in item["links"]),
            item["summary"]["text"],
        )
        for item in raw["memories"]
    )
    proposed = raw["proposal"]
    proposal = (
        EmergentProposal(
            request.context.participant_id,
            request.context.interview_id,
            InterviewTopic.from_dict(proposed["topic"]),
            proposed["parent_topic_id"],
            references(proposed["evidence_ids"]),
            request.source_revision,
            request.source_digest,
            request.digest,
        )
        if proposed is not None
        else None
    )
    if used != set(evidence):
        raise InterviewContractError("analysis contains unused evidence declarations")
    return AnalysisResult(request, rows, memories, proposal)


def parse_question_completion(request: InterviewStageRequest, response: Any) -> QuestionResult:
    raw = _completion(request, response, "question")
    _same(raw["target"], request.target.link.to_dict(), "question target")
    return QuestionResult(request, raw["text"], raw["memory_ids"])


@dataclass(frozen=True, slots=True)
class CompletionRecord:
    """Bound original completion plus a locally reconstructed normalized result.

    Preserving an envelope permits completion checks during replay; it is not
    cryptographic proof that a remote provider sent it. No status is synthesized.
    """

    request: InterviewStageRequest
    response: Mapping[str, Any]
    result: AnalysisResult | QuestionResult = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, InterviewStageRequest):
            raise InterviewContractError("completion requires a bound stage request")
        if not isinstance(self.response, Mapping):
            raise InterviewContractError("completion requires the original provider envelope")
        original = load_interview_json(contract_json(self.response))
        parser = (
            parse_analysis_completion
            if self.request.stage == "analysis"
            else parse_question_completion
        )
        result = parser(self.request, original)
        object.__setattr__(self, "response", _freeze_json(original, "original completion"))
        object.__setattr__(self, "result", result)
        contract_json(self.to_dict())

    @property
    def response_digest(self) -> str:
        return contract_digest(self.response)

    def _body(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-completion/v1",
            "request_digest": self.request.digest,
            "response": _thaw_json(self.response),
            "response_digest": self.response_digest,
            "result": self.result.to_dict(),
        }

    @property
    def digest(self) -> str:
        return contract_digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value: Any, *, request: InterviewStageRequest) -> CompletionRecord:
        data = _versioned(
            value,
            "interview-completion",
            {"request_digest", "response", "response_digest", "result", "digest"},
        )
        result = cls(request, data["response"])
        _same(data, result.to_dict(), "completion record")
        return result

    @classmethod
    def from_json(cls, value: str | bytes, *, request: InterviewStageRequest) -> CompletionRecord:
        return cls.from_dict(load_interview_json(value), request=request)
