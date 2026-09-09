"""Pure, source-bound interview contracts; no scheduler, journal or model calls.

Quotes prove exact source linkage, not factual truth or semantic entailment.
Runtime callers must authenticate supplied participant answers and journal heads.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .models import _freeze_json, _thaw_json
from .sessions import _json, _load, _positive, _text
from .task_data import generation_settings

MAX_CONTRACT_BYTES = 16 * 1024 * 1024
MAX_ANSWER_BYTES = 1024 * 1024
MAX_SOURCES = 10_000
_MAX_INTEGER = 2**63 - 1


class InterviewContractError(ValueError):
    """An interview contract is malformed, ambiguous or not source-bound."""


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int = _MAX_INTEGER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise InterviewContractError(f"{name} must be a bounded integer")
    return value


def _string(value: Any, name: str, *, limit: int = MAX_ANSWER_BYTES, empty: bool = False) -> str:
    try:
        if not isinstance(value, str):
            raise ValueError
        if not empty:
            _text(value, name)
        if len(value.encode("utf-8")) > limit:
            raise ValueError
    except (ValueError, TypeError, UnicodeError):
        raise InterviewContractError(f"{name} must be bounded UTF-8 text") from None
    return value


def _identifier(value: Any, name: str = "identifier") -> str:
    return _string(value, name, limit=512)


def _sha(value: Any, name: str = "digest") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise InterviewContractError(f"{name} must be lowercase SHA-256 hexadecimal")
    return value


def _closed(value: Any, names: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != names:
        raise InterviewContractError(f"invalid closed {name} fields")
    return value


def _versioned(value: Any, name: str, fields: set[str]) -> Mapping[str, Any]:
    data = _closed(value, fields | {"format"}, name)
    if data["format"] != f"promptwitness.{name}/v1":
        raise InterviewContractError(f"unsupported {name} format")
    return data


def _array(
    value: Any, name: str, *, minimum: int = 0, maximum: int = MAX_SOURCES
) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or not minimum <= len(value) <= maximum:
        raise InterviewContractError(f"{name} must be a bounded array")
    return tuple(value)


def _portable(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise InterviewContractError("contract exceeds 32 nesting levels")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        _integer(value, "JSON integer", minimum=-_MAX_INTEGER)
    elif type(value) is float:
        if not math.isfinite(value):
            raise InterviewContractError("contract numbers must be finite")
    elif isinstance(value, str):
        _string(value, "JSON string", limit=MAX_CONTRACT_BYTES, empty=True)
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            _string(key, "JSON key", limit=MAX_CONTRACT_BYTES, empty=True)
            _portable(nested, depth + 1)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _portable(nested, depth + 1)
    else:
        raise InterviewContractError("contract must contain JSON values")


def contract_json(value: Any) -> str:
    """Reuse canonical JSON helpers after applying this contract's stricter bounds."""
    try:
        _portable(value)
        result = _json(value)
        if len(result.encode("utf-8")) > MAX_CONTRACT_BYTES:
            raise InterviewContractError("contract exceeds byte limit")
        return result
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        if isinstance(exc, InterviewContractError):
            raise
        raise InterviewContractError("invalid finite interview JSON") from None


def contract_digest(value: Any) -> str:
    return hashlib.sha256(contract_json(value).encode("utf-8")).hexdigest()


def load_interview_json(raw: str | bytes) -> Any:
    """Reject duplicate keys, non-finite numbers and oversized/ambiguous JSON."""
    try:
        if isinstance(raw, bytes):
            if len(raw) > MAX_CONTRACT_BYTES:
                raise ValueError
            raw = raw.decode("utf-8")
        _string(raw, "contract JSON", limit=MAX_CONTRACT_BYTES, empty=True)
        value = _load(raw)
        contract_json(value)
        return value
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise InterviewContractError("invalid or oversized interview JSON") from None


@dataclass(frozen=True, slots=True)
class InterviewCriterion:
    criterion_id: str
    description: str

    def __post_init__(self) -> None:
        _identifier(self.criterion_id, "criterion ID")
        _string(self.description, "criterion description", limit=16384)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.criterion_id, "description": self.description}

    @classmethod
    def from_dict(cls, value: Any) -> InterviewCriterion:
        data = _closed(value, {"id", "description"}, "criterion")
        return cls(data["id"], data["description"])


@dataclass(frozen=True, slots=True)
class InterviewTopic:
    topic_id: str
    description: str
    criteria: tuple[InterviewCriterion, ...]

    def __post_init__(self) -> None:
        _identifier(self.topic_id, "topic ID")
        _string(self.description, "topic description", limit=16384)
        values = _array(self.criteria, "criteria", minimum=1, maximum=128)
        if any(not isinstance(value, InterviewCriterion) for value in values):
            raise InterviewContractError("criteria require InterviewCriterion values")
        if len({value.criterion_id for value in values}) != len(values):
            raise InterviewContractError("criterion IDs must be unique")
        object.__setattr__(self, "criteria", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.topic_id,
            "description": self.description,
            "criteria": [item.to_dict() for item in self.criteria],
        }

    @classmethod
    def from_dict(cls, value: Any) -> InterviewTopic:
        data = _closed(value, {"id", "description", "criteria"}, "topic")
        values = _array(data["criteria"], "criteria", minimum=1, maximum=128)
        return cls(
            data["id"],
            data["description"],
            tuple(InterviewCriterion.from_dict(item) for item in values),
        )


@dataclass(frozen=True, slots=True)
class InterviewBudgets:
    participant_turns: int = 16
    provider_calls: int = 40
    question_bytes: int = 8192
    answer_bytes: int = 65536
    context_bytes: int = 262144
    followups_per_topic: int = 2
    emergent_topics: int = 1

    def __post_init__(self) -> None:
        for name in ("participant_turns", "provider_calls"):
            _integer(getattr(self, name), name, minimum=1, maximum=10_000)
        for name in ("question_bytes", "answer_bytes"):
            _integer(getattr(self, name), name, minimum=1, maximum=MAX_ANSWER_BYTES)
        _integer(self.context_bytes, "context bytes", minimum=1, maximum=MAX_CONTRACT_BYTES)
        _integer(self.followups_per_topic, "followups", maximum=1000)
        _integer(self.emergent_topics, "emergent topics", maximum=32)

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Any) -> InterviewBudgets:
        return cls(**_closed(value, set(cls.__dataclass_fields__), "budgets"))


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    top_k: int = 8
    context_bytes: int = 16384
    k1: float = 1.2
    b: float = 0.75
    include_model_summaries: bool = True

    def __post_init__(self) -> None:
        _integer(self.top_k, "retrieval top_k", maximum=1000)
        _integer(
            self.context_bytes, "retrieval context bytes", minimum=1, maximum=MAX_CONTRACT_BYTES
        )
        for name, minimum, maximum in (("k1", 0, 10), ("b", 0, 1)):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not minimum <= value <= maximum
                or not math.isfinite(value)
                or (name == "k1" and value == 0)
            ):
                raise InterviewContractError(f"invalid BM25 {name}")
            object.__setattr__(self, name, float(value))
        if type(self.include_model_summaries) is not bool:
            raise InterviewContractError("include_model_summaries must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-retrieval/v1",
            "tokenizer": "unicode-alnum-casefold/v1",
            "top_k": self.top_k,
            "context_bytes": self.context_bytes,
            "k1": self.k1,
            "b": self.b,
            "include_model_summaries": self.include_model_summaries,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RetrievalPolicy:
        data = _versioned(
            value,
            "interview-retrieval",
            {"tokenizer", "top_k", "context_bytes", "k1", "b", "include_model_summaries"},
        )
        if data["tokenizer"] != "unicode-alnum-casefold/v1":
            raise InterviewContractError("unsupported retrieval tokenizer")
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class InterviewPlan:
    plan_id: str
    version: str
    purpose: str
    topics: tuple[InterviewTopic, ...]
    budgets: InterviewBudgets = field(default_factory=InterviewBudgets)
    retrieval: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    analyst_prompt_version: str = "analysis/v1"
    questioner_prompt_version: str = "question/v1"
    analyst_generation: Mapping[str, Any] = field(default_factory=lambda: {"max_tokens": 512})
    questioner_generation: Mapping[str, Any] = field(default_factory=lambda: {"max_tokens": 512})

    def __post_init__(self) -> None:
        for name in ("plan_id", "version", "analyst_prompt_version", "questioner_prompt_version"):
            _identifier(getattr(self, name), name)
        _string(self.purpose, "plan purpose", limit=65536)
        values = _array(self.topics, "topics", minimum=1, maximum=128)
        if any(not isinstance(item, InterviewTopic) for item in values):
            raise InterviewContractError("topics require InterviewTopic values")
        ids = [criterion.criterion_id for topic in values for criterion in topic.criteria]
        if (
            len({topic.topic_id for topic in values}) != len(values)
            or len(ids) != len(set(ids))
            or len(ids) > 4096
        ):
            raise InterviewContractError("plan topic and criterion IDs must be unique and bounded")
        if not isinstance(self.budgets, InterviewBudgets) or not isinstance(
            self.retrieval, RetrievalPolicy
        ):
            raise InterviewContractError("plan requires typed budgets and retrieval policy")
        if self.retrieval.context_bytes > self.budgets.context_bytes:
            raise InterviewContractError("retrieval budget cannot exceed plan context budget")
        object.__setattr__(self, "topics", values)
        for name in ("analyst_generation", "questioner_generation"):
            try:
                generation = generation_settings(getattr(self, name))
                _positive(generation["max_tokens"], "max_tokens")
                _integer(generation["max_tokens"], "max_tokens", minimum=1, maximum=1_000_000)
                contract_json(generation)
            except (ValueError, TypeError, OverflowError):
                raise InterviewContractError("invalid nonsecret generation settings") from None
            object.__setattr__(self, name, _freeze_json(generation, name))
        contract_json(self.to_dict())

    @property
    def required_criteria(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (topic.topic_id, item.criterion_id) for topic in self.topics for item in topic.criteria
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-plan/v1",
            "id": self.plan_id,
            "version": self.version,
            "purpose": self.purpose,
            "topics": [item.to_dict() for item in self.topics],
            "budgets": self.budgets.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "analyst_prompt_version": self.analyst_prompt_version,
            "questioner_prompt_version": self.questioner_prompt_version,
            "analyst_generation": _thaw_json(self.analyst_generation),
            "questioner_generation": _thaw_json(self.questioner_generation),
        }

    @property
    def digest(self) -> str:
        return contract_digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> InterviewPlan:
        data = _versioned(
            value,
            "interview-plan",
            {
                "id",
                "version",
                "purpose",
                "topics",
                "budgets",
                "retrieval",
                "analyst_prompt_version",
                "questioner_prompt_version",
                "analyst_generation",
                "questioner_generation",
            },
        )
        topics = _array(data["topics"], "topics", minimum=1, maximum=128)
        return cls(
            data["id"],
            data["version"],
            data["purpose"],
            tuple(InterviewTopic.from_dict(item) for item in topics),
            InterviewBudgets.from_dict(data["budgets"]),
            RetrievalPolicy.from_dict(data["retrieval"]),
            data["analyst_prompt_version"],
            data["questioner_prompt_version"],
            data["analyst_generation"],
            data["questioner_generation"],
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> InterviewPlan:
        return cls.from_dict(load_interview_json(value))


@dataclass(frozen=True, slots=True)
class ParticipantAnswer:
    interview_id: str
    participant_id: str
    answer_id: str
    question_id: str
    text: str
    revision: int

    def __post_init__(self) -> None:
        for name in ("interview_id", "participant_id", "answer_id", "question_id"):
            _identifier(getattr(self, name), name)
        _string(self.text, "answer text", empty=True)
        _integer(self.revision, "answer revision", minimum=1)
        contract_json(self.to_dict())

    @property
    def text_digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def digest(self) -> str:
        return contract_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-answer/v1",
            "source_kind": "participant_answer",
            "interview_id": self.interview_id,
            "participant_id": self.participant_id,
            "answer_id": self.answer_id,
            "question_id": self.question_id,
            "text": self.text,
            "text_digest": self.text_digest,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ParticipantAnswer:
        data = _versioned(
            value,
            "interview-answer",
            {
                "source_kind",
                "interview_id",
                "participant_id",
                "answer_id",
                "question_id",
                "text",
                "text_digest",
                "revision",
            },
        )
        if data["source_kind"] != "participant_answer":
            raise InterviewContractError("only participant answers can supply interview evidence")
        result = cls(*(data[name] for name in cls.__dataclass_fields__))
        if data["text_digest"] != result.text_digest:
            raise InterviewContractError("answer text digest does not match")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> ParticipantAnswer:
        return cls.from_dict(load_interview_json(value))


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    interview_id: str
    participant_id: str
    answer_id: str
    answer_digest: str
    answer_revision: int
    start: int
    end: int
    quote: str

    def __post_init__(self) -> None:
        for name in ("interview_id", "participant_id", "answer_id"):
            _identifier(getattr(self, name), name)
        _sha(self.answer_digest, "answer digest")
        _integer(self.answer_revision, "answer revision", minimum=1)
        _integer(self.start, "evidence start", maximum=MAX_ANSWER_BYTES)
        _integer(self.end, "evidence end", minimum=self.start + 1, maximum=MAX_ANSWER_BYTES)
        _string(self.quote, "quoted evidence", empty=True)
        if len(self.quote) != self.end - self.start:
            raise InterviewContractError("quoted evidence length must match codepoint interval")
        contract_json(self.to_dict())

    def _body(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-evidence/v1",
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }

    @property
    def evidence_id(self) -> str:
        return "ev-" + contract_digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "id": self.evidence_id}

    def validate(
        self,
        answer: ParticipantAnswer,
        *,
        participant_id: str | None = None,
        max_revision: int | None = None,
    ) -> None:
        if not isinstance(answer, ParticipantAnswer):
            raise InterviewContractError("evidence requires an existing participant answer")
        if participant_id is not None:
            _identifier(participant_id, "participant ID")
            if self.participant_id != participant_id:
                raise InterviewContractError("evidence belongs to another participant")
        if max_revision is not None:
            _integer(max_revision, "source head revision", minimum=1)
            if self.answer_revision > max_revision:
                raise InterviewContractError("evidence is newer than the pinned source head")
        if (
            self.interview_id,
            self.participant_id,
            self.answer_id,
            self.answer_digest,
            self.answer_revision,
        ) != (
            answer.interview_id,
            answer.participant_id,
            answer.answer_id,
            answer.digest,
            answer.revision,
        ):
            raise InterviewContractError("evidence source identity does not match its answer")
        if self.end > len(answer.text) or answer.text[self.start : self.end] != self.quote:
            raise InterviewContractError("evidence quote is not the exact source codepoint slice")

    @classmethod
    def from_answer(cls, answer: ParticipantAnswer, start: int, end: int) -> EvidenceRef:
        if not isinstance(answer, ParticipantAnswer):
            raise InterviewContractError("evidence requires a participant answer")
        _integer(start, "start", maximum=len(answer.text))
        _integer(end, "end", minimum=start + 1, maximum=len(answer.text))
        return cls(
            answer.interview_id,
            answer.participant_id,
            answer.answer_id,
            answer.digest,
            answer.revision,
            start,
            end,
            answer.text[start:end],
        )

    @classmethod
    def from_dict(cls, value: Any) -> EvidenceRef:
        data = _versioned(value, "interview-evidence", set(cls.__dataclass_fields__) | {"id"})
        result = cls(*(data[name] for name in cls.__dataclass_fields__))
        if data["id"] != result.evidence_id:
            raise InterviewContractError("evidence content identity does not match")
        return result


@dataclass(frozen=True, slots=True)
class BoundEvidence:
    reference: EvidenceRef
    answer: ParticipantAnswer

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvidenceRef):
            raise InterviewContractError("bound evidence requires an EvidenceRef")
        self.reference.validate(self.answer)

    @property
    def evidence_id(self) -> str:
        return self.reference.evidence_id

    def to_dict(self) -> dict[str, Any]:
        return self.reference.to_dict()

    @classmethod
    def from_answer(cls, answer: ParticipantAnswer, start: int, end: int) -> BoundEvidence:
        return cls(EvidenceRef.from_answer(answer, start, end), answer)


def answer_catalog(
    answers: Sequence[ParticipantAnswer],
) -> dict[tuple[str, str], ParticipantAnswer]:
    result = {}
    for answer in _array(answers, "answer catalog"):
        if not isinstance(answer, ParticipantAnswer):
            raise InterviewContractError("answer catalog requires participant answers")
        key = (answer.interview_id, answer.answer_id)
        if key in result:
            raise InterviewContractError("duplicate or ambiguous composite answer identity")
        result[key] = answer
    return result


def bind_evidence(values: Any, answers: Sequence[ParticipantAnswer]) -> tuple[BoundEvidence, ...]:
    catalog = answer_catalog(answers)
    result = []
    for value in _array(values, "evidence", maximum=128):
        reference = EvidenceRef.from_dict(value)
        source = catalog.get((reference.interview_id, reference.answer_id))
        if source is None:
            raise InterviewContractError("evidence answer is missing from the supplied catalog")
        result.append(BoundEvidence(reference, source))
    return tuple(result)


def _consistent_sources(values: Iterable[BoundEvidence]) -> None:
    sources: dict[tuple[str, str], str] = {}
    for value in values:
        answer = value.answer
        key = (answer.interview_id, answer.answer_id)
        digest = answer.digest
        if key in sources and sources[key] != digest:
            raise InterviewContractError("conflicting answers share a composite source identity")
        sources[key] = digest


def _bound(
    values: Any, participant_id: str, *, interview_id: str | None = None, required: bool = True
) -> tuple[BoundEvidence, ...]:
    result = _array(values, "bound evidence", minimum=int(required), maximum=128)
    if any(not isinstance(value, BoundEvidence) for value in result):
        raise InterviewContractError("evidence must be validated BoundEvidence values")
    for value in result:
        value.reference.validate(value.answer, participant_id=participant_id)
        if interview_id is not None and value.answer.interview_id != interview_id:
            raise InterviewContractError("assessment evidence belongs to another interview")
    _consistent_sources(result)
    if len({value.evidence_id for value in result}) != len(result):
        raise InterviewContractError("duplicate evidence reference IDs")
    return tuple(sorted(result, key=lambda item: item.evidence_id))


@dataclass(frozen=True, slots=True, order=True)
class CriterionLink:
    topic_id: str
    criterion_id: str

    def __post_init__(self) -> None:
        _identifier(self.topic_id, "topic ID")
        _identifier(self.criterion_id, "criterion ID")

    def to_dict(self) -> dict[str, str]:
        return {"topic_id": self.topic_id, "criterion_id": self.criterion_id}

    @classmethod
    def from_dict(cls, value: Any) -> CriterionLink:
        data = _closed(value, {"topic_id", "criterion_id"}, "criterion link")
        return cls(data["topic_id"], data["criterion_id"])


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    participant_id: str
    evidence: tuple[BoundEvidence, ...]
    links: tuple[CriterionLink, ...]
    model_summary: str = ""
    supersedes: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.participant_id, "participant ID")
        object.__setattr__(self, "evidence", _bound(self.evidence, self.participant_id))
        links = _array(self.links, "memory links", minimum=1, maximum=128)
        if any(not isinstance(item, CriterionLink) for item in links) or len(set(links)) != len(
            links
        ):
            raise InterviewContractError("memory links must be unique CriterionLink values")
        object.__setattr__(self, "links", tuple(sorted(links)))
        _string(self.model_summary, "model-proposed summary", limit=32768, empty=True)
        if self.supersedes is not None:
            if (
                not isinstance(self.supersedes, str)
                or re.fullmatch(r"mem-[0-9a-f]{64}", self.supersedes) is None
            ):
                raise InterviewContractError("supersedes must be a content-derived memory ID")
            if self.supersedes == self.memory_id:
                raise InterviewContractError("memory cannot supersede itself")
        contract_json(self.to_dict())

    def _body(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-memory/v1",
            "participant_id": self.participant_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "links": [item.to_dict() for item in self.links],
            "summary": {"kind": "model_proposed", "text": self.model_summary},
            "supersedes": self.supersedes,
        }

    @property
    def memory_id(self) -> str:
        return "mem-" + contract_digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "id": self.memory_id}

    @classmethod
    def from_dict(cls, value: Any, *, answers: Sequence[ParticipantAnswer]) -> MemoryRecord:
        data = _versioned(
            value,
            "interview-memory",
            {"id", "participant_id", "evidence", "links", "summary", "supersedes"},
        )
        summary = _closed(data["summary"], {"kind", "text"}, "memory summary")
        if summary["kind"] != "model_proposed":
            raise InterviewContractError("a model summary must remain explicitly labelled")
        if data["supersedes"] == data["id"]:
            raise InterviewContractError("memory cannot supersede itself")
        links = tuple(
            CriterionLink.from_dict(item)
            for item in _array(data["links"], "memory links", minimum=1, maximum=128)
        )
        result = cls(
            data["participant_id"],
            bind_evidence(data["evidence"], answers),
            links,
            summary["text"],
            data["supersedes"],
        )
        if data["id"] != result.memory_id:
            raise InterviewContractError("memory content identity does not match")
        return result

    @classmethod
    def from_json(cls, value: str | bytes, *, answers: Sequence[ParticipantAnswer]) -> MemoryRecord:
        return cls.from_dict(load_interview_json(value), answers=answers)


@dataclass(frozen=True, slots=True)
class CriterionAssessment:
    participant_id: str
    interview_id: str
    link: CriterionLink
    status: str
    evidence: tuple[BoundEvidence, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        _identifier(self.participant_id, "participant ID")
        _identifier(self.interview_id, "interview ID")
        if not isinstance(self.link, CriterionLink) or self.status not in (
            "covered",
            "partial",
            "unanswered",
        ):
            raise InterviewContractError("invalid criterion assessment")
        object.__setattr__(
            self,
            "evidence",
            _bound(
                self.evidence,
                self.participant_id,
                interview_id=self.interview_id,
                required=self.status != "unanswered",
            ),
        )
        _string(self.rationale, "assessment rationale", limit=16384, empty=True)
        contract_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.criterion-assessment/v1",
            "participant_id": self.participant_id,
            "interview_id": self.interview_id,
            "link": self.link.to_dict(),
            "status": self.status,
            "evidence": [item.to_dict() for item in self.evidence],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: Any, *, answers: Sequence[ParticipantAnswer]) -> CriterionAssessment:
        data = _versioned(
            value,
            "criterion-assessment",
            {"participant_id", "interview_id", "link", "status", "evidence", "rationale"},
        )
        return cls(
            data["participant_id"],
            data["interview_id"],
            CriterionLink.from_dict(data["link"]),
            data["status"],
            bind_evidence(data["evidence"], answers),
            data["rationale"],
        )


@dataclass(frozen=True, slots=True)
class AssessedCoverage:
    required_total: int
    required_covered: int
    required_partial: int
    emergent_total: int
    emergent_covered: int
    emergent_partial: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), name, maximum=8192)
        if self.required_total < 1:
            raise InterviewContractError("required coverage must retain the original denominator")
        for prefix in ("required", "emergent"):
            if getattr(self, prefix + "_covered") + getattr(self, prefix + "_partial") > getattr(
                self, prefix + "_total"
            ):
                raise InterviewContractError("coverage counts exceed their denominator")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"format": "promptwitness.assessed-coverage/v1"}
        for prefix in ("required", "emergent"):
            total = getattr(self, prefix + "_total")
            covered = getattr(self, prefix + "_covered")
            partial = getattr(self, prefix + "_partial")
            result[prefix] = {
                "total": total,
                "covered": covered,
                "partial": partial,
                "unanswered": total - covered - partial,
                "assessed_coverage": covered / total if total else None,
            }
        return result


def assessed_coverage(
    plan: InterviewPlan,
    assessments: Sequence[CriterionAssessment],
    *,
    accepted_emergent: Sequence[InterviewTopic] = (),
) -> AssessedCoverage:
    """Pure report; explicit decisions do not imply factual accuracy or completion."""
    if not isinstance(plan, InterviewPlan):
        raise InterviewContractError("coverage requires an original InterviewPlan")
    emergent = _array(
        accepted_emergent, "accepted emergent topics", maximum=plan.budgets.emergent_topics
    )
    if any(not isinstance(topic, InterviewTopic) for topic in emergent):
        raise InterviewContractError("accepted emergent topics require InterviewTopic values")
    topic_ids = [topic.topic_id for topic in (*plan.topics, *emergent)]
    required = {CriterionLink(*pair) for pair in plan.required_criteria}
    extra = {
        CriterionLink(topic.topic_id, criterion.criterion_id)
        for topic in emergent
        for criterion in topic.criteria
    }
    all_criteria = [
        criterion.criterion_id
        for topic in (*plan.topics, *emergent)
        for criterion in topic.criteria
    ]
    if len(topic_ids) != len(set(topic_ids)) or len(all_criteria) != len(set(all_criteria)):
        raise InterviewContractError("emergent topics/criteria cannot change original identities")
    rows = _array(assessments, "criterion assessments", maximum=8192)
    if any(not isinstance(item, CriterionAssessment) for item in rows):
        raise InterviewContractError("coverage requires validated CriterionAssessment values")
    if len({(item.participant_id, item.interview_id) for item in rows}) > 1:
        raise InterviewContractError("coverage cannot merge participants or interviews")
    if len({item.link for item in rows}) != len(rows):
        raise InterviewContractError("duplicate criterion decisions")
    if any(item.link not in required | extra for item in rows):
        raise InterviewContractError("assessment refers to an unknown or unaccepted criterion")
    _consistent_sources(evidence for item in rows for evidence in item.evidence)
    return AssessedCoverage(
        len(required),
        sum(item.link in required and item.status == "covered" for item in rows),
        sum(item.link in required and item.status == "partial" for item in rows),
        len(extra),
        sum(item.link in extra and item.status == "covered" for item in rows),
        sum(item.link in extra and item.status == "partial" for item in rows),
    )
