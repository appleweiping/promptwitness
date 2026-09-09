"""Participant-scoped, immutable memory snapshots and deterministic lexical BM25.

Declared source heads are not authenticated here: the integrating journal must
verify ancestry and membership before constructing these pure snapshots.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .interview_models import (
    MAX_ANSWER_BYTES,
    MAX_CONTRACT_BYTES,
    InterviewContractError,
    MemoryRecord,
    ParticipantAnswer,
    RetrievalPolicy,
    _array,
    _closed,
    _identifier,
    _integer,
    _sha,
    _string,
    _versioned,
    answer_catalog,
    contract_digest,
    contract_json,
    load_interview_json,
)


@dataclass(frozen=True, slots=True)
class SourceHead:
    interview_id: str
    revision: int
    digest: str

    def __post_init__(self) -> None:
        _identifier(self.interview_id, "source interview ID")
        _integer(self.revision, "source head revision", minimum=1)
        _sha(self.digest, "source head digest")

    def to_dict(self) -> dict[str, Any]:
        return {"interview_id": self.interview_id, "revision": self.revision, "digest": self.digest}

    @classmethod
    def from_dict(cls, value: Any) -> SourceHead:
        data = _closed(value, {"interview_id", "revision", "digest"}, "source head")
        return cls(data["interview_id"], data["revision"], data["digest"])


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A declared owning revision, separate from the content-derived memory ID.

    The journal must prove this membership; no self-referential event digest is
    inserted into memory content. Corrections remain distinct visible records.
    """

    memory: MemoryRecord
    interview_id: str
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.memory, MemoryRecord):
            raise InterviewContractError("memory entry requires a validated MemoryRecord")
        _identifier(self.interview_id, "owning interview ID")
        _integer(self.revision, "owning memory revision", minimum=1)
        if any(
            item.answer.interview_id == self.interview_id and item.answer.revision > self.revision
            for item in self.memory.evidence
        ):
            raise InterviewContractError("memory cannot precede its own interview source answer")
        contract_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "interview_id": self.interview_id,
            "revision": self.revision,
            "memory": self.memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any, *, answers: Sequence[ParticipantAnswer]) -> MemoryEntry:
        data = _closed(value, {"interview_id", "revision", "memory"}, "memory entry")
        return cls(
            MemoryRecord.from_dict(data["memory"], answers=answers),
            data["interview_id"],
            data["revision"],
        )


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    participant_id: str
    heads: tuple[SourceHead, ...]
    entries: tuple[MemoryEntry, ...]

    def __post_init__(self) -> None:
        _identifier(self.participant_id, "participant ID")
        heads = _array(self.heads, "source heads", maximum=128)
        entries = _array(self.entries, "memory entries")
        if any(not isinstance(head, SourceHead) for head in heads):
            raise InterviewContractError("snapshot heads require SourceHead values")
        if len({head.interview_id for head in heads}) != len(heads):
            raise InterviewContractError("duplicate source head identity")
        if any(not isinstance(entry, MemoryEntry) for entry in entries):
            raise InterviewContractError("snapshot entries require MemoryEntry values")
        if len({entry.memory.memory_id for entry in entries}) != len(entries):
            raise InterviewContractError("duplicate memory identity")
        by_interview = {head.interview_id: head for head in heads}
        sources: dict[tuple[str, str], ParticipantAnswer] = {}
        for entry in entries:
            if entry.memory.participant_id != self.participant_id:
                raise InterviewContractError("snapshot cannot include another participant's memory")
            head = by_interview.get(entry.interview_id)
            if head is None or entry.revision > head.revision:
                raise InterviewContractError("memory entry is outside the selected source heads")
            for evidence in entry.memory.evidence:
                head = by_interview.get(evidence.answer.interview_id)
                if head is None:
                    raise InterviewContractError(
                        "memory evidence is outside selected source interviews"
                    )
                evidence.reference.validate(
                    evidence.answer, participant_id=self.participant_id, max_revision=head.revision
                )
                key = (evidence.answer.interview_id, evidence.answer.answer_id)
                previous = sources.get(key)
                if previous is not None and previous.digest != evidence.answer.digest:
                    raise InterviewContractError("ambiguous composite answer identity in snapshot")
                sources[key] = evidence.answer
        object.__setattr__(self, "heads", tuple(sorted(heads, key=lambda item: item.interview_id)))
        object.__setattr__(
            self, "entries", tuple(sorted(entries, key=lambda item: item.memory.memory_id))
        )
        contract_json(self.to_dict())

    @property
    def answers(self) -> tuple[ParticipantAnswer, ...]:
        sources = {
            (item.answer.interview_id, item.answer.answer_id): item.answer
            for entry in self.entries
            for item in entry.memory.evidence
        }
        return tuple(sources[key] for key in sorted(sources))

    def _body(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-memory-snapshot/v1",
            "participant_id": self.participant_id,
            "heads": [head.to_dict() for head in self.heads],
            "entries": [entry.to_dict() for entry in self.entries],
            "answers": [answer.to_dict() for answer in self.answers],
        }

    @property
    def digest(self) -> str:
        return contract_digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "digest": self.digest}

    @classmethod
    def build(
        cls, participant_id: str, heads: Sequence[SourceHead], entries: Sequence[MemoryEntry]
    ) -> MemorySnapshot:
        """Filter a caller-supplied catalog before any term counts or ranking.

        Entries created after selected heads or owning/unquoted interviews not
        explicitly selected are excluded, never partially stripped of evidence.
        Selected entries containing future source evidence are invalid.
        """
        _identifier(participant_id, "participant ID")
        normalized_heads = _array(heads, "source heads", maximum=128)
        if any(not isinstance(head, SourceHead) for head in normalized_heads):
            raise InterviewContractError("snapshot heads require SourceHead values")
        by_interview = {head.interview_id: head for head in normalized_heads}
        if len(by_interview) != len(normalized_heads):
            raise InterviewContractError("duplicate source head identity")
        selected = []
        seen = set()
        for entry in _array(entries, "memory catalog"):
            if not isinstance(entry, MemoryEntry):
                raise InterviewContractError("catalog requires MemoryEntry values")
            if entry.memory.memory_id in seen:
                raise InterviewContractError("duplicate catalog memory identity")
            seen.add(entry.memory.memory_id)
            if entry.memory.participant_id != participant_id:
                continue
            head = by_interview.get(entry.interview_id)
            if head is None or entry.revision > head.revision:
                continue
            if any(item.answer.interview_id not in by_interview for item in entry.memory.evidence):
                continue
            selected.append(entry)
        return cls(participant_id, normalized_heads, tuple(selected))

    @classmethod
    def from_dict(cls, value: Any) -> MemorySnapshot:
        data = _versioned(
            value,
            "interview-memory-snapshot",
            {"participant_id", "heads", "entries", "answers", "digest"},
        )
        sources = tuple(
            ParticipantAnswer.from_dict(item)
            for item in _array(data["answers"], "snapshot answers")
        )
        answer_catalog(sources)
        heads = tuple(
            SourceHead.from_dict(item)
            for item in _array(data["heads"], "source heads", maximum=128)
        )
        entries = tuple(
            MemoryEntry.from_dict(item, answers=sources)
            for item in _array(data["entries"], "memory entries")
        )
        result = cls(data["participant_id"], heads, entries)
        if {source.digest for source in sources} != {source.digest for source in result.answers}:
            raise InterviewContractError(
                "snapshot answer catalog contains unused or out-of-scope answers"
            )
        if data["digest"] != result.digest:
            raise InterviewContractError("snapshot content digest does not match")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> MemorySnapshot:
        return cls.from_dict(load_interview_json(value))


def memory_tokens(text: str) -> tuple[str, ...]:
    """Unicode alphanumeric runs after casefold; no NFC or semantic expansion."""
    _string(text, "retrieval text", limit=MAX_CONTRACT_BYTES, empty=True)
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


@dataclass(frozen=True, slots=True)
class MemoryHit:
    memory_id: str
    score: float
    omission: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.memory_id, str)
            or re.fullmatch(r"mem-[0-9a-f]{64}", self.memory_id) is None
        ):
            raise InterviewContractError("retrieval hit requires a memory content ID")
        if (
            type(self.score) not in (float, int)
            or not 0 < self.score <= sys.float_info.max
            or not math.isfinite(self.score)
        ):
            raise InterviewContractError("matched memory scores must be finite and positive")
        object.__setattr__(self, "score", float(self.score))
        if self.omission not in (None, "top_k", "context_bytes"):
            raise InterviewContractError("unsupported retrieval omission reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "score": self.score,
            "selected": self.omission is None,
            "omission": self.omission,
        }


def _context(snapshot_digest: str, entries: Sequence[MemoryEntry]) -> str:
    return contract_json(
        {
            "format": "promptwitness.interview-memory-context/v1",
            "memories": [entry.to_dict() for entry in entries],
            "snapshot_digest": snapshot_digest,
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryRetrieval:
    snapshot: MemorySnapshot
    policy: RetrievalPolicy
    query: str
    hits: tuple[MemoryHit, ...]
    selected: tuple[MemoryEntry, ...]
    context_prefix: str = ""
    context_suffix: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MemorySnapshot) or not isinstance(
            self.policy, RetrievalPolicy
        ):
            raise InterviewContractError("retrieval requires a scoped snapshot and policy")
        _string(self.query, "query", limit=MAX_ANSWER_BYTES, empty=True)
        _string(self.context_prefix, "context prefix", limit=MAX_CONTRACT_BYTES, empty=True)
        _string(self.context_suffix, "context suffix", limit=MAX_CONTRACT_BYTES, empty=True)
        hits = _array(self.hits, "retrieval hits")
        selected = _array(self.selected, "selected memories", maximum=self.policy.top_k)
        if any(not isinstance(hit, MemoryHit) for hit in hits) or any(
            not isinstance(entry, MemoryEntry) for entry in selected
        ):
            raise InterviewContractError("invalid typed retrieval hits/entries")
        if len({hit.memory_id for hit in hits}) != len(hits):
            raise InterviewContractError("duplicate retrieval hit")
        if list(hits) != sorted(hits, key=lambda item: (-item.score, item.memory_id)):
            raise InterviewContractError("retrieval hits must retain score/ID order")
        available = {entry.memory.memory_id: entry for entry in self.snapshot.entries}
        if any(hit.memory_id not in available for hit in hits):
            raise InterviewContractError("retrieval hit is outside its snapshot")
        if tuple(hit.memory_id for hit in hits if hit.omission is None) != tuple(
            entry.memory.memory_id for entry in selected
        ):
            raise InterviewContractError("retrieval selections do not match hit decisions")
        if any(
            contract_digest(entry.to_dict())
            != contract_digest(available[entry.memory.memory_id].to_dict())
            for entry in selected
        ):
            raise InterviewContractError("selected entry differs from its source snapshot")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "selected", selected)
        if self.context_bytes > self.policy.context_bytes:
            raise InterviewContractError(
                "retrieval context exceeds the complete supplied byte budget"
            )
        contract_json(self.to_dict())

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(entry.memory.memory_id for entry in self.selected)

    @property
    def context(self) -> str:
        return (
            self.context_prefix
            + _context(self.snapshot.digest, self.selected)
            + self.context_suffix
        )

    @property
    def context_bytes(self) -> int:
        return len(self.context.encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.interview-memory-retrieval/v1",
            "snapshot_digest": self.snapshot.digest,
            "policy": self.policy.to_dict(),
            "query_digest": hashlib.sha256(self.query.encode("utf-8")).hexdigest(),
            "considered": len(self.snapshot.entries),
            "matched": len(self.hits),
            "selected": len(self.selected),
            "omitted_no_overlap": len(self.snapshot.entries) - len(self.hits),
            "omitted_top_k": sum(hit.omission == "top_k" for hit in self.hits),
            "omitted_context_bytes": sum(hit.omission == "context_bytes" for hit in self.hits),
            "hits": [hit.to_dict() for hit in self.hits],
            "memory_ids": list(self.memory_ids),
            "context": self.context,
            "context_bytes": self.context_bytes,
        }


class BM25MemoryIndex:
    """Frozen scoped corpus statistics; lexical matching, not vector semantics."""

    __slots__ = (
        "_average",
        "_counts",
        "_entry_bytes",
        "_frequency",
        "_lengths",
        "_policy",
        "_snapshot",
    )

    def __init__(self, snapshot: MemorySnapshot, policy: RetrievalPolicy | None = None) -> None:
        if not isinstance(snapshot, MemorySnapshot) or (
            policy is not None and not isinstance(policy, RetrievalPolicy)
        ):
            raise InterviewContractError("BM25 requires a MemorySnapshot and RetrievalPolicy")
        self._snapshot = snapshot
        self._policy = policy if policy is not None else RetrievalPolicy()
        self._counts = []
        self._lengths = []
        self._frequency: Counter[str] = Counter()
        for entry in snapshot.entries:
            pieces = [item.reference.quote for item in entry.memory.evidence]
            if self.policy.include_model_summaries:
                pieces.append(entry.memory.model_summary)
            counts = Counter(memory_tokens("\n".join(pieces)))
            self._counts.append(counts)
            self._lengths.append(sum(counts.values()))
            self._frequency.update(counts.keys())
        self._average = (
            math.fsum(self._lengths) / len(snapshot.entries) if snapshot.entries else 0.0
        )
        self._entry_bytes = [
            len(contract_json(entry.to_dict()).encode("utf-8")) for entry in snapshot.entries
        ]

    @property
    def snapshot(self) -> MemorySnapshot:
        return self._snapshot

    @property
    def policy(self) -> RetrievalPolicy:
        return self._policy

    def search(
        self, query: str, *, context_prefix: str = "", context_suffix: str = ""
    ) -> MemoryRetrieval:
        """Rank positive matches, skip oversized rows, then fill up to top_k.

        The byte limit includes both caller-supplied context strings and the
        complete canonical memory envelope. Nothing is silently truncated.
        """
        _string(query, "query", limit=MAX_ANSWER_BYTES, empty=True)
        _string(context_prefix, "context prefix", limit=MAX_CONTRACT_BYTES, empty=True)
        _string(context_suffix, "context suffix", limit=MAX_CONTRACT_BYTES, empty=True)
        used = len(
            (context_prefix + _context(self.snapshot.digest, ()) + context_suffix).encode("utf-8")
        )
        if used > self.policy.context_bytes:
            raise InterviewContractError("fixed context and empty envelope exceed the byte budget")
        terms = sorted(set(memory_tokens(query)))
        count = len(self.snapshot.entries)
        ranked = []
        if self._average:
            for index, frequencies in enumerate(self._counts):
                contributions = []
                for term in terms:
                    tf = frequencies.get(term, 0)
                    if not tf:
                        continue
                    df = self._frequency[term]
                    idf = math.log1p((count - df + 0.5) / (df + 0.5))
                    denominator = tf + self.policy.k1 * (
                        1 - self.policy.b + self.policy.b * self._lengths[index] / self._average
                    )
                    contributions.append(idf * tf * (self.policy.k1 + 1) / denominator)
                score = math.fsum(contributions)
                if score > 0:
                    ranked.append((score, self.snapshot.entries[index].memory.memory_id, index))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        selected: list[MemoryEntry] = []
        for score, memory_id, index in ranked:
            omission = None
            addition = self._entry_bytes[index] + int(bool(selected))
            if len(selected) >= self.policy.top_k:
                omission = "top_k"
            elif used + addition > self.policy.context_bytes:
                omission = "context_bytes"
            else:
                selected.append(self.snapshot.entries[index])
                used += addition
            hits.append(MemoryHit(memory_id, score, omission))
        result = MemoryRetrieval(
            self.snapshot,
            self.policy,
            query,
            tuple(hits),
            tuple(selected),
            context_prefix,
            context_suffix,
        )
        if result.context_bytes != used:
            raise InterviewContractError("context byte accounting does not match rendered content")
        return result
