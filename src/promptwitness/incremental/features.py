"""Traceable frozen CPU features; no candidate response or audit labels accepted."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

from promptwitness.diff import DiffOptions, MessageAlignment, compare_prompts
from promptwitness.models import ChangeKind, PromptDocument

from .sampling import digest

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
TEXT_DIMENSIONS = 64
FEATURE_VERSION = "delta-v1.1-hash64-unigram-bigram"


@dataclass(frozen=True, slots=True)
class ParentTrace:
    """Only an already observed parent trace, never the candidate's execution."""

    correct: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    allocated_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.correct is not None and (
            type(self.correct) is not int or self.correct not in (0, 1)
        ):
            raise ValueError("parent correctness must be binary or missing")
        for field in ("input_tokens", "output_tokens"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("parent token count must be nonnegative or missing")
        if self.allocated_seconds is not None and (
            isinstance(self.allocated_seconds, bool)
            or not math.isfinite(self.allocated_seconds)
            or self.allocated_seconds < 0
        ):
            raise ValueError("parent cost must be finite, nonnegative or missing")


@dataclass(frozen=True, slots=True)
class FeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    source_map: tuple[tuple[str, str], ...]
    structured: bool
    version: str = FEATURE_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.names, tuple)
            or not isinstance(self.values, tuple)
            or not self.names
        ):
            raise ValueError("immutable nonempty feature vector required")
        if len(self.names) != len(self.values) or len(set(self.names)) != len(self.names):
            raise ValueError("unaligned or duplicate feature names")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in self.values
        ):
            raise ValueError("non-finite or ambiguous features")
        if type(self.structured) is not bool or self.version != FEATURE_VERSION:
            raise ValueError("unsupported frozen feature specification")

    @property
    def sha256(self) -> str:
        return digest(
            {
                "version": self.version,
                "names": self.names,
                "values": self.values,
                "structured": self.structured,
                "source_map": self.source_map,
            }
        )


def _text_features(text: str) -> tuple[float, ...]:
    tokens = _TOKEN.findall(text[:65536])[:8192]
    grams = tokens + [a + "\0" + b for a, b in pairwise(tokens)]
    values = [0.0] * TEXT_DIMENSIONS
    for gram in grams:
        code = hashlib.blake2b(gram.encode("utf-8"), digest_size=8, person=b"pwdelta1").digest()
        number = int.from_bytes(code, "big")
        values[number % TEXT_DIMENSIONS] += 1 if number & 64 else -1
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return tuple(v / norm for v in values)


def extract_features(
    parent: PromptDocument,
    candidate: PromptDocument,
    input_text: str,
    *,
    trace: ParentTrace | None = None,
    structured: bool = True,
) -> FeatureVector:
    """Same frozen text/input/parent features in structure and text-only controls."""
    if not isinstance(input_text, str) or type(structured) is not bool:
        raise ValueError("text input and explicit structured flag required")
    old_text = "\n".join(m.content for m in parent.messages)
    new_text = "\n".join(m.content for m in candidate.messages)
    names: list[str] = []
    values: list[float] = []
    for prefix, text in (
        ("parent_text", old_text),
        ("candidate_text", new_text),
        ("input_text", input_text),
    ):
        names.extend(f"{prefix}.hash.{i}" for i in range(TEXT_DIMENSIONS))
        values.extend(_text_features(text))
    scalars = {
        "parent_characters": math.log1p(len(old_text)),
        "candidate_characters": math.log1p(len(new_text)),
        "input_characters": math.log1p(len(input_text)),
    }
    active = trace or ParentTrace()
    for field in ("correct", "input_tokens", "output_tokens", "allocated_seconds"):
        value = getattr(active, field)
        scalars[f"parent_trace.{field}.missing"] = float(value is None)
        scalars[f"parent_trace.{field}"] = (
            0.0 if value is None else (float(value) if field == "correct" else math.log1p(value))
        )
    names.extend(scalars)
    values.extend(scalars.values())
    source_map: tuple[tuple[str, str], ...] = ()
    if structured:
        alignment = (
            MessageAlignment.ID
            if all(m.message_id for m in (*parent.messages, *candidate.messages))
            else MessageAlignment.POSITIONAL
        )
        report = compare_prompts(
            parent, candidate, DiffOptions(message_alignment=alignment, include_metadata=False)
        )
        counts = Counter(change.kind for change in report.changes)
        source_map = tuple((change.kind.value, change.path) for change in report.changes)
        for kind in ChangeKind:
            if kind == ChangeKind.METADATA:
                continue
            count = float(counts[kind])
            names.extend(
                (
                    f"structure.{kind.value}",
                    f"interaction.{kind.value}.input_length",
                    f"interaction.{kind.value}.parent_correct",
                    f"interaction.{kind.value}.parent_trace_missing",
                )
            )
            values.extend(
                (
                    count,
                    count * scalars["input_characters"],
                    count * (active.correct or 0),
                    count * scalars["parent_trace.correct.missing"],
                )
            )
    return FeatureVector(tuple(names), tuple(values), source_map, structured)
