"""Research scoring adapters; unavailable execution is never a wrong answer.

Native dependencies are passed explicitly so the zero-dependency public package
does not import benchmark SDKs. The live binding/check is in check_strict_scorers.
Unit fixtures here verify mechanics, not model quality or full scorer admission.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class ScoringError(RuntimeError):
    """Missing execution, invalid annotations or a broken scorer; not score zero."""


class UnsupportedScoring(ScoringError):
    """The requested native category or constraint is not supported."""


@dataclass(frozen=True)
class BinaryScore:
    value: int
    profile: str
    constraint_flags: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value not in (0, 1):
            raise ScoringError("a binary observed score is required")


class Instruction(Protocol):
    def build_description(self, **kwargs: Any) -> Any: ...

    def check_following(self, response: str) -> Any: ...


def completed_text(response: str | None, status: str) -> str:
    """A genuinely completed empty text is observed; a missing text is not."""
    if status != "completed" or not isinstance(response, str):
        raise ScoringError("only a completed text response may be scored")
    return response


def score_hotpotqa(
    response: str | None,
    status: str,
    gold_answer: str,
    exact_match: Callable[[str, str], bool],
) -> BinaryScore:
    """Official answer EM, not supporting-fact or joint EM, F1 or model judging."""
    text = completed_text(response, status)
    if not isinstance(gold_answer, str) or not gold_answer.strip():
        raise ScoringError("a dataset-provided nonempty answer is required")
    try:
        matched = exact_match(text, gold_answer)
    except Exception as exc:
        raise ScoringError("official HotpotQA answer scorer failed") from exc
    if type(matched) is not bool:
        raise ScoringError("official answer EM must return a Boolean")
    return BinaryScore(int(matched), "hotpotqa_official_answer_EM")


def _constraints(
    identifiers: Sequence[str],
    arguments: Sequence[Mapping[str, Any] | None],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not identifiers or isinstance(identifiers, str) or len(identifiers) != len(arguments):
        raise ScoringError("nonempty aligned constraints are required")
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ScoringError("invalid constraint identifier")
    if any(argument is not None and not isinstance(argument, Mapping) for argument in arguments):
        raise ScoringError("constraint arguments must be mappings or None")
    return list(identifiers), [
        copy.deepcopy({key: value for key, value in (argument or {}).items() if value is not None})
        for argument in arguments
    ]


def score_iftrain(
    response: str | None,
    status: str,
    identifiers: Sequence[str],
    arguments: Sequence[Mapping[str, Any] | None],
    registry: Mapping[str, Callable[[str], Instruction]],
) -> BinaryScore:
    """Pinned IFTrain checkers, with charter's all-constraint/raw-text adaptation.

    Unlike upstream IFEvalVerifier, do not remove thinking sections and do not
    average fractional rewards. Registry construction/check failures propagate.
    """
    text = completed_text(response, status)
    ids, kwargs = _constraints(identifiers, arguments)
    if any(identifier not in registry for identifier in ids):
        raise UnsupportedScoring("unknown IFTrain constraint")
    flags = []
    for identifier, argument in zip(ids, kwargs, strict=True):
        try:
            checker = registry[identifier](identifier)
            checker.build_description(**argument)
            # Preserve the native checker's truth test, including empty-text failure.
            flags.append(bool(text.strip() and checker.check_following(text)))
        except Exception as exc:
            raise ScoringError("official IFTrain checker failed") from exc
    return BinaryScore(int(all(flags)), "iftrain_raw_text_all_constraints", tuple(flags))


def score_ifbench(
    response: str | None,
    status: str,
    prompt: str,
    identifiers: Sequence[str],
    arguments: Sequence[Mapping[str, Any] | None],
    strict: Callable[[str, list[str], list[dict[str, Any]], str], Sequence[bool]],
) -> BinaryScore:
    """Call native strict evaluation, including its prompt-dependent reconstruction."""
    text = completed_text(response, status)
    ids, kwargs = _constraints(identifiers, arguments)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ScoringError("the original instruction prompt is required")
    try:
        flags = tuple(strict(prompt, ids, kwargs, text))
    except UnsupportedScoring:
        raise
    except Exception as exc:
        raise ScoringError("official IFBench strict scorer failed") from exc
    if len(flags) != len(ids) or any(type(flag) is not bool for flag in flags):
        raise ScoringError("native strict result has missing or invalid constraint flags")
    return BinaryScore(int(all(flags)), "ifbench_official_strict_prompt_accuracy", flags)


BFCL_CATEGORIES = frozenset(
    {"simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"}
)


def score_bfcl(
    response: str | None,
    status: str,
    category: str,
    functions: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]] | None,
    checker: Callable[
        [list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str], Mapping[str, Any]
    ],
) -> BinaryScore:
    """BFCL-derived Python offline correctness with the fixed name/arguments interface.

    Decode JSON only: never the upstream Python-expression decoder (which uses
    eval). Keep exact function names; native AST checker handles parameters and
    parallel order. Irrelevance follows native absence-of-decodable-call scoring,
    so an invalid JSON answer can pass *irrelevance*, not the other categories.
    This JSON-only decoder is an explicit adaptation shared by every control.
    """
    text = completed_text(response, status)
    if category not in BFCL_CATEGORIES:
        raise UnsupportedScoring("unsupported BFCL offline category")
    if not functions or any(not isinstance(function, dict) for function in functions):
        raise ScoringError("BFCL tool descriptions are required")
    try:
        calls = json.loads(text)
    except json.JSONDecodeError:
        calls = None
    valid_format = isinstance(calls, list) and all(
        isinstance(call, dict)
        and set(call) == {"name", "arguments"}
        and isinstance(call["name"], str)
        and bool(call["name"])
        and isinstance(call["arguments"], dict)
        for call in calls
    )
    if category == "irrelevance":
        return BinaryScore(int(not (valid_format and calls)), "bfcl_json_native_irrelevance")
    if not possible_answers or any(not isinstance(answer, dict) for answer in possible_answers):
        raise ScoringError("BFCL dataset-provided possible answers are required")
    if not valid_format or not calls:
        return BinaryScore(0, "bfcl_json_native_AST")
    decoded = [{call["name"]: call["arguments"]} for call in calls]
    try:
        result = checker(
            copy.deepcopy(functions), decoded, copy.deepcopy(possible_answers), category
        )
    except Exception as exc:
        raise ScoringError("official BFCL AST checker failed") from exc
    if not isinstance(result, Mapping) or type(result.get("valid")) is not bool:
        raise ScoringError("official AST checker did not return a valid Boolean")
    return BinaryScore(int(result["valid"]), "bfcl_json_native_AST")
