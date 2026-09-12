"""Exact inert TSV multiset metrics and explicitly declared bullet-trace agreement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .procedure_common_score import ProcedureScoreLimitError, ScoreWork, prefix_fraction, result


def _read_tsv(text: str, work: ScoreWork) -> list[list[str]] | None:
    """A local four-state quoted-field parser, independent of global csv settings."""
    rows: list[list[str]] = []
    row: list[str] = []
    cell: list[str] = []
    cell_bytes = 0
    state = "start"

    def add(char: str) -> None:
        nonlocal cell_bytes
        codepoint = ord(char)
        cell_bytes += (
            1 if codepoint < 128 else 2 if codepoint < 2048 else 3 if codepoint < 65536 else 4
        )
        if cell_bytes > work.limits.max_line_bytes:
            raise ProcedureScoreLimitError("procedural scoring cell byte limit exceeded")
        cell.append(char)

    for char in text:
        if state == "quoted":
            if char == '"':
                state = "closed"
            else:
                add(char)
            continue
        if state == "closed" and char == '"':
            add(char)
            state = "quoted"
            continue
        if char in ("\t", "\n"):
            row.append("".join(cell))
            cell, cell_bytes, state = [], 0, "start"
            if char == "\n":
                rows.append(row)
                row = []
            continue
        if state == "closed" or char == "\r":
            return None
        if state == "start" and char == '"':
            state = "quoted"
        else:
            add(char)
            state = "bare"
    if state == "quoted":
        return None
    if row or cell or state != "start":
        row.append("".join(cell))
        rows.append(row)
    return rows


def _table(text: str, work: ScoreWork, *, fenced: bool) -> list[list[str]] | None:
    text = text.replace("\r\n", "\n")
    if fenced and "```" in text:
        stripped = text.strip()
        if not stripped.startswith("```tsv\n") or not stripped.endswith("\n```"):
            return None
        if stripped.count("```") != 2:
            return None
        text = stripped[7:-4]
    work.lines(text)
    # Upper bound before creating row/cell objects (quoted delimiters may
    # conservatively count extra). The budget is shared with source and output.
    work.use(text.count("\t") + 2 * (text.count("\n") + 1))
    rows = _read_tsv(text, work)
    if not rows or not rows[0]:
        return None
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        return None
    return rows


def score_tsv(
    data: Mapping[str, Any], reference: Mapping[str, Any], prediction: str, work: ScoreWork
) -> dict[str, Any]:
    gold = _table(reference["tsv"], work, fenced=False)
    predicted = _table(prediction, work, fenced=True)
    header = list(data["header"])
    gold_valid = gold is not None and gold[0] == header
    header_valid = predicted is not None and predicted[0] == header
    metrics = {
        "row_precision": 0.0,
        "row_recall": 0.0,
        "row_f1": 0.0,
        "table_exact_match": 0.0,
        "format_valid": float(predicted is not None),
        "header_valid": float(header_valid),
        "reference_valid": float(gold_valid),
    }
    if gold_valid and header_valid and gold is not None and predicted is not None:
        expected = Counter(tuple(row) for row in gold[1:])
        actual = Counter(tuple(row) for row in predicted[1:])
        matches = sum((expected & actual).values())
        n_gold, n_predicted = len(gold) - 1, len(predicted) - 1
        precision = matches / n_predicted if n_predicted else float(not n_gold)
        recall = matches / n_gold if n_gold else 1.0
        metrics.update(
            row_precision=precision,
            row_recall=recall,
            row_f1=2 * matches / (n_gold + n_predicted) if n_gold + n_predicted else 1.0,
            table_exact_match=float(actual == expected),
        )
    return result(
        metrics,
        "row_f1" if gold_valid else None,
        {"header": predicted[0], "rows": predicted[1:]} if predicted is not None else None,
        diagnostics={"failure": None if header_valid else "table_format_or_header"},
        unsupported={} if gold_valid else {"row_f1": "invalid_declared_reference"},
    )


def _bullets(lines: list[str]) -> list[str]:
    # Preserve case, punctuation, negation, prepositions and every semantic
    # token. Only the bullet marker and runs of whitespace are normalized.
    return [" ".join(line.strip()[1:].split()) for line in lines if line.strip().startswith("-")]


def score_tom(reference: Mapping[str, Any], prediction: str, work: ScoreWork) -> dict[str, Any]:
    expected = _bullets(list(reference["trace"]))
    actual = _bullets(work.lines(prediction))
    work.use(len(expected) + len(actual))
    gold_valid = bool(expected) and all(expected)
    valid = bool(actual) and all(actual)
    return result(
        {
            "declared_trace_exact_match": float(valid and actual == expected),
            "declared_trace_prefix_fraction": prefix_fraction(actual, expected)
            if gold_valid and valid
            else 0.0,
            "format_valid": float(valid),
            "reference_valid": float(gold_valid),
        },
        "declared_trace_exact_match" if gold_valid else None,
        actual,
        diagnostics={"scope": "declared_bullet_trace_not_semantic_entailment"},
        unsupported={"belief_entailment": "semantic_judge_not_implemented"}
        | ({} if gold_valid else {"declared_trace_exact_match": "invalid_declared_reference"}),
    )


def score_code(prediction: str) -> dict[str, Any]:
    stripped = prediction.replace("\r\n", "\n").strip()
    fenced = (
        (stripped.startswith("```cpp\n") or stripped.startswith("```c++\n"))
        and stripped.endswith("\n```")
        and stripped.count("```") == 2
    )
    return result(
        {"code_fence_present": float(fenced)},
        None,
        {"code_fence_present": fenced},
        diagnostics={"execution_attempted": False},
        # A metric name, not a password or any other credential.
        unsupported={"executable_test_pass": "isolated_execution_sandbox_not_implemented"},
    )
