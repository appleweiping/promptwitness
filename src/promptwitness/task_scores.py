"""Explicit family-specific local metrics, not a benchmark-equivalence claim."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .sessions import _load

METRIC_CAPABILITIES = {
    "recall": {"primary": "value_recall", "unsupported": []},
    "rag": {"primary": "normalized_exact_match", "unsupported": []},
    "rerank": {"primary": "ndcg_at_k", "unsupported": []},
    "icl": {"primary": "label_exact_match", "unsupported": []},
    "cite": {"primary": None, "unsupported": ["citation_entailment", "answer_correctness"]},
    "longqa": {"primary": None, "unsupported": ["factual_correctness"]},
    "summ": {"primary": None, "unsupported": ["summary_keypoint_coverage"]},
}


def response_text(response: Any) -> str:
    """Accept text or one unambiguous completed text response, never tool calls."""
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        if response.get("error") or response.get("tool_calls") or response.get("function_call"):
            raise ValueError("provider returned an error or a tool call")
        if "status" in response and response["status"] != "completed":
            raise ValueError("provider response is not completed")
        if isinstance(response.get("output_text"), str):
            if (
                response.get("status") == "completed"
                and not response.get("choices")
                and not response.get("output")
            ):
                return str(response["output_text"])
            raise ValueError("output_text requires an unambiguous completed response")
        choices = response.get("choices")
        if isinstance(choices, list) and len(choices) == 1:
            choice = choices[0]
            if isinstance(choice, Mapping) and choice.get("finish_reason") == "stop":
                message = choice.get("message")
                if (
                    isinstance(message, Mapping)
                    and not message.get("tool_calls")
                    and not message.get("function_call")
                    and not message.get("refusal")
                    and message.get("role", "assistant") == "assistant"
                    and isinstance(message.get("content"), str)
                ):
                    return str(message["content"])
    raise ValueError("provider must return one complete text answer")


def _normalize(value: str) -> str:
    value = value.casefold().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def _f1(prediction: str, expected: str) -> float:
    predicted = Counter(_normalize(prediction).split())
    gold = Counter(_normalize(expected).split())
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((predicted & gold).values())
    return 2 * overlap / (sum(predicted.values()) + sum(gold.values()))


def _array(prediction: str) -> list[str] | None:
    try:
        value = _load(prediction.strip())
    except ValueError:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return None
    return value


def _gain_ratio(grade: float, maximum: float) -> float:
    # Scale before discounting, keeping even the smallest subnormal relevance
    # grades out of DCG arithmetic. expm1(x)/x tends to one as x tends to zero.
    def relative_expm1(value: float) -> float:
        return math.expm1(value) / value if value else 1.0

    return (
        (grade / maximum)
        * relative_expm1(math.log(2) * grade)
        / relative_expm1(math.log(2) * maximum)
    )


def score_task(item: Mapping[str, Any], prediction: str) -> dict[str, Any]:
    """Keep primary metrics separate from diagnostics and unsupported judges.

    All exact-match and retrieval metrics are local definitions on [0, 1].
    Citation entailment, long-form factual correctness, and summary coverage
    require a separate calibrated judge and have no surrogate primary score.
    """
    family = item["family"]
    record = item["record"]
    metrics: dict[str, float] = {}
    unsupported: dict[str, str] = {}
    processed: Any = prediction.strip()
    primary: str | None = None
    if family == "recall":
        values = _array(prediction)
        gold = {value.strip().casefold() for value in record["answers"]}
        predicted = {value.strip().casefold() for value in values} if values is not None else set()
        metrics = {
            "value_recall": len(predicted & gold) / len(gold),
            "value_precision": len(predicted & gold) / len(predicted) if predicted else 0.0,
            "format_valid": float(values is not None and len(predicted) == len(values)),
        }
        if not metrics["format_valid"]:
            metrics["value_recall"] = 0.0
        primary = "value_recall"
        processed = values
    elif family in {"rag", "longqa", "summ"}:
        processed = re.sub(r"^answer\s*:\s*", "", prediction.strip(), flags=re.IGNORECASE)
        metrics = {
            "normalized_exact_match": max(
                float(_normalize(processed) == _normalize(answer)) for answer in record["answers"]
            ),
            "normalized_token_f1": max(_f1(processed, answer) for answer in record["answers"]),
        }
        if family == "rag":
            primary = "normalized_exact_match"
        else:
            unsupported[
                "factual_correctness" if family == "longqa" else "summary_keypoint_coverage"
            ] = "model_judge_not_implemented"
    elif family == "rerank":
        ranking = _array(prediction)
        relevance = record["relevance"]
        valid = (
            ranking is not None
            and len(ranking) == len(relevance)
            and set(ranking) == set(relevance)
        )
        k = min(item["ranking_k"], len(relevance))
        ideal = sorted(relevance.values(), reverse=True)[:k]
        maximum = max(ideal)
        ideal_dcg = sum(
            _gain_ratio(grade, maximum) / math.log2(index + 2) for index, grade in enumerate(ideal)
        )
        dcg = (
            sum(
                _gain_ratio(relevance[doc_id], maximum) / math.log2(index + 2)
                for index, doc_id in enumerate(ranking[:k])
            )
            if valid and ranking
            else 0.0
        )
        reciprocal_rank = (
            next(
                (
                    1 / (index + 1)
                    for index, doc_id in enumerate(ranking[:k])
                    if relevance[doc_id] > 0
                ),
                0.0,
            )
            if valid and ranking
            else 0.0
        )
        metrics = {
            "ndcg_at_k": dcg / ideal_dcg,
            "mrr_at_k": reciprocal_rank,
            "ranking_valid": float(valid),
        }
        primary = "ndcg_at_k"
        processed = ranking
    elif family == "icl":
        processed = prediction.strip()
        metrics = {
            "label_exact_match": float(processed == record["answers"][0]),
            "label_valid": float(processed in record["labels"]),
        }
        primary = "label_exact_match"
    elif family == "cite":
        citations = re.findall(r"\[([^\[\]\n]+)\]", prediction)
        known = {doc["id"] for doc in record["documents"]}
        metrics = {
            "citation_id_valid_fraction": sum(value in known for value in citations)
            / len(citations)
            if citations
            else 0.0,
            "has_citations": float(bool(citations)),
        }
        processed = {"answer": prediction, "citation_ids": citations}
        unsupported = {
            "citation_entailment": "model_judge_not_implemented",
            "answer_correctness": "model_judge_not_implemented",
        }
    else:
        raise ValueError(f"unknown task family: {family}")
    return {
        "metrics": metrics,
        "primary_metric": primary,
        "primary_score": metrics[primary] if primary else None,
        "score_status": "scored" if primary else "unsupported",
        "unsupported_metrics": unsupported,
        "processed": processed,
    }
