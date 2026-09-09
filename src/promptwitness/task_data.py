"""Local task-suite ingestion and deterministic, gold-free prompt planning."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import _freeze_json, _thaw_json
from .sessions import _json, _load, _positive, _text

FAMILIES = ("recall", "rag", "rerank", "cite", "longqa", "summ", "icl")
_INSTRUCTIONS = {
    "recall": "Retrieve requested values from the context. Return only a JSON array of strings.",
    "rag": "Answer the question using the supplied documents. Return only the concise answer.",
    "rerank": "Rank all document IDs by relevance to the query. Return only a JSON array of IDs.",
    "cite": "Answer from the documents and cite supporting document IDs in square brackets.",
    "longqa": "Read the complete context and answer the question using its evidence.",
    "summ": "Summarize the supplied context in response to the request, preserving its facts.",
    "icl": "Infer the classification from the labeled examples. Return only one allowed label.",
}


def digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def generation_settings(raw: Any) -> dict[str, Any]:
    """Validate only supported generation fields; never allow request overrides."""
    if not isinstance(raw, Mapping) or set(raw) - {"max_tokens", "temperature", "top_p", "seed"}:
        raise ValueError("generation supports only max_tokens, temperature, top_p, and seed")
    result = dict(raw)
    _positive(result.get("max_tokens"), "generation.max_tokens")
    for name, upper in (("temperature", 2), ("top_p", 1)):
        if name in result:
            value = result[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= upper
            ):
                raise ValueError(f"generation.{name} must be between 0 and {upper}")
    if "seed" in result and (
        isinstance(result["seed"], bool) or not isinstance(result["seed"], int)
    ):
        raise ValueError("generation.seed must be an integer")
    return result


def _strings(raw: Any, name: str, *, empty: bool = False) -> list[str]:
    if not isinstance(raw, (list, tuple)) or (not raw and not empty):
        raise ValueError(f"{name} must be an array" + ("" if empty else " with at least one entry"))
    return [_text(item, name) for item in raw]


def _fields(raw: Any, allowed: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) - allowed:
        raise ValueError(f"invalid {name} fields")
    return raw


def normalize_record(raw: Any, family: str, adapter: str) -> dict[str, Any]:
    """Adapt local records without downloading datasets or importing their code."""
    if family not in FAMILIES:
        raise ValueError("unsupported task family")
    if not isinstance(raw, Mapping):
        raise ValueError("task record must be an object")
    if adapter == "kilt":
        if family != "rag":
            raise ValueError("kilt adapter requires the rag family")
        raw = {
            "id": raw.get("id"),
            "question": raw.get("question"),
            "answers": raw.get("answers"),
            "documents": [
                {"id": str(index), "text": item["text"], "title": item.get("title", "")}
                for index, item in enumerate(raw.get("ctxs", []), 1)
            ],
        }
    elif adapter == "msmarco":
        if family != "rerank":
            raise ValueError("msmarco adapter requires the rerank family")
        raw = {
            "id": raw.get("qid"),
            "question": raw.get("query"),
            "documents": [
                {"id": item["id"], "text": item["text"], "title": item.get("title", "")}
                for item in raw.get("ctxs", [])
            ],
            "relevance": {item["id"]: item["label"] for item in raw.get("ctxs", [])},
        }
    elif adapter == "processed":
        if family not in {"recall", "rag", "longqa", "summ"}:
            raise ValueError("processed adapter supports recall, rag, longqa, and summ")
        answers = raw.get("answer")
        raw = {
            "id": raw.get("id"),
            "question": raw.get("question"),
            "context": raw.get("context"),
            "answers": [answers] if isinstance(answers, str) else answers,
        }
    elif adapter != "native":
        raise ValueError(f"unsupported task adapter: {adapter}")
    _fields(
        raw,
        {"id", "question", "context", "documents", "answers", "relevance", "examples", "labels"},
        "record",
    )
    result: dict[str, Any] = {
        "id": _text(raw.get("id"), "record.id"),
        "question": _text(raw.get("question"), "record.question"),
    }
    documents = raw.get("documents", [])
    if not isinstance(documents, (list, tuple)):
        raise ValueError("documents must be an array")
    docs = []
    for item in documents:
        _fields(item, {"id", "text", "title"}, "document")
        title = item.get("title", "")
        if not isinstance(title, str):
            raise ValueError("document.title must be a string")
        docs.append(
            {
                "id": _text(item.get("id"), "document.id"),
                "text": _text(item.get("text"), "document.text"),
                "title": title,
            }
        )
    if len({item["id"] for item in docs}) != len(docs):
        raise ValueError("document IDs must be unique")
    context = raw.get("context", "")
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    if context and docs:
        raise ValueError("supply context or documents, not both")
    result.update(context=context, documents=docs)
    if family != "icl" and not (context.strip() or docs):
        raise ValueError("task needs non-empty context or documents")
    if family in {"rerank", "cite"} and not docs:
        raise ValueError(f"{family} requires documents with IDs")
    result["answers"] = _strings(
        raw.get("answers", []), "answers", empty=family in {"cite", "rerank"}
    )
    if family == "rerank":
        relevance = raw.get("relevance")
        if not isinstance(relevance, Mapping) or set(relevance) != {item["id"] for item in docs}:
            raise ValueError("relevance must label every document exactly once")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 4
            for v in relevance.values()
        ):
            raise ValueError("relevance grades must be finite numbers from 0 to 4")
        if not any(relevance.values()):
            raise ValueError("ranking task needs at least one relevant document")
        result["relevance"] = dict(relevance)
    elif "relevance" in raw:
        raise ValueError("relevance is only valid for rerank")
    if family == "icl":
        if context or docs:
            raise ValueError("ICL context must be represented as labeled examples")
        labels = _strings(raw.get("labels"), "labels")
        if (
            len(set(labels)) != len(labels)
            or len(result["answers"]) != 1
            or result["answers"][0] not in labels
        ):
            raise ValueError("ICL requires unique labels and one answer from those labels")
        examples = raw.get("examples")
        if not isinstance(examples, (list, tuple)) or not examples:
            raise ValueError("ICL requires labeled examples")
        normalized = []
        for item in examples:
            _fields(item, {"id", "text", "label"}, "example")
            example = {k: _text(item.get(k), f"example.{k}") for k in ("id", "text", "label")}
            if example["id"] == result["id"] or example["text"] == result["question"]:
                raise ValueError("ICL test record cannot be its own example")
            if example["label"] not in labels:
                raise ValueError("example label must be an allowed label")
            normalized.append(example)
        if len({item["id"] for item in normalized}) != len(normalized):
            raise ValueError("example IDs must be unique")
        result.update(labels=labels, examples=normalized)
    elif "examples" in raw or "labels" in raw:
        raise ValueError("examples and labels are only valid for icl")
    return result


def _messages(record: Mapping[str, Any], family: str, seed: str) -> list[dict[str, str]]:
    context = record["context"]
    if record["documents"]:
        context = "\n".join(_json(item) for item in record["documents"])
    if family == "icl":
        ordered = sorted(
            record["examples"], key=lambda item: digest([seed, record["id"], item["id"]])
        )
        context = (
            "Allowed labels: "
            + _json(record["labels"])
            + "\nExamples:\n"
            + "\n".join(_json({"text": item["text"], "label": item["label"]}) for item in ordered)
        )
    return [
        {"role": "system", "content": _INSTRUCTIONS[family]},
        {"role": "user", "content": f"Context:\n{context}\n\nRequest:\n{record['question']}"},
    ]


@dataclass(frozen=True, slots=True)
class TaskSuitePlan:
    """Immutable materialized inputs/configuration, including gold only off wire."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_plan(self.payload)
        object.__setattr__(self, "payload", _freeze_json(self.payload, "task plan"))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = _thaw_json(self.payload)
        return value

    @property
    def digest(self) -> str:
        return digest(self.payload)


def _validate_plan(raw: Any) -> None:
    """Revalidate public construction: only generated gold-free prompts can run."""
    expected = {
        "format",
        "implementation",
        "suite_id",
        "manifest_sha256",
        "seed",
        "budgets",
        "counter",
        "generation",
        "tasks",
        "items",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("invalid task plan fields")
    if (
        raw["format"] != "promptwitness.task-plan/v1"
        or raw["implementation"] != "task-prompts-scores/v1"
    ):
        raise ValueError("unsupported task plan implementation")
    for name in ("suite_id", "seed", "manifest_sha256"):
        _text(raw[name], name)
    budgets = raw["budgets"]
    if not isinstance(budgets, (list, tuple)) or not budgets:
        raise ValueError("plan budgets must be a non-empty array")
    for budget in budgets:
        _positive(budget, "input budget")
    if len(set(budgets)) != len(budgets):
        raise ValueError("plan input budgets must be unique")
    generation_settings(raw["generation"])
    identity = raw["counter"]
    if not isinstance(identity, Mapping):
        raise ValueError("plan counter identity must be an object")
    for name in ("unit", "implementation"):
        _text(identity.get(name), f"counter.{name}")
    tasks, items = raw["tasks"], raw["items"]
    if (
        not isinstance(tasks, (list, tuple))
        or not tasks
        or not isinstance(items, (list, tuple))
        or not items
    ):
        raise ValueError("plan tasks and items must be non-empty arrays")
    index = {}
    for task in tasks:
        if not isinstance(task, Mapping) or set(task) != {
            "id",
            "family",
            "adapter",
            "input_sha256",
            "available",
            "selected",
            "sampled_out",
        }:
            raise ValueError("invalid planned task fields")
        name = _text(task["id"], "task.id")
        if name in index or task["family"] not in FAMILIES:
            raise ValueError("plan task IDs must be unique and families supported")
        _text(task["input_sha256"], "input_sha256")
        _positive(task["available"], "available records")
        _positive(task["selected"], "selected records")
        sampled_out = task["sampled_out"]
        if (
            isinstance(sampled_out, bool)
            or not isinstance(sampled_out, int)
            or sampled_out < 0
            or task["available"] != task["selected"] + sampled_out
        ):
            raise ValueError("invalid plan sample accounting")
        if task["adapter"] not in {"native", "kilt", "msmarco", "processed"}:
            raise ValueError("unsupported planned adapter")
        index[name] = task
    seen: set[str] = set()
    record_budgets: dict[tuple[str, str], set[int]] = {}
    record_digests: dict[tuple[str, str], str] = {}
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {
            "id",
            "task_id",
            "record_id",
            "family",
            "budget",
            "input_units",
            "messages",
            "record",
            "ranking_k",
            "skip_reason",
        }:
            raise ValueError("invalid planned item fields")
        _positive(item["budget"], "planned item budget")
        task = index.get(item["task_id"])
        if task is None or item["family"] != task["family"] or item["budget"] not in budgets:
            raise ValueError("item does not belong to its planned task/budget")
        normalized = normalize_record(item["record"], item["family"], "native")
        if _json(normalized) != _json(item["record"]) or normalized["id"] != item["record_id"]:
            raise ValueError("planned record is not canonical")
        if _json(item["messages"]) != _json(_messages(normalized, item["family"], raw["seed"])):
            raise ValueError("planned messages differ from the gold-free task prompt")
        _positive(item["input_units"], "input units")
        _positive(item["ranking_k"], "ranking_k")
        if identity == {
            "unit": "utf8_bytes",
            "implementation": "canonical-messages-json/v1",
        } and item["input_units"] != len(_json(item["messages"]).encode("utf-8")):
            raise ValueError("planned byte count does not match full messages")
        if item["skip_reason"] != (
            "input_budget_exceeded" if item["input_units"] > item["budget"] else None
        ):
            raise ValueError("planned skip status does not match input budget")
        item_id = digest([item["task_id"], item["record_id"], item["budget"]])
        if item["id"] != item_id or item_id in seen:
            raise ValueError("planned item identity must be correct and unique")
        seen.add(item_id)
        key = (item["task_id"], item["record_id"])
        record_budgets.setdefault(key, set()).add(item["budget"])
        signature = digest([item["record"], item["input_units"], item["ranking_k"]])
        if key in record_digests and record_digests[key] != signature:
            raise ValueError("same record changed between input budgets")
        record_digests[key] = signature
    if any(value != set(budgets) for value in record_budgets.values()):
        raise ValueError("each selected record needs every input budget")
    if any(
        sum(key[0] == name for key in record_budgets) != task["selected"]
        for name, task in index.items()
    ):
        raise ValueError("planned items differ from selected record accounting")


def load_task_suite(
    path: str | Path,
    *,
    counter: Callable[[Sequence[Mapping[str, str]]], int] | None = None,
    counter_identity: Mapping[str, Any] | None = None,
) -> TaskSuitePlan:
    """Plan local JSON/JSONL tasks at each input budget without truncating evidence.

    The default counts the UTF-8 bytes of the entire canonical messages JSON,
    not model tokens. A custom counter must count the full chat prompt including
    framing and identify its tokenizer/template revision. Generation limits are
    separate and are sent to the provider, never silently subtracted from bytes.
    """
    source = Path(path).resolve()
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("suite manifest exceeds 1 MiB")
    content = source.read_bytes()
    manifest = _load(content.decode("utf-8"))
    _fields(
        manifest, {"format", "suite_id", "budgets", "tasks", "seed", "generation"}, "task suite"
    )
    if manifest.get("format") != "promptwitness.task-suite/v1":
        raise ValueError("unsupported task-suite format")
    suite_id = _text(manifest.get("suite_id"), "suite_id")
    budgets = manifest.get("budgets")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("budgets must be a non-empty array")
    budgets = [_positive(item, "input budget") for item in budgets]
    if len(set(budgets)) != len(budgets):
        raise ValueError("input budgets must be unique")
    seed = manifest.get("seed", "0")
    if not isinstance(seed, (str, int)) or isinstance(seed, bool):
        raise ValueError("seed must be a string or integer")
    generation = generation_settings(
        manifest.get("generation", {"max_tokens": 256, "temperature": 0})
    )
    if counter is None:
        if counter_identity is not None:
            raise ValueError("counter_identity requires a counter")
        identity: Mapping[str, Any] = {
            "unit": "utf8_bytes",
            "implementation": "canonical-messages-json/v1",
        }
    else:
        if not callable(counter) or not isinstance(counter_identity, Mapping):
            raise ValueError("custom counter requires a callable and counter_identity")
        for field in ("unit", "implementation"):
            _text(counter_identity.get(field), f"counter_identity.{field}")
        identity = counter_identity
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("tasks must be a non-empty array")
    tasks = []
    items = []
    task_ids: set[str] = set()
    for task in raw_tasks:
        _fields(task, {"id", "family", "path", "adapter", "limit", "ranking_k"}, "task")
        task_id = _text(task.get("id"), "task.id")
        family = task.get("family")
        if family not in FAMILIES or task_id in task_ids:
            raise ValueError("task family must be supported and task IDs unique")
        task_ids.add(task_id)
        filename = _text(task.get("path"), "task.path")
        input_path = (source.parent / filename).resolve()
        if input_path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("task input exceeds the 64 MiB local-file limit")
        raw_bytes = input_path.read_bytes()
        text = raw_bytes.decode("utf-8")
        rows = (
            [_load(line) for line in text.split("\n") if line.strip()]
            if input_path.suffix.lower() == ".jsonl"
            else _load(text)
        )
        if not isinstance(rows, list) or not rows:
            raise ValueError("task input must be a non-empty JSON array or JSONL records")
        adapter = task.get("adapter", "native")
        try:
            records = [normalize_record(row, family, adapter) for row in rows]
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"invalid {adapter} record structure") from error
        if len({record["id"] for record in records}) != len(records):
            raise ValueError("record IDs must be unique within a task")
        limit = _positive(task.get("limit", len(records)), "task.limit")
        ranking_k = _positive(task.get("ranking_k", 10), "ranking_k")
        selected = sorted(records, key=lambda record: digest([str(seed), task_id, record["id"]]))[
            :limit
        ]
        tasks.append(
            {
                "id": task_id,
                "family": family,
                "adapter": adapter,
                "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "available": len(records),
                "selected": len(selected),
                "sampled_out": len(records) - len(selected),
            }
        )
        for record in selected:
            messages = _messages(record, family, str(seed))
            units = (
                len(_json(messages).encode("utf-8"))
                if counter is None
                else counter(_freeze_json(messages, "messages"))
            )
            _positive(units, "measured input units")
            for budget in budgets:
                items.append(
                    {
                        "id": digest([task_id, record["id"], budget]),
                        "task_id": task_id,
                        "record_id": record["id"],
                        "family": family,
                        "budget": budget,
                        "input_units": units,
                        "messages": messages,
                        "record": record,
                        "ranking_k": ranking_k,
                        "skip_reason": "input_budget_exceeded" if units > budget else None,
                    }
                )
    return TaskSuitePlan(
        {
            "format": "promptwitness.task-plan/v1",
            "implementation": "task-prompts-scores/v1",
            "suite_id": suite_id,
            "manifest_sha256": hashlib.sha256(content).hexdigest(),
            "seed": str(seed),
            "budgets": budgets,
            "counter": identity,
            "generation": generation,
            "tasks": tasks,
            "items": items,
        }
    )
