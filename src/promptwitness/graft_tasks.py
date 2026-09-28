"""GReaTer-protocol tasks for GRAFT: 21 BBH tasks, GSM8K and FOLIO.

Data are the GReaTer repository files at commit 42a22d9 (BBH/FOLIO CSV with
``goal,target,final_target``; GSM8K JSONL). Splits follow GReaTer's sizes with a
fixed seed: BBH 50 train / 100 dev / remaining (<=100) test, GSM8K 100/100/1319,
FOLIO 50 (from its train file) / 100 dev / 203 test. Every method is evaluated by
the same two-stage reader: greedy reasoning under the prompt, then a task-typed
extractor appended inside the assistant turn and a short greedy answer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .models import Message, PromptDocument
from .structured_prompt import BlockKind, PromptBlock, StructuredPrompt

Kind = Literal["mc", "binary", "integer"]
SPLIT_SEED = 20260928

BINARY = {
    "boolean_expressions": ("True", "False"),
    "causal_judgement": ("Yes", "No"),
    "formal_fallacies": ("valid", "invalid"),
    "navigate": ("Yes", "No"),
    "sports_understanding": ("yes", "no"),
    "web_of_lies": ("Yes", "No"),
}
INTEGER = ("object_counting", "multistep_arithmetic_two", "gsm8k")
BBH_TASKS = (
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects", "web_of_lies",
)
ALL_TASKS = BBH_TASKS + ("gsm8k", "folio")


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    question: str
    answer: str  # canonical gold string: "(B)", "Yes", "12"

    @property
    def row_id(self) -> str:
        """Alias used by the structured-gradient pilot record format."""
        return self.example_id


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    kind: Kind
    options: tuple[str, ...] = ()

    @property
    def extractor(self) -> str:
        if self.kind == "mc":
            return "\nTherefore, the final answer is ("
        if self.kind == "binary":
            return f"\nTherefore, the final answer ({self.options[0]} or {self.options[1]}) is:"
        return "\nTherefore, the final answer (a single integer) is:"

    def target(self, answer: str) -> str:
        """Gold continuation after the extractor, used for the answer loss."""
        if self.kind == "mc":
            return answer.strip("()") + ")"
        return " " + answer

    def parse(self, continuation: str) -> str | None:
        # Tolerate leading markup the reader may emit after the extractor: quotes,
        # asterisks, backticks, dollar signs and spaces.
        text = re.sub(r"^[\s\"'`*$]+", "", continuation)
        if self.kind == "mc":
            match = re.match(r"\(?\s*([A-R])\b", text)
            return f"({match.group(1)})" if match else None
        if self.kind == "binary":
            lowered = text.lower()
            for option in sorted(self.options, key=len, reverse=True):
                if lowered.startswith(option.lower()):
                    return option
            return None
        match = re.match(r"\$?\s*(-?[\d,]*\d(?:\.\d+)?)", text)
        if not match:
            return None
        value = match.group(1).replace(",", "")
        return str(int(float(value))) if float(value).is_integer() else value

    def correct(self, continuation: str, answer: str) -> bool:
        parsed = self.parse(continuation)
        if parsed is None:
            return False
        if self.kind == "binary":
            return parsed.lower() == answer.lower()
        return parsed == answer


def spec(name: str) -> TaskSpec:
    if name in BINARY:
        return TaskSpec(name, "binary", BINARY[name])
    if name in INTEGER:
        return TaskSpec(name, "integer")
    if name in BBH_TASKS or name == "folio":
        return TaskSpec(name, "mc")
    raise KeyError(name)


def _csv(path: Path, task: str) -> list[Example]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for index, row in enumerate(rows):
        question, answer = row["goal"].strip(), row["final_target"].strip()
        if not question or not answer:
            raise ValueError(f"{path}: empty row {index}")
        out.append(Example(f"{task}:{path.stem}:{index:04d}", question, answer))
    return out


def _gsm8k(path: Path, split: str) -> list[Example]:
    out = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Example(f"gsm8k:{split}:{index:04d}", row["question"].strip(), answer))
    return out


def _seeded(task: str) -> random.Random:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{task}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def load_splits(data_dir: Path, task: str) -> dict[str, list[Example]]:
    """Deterministic train/dev/test splits in GReaTer's sizes."""
    if task == "gsm8k":
        pool = _gsm8k(data_dir / "gsm8k" / "train.jsonl", "train")
        _seeded(task).shuffle(pool)
        return {"train": pool[:100], "dev": pool[100:200],
                "test": _gsm8k(data_dir / "gsm8k" / "test.jsonl", "test")}
    if task == "folio":
        train = _csv(data_dir / "FOLIO" / "train.csv", task)
        _seeded(task).shuffle(train)
        return {"train": train[:50], "dev": _csv(data_dir / "FOLIO" / "dev.csv", task),
                "test": _csv(data_dir / "FOLIO" / "test.csv", task)}
    if task not in BBH_TASKS:
        raise KeyError(task)
    rows = _csv(data_dir / "BBH" / f"{task}.csv", task)
    _seeded(task).shuffle(rows)
    rest = rows[50:]
    dev = min(100, len(rest) // 2) if len(rest) < 200 else 100
    return {"train": rows[:50], "dev": rest[:dev], "test": rest[dev : dev + 100]}


def initial_prompt(*, insertion_slot: bool = True) -> StructuredPrompt:
    """GReaTer's initial instruction, split into typed PromptWitness blocks.

    GReaTer starts every task from "Use proper logical reasoning and think step by
    step. Finally give the actual correct answer." Here the same words form a
    strategy block and an output block, with an empty optional procedure slot
    between them that insertion edits may fill.
    """
    pieces: list[tuple[str, BlockKind, str, bool]] = [
        ("input", BlockKind.INPUT_DATA, "{{input}}\n\n", False),
        ("strategy", BlockKind.REASONING_POLICY,
         "Use proper logical reasoning and think step by step.\n", True),
    ]
    if insertion_slot:
        pieces.append(("procedure", BlockKind.REASONING_POLICY, "", True))
    pieces.append(("output", BlockKind.OUTPUT_INSTRUCTION,
                   "Finally give the actual correct answer.\n", True))
    blocks: list[PromptBlock] = []
    content = ""
    for block_id, kind, text, editable in pieces:
        start = len(content)
        content += text
        blocks.append(PromptBlock(block_id, kind, text, editable, "user", start, len(content)))
    document = PromptDocument("graft-greater-init", (Message("user", content, message_id="user"),))
    return StructuredPrompt(document, tuple(blocks))


_PREAMBLE = re.compile(
    r"^\s*(here(?:'s| is| are)|sure|certainly|okay|ok|below is|revised|new|improved|updated|"
    r"rewritten|the (?:new|revised|improved|updated))\b[^\n]*:\s*$",
    re.IGNORECASE,
)
_LABEL = re.compile(
    r"^\s*(?:\*\*)?(?:block(?: type)?|new block|revised block|task_instruction|reasoning_policy|"
    r"output_instruction|strategy|procedure|output|instruction)(?:\*\*)?\s*:\s*(?:\*\*)?\s*$",
    re.IGNORECASE,
)


def clean_proposal(text: str) -> str:
    """Strip proposer meta-text so only the block content becomes a candidate.

    Removes chatty preambles ("Here is the revised block:"), bare type labels
    ("reasoning_policy:"), markdown emphasis, surrounding quotes and code fences.
    Returns "" when nothing substantive is left.
    """
    if "NEW BLOCK:" in text:
        text = text.split("NEW BLOCK:")[-1]
    lines = [line.rstrip() for line in text.replace("```", "").splitlines()]
    kept = [line for line in lines if not _PREAMBLE.match(line) and not _LABEL.match(line)]
    body = "\n".join(kept).strip().strip('"').strip("'").strip()
    body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body + "\n" if body else ""


def flat_prompt(text: str) -> StructuredPrompt:
    """A one-block prompt (for published token-level prompts and ZS-CoT baselines)."""
    content = "{{input}}\n\n" + text.strip() + "\n"
    blocks = (
        PromptBlock("input", BlockKind.INPUT_DATA, "{{input}}\n\n", False, "user", 0, 11),
        PromptBlock("instruction", BlockKind.TASK_INSTRUCTION, content[11:], False, "user",
                    11, len(content)),
    )
    document = PromptDocument("graft-flat", (Message("user", content, message_id="user"),))
    return StructuredPrompt(document, blocks)
