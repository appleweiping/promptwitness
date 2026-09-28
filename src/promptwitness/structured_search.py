"""Shared block proposal and strict multiple-choice scoring for the pilot."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .gradient_backend import GradientObservation
from .models import Message, PromptDocument
from .structured_prompt import BlockKind, PromptBlock, StructuredPrompt


@dataclass(frozen=True, slots=True)
class TaskScore:
    correct: bool
    format_valid: bool
    predicted: str | None
    answer: str


def initial_prompt() -> StructuredPrompt:
    """One stable four-block prompt shared by both BBH pilot tasks."""
    pieces = (
        ("input", BlockKind.INPUT_DATA, "Question:\n{{input}}\n\n", False, ()),
        (
            "task",
            BlockKind.TASK_INSTRUCTION,
            "Solve the multiple-choice question using its supplied options.\n",
            True,
            (),
        ),
        (
            "reasoning",
            BlockKind.REASONING_POLICY,
            "Give at most two concise reasoning sentences before the final answer.\n",
            True,
            (),
        ),
        (
            "output",
            BlockKind.OUTPUT_INSTRUCTION,
            "End with one line in the form Final answer: (X), "
            "where X is the correct option letter.\n",
            True,
            ("Final answer: (X)",),
        ),
    )
    blocks: list[PromptBlock] = []
    content = ""
    for block_id, kind, text, editable, required in pieces:
        start = len(content)
        content += text
        blocks.append(
            PromptBlock(block_id, kind, text, editable, "question", start, len(content), required)
        )
    document = PromptDocument(
        "structured-gradient-bbh-v1", (Message("user", content, message_id="question"),)
    )
    return StructuredPrompt(document, tuple(blocks))


def score_multiple_choice(response: str, answer: str, *, max_letter: str) -> TaskScore:
    """Require the final nonempty line to contain one legal option letter."""
    final_line = next(
        (line.strip() for line in reversed(response.splitlines()) if line.strip()), ""
    )
    match = re.fullmatch(r"Final answer:\s*\(([A-Z])\)", final_line)
    valid = bool(match and "A" <= match.group(1) <= max_letter)
    prediction = f"({match.group(1)})" if valid and match else None
    return TaskScore(prediction == answer if valid else False, valid, prediction, answer)


def ranked_blocks(sensitivities: dict[str, float], count: int = 2) -> tuple[str, ...]:
    """Length-normalized block choice: mean of mapped token gradient norms."""
    return tuple(
        key
        for key, _ in sorted(sensitivities.items(), key=lambda item: (-item[1], item[0]))[:count]
    )


def first_order_rewrite_delta(
    model: Any,
    tokenizer: Any,
    parent: StructuredPrompt,
    candidate: StructuredPrompt,
    block_id: str,
    rows_and_gradients: tuple[tuple[str, GradientObservation], ...],
) -> float | None:
    """Mean g·(e_new-e_old), only for truly aligned full-render token positions.

    A changed token count, shifted position, or any changed token outside this
    block makes the linearization undefined for this implementation.
    """
    import torch

    estimates: list[float] = []
    weights = model.get_input_embeddings().weight.detach()
    for question, observation in rows_and_gradients:
        old = parent.render_tokens(tokenizer, {"input": question})
        new = candidate.render_tokens(tokenizer, {"input": question})
        positions = old.block_positions[block_id]
        if (
            len(old.input_ids) != len(new.input_ids)
            or positions != new.block_positions[block_id]
            or len(positions) != len(observation.token_gradients.get(block_id, ()))
        ):
            return None
        selected = set(positions)
        if any(
            old_id != new_id
            for index, (old_id, new_id) in enumerate(zip(old.input_ids, new.input_ids, strict=True))
            if index not in selected
        ):
            return None
        total = 0.0
        for index, gradient in zip(positions, observation.token_gradients[block_id], strict=True):
            direction = (
                weights[new.input_ids[index]].float() - weights[old.input_ids[index]].float()
            )
            total += float(torch.tensor(gradient, device=direction.device).dot(direction).item())
        estimates.append(total)
    return sum(estimates) / len(estimates) if estimates else None


def propose_block_rewrites(
    model: Any,
    tokenizer: Any,
    prompt: StructuredPrompt,
    block_id: str,
    *,
    count: int = 3,
    variant_start: int = 0,
) -> tuple[tuple[str, int, int, float], ...]:
    """Use the same local model and fixed unlabeled template for every block arm."""
    import torch

    block = next(block for block in prompt.blocks if block.block_id == block_id)
    if not block.editable or count < 1 or variant_start < 0 or variant_start + count > 3:
        raise ValueError("expected an editable block and one to three proposals")
    variants = (
        "Make the instruction shorter while retaining its requirements.",
        "Make the instruction more explicit while retaining its requirements.",
        "Rephrase the instruction with different wording while retaining its requirements.",
    )
    device = next(model.parameters()).device
    proposals: list[tuple[str, int, int, float]] = []
    for variant in variants[variant_start : variant_start + count]:
        request = (
            "Rewrite exactly one prompt instruction block. Return only the rewritten block text. "
            "Do not add examples, answers, task inputs, tool calls, or new constraints. "
            "Preserve every required literal exactly.\n"
            f"Block type: {block.kind.value}\n"
            f"Required literals: {list(block.required_literals)}\n"
            f"Original block: {block.text}\n"
            f"Edit style: {variant}"
        )
        messages = [{"role": "user", "content": request}]
        encoding = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        ids = encoding["input_ids"] if isinstance(encoding, Mapping) else encoding
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        started = perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output[0, input_ids.shape[1] :].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip().strip('"')
        proposals.append((text, len(ids), len(new_ids), perf_counter() - started))
    return tuple(proposals)
