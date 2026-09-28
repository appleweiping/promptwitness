"""Editable semantic blocks with token positions from the actual chat template."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .models import PromptDocument
from .variables import inspect_variables, render_template


class BlockKind(str, Enum):
    TASK_INSTRUCTION = "task_instruction"
    REASONING_POLICY = "reasoning_policy"
    OUTPUT_INSTRUCTION = "output_instruction"
    INPUT_DATA = "input_data"
    EXAMPLE = "example"


@dataclass(frozen=True, slots=True)
class PromptBlock:
    block_id: str
    kind: BlockKind
    text: str
    editable: bool
    message_id: str
    source_start: int
    source_end: int
    required_literals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.block_id or not self.message_id:
            raise ValueError("blocks require stable block and message IDs")
        if self.source_start < 0 or self.source_end < self.source_start:
            raise ValueError("invalid block source span")
        if self.source_end - self.source_start != len(self.text):
            raise ValueError("block source span must match its text")
        if self.kind in (BlockKind.INPUT_DATA, BlockKind.EXAMPLE) and self.editable:
            raise ValueError("input data and examples are frozen in this pilot")
        if self.editable and inspect_variables(self.text).names:
            raise ValueError("editable blocks cannot contain task variables")
        if any(not literal or literal not in self.text for literal in self.required_literals):
            raise ValueError("required literals must occur in the source block")


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    text: str
    input_ids: tuple[int, ...]
    block_positions: Mapping[str, tuple[int, ...]]
    boundary_positions: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    block_char_spans: Mapping[str, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class StructuredPrompt:
    document: PromptDocument
    blocks: tuple[PromptBlock, ...]

    def __post_init__(self) -> None:
        if len({block.block_id for block in self.blocks}) != len(self.blocks):
            raise ValueError("block IDs must be unique")
        if any(message.message_id is None for message in self.document.messages):
            raise ValueError("structured prompts require message IDs")
        if any(message.content_parts or message.name for message in self.document.messages):
            raise ValueError("pilot supports text-only unnamed messages")
        if self.document.tools:
            raise ValueError("pilot does not render tool interfaces")
        by_message: dict[str, list[PromptBlock]] = {}
        for block in self.blocks:
            by_message.setdefault(block.message_id, []).append(block)
        message_ids = {message.message_id for message in self.document.messages}
        if set(by_message) != message_ids:
            raise ValueError("blocks must cover every message and no unknown message")
        for message in self.document.messages:
            message_id = message.message_id
            if message_id is None:
                raise ValueError("structured prompts require message IDs")
            parts = sorted(by_message[message_id], key=lambda block: block.source_start)
            cursor = 0
            for block in parts:
                if block.source_start != cursor:
                    raise ValueError("block spans must partition each message without gaps")
                cursor = block.source_end
            if (
                cursor != len(message.content)
                or "".join(part.text for part in parts) != message.content
            ):
                raise ValueError("blocks must reproduce their source message")

    def replace_block(self, block_id: str, text: str) -> StructuredPrompt:
        """Rewrite exactly one block, preserving variables and all frozen material."""
        old = next((block for block in self.blocks if block.block_id == block_id), None)
        if old is None:
            raise KeyError(block_id)
        if not old.editable:
            raise ValueError("block is frozen")
        if not text.strip() or inspect_variables(text).malformed or inspect_variables(text).names:
            raise ValueError("rewrite must be nonempty and cannot add variables")
        if any(literal not in text for literal in old.required_literals):
            raise ValueError("rewrite removed a required literal")
        delta = len(text) - len(old.text)
        blocks = tuple(
            replace(block, text=text, source_end=block.source_start + len(text))
            if block.block_id == block_id
            else replace(
                block,
                source_start=block.source_start + delta,
                source_end=block.source_end + delta,
            )
            if block.message_id == old.message_id and block.source_start >= old.source_end
            else block
            for block in self.blocks
        )
        messages = tuple(
            replace(
                message,
                content=(
                    message.content[: old.source_start] + text + message.content[old.source_end :]
                ),
            )
            if message.message_id == old.message_id
            else message
            for message in self.document.messages
        )
        return StructuredPrompt(replace(self.document, messages=messages), blocks)

    def render_tokens(self, tokenizer: Any, values: Mapping[str, object]) -> TokenizedPrompt:
        """Map blocks through a fully rendered chat template and full-string tokenization.

        Tokens touching two blocks or template delimiters are excluded from all
        editable positions. Fast-tokenizer offsets and exact ID parity are required.
        """
        plain_messages: list[dict[str, str]] = []
        marked_messages: list[dict[str, str]] = []
        markers: list[tuple[str, str, str]] = []
        for message in self.document.messages:
            plain = render_template(message.content, values)
            plain_messages.append({"role": message.role, "content": plain})
            marked_parts: list[str] = []
            parts = sorted(
                (block for block in self.blocks if block.message_id == message.message_id),
                key=lambda block: block.source_start,
            )
            for block in parts:
                index = len(markers)
                start = f"\ue000PW_START_{index}\ue001"
                end = f"\ue000PW_END_{index}\ue001"
                rendered = render_template(block.text, values)
                if start in plain or end in plain:
                    raise ValueError("marker collision in prompt content")
                markers.append((block.block_id, start, end))
                marked_parts.append(start + rendered + end)
            marked_messages.append({"role": message.role, "content": "".join(marked_parts)})

        plain_text = tokenizer.apply_chat_template(
            plain_messages, tokenize=False, add_generation_prompt=True
        )
        marked_text = tokenizer.apply_chat_template(
            marked_messages, tokenize=False, add_generation_prompt=True
        )
        spans: dict[str, tuple[int, int]] = {}
        cursor = 0
        removed = 0
        for block_id, start, end in markers:
            start_at = marked_text.find(start, cursor)
            end_at = marked_text.find(end, start_at + len(start))
            if start_at < 0 or end_at < 0:
                raise ValueError("chat template did not preserve block markers")
            lo = start_at - removed
            hi = end_at - removed - len(start)
            spans[block_id] = (lo, hi)
            removed += len(start) + len(end)
            cursor = end_at + len(end)
        stripped = marked_text
        for _, start, end in markers:
            stripped = stripped.replace(start, "", 1).replace(end, "", 1)
        if stripped != plain_text:
            raise ValueError("marked and plain chat renderings differ")
        encoded = tokenizer(plain_text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = tuple(int(value) for value in encoded["input_ids"])
        template_encoding = tokenizer.apply_chat_template(
            plain_messages, tokenize=True, add_generation_prompt=True
        )
        template_values = (
            template_encoding["input_ids"]
            if isinstance(template_encoding, Mapping)
            else template_encoding
        )
        template_ids = tuple(int(value) for value in template_values)
        if input_ids != template_ids:
            raise ValueError("full-string tokens differ from chat-template token IDs")
        positions: dict[str, list[int]] = {block.block_id: [] for block in self.blocks}
        boundary: list[int] = []
        for index, (lo, hi) in enumerate(encoded["offset_mapping"]):
            owners = [
                block_id
                for block_id, (start, end) in spans.items()
                if hi > lo and lo >= start and hi <= end
            ]
            if len(owners) == 1:
                positions[owners[0]].append(index)
            else:
                boundary.append(index)
        return TokenizedPrompt(
            plain_text,
            input_ids,
            {key: tuple(value) for key, value in positions.items()},
            tuple(boundary),
            tuple((int(lo), int(hi)) for lo, hi in encoded["offset_mapping"]),
            spans,
        )
