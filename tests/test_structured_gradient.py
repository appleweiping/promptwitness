"""Focused checks for structural mapping and genuine input autograd."""

from __future__ import annotations

import re

import pytest

from promptwitness.models import Message, PromptDocument
from promptwitness.structured_prompt import BlockKind, PromptBlock, StructuredPrompt


class TinyTokenizer:
    eos_token_id = 0

    def __init__(self, merge_words: bool = False) -> None:
        self.merge_words = merge_words
        self.ids: dict[str, int] = {}

    def _parts(self, text: str) -> list[tuple[str, int, int]]:
        pattern = r"[A-Za-z]+|[^A-Za-z]" if self.merge_words else r"."
        return [(match.group(), *match.span()) for match in re.finditer(pattern, text, re.DOTALL)]

    def _id(self, piece: str) -> int:
        if piece not in self.ids:
            self.ids[piece] = len(self.ids) + 1
        return self.ids[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(piece) for piece, _, _ in self._parts(text)]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        reverse = {value: key for key, value in self.ids.items()}
        return "".join(reverse[index] for index in ids)

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        parts = self._parts(text)
        return {
            "input_ids": [self._id(piece) for piece, _, _ in parts],
            "offset_mapping": [(lo, hi) for _, lo, hi in parts],
        }

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str | list[int]:
        text = "".join(f"<{row['role']}>{row['content']}</{row['role']}>" for row in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


def prompt() -> StructuredPrompt:
    content = "Task: decide. Think carefully. Answer (A) or (B). {{input}}"
    pieces = [
        ("task", BlockKind.TASK_INSTRUCTION, "Task: decide. ", True),
        ("reason", BlockKind.REASONING_POLICY, "Think carefully. ", True),
        ("output", BlockKind.OUTPUT_INSTRUCTION, "Answer (A) or (B). ", True),
        ("input", BlockKind.INPUT_DATA, "{{input}}", False),
    ]
    blocks = []
    cursor = 0
    for block_id, kind, text, editable in pieces:
        blocks.append(
            PromptBlock(block_id, kind, text, editable, "user-1", cursor, cursor + len(text))
        )
        cursor += len(text)
    assert cursor == len(content)
    document = PromptDocument("pilot", (Message("user", content, message_id="user-1"),))
    return StructuredPrompt(document, tuple(blocks))


def test_rewrite_freezes_input_and_preserves_source_positions() -> None:
    original = prompt()
    changed = original.replace_block("reason", "Reason in steps. ")
    assert changed.document.messages[0].message_id == "user-1"
    assert changed.document.messages[0].content.endswith("{{input}}")
    assert changed.blocks[-1].source_end == len(changed.document.messages[0].content)
    with pytest.raises(ValueError, match="frozen"):
        changed.replace_block("input", "secret label")
    with pytest.raises(ValueError, match="variables"):
        changed.replace_block("task", "Use {{answer}}")


def test_mapping_excludes_cross_block_token() -> None:
    original = StructuredPrompt(
        PromptDocument("pilot", (Message("user", "Actions", message_id="u"),)),
        (
            PromptBlock("a", BlockKind.TASK_INSTRUCTION, "Act", True, "u", 0, 3),
            PromptBlock("b", BlockKind.REASONING_POLICY, "ions", True, "u", 3, 7),
        ),
    )
    mapped = original.render_tokens(TinyTokenizer(merge_words=True), {})
    assert mapped.block_positions["a"] == ()
    assert mapped.block_positions["b"] == ()
    assert mapped.boundary_positions


def test_mapping_accepts_chat_template_encoding_object() -> None:
    class EncodingTokenizer(TinyTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):  # type: ignore[no-untyped-def]
            value = super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
            return {"input_ids": value} if tokenize else value

    mapped = prompt().render_tokens(EncodingTokenizer(), {"input": "Which choice?"})
    assert mapped.block_positions["task"]
    assert mapped.block_positions["reason"]


def test_multiple_choice_score_requires_final_line() -> None:
    from promptwitness.structured_search import score_multiple_choice

    assert score_multiple_choice("Reasoning\nFinal answer: (B)", "(B)", max_letter="C").correct
    invalid = score_multiple_choice("Final answer: (B)\nextra", "(B)", max_letter="C")
    assert not invalid.format_valid
    assert not score_multiple_choice("Final answer: (F)", "(F)", max_letter="C").correct


def test_real_autograd_covers_full_answer_and_leaves_weights_frozen() -> None:
    torch = pytest.importorskip("torch")
    from promptwitness.gradient_backend import FrozenGradientBackend

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 8)
            self.readout = torch.nn.Linear(8, 256, bias=False)

        def get_input_embeddings(self):  # type: ignore[no-untyped-def]
            return self.embedding

        def forward(self, *, inputs_embeds, use_cache):  # type: ignore[no-untyped-def]
            logits = self.readout(inputs_embeds.cumsum(dim=1))
            return type("Output", (), {"logits": logits})()

        def generate(self, *, input_ids, max_new_tokens, do_sample, pad_token_id):  # type: ignore[no-untyped-def]
            reason_id = tokenizer.encode("R")[0]
            return torch.cat(
                [input_ids, torch.tensor([[reason_id]], device=input_ids.device)], dim=1
            )

    tokenizer = TinyTokenizer()
    model = ToyModel()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    observation = FrozenGradientBackend(model, tokenizer).observe(
        prompt(), {"input": "Which choice?"}, "(A)", finite_difference_block="reason"
    )
    assert len(observation.answer_tokens) == 3
    assert observation.block_sensitivity["reason"] > 0
    assert observation.finite_difference is not None
    assert observation.finite_difference["measured"] == pytest.approx(
        observation.finite_difference["predicted"], rel=0.1, abs=0.01
    )
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
