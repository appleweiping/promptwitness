"""Deferred evaluation: prompt block round trip and the vLLM reader's output handling."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

from promptwitness import graft_tasks
from promptwitness.graft_runtime import Ledger
from promptwitness.graft_vllm import VllmReader


def test_prompt_blocks_round_trip_after_edits() -> None:
    prompt = graft_tasks.initial_prompt()
    edited = prompt.replace_block("procedure", "List every constraint first.\n")
    edited = edited.replace_block("output", "State the final option.\n")
    for p in (prompt, edited):
        back = graft_tasks.prompt_from_blocks(graft_tasks.prompt_blocks(p))
        assert back.document.messages[0].content == p.document.messages[0].content
        assert [(b.block_id, b.kind, b.text, b.editable, b.source_start, b.source_end) for b in back.blocks] == \
               [(b.block_id, b.kind, b.text, b.editable, b.source_start, b.source_end) for b in p.blocks]


class _Tokenizer:
    eos_token_id = 2
    unk_token_id = 0

    def convert_tokens_to_ids(self, name: str) -> int:
        return {"<|eot_id|>": 9}.get(name, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) if i > 31 else "" for i in ids)


@dataclass
class _Completion:
    token_ids: list[int]
    finish_reason: str


@dataclass
class _Output:
    outputs: list[_Completion]


class _LLM:
    def __init__(self, replies: list[_Completion]) -> None:
        self.replies = replies
        self.seen: list[list[int]] = []

    def generate(self, prompts: list[dict], params: object, use_tqdm: bool = False) -> list[_Output]:
        self.seen.extend(p["prompt_token_ids"] for p in prompts)
        return [_Output([r]) for r in self.replies[: len(prompts)]]


def _fake_vllm() -> None:
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}  # type: ignore[attr-defined]
    sys.modules["vllm"], sys.modules["vllm.inputs"] = vllm, inputs


def test_vllm_reader_strips_stop_tokens_and_flags_truncation() -> None:
    _fake_vllm()
    replies = [_Completion([72, 105, 9], "stop"), _Completion([72, 105, 106], "length"),
               _Completion([65, 2], "stop")]
    llm = _LLM(replies)
    reader = VllmReader("unused", _Tokenizer(), graft_tasks.spec("date_understanding"), Ledger(), 16, llm=llm)
    assert reader.stop == [2, 9]
    gens, _, generated = reader.generate([[1, 2], [3], [4]], 16)
    assert [g.token_ids for g in gens] == [(72, 105), (72, 105, 106), (65,)]
    assert [g.ended for g in gens] == [True, False, True]
    assert [g.text for g in gens] == ["Hi", "Hij", "A"]
    assert generated == 3 + 3 + 2
    assert llm.seen == [[1, 2], [3], [4]]
