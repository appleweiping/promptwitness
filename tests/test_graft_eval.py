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


def test_token_pools_are_single_token_substitutions() -> None:
    import os
    import random
    from pathlib import Path

    import pytest

    path = os.environ.get("PW_TEST_TOKENIZER")
    if path is None:
        pytest.skip("set PW_TEST_TOKENIZER to a local path")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research" / "graft"))
    from decision_study import token_pools
    from promptwitness.graft_runtime import load_tokenizer
    from promptwitness.superposed_gates import _block_token_span

    tokenizer = load_tokenizer(path)
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = transformers.LlamaForCausalLM(config).float().eval()
    prompt = graft_tasks.initial_prompt()
    rows = [graft_tasks.Example("a", "Which is first? (A) x (B) y", "(A)"),
            graft_tasks.Example("b", "Is the sky green?", "No")]
    pools = token_pools(model, tokenizer, prompt, rows, 12, 10, random.Random(0))
    assert sum(map(len, pools.values())) == 12
    for slot, texts in pools.items():
        _, lo, hi = _block_token_span(prompt, tokenizer, {"input": rows[0].question}, slot)
        base = _block_token_span(prompt, tokenizer, {"input": rows[0].question}, slot)[0][lo:hi]
        for text in texts:
            edited = prompt.replace_block(slot, text)
            ids, lo2, hi2 = _block_token_span(edited, tokenizer, {"input": rows[0].question}, slot)
            new = ids[lo2:hi2]
            assert len(new) == len(base) and sum(a != b for a, b in zip(new, base)) == 1
