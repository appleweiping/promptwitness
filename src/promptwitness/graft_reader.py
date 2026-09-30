"""The shared two-stage reader through a running vLLM generation server.

Same rendering, stop tokens, extractor, answer length and parser as ``VllmReader`` and the
HF ``Runner``; generation is sent to a ``graft_genserver`` process
(``graft_runtime.use_remote_generation``), so evaluation can reuse the engine that already
serves a search queue. Every method and fixed prompt of one model in one comparison must be
read through the same path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import graft_tasks
from .graft_runtime import ANSWER_TOKENS, Ledger, generate_batch
from .structured_prompt import StructuredPrompt


class RemoteReader:
    def __init__(self, tokenizer: Any, spec: graft_tasks.TaskSpec, ledger: Ledger, max_new_tokens: int) -> None:
        self.tokenizer, self.spec, self.ledger = tokenizer, spec, ledger
        self.max_new_tokens = max_new_tokens
        self.extractor = tokenizer.encode(spec.extractor, add_special_tokens=False)

    def ids(self, prompt: StructuredPrompt, example: graft_tasks.Example) -> list[int]:
        return list(prompt.render_tokens(self.tokenizer, {"input": example.question}).input_ids)

    def evaluate(self, prompt: StructuredPrompt, examples: Sequence[graft_tasks.Example], phase: str,
                 *, max_new_tokens: int | None = None) -> dict[str, Any]:
        prompts = [self.ids(prompt, e) for e in examples]
        gens, seconds, tokens = generate_batch(None, self.tokenizer, prompts,
                                               max_new_tokens=max_new_tokens or self.max_new_tokens)
        self.ledger.add(phase + "_reasoning", seconds, generated=tokens, prompt_tokens=sum(map(len, prompts)))
        reads, seconds, tokens = generate_batch(None, self.tokenizer,
                                                [p + list(g.token_ids) + self.extractor for p, g in zip(prompts, gens)],
                                                max_new_tokens=ANSWER_TOKENS)
        self.ledger.add(phase + "_answer", seconds, generated=tokens)
        correct = [self.spec.correct(r.text, e.answer) for r, e in zip(reads, examples)]
        return {"accuracy": sum(correct) / len(correct), "correct": correct,
                "reads": [r.text for r in reads], "reasoning": [list(g.token_ids) for g in gens],
                "truncated": sum(not g.ended for g in gens)}
