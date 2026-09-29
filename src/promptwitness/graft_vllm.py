"""The shared two-stage reader on vLLM, for final dev/test evaluations.

Token-level prompts are rendered exactly as on the Hugging Face path
(``StructuredPrompt.render_tokens``) and passed to vLLM as token ids; reasoning is
greedy up to ``max_new_tokens`` and stops at the same end-of-turn tokens, then the
task extractor is appended inside the assistant turn and a short greedy answer is read.
Greedy decoding in different kernels can diverge on near-ties, so every method in one
comparison is evaluated with the same engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

from . import graft_tasks
from .graft_runtime import ANSWER_TOKENS, Generation, Ledger, stop_tokens
from .structured_prompt import StructuredPrompt


class VllmReader:
    def __init__(self, model_path: str, tokenizer: Any, spec: graft_tasks.TaskSpec, ledger: Ledger,
                 max_new_tokens: int, *, llm: Any = None, gpu_memory_utilization: float = 0.5,
                 max_model_len: int = 4096) -> None:
        if llm is None:
            from vllm import LLM

            llm = LLM(model=model_path, tokenizer=model_path, dtype="bfloat16", seed=0,
                      gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                      enable_prefix_caching=True)
        self.llm, self.tokenizer, self.spec, self.ledger = llm, tokenizer, spec, ledger
        self.max_new_tokens = max_new_tokens
        self.stop = sorted(stop_tokens(tokenizer))
        self.extractor = tokenizer.encode(spec.extractor, add_special_tokens=False)

    def ids(self, prompt: StructuredPrompt, example: graft_tasks.Example) -> list[int]:
        return list(prompt.render_tokens(self.tokenizer, {"input": example.question}).input_ids)

    def generate(self, prompts: Sequence[Sequence[int]], max_new_tokens: int
                 ) -> tuple[list[Generation], float, int]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=self.stop,
                                skip_special_tokens=True)
        started = perf_counter()
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in prompts], params,
                                    use_tqdm=False)
        results, generated = [], 0
        for out in outputs:
            tokens = list(out.outputs[0].token_ids)
            ended = (bool(tokens) and tokens[-1] in self.stop) or out.outputs[0].finish_reason == "stop"
            while tokens and tokens[-1] in self.stop:
                tokens.pop()
            generated += len(tokens) + int(ended)
            results.append(Generation(tuple(tokens), self.tokenizer.decode(tokens, skip_special_tokens=True),
                                      ended))
        return results, perf_counter() - started, generated

    def evaluate(self, prompt: StructuredPrompt, examples: Sequence[graft_tasks.Example], phase: str,
                 *, max_new_tokens: int | None = None) -> dict[str, Any]:
        prompts = [self.ids(prompt, e) for e in examples]
        gens, seconds, tokens = self.generate(prompts, max_new_tokens or self.max_new_tokens)
        self.ledger.add(phase + "_reasoning", seconds, generated=tokens, prompt_tokens=sum(map(len, prompts)))
        reads, seconds, tokens = self.generate([p + list(g.token_ids) + self.extractor
                                                for p, g in zip(prompts, gens)], ANSWER_TOKENS)
        self.ledger.add(phase + "_answer", seconds, generated=tokens)
        correct = [self.spec.correct(r.text, e.answer) for r, e in zip(reads, examples)]
        return {"accuracy": sum(correct) / len(correct), "correct": correct,
                "reads": [r.text for r in reads], "reasoning": [list(g.token_ids) for g in gens],
                "truncated": sum(not g.ended for g in gens)}
