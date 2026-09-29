"""Batched generation and fixed-reasoning answer losses for GRAFT experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any


def load_tokenizer(path: str) -> Any:
    """Tokenizer with model-specific chat-template defaults fixed for all experiments.

    Qwen3 templates default to thinking mode; every experiment uses non-thinking mode
    so that prompts, reasoning and extraction have the same shape across models.
    """
    import functools
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    config = Path(path) / "config.json"
    model_type = json.loads(config.read_text()).get("model_type", "") if config.exists() else ""
    if model_type.startswith("qwen3"):
        tokenizer.apply_chat_template = functools.partial(
            tokenizer.apply_chat_template, enable_thinking=False)
    return tokenizer


class Ledger:
    """Wall-clock seconds and token counts per phase (the cost ledger of every run)."""

    def __init__(self) -> None:
        self.phases: dict[str, dict[str, float]] = {}

    def add(self, phase: str, seconds: float, **counts: float) -> None:
        bucket = self.phases.setdefault(phase, {"seconds": 0.0, "calls": 0})
        bucket["seconds"] += seconds
        bucket["calls"] += 1
        for name, value in counts.items():
            bucket[name] = bucket.get(name, 0) + value


@dataclass(frozen=True, slots=True)
class Generation:
    token_ids: tuple[int, ...]  # without trailing EOS/padding
    text: str
    ended: bool


def stop_tokens(tokenizer: Any) -> set[int]:
    """EOS plus the end-of-turn tokens of the chat templates in use (shared by all engines)."""
    stop = {tokenizer.eos_token_id}
    for name in ("<|eot_id|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>"):
        token = tokenizer.convert_tokens_to_ids(name)
        if isinstance(token, int) and token >= 0 and token != tokenizer.unk_token_id:
            stop.add(token)
    return stop


# Tokens read after the extractor: long enough for LaTeX-wrapped answers such as
# "$$ \boxed{70,000} $$" (8 tokens truncated them).
ANSWER_TOKENS = 16
_REMOTE: Any = None  # graft_genserver.GenClient when generation is served by vLLM


def use_remote_generation(address: str | None) -> None:
    """Route generate_batch, sample_reasonings and remote_sample to a graft_genserver process."""
    global _REMOTE
    if address:
        from .graft_genserver import GenClient

        _REMOTE = GenClient(address)
    else:
        _REMOTE = None


def remote_sample(prompts: Sequence[Sequence[int]], *, max_new_tokens: int, stop: Sequence[int],
                  temperature: float, top_p: float, seed: int) -> list[list[int]] | None:
    """Sampled continuations from the generation server, or None when none is configured."""
    if _REMOTE is None:
        return None
    outs = _REMOTE.generate(prompts, max_new_tokens=max_new_tokens, stop=stop, temperature=temperature,
                            top_p=top_p, seeds=[seed * 1000 + i for i in range(len(prompts))])
    return [tokens for tokens, _ in outs]


def sampling_stops(model: Any, tokenizer: Any) -> list[int]:
    """Stop tokens for sampled generation: the model's generation-config EOS ids plus the
    chat end-of-turn tokens (Gemma-2's generation config lists only ``<eos>``, so without
    ``<end_of_turn>`` samples would run past the end of the assistant turn)."""
    stops = set(stop_tokens(tokenizer))
    config_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(config_eos, int):
        stops.add(config_eos)
    elif config_eos:
        stops.update(config_eos)
    return sorted(stops)


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    batch_size: int = 16,
) -> tuple[list[Generation], float, int]:
    """Greedy, left-padded batched generation. Returns generations, seconds, new tokens."""
    if _REMOTE is not None:
        started = perf_counter()
        outs = _REMOTE.generate(prompts, max_new_tokens=max_new_tokens, stop=sorted(stop_tokens(tokenizer)))
        gens = [Generation(tuple(tokens), tokenizer.decode(tokens, skip_special_tokens=True), ended)
                for tokens, ended in outs]
        return gens, perf_counter() - started, sum(len(g.token_ids) + int(g.ended) for g in gens)
    import torch

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    stop = stop_tokens(tokenizer)
    device = next(model.parameters()).device
    results: list[Generation] = []
    generated = 0
    started = perf_counter()
    for begin in range(0, len(prompts), batch_size):
        chunk = [list(ids) for ids in prompts[begin : begin + batch_size]]
        width = max(len(ids) for ids in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, sequence in enumerate(chunk):
            ids[row, width - len(sequence) :] = torch.tensor(sequence)
            mask[row, width - len(sequence) :] = 1
        with torch.no_grad():
            out = model.generate(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad,
                eos_token_id=sorted(stop),
            )
        for row in out[:, width:].tolist():
            ended = False
            tokens: list[int] = []
            for token in row:
                if token in stop:
                    ended = True
                    break
                tokens.append(token)
            generated += len(tokens) + int(ended)
            results.append(
                Generation(tuple(tokens), tokenizer.decode(tokens, skip_special_tokens=True), ended)
            )
    return results, perf_counter() - started, generated


@dataclass(frozen=True, slots=True)
class ScoredTarget:
    """Tokens scored after the prompt, with per-token weights.

    ``tail`` precedes the scored tokens and is not scored. For GReaTer's objective the
    tail is reasoning+extractor and only the answer is scored; for the verified
    objective the tail is empty and a gold-consistent reasoning plus the answer are
    scored with equal total weight (the extractor tokens weigh 0).
    """

    tail: tuple[int, ...]
    scored: tuple[int, ...]
    weights: tuple[float, ...]
    source: str  # "answer", "greedy_correct", "sampled_correct", "fallback", "fork"
    contrast: tuple[int, ...] = ()  # decision margin: loss = logit[contrast] - logit[scored]
    row: int = -1  # index of the minibatch example this target belongs to


def answer_target(reasoning: Sequence[int], extractor: Sequence[int],
                  answer: Sequence[int]) -> ScoredTarget:
    return ScoredTarget(tuple(reasoning) + tuple(extractor), tuple(answer),
                        tuple([1.0] * len(answer)), "answer")


def verified_targets(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    greedy: Sequence[Generation],
    greedy_correct: Sequence[bool],
    extractor: Sequence[int],
    answers: Sequence[Sequence[int]],
    judge: Any,
    *,
    samples: int = 4,
    max_new_tokens: int = 384,
    seed: int = 0,
) -> tuple[list[ScoredTarget], float]:
    """Gold-consistent self-generated reasoning per example (verified objective).

    Greedy reasoning if it reaches the gold answer; otherwise the first of ``samples``
    sampled reasonings that does (``judge(index, continuation) -> bool`` reads the
    extracted answer); otherwise fall back to GReaTer's answer-only objective.
    """
    import torch

    started = perf_counter()
    chosen: list[list[int] | None] = [
        list(g.token_ids) if ok else None for g, ok in zip(greedy, greedy_correct)
    ]
    sources = ["greedy_correct" if ok else "fallback" for ok in greedy_correct]
    todo = [i for i, c in enumerate(chosen) if c is None]
    if todo and samples > 0:
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        device = next(model.parameters()).device
        batch = [list(prompts[i]) for i in todo for _ in range(samples)]
        width = max(map(len, batch))
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(batch):
            ids[row, width - len(seq):] = torch.tensor(seq)
            mask[row, width - len(seq):] = 1
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                                 top_p=0.95, pad_token_id=pad)
        drafts: list[list[int]] = []
        for row in out[:, width:].tolist():
            tokens = []
            for token in row:
                if token == eos or token == pad:
                    break
                tokens.append(token)
            drafts.append(tokens)
        reads, _, _ = generate_batch(
            model, tokenizer, [batch[r] + drafts[r] + list(extractor) for r in range(len(batch))],
            max_new_tokens=8)
        for position, index in enumerate(todo):
            for k in range(samples):
                r = position * samples + k
                if drafts[r] and judge(index, reads[r].text):
                    chosen[index] = drafts[r]
                    sources[index] = "sampled_correct"
                    break
    targets: list[ScoredTarget] = []
    for index, answer in enumerate(answers):
        reasoning = chosen[index]
        if reasoning is None:
            targets.append(ScoredTarget(
                tuple(greedy[index].token_ids) + tuple(extractor), tuple(answer),
                tuple([1.0] * len(answer)), "fallback"))
            continue
        scored = tuple(reasoning) + tuple(extractor) + tuple(answer)
        weights = ([0.5 / len(reasoning)] * len(reasoning) + [0.0] * len(extractor)
                   + [0.5 / len(answer)] * len(answer))
        targets.append(ScoredTarget((), scored, tuple(weights), sources[index]))
    return targets, perf_counter() - started


def sample_reasonings(model: Any, tokenizer: Any, prompts: list[list[int]], samples: int,
                      max_new_tokens: int, seed: int) -> list[list[list[int]]]:
    flat = [p for p in prompts for _ in range(samples)]
    remote = remote_sample(flat, max_new_tokens=max_new_tokens, stop=sampling_stops(model, tokenizer),
                           temperature=0.7, top_p=0.95, seed=seed)
    if remote is not None:
        return [remote[i * samples:(i + 1) * samples] for i in range(len(prompts))]
    import torch

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    stops = sampling_stops(model, tokenizer)
    cut = set(stops) | {eos, pad}
    device = next(model.parameters()).device
    batch = [p for p in prompts for _ in range(samples)]
    out_all: list[list[int]] = []
    for begin in range(0, len(batch), 24):
        chunk = batch[begin:begin + 24]
        width = max(map(len, chunk))
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(chunk):
            ids[row, width - len(seq):] = torch.tensor(seq)
            mask[row, width - len(seq):] = 1
        torch.manual_seed(seed + begin)
        with torch.no_grad():
            gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                                 top_p=0.95, pad_token_id=pad, eos_token_id=stops)
        for row in gen[:, width:].tolist():
            tokens = []
            for token in row:
                if token in cut:
                    break
                tokens.append(token)
            out_all.append(tokens)
    return [out_all[i * samples:(i + 1) * samples] for i in range(len(prompts))]


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None


def fork_targets(greedy: Sequence[Sequence[int]], greedy_correct: Sequence[bool],
                 samples: Sequence[Sequence[Sequence[int]]], sample_correct: Sequence[Sequence[bool]],
                 *, max_forks: int = 3) -> list[ScoredTarget]:
    """Decision forks between the greedy reasoning and self-samples of opposite correctness.

    If greedy is wrong: its first divergence from each correct sample; if greedy is right:
    from each incorrect sample. Target: margin CE(correct token) - CE(incorrect token).
    """
    targets: list[ScoredTarget] = []
    for row, (g, g_ok) in enumerate(zip(greedy, greedy_correct)):
        found: dict[tuple[int, int, int], ScoredTarget] = {}
        for d, ok in zip(samples[row], sample_correct[row]):
            if not d or ok == g_ok:
                continue
            good, bad = (list(g), list(d)) if g_ok else (list(d), list(g))
            t = first_divergence(good, bad)
            if t is not None and t < len(good) and t < len(bad):
                found.setdefault((t, good[t], bad[t]), ScoredTarget(
                    tuple(good[:t]), (good[t],), (1.0,), "fork", (bad[t],), row))
        targets.extend(list(found.values())[:max_forks])
    return targets


def answer_losses(
    model: Any,
    sequences: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    batch_size: int = 8,
    weights: Sequence[Sequence[float]] | None = None,
) -> tuple[list[float], float]:
    """Answer CE for (context_ids, answer_ids) pairs, right-padded batches.

    With ``weights`` (one list per pair, aligned with answer_ids) the loss is the
    weighted mean of per-token cross-entropies; otherwise the plain mean.
    """
    import torch
    import torch.nn.functional as functional

    device = next(model.parameters()).device
    losses: list[float] = []
    started = perf_counter()
    for begin in range(0, len(sequences), batch_size):
        chunk = sequences[begin : begin + batch_size]
        width = max(len(context) + len(answer) for context, answer in chunk)
        ids = torch.zeros((len(chunk), width), dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, (context, answer) in enumerate(chunk):
            full = list(context) + list(answer)
            ids[row, : len(full)] = torch.tensor(full)
            mask[row, : len(full)] = 1
        with torch.no_grad():
            logits = model(
                input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False
            ).logits
        for row, (context, answer) in enumerate(chunk):
            start = len(context)
            pred = logits[row, start - 1 : start - 1 + len(answer)].float()
            target = torch.tensor(list(answer), device=device)
            if weights is None:
                losses.append(float(functional.cross_entropy(pred, target).item()))
            else:
                w = torch.tensor(list(weights[begin + row]), device=device)
                per = functional.cross_entropy(pred, target, reduction="none")
                losses.append(float((per * w).sum() / w.sum()))
    return losses, perf_counter() - started
