"""Batched generation and fixed-reasoning answer losses for GRAFT experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True, slots=True)
class Generation:
    token_ids: tuple[int, ...]  # without trailing EOS/padding
    text: str
    ended: bool


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    batch_size: int = 16,
) -> tuple[list[Generation], float, int]:
    """Greedy, left-padded batched generation. Returns generations, seconds, new tokens."""
    import torch

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    stop = {eos}
    for name in ("<|eot_id|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>"):
        token = tokenizer.convert_tokens_to_ids(name)
        if isinstance(token, int) and token >= 0 and token != tokenizer.unk_token_id:
            stop.add(token)
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
    source: str  # "answer", "greedy_correct", "sampled_correct", "fallback"


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
