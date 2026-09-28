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


def answer_losses(
    model: Any,
    sequences: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    batch_size: int = 8,
) -> tuple[list[float], float]:
    """Mean answer CE for (context_ids, answer_ids) pairs, right-padded batches."""
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
            losses.append(float(functional.cross_entropy(pred, target).item()))
    return losses, perf_counter() - started
