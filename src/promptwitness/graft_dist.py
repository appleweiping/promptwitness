"""Distributional edit scores: exact likelihood ratios of the incumbent's own sampled reasoning.

An edit P -> P' changes expected accuracy by a reasoning-distribution term
E_{r~pi_P}[(w(r) - 1) c] and an answer read-off term (see research/GRAFT_THEORY.md). With
S reasonings sampled once under the incumbent and exact teacher-forced log-likelihoods under
every candidate, the tempered self-normalized importance estimate

    J_beta(P' | x) = sum_s w_s^beta c_s / sum_s w_s^beta,   w_s = pi_P'^tau(r_s|x) / pi_P^tau(r_s|x)

scores all candidates from shared samples, without generating or reading under any candidate
(``c_s``: correctness of the incumbent's read of sample s). beta is chosen without targets
from the effective sample size. Validated by the H-dist / H-inc studies (is_study.py,
analyze_dist.py); this module holds the pieces the search loop uses.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from .graft_runtime import remote_sample


def logsumexp(values: Sequence[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def snis(logw: Sequence[float], values: Sequence[float], beta: float) -> float:
    """Tempered self-normalized importance estimate of the mean of ``values``."""
    if beta == 0.0:
        return statistics.mean(values)
    scaled = [beta * v for v in logw]
    norm = logsumexp(scaled)
    return sum(math.exp(s - norm) * x for s, x in zip(scaled, values))


def ess(logw: Sequence[float], beta: float) -> float:
    if beta == 0.0:
        return float(len(logw))
    a = [beta * v for v in logw]
    return math.exp(2 * logsumexp(a) - logsumexp([2 * v for v in a]))


def choose_beta(logw_by_edit: Sequence[Sequence[Sequence[float]]], samples: int,
                betas: Sequence[float] = (1.0, 0.5, 0.25)) -> float:
    """Largest beta whose median (over edits) of the median (over rows) ESS is >= S/4."""
    for beta in betas:
        per_edit = [statistics.median(ess(row, beta) for row in rows) for rows in logw_by_edit]
        if per_edit and statistics.median(per_edit) >= samples / 4:
            return beta
    return betas[-1]


def sample_traces(model: Any, tokenizer: Any, prompts: list[list[int]], *, max_new_tokens: int, tau: float,
                  stops: list[int], seed: int, batch: int = 16) -> list[tuple[list[int], bool]]:
    """Untruncated temperature samples (top_k=0, top_p=1): server if configured, else HF."""
    remote = remote_sample(prompts, max_new_tokens=max_new_tokens, stop=stops, temperature=tau, top_p=1.0,
                           seed=seed, with_ended=True)
    if remote is not None:
        return [(list(tokens), bool(ended)) for tokens, ended in remote]
    import torch

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    out: list[tuple[list[int], bool]] = []
    for begin in range(0, len(prompts), batch):
        chunk = prompts[begin:begin + batch]
        width = max(map(len, chunk))
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(chunk):
            ids[row, width - len(seq):] = torch.tensor(seq)
            mask[row, width - len(seq):] = 1
        torch.manual_seed(seed * 1000 + begin)
        with torch.no_grad():
            gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=True, temperature=tau,
                                 top_k=0, top_p=1.0, pad_token_id=pad, eos_token_id=stops)
        for row in gen[:, width:].tolist():
            tokens: list[int] = []
            ended = False
            for token in row:
                if token in stops:
                    ended = True
                    break
                tokens.append(token)
            out.append((tokens, ended))
    return out


def trace_logprobs(model: Any, items: list[tuple[list[int], int, int, bool]], *, tau: float, stops: list[int],
                   batch: int = 8) -> list[float]:
    """Tempered log-probability of each sample: ``(ids, reason_start, reason_end, ended)``.

    Sums log softmax(z / tau) over the reasoning tokens, plus log P(any stop token) at the end
    when the sample stopped there. Right-padded batches; float32 softmax.
    """
    import torch

    device = next(model.parameters()).device
    stop_index = torch.tensor(sorted(set(stops)), device=device)
    out: list[float] = []
    for begin in range(0, len(items), batch):
        chunk = items[begin:begin + batch]
        width = max(len(item[0]) for item in chunk)
        ids = torch.zeros((len(chunk), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, item in enumerate(chunk):
            ids[row, :len(item[0])] = torch.tensor(item[0])
            mask[row, :len(item[0])] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
            for row, (_, lo, hi, ended) in enumerate(chunk):
                value = 0.0
                if hi > lo:
                    z = torch.log_softmax(logits[row, lo - 1:hi - 1].float() / tau, dim=-1)
                    value = float(z.gather(1, ids[row, lo:hi, None]).sum())
                if ended:
                    zs = torch.log_softmax(logits[row, hi - 1].float() / tau, dim=-1)
                    value += float(torch.logsumexp(zs[stop_index], dim=0))
                out.append(value)
        del logits
    return out
