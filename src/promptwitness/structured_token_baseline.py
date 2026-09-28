"""A constrained token-coordinate adaptation of GReaTer for the pilot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .gradient_backend import GradientObservation
from .structured_prompt import StructuredPrompt


@dataclass(frozen=True, slots=True)
class TokenProposal:
    prompt: StructuredPrompt
    block_id: str
    position: int
    old_token: int
    new_token: int
    predicted_delta: float


def propose_token_replacements(
    model: Any,
    tokenizer: Any,
    parent: StructuredPrompt,
    rows_and_gradients: tuple[tuple[str, GradientObservation], ...],
    round_index: int,
    *,
    topk: int = 11,
    shortlist: int = 6,
    on_prefix_forward: Callable[[str, int], None] | None = None,
) -> tuple[tuple[TokenProposal, ...], int]:
    """LM top-k proposals, gradient ranking, strict one-token render parity.

    This follows the core proposal/gradient/update path of the official code,
    but uses PromptWitness's frozen document contract and a different model.
    It is an adapted reproduction, not the published experiment setting.
    """
    import torch

    if not rows_and_gradients or topk < 1 or shortlist < 1:
        raise ValueError("nonempty gradient batch and positive candidate limits required")
    first = parent.render_tokens(tokenizer, {"input": rows_and_gradients[0][0]})
    coordinates = [
        (block.block_id, position)
        for block in parent.blocks
        if block.editable
        for position in first.block_positions[block.block_id]
    ]
    if not coordinates:
        raise ValueError("no editable token coordinates")
    # Spread the eight allowed search rounds over the full editable prompt.
    block_id, position = coordinates[(round_index * len(coordinates) // 8) % len(coordinates)]
    block_positions = first.block_positions[block_id]
    ordinal = block_positions.index(position)
    block = next(block for block in parent.blocks if block.block_id == block_id)
    char_start, _ = first.block_char_spans[block_id]
    token_start, token_end = first.token_offsets[position]
    relative_start = token_start - char_start
    relative_end = token_end - char_start
    old_token = first.input_ids[position]
    device = next(model.parameters()).device

    proposal_ids: set[int] = set()
    proposal_forward_tokens = 0
    for question, _ in rows_and_gradients:
        mapped = parent.render_tokens(tokenizer, {"input": question})
        mapped_position = mapped.block_positions[block_id][ordinal]
        if mapped.input_ids[mapped_position] != old_token:
            raise ValueError("selected token differs across task-conditioned prefixes")
        prefix = torch.tensor([mapped.input_ids[:mapped_position]], dtype=torch.long, device=device)
        if on_prefix_forward is not None:
            on_prefix_forward("attempt", prefix.shape[1])
        with torch.no_grad():
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1]
        if on_prefix_forward is not None:
            on_prefix_forward("success", prefix.shape[1])
        proposal_forward_tokens += prefix.shape[1]
        proposal_ids.update(int(item) for item in logits.topk(topk).indices.tolist())

    gradients = [
        observation.token_gradients[block_id][ordinal] for _, observation in rows_and_gradients
    ]
    mean_gradient = torch.tensor(gradients, device=device, dtype=torch.float32).mean(dim=0)
    weights = model.get_input_embeddings().weight.detach()
    candidates: list[TokenProposal] = []
    for new_token in sorted(proposal_ids):
        if new_token == old_token:
            continue
        replacement = tokenizer.decode(
            [new_token], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not replacement or not replacement.isascii() or not replacement.isprintable():
            continue
        new_text = block.text[:relative_start] + replacement + block.text[relative_end:]
        try:
            candidate = parent.replace_block(block_id, new_text)
        except ValueError:
            continue
        aligned = True
        for question, _ in rows_and_gradients:
            old = parent.render_tokens(tokenizer, {"input": question})
            new = candidate.render_tokens(tokenizer, {"input": question})
            row_position = old.block_positions[block_id][ordinal]
            if (
                len(old.input_ids) != len(new.input_ids)
                or new.input_ids[row_position] != new_token
                or any(
                    before != after
                    for index, (before, after) in enumerate(
                        zip(old.input_ids, new.input_ids, strict=True)
                    )
                    if index != row_position
                )
            ):
                aligned = False
                break
        if not aligned:
            continue
        direction = weights[new_token].float() - weights[old_token].float()
        delta = float(mean_gradient.dot(direction).item())
        candidates.append(TokenProposal(candidate, block_id, position, old_token, new_token, delta))
    candidates.sort(key=lambda item: (item.predicted_delta, item.new_token))
    return tuple(candidates[:shortlist]), proposal_forward_tokens
