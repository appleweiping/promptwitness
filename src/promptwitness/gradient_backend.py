"""White-box conditional answer gradients for a frozen causal language model.

The sampled reasoning is fixed discrete text during differentiation. This is
not a gradient through the sampling decisions that produced that reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

from .structured_prompt import StructuredPrompt, TokenizedPrompt


@dataclass(frozen=True, slots=True)
class GradientObservation:
    loss: float
    reasoning: str
    answer_tokens: tuple[int, ...]
    block_sensitivity: dict[str, float]
    token_gradients: dict[str, tuple[tuple[float, ...], ...]]
    token_positions: dict[str, tuple[int, ...]]
    generation_seconds: float
    gradient_seconds: float
    generated_tokens: int
    forward_tokens: int
    backward_tokens: int
    finite_difference: dict[str, float] | None = None


class FrozenGradientBackend:
    """Calculate d(answer CE)/d(prompt input embedding), with frozen weights."""

    def __init__(self, model: Any, tokenizer: Any, *, extractor: str = "\nFinal answer: ") -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.extractor = extractor
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _device(self) -> Any:
        return next(self.model.parameters()).device

    def _loss(self, embeddings: Any, answer_start: int, answer_ids: Any) -> Any:
        import torch.nn.functional as functional

        output = self.model(inputs_embeds=embeddings, use_cache=False)
        logits = output.logits[:, answer_start - 1 : answer_start + answer_ids.numel() - 1]
        return functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), answer_ids.reshape(-1)
        )

    def generate_response(
        self,
        prompt: StructuredPrompt,
        values: dict[str, object],
        *,
        max_new_tokens: int = 96,
    ) -> tuple[str, int, float]:
        """Generate fresh task output; no gold answer is an input to this method."""
        import torch

        rendered = prompt.render_tokens(self.tokenizer, values)
        input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=self._device())
        started = perf_counter()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0, input_ids.shape[1] :].tolist()
        return (
            self.tokenizer.decode(output_ids, skip_special_tokens=True),
            len(output_ids),
            perf_counter() - started,
        )

    def observe(
        self,
        prompt: StructuredPrompt,
        values: dict[str, object],
        gold_answer: str,
        *,
        max_reasoning_tokens: int = 96,
        finite_difference_block: str | None = None,
        finite_difference_epsilon: float = 0.1,
    ) -> GradientObservation:
        import torch

        if not gold_answer or max_reasoning_tokens < 1:
            raise ValueError("gold answer and positive reasoning token limit are required")
        if finite_difference_epsilon <= 0:
            raise ValueError("finite-difference epsilon must be positive")
        rendered: TokenizedPrompt = prompt.render_tokens(self.tokenizer, values)
        if not rendered.input_ids:
            raise ValueError("rendered prompt has no tokens")
        device = self._device()
        prompt_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
        started = perf_counter()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=max_reasoning_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        reasoning_ids = generated[0, prompt_ids.shape[1] :].tolist()
        eos = self.tokenizer.eos_token_id
        while reasoning_ids and eos is not None and reasoning_ids[-1] == eos:
            reasoning_ids.pop()
        generation_seconds = perf_counter() - started
        reasoning = self.tokenizer.decode(reasoning_ids, skip_special_tokens=True)
        extractor_ids = self.tokenizer.encode(self.extractor, add_special_tokens=False)
        answer_ids_list = self.tokenizer.encode(gold_answer, add_special_tokens=False)
        if not extractor_ids or not answer_ids_list:
            raise ValueError("extractor and gold answer must each tokenize to nonempty sequences")
        prefix = list(rendered.input_ids) + reasoning_ids + extractor_ids
        all_ids = torch.tensor([prefix + answer_ids_list], dtype=torch.long, device=device)
        answer_ids = torch.tensor(answer_ids_list, dtype=torch.long, device=device)
        answer_start = len(prefix)

        started = perf_counter()
        with torch.enable_grad():
            embeddings = self.model.get_input_embeddings()(all_ids).detach().requires_grad_(True)
            loss = self._loss(embeddings, answer_start, answer_ids)
            loss.backward()
            assert embeddings.grad is not None
            gradients = embeddings.grad[0, : len(rendered.input_ids)].detach().float()
        gradient_seconds = perf_counter() - started
        loss_value = float(loss.detach().float().item())
        if not isfinite(loss_value) or not bool(torch.isfinite(gradients).all()):
            raise ValueError("answer loss or input gradients are non-finite")
        token_gradients: dict[str, tuple[tuple[float, ...], ...]] = {}
        sensitivity: dict[str, float] = {}
        for block in prompt.blocks:
            positions = rendered.block_positions[block.block_id] if block.editable else ()
            if not positions:
                continue
            selected = gradients[list(positions)]
            token_gradients[block.block_id] = tuple(
                tuple(float(value) for value in row) for row in selected.tolist()
            )
            sensitivity[block.block_id] = float(selected.norm(dim=-1).mean().item())
        if not sensitivity or not any(value > 0 for value in sensitivity.values()):
            raise ValueError("no nonzero editable-block gradient")

        finite_difference: dict[str, float] | None = None
        if finite_difference_block is not None:
            positions = rendered.block_positions.get(finite_difference_block, ())
            if not positions:
                raise ValueError("finite-difference block has no editable token")
            position = positions[0]
            direction = gradients[position]
            direction = direction / direction.norm()
            epsilon = finite_difference_epsilon
            with torch.no_grad():
                plus = embeddings.detach().clone()
                minus = embeddings.detach().clone()
                plus[0, position] += epsilon * direction.to(plus.dtype)
                minus[0, position] -= epsilon * direction.to(minus.dtype)
                plus_loss = self._loss(plus, answer_start, answer_ids)
                minus_loss = self._loss(minus, answer_start, answer_ids)
            measured = float(((plus_loss - minus_loss) / (2 * epsilon)).float().item())
            predicted = float(gradients[position].dot(direction).item())
            finite_difference = {
                "epsilon": epsilon,
                "predicted": predicted,
                "measured": measured,
            }

        return GradientObservation(
            loss=loss_value,
            reasoning=reasoning,
            answer_tokens=tuple(answer_ids_list),
            block_sensitivity=sensitivity,
            token_gradients=token_gradients,
            token_positions={key: tuple(value) for key, value in rendered.block_positions.items()},
            generation_seconds=generation_seconds,
            gradient_seconds=gradient_seconds,
            generated_tokens=len(reasoning_ids),
            forward_tokens=len(prefix) + len(answer_ids_list),
            backward_tokens=len(prefix) + len(answer_ids_list),
            finite_difference=finite_difference,
        )
