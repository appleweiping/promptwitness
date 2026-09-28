"""Superposed prompt blocks with differentiable attention gates and slot offsets (GRAFT).

Every editable block of a :class:`StructuredPrompt` may carry candidate rewrites of
arbitrary token length. All candidates are placed in *parallel slots* of one token
sequence: a candidate starts at its incumbent's first position, sees only the prefix
before that slot, and is read by downstream tokens through a scalar gate.

Attention with gates is ``p_tu = G_tu exp(s_tu - m_t) / sum_v G_tv exp(s_tv - m_t)``.
The gate multiplies after the max shift, so ``G = 0`` removes a key exactly and the
right derivative at ``G = 0`` is exact.

Length changes are handled by one continuous offset ``w_j`` per slot: every token
after slot ``j`` is moved by ``w_j`` positions, implemented as an extra RoPE rotation
of its query and key (rotations compose and scores depend only on position
differences). Hence, in ``mode="exact"``:

* base point (incumbents 1, candidates 0, offsets 0) = the incumbent prompt;
* vertex (incumbent 0, candidate 1, ``w_j = len(c) - len(b)``) = the rewritten prompt;
* deletion vertex (incumbent 0, ``w_j = -len(b)``) = the prompt without the block;

each up to floating-point accumulation order. ``mode="right"``/``"left"`` drop the
offsets and are ablations whose vertices are only surrogates.

The sampled reasoning is fixed discrete text, as in GReaTer; the derivative is not a
derivative through the sampling decisions that produced it.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .structured_prompt import StructuredPrompt

Mode = Literal["exact", "right", "left"]
ATTENTION_NAME = "promptwitness_gated"


class _ActiveGate:
    """Process-wide slot for the gate context.

    Not a ContextVar: checkpointed layers are recomputed on autograd's device worker
    thread, which does not inherit context variables. One superposed pass at a time.
    """

    def __init__(self) -> None:
        self.value: _GateContext | None = None
        self.lock = threading.Lock()

    def get(self) -> _GateContext | None:
        return self.value

    def set(self, context: _GateContext) -> _GateContext | None:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("another superposed pass is active")
        previous, self.value = self.value, context
        return previous

    def reset(self, previous: _GateContext | None) -> None:
        self.value = previous
        self.lock.release()


_ACTIVE = _ActiveGate()


@dataclass(frozen=True, slots=True)
class GateSpec:
    gate_id: int
    kind: Literal["incumbent", "candidate"]
    block_id: str
    candidate_index: int | None
    token_count: int
    base_value: float


@dataclass(slots=True)
class SuperposedSequence:
    input_ids: list[int]
    position_ids: list[int]
    visible: Any  # torch.BoolTensor [T, T]
    gate_index: Any  # torch.LongTensor [T, T]
    answer_start: int
    answer_ids: list[int]
    gates: list[GateSpec]
    unaligned: list[tuple[str, int, str]] = field(default_factory=list)
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    slots: list[str] = field(default_factory=list)  # offset index -> block id
    offset_matrix: Any = None  # torch.FloatTensor [T, S]; token t moves by M[t] @ w
    mode: str = "exact"
    orig: list[int] = field(default_factory=list)  # logical order; -1 for candidates
    owner: list[str | None] = field(default_factory=list)  # candidate slot per token
    answer_weights: list[float] | None = None  # per-token CE weights over answer_ids

    def base_values(self) -> list[float]:
        return [1.0] + [gate.base_value for gate in self.gates]

    def base_offsets(self) -> list[float]:
        return [0.0] * len(self.slots)

    def gate_for(self, block_id: str, candidate_index: int | None) -> GateSpec:
        for gate in self.gates:
            if gate.block_id == block_id and gate.candidate_index == candidate_index:
                return gate
        raise KeyError((block_id, candidate_index))

    def vertex(
        self, block_id: str, candidate_index: int | None
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Gate and offset overrides that replace (or, with ``None``, delete) a block.

        For an empty insertion slot there is no incumbent: the vertex inserts.
        """
        incumbents = [g for g in self.gates if g.block_id == block_id and g.kind == "incumbent"]
        gates: dict[int, float] = {}
        delta = 0
        if incumbents:
            gates[incumbents[0].gate_id] = 0.0
            delta = -incumbents[0].token_count
        elif candidate_index is None:
            raise ValueError("an empty slot cannot be deleted")
        if candidate_index is not None:
            candidate = self.gate_for(block_id, candidate_index)
            gates[candidate.gate_id] = 1.0
            delta += candidate.token_count
        offsets = {self.slots.index(block_id): float(delta)} if self.mode == "exact" else {}
        return gates, offsets


def _rotate_half(x: Any) -> Any:
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _GateContext:
    def __init__(
        self,
        visible: Any,
        gate_index: Any,
        gate_values: Any,
        offsets: Any = None,
        offset_matrix: Any = None,
        inv_freq: Any = None,
    ) -> None:
        import torch

        self.visible = visible
        # Built here, outside every checkpointed layer, so recomputation during
        # backward reuses one graph node instead of re-saving it per layer.
        self.gate_matrix = gate_values[gate_index].float()
        self.cos: Any = None
        self.sin: Any = None
        self.patch: Any = None  # set by superposed_patching for renormalized estimates
        if offsets is not None and offset_matrix is not None and offset_matrix.shape[1]:
            shift = offset_matrix.float() @ offsets.float()  # [T]
            angles = shift[:, None] * inv_freq.float()[None, :]  # [T, D/2]
            angles = torch.cat((angles, angles), dim=-1)  # rotate_half layout
            self.cos, self.sin = torch.cos(angles), torch.sin(angles)


def _block_token_span(
    prompt: StructuredPrompt, tokenizer: Any, values: Mapping[str, object], block_id: str
) -> tuple[tuple[int, ...], int, int]:
    """Token span ``[lo, hi)`` of a block; an empty block gives ``lo == hi``.

    An empty (optional) block marks an insertion point and must fall on a clean
    token boundary.
    """
    rendered = prompt.render_tokens(tokenizer, values)
    positions = rendered.block_positions[block_id]
    if not positions:
        char_lo, char_hi = rendered.block_char_spans[block_id]
        if char_lo != char_hi:
            raise ValueError(f"block {block_id!r} has no whole token")
        for index, (start, end) in enumerate(rendered.token_offsets):
            if start >= char_lo:
                if index and rendered.token_offsets[index - 1][1] > char_lo:
                    raise ValueError(f"insertion slot {block_id!r} splits a token")
                return rendered.input_ids, index, index
        return rendered.input_ids, len(rendered.input_ids), len(rendered.input_ids)
    lo, hi = positions[0], positions[-1] + 1
    if tuple(range(lo, hi)) != positions:
        raise ValueError(f"block {block_id!r} tokens are not contiguous")
    return rendered.input_ids, lo, hi


def candidate_token_ids(
    prompt: StructuredPrompt,
    tokenizer: Any,
    values: Mapping[str, object],
    block_id: str,
    text: str,
) -> tuple[int, ...] | None:
    """Tokens of a rewrite when it partitions cleanly between unchanged prefix/suffix.

    Returns ``None`` if the rewritten full rendering does not keep the incumbent's
    prefix tokens before the slot and suffix tokens after it (a boundary merge).
    """
    base_ids, lo, hi = _block_token_span(prompt, tokenizer, values, block_id)
    try:
        rewritten = prompt.replace_block(block_id, text)
    except ValueError:
        return None
    new_ids = rewritten.render_tokens(tokenizer, values).input_ids
    suffix = base_ids[hi:]
    if tuple(new_ids[:lo]) != tuple(base_ids[:lo]):
        return None
    if suffix and tuple(new_ids[len(new_ids) - len(suffix) :]) != tuple(suffix):
        return None
    middle = tuple(new_ids[lo : len(new_ids) - len(suffix)])
    return middle or None


def build_superposed(
    prompt: StructuredPrompt,
    tokenizer: Any,
    values: Mapping[str, object],
    candidates: Mapping[str, Sequence[str]],
    tail_ids: Sequence[int],
    answer_ids: Sequence[int],
    *,
    mode: Mode = "exact",
    gate_incumbents: Sequence[str] | None = None,
    answer_weights: Sequence[float] | None = None,
) -> SuperposedSequence:
    """Lay out incumbent prompt, parallel candidate slots, then reasoning+extractor+answer.

    ``tail_ids`` precede the scored tokens ``answer_ids`` (GReaTer: reasoning plus
    extractor, then the answer). ``answer_weights`` optionally weights the per-token
    cross-entropy of ``answer_ids`` (e.g. verified reasoning and answer scored, extractor
    tokens weighted 0). ``gate_incumbents`` lists incumbent blocks that receive a gate
    (deletion effect); by default every non-empty editable block.
    """
    import torch

    if not answer_ids:
        raise ValueError("answer tokens are required")
    if answer_weights is not None and (
        len(answer_weights) != len(answer_ids) or not any(answer_weights)
    ):
        raise ValueError("answer weights must match the answer and not all be zero")
    base_ids = list(prompt.render_tokens(tokenizer, values).input_ids)
    n_prompt = len(base_ids)
    editable = [block.block_id for block in prompt.blocks if block.editable and block.text]
    gated_blocks = list(gate_incumbents) if gate_incumbents is not None else editable
    spans: dict[str, tuple[int, int]] = {}
    for block_id in list(dict.fromkeys(list(gated_blocks) + list(candidates))):
        _, lo, hi = _block_token_span(prompt, tokenizer, values, block_id)
        spans[block_id] = (lo, hi)
    if any(spans[block_id][0] == spans[block_id][1] for block_id in gated_blocks):
        raise ValueError("an empty insertion slot has no incumbent to gate")
    ungated = [b for b in candidates if spans[b][0] != spans[b][1] and b not in gated_blocks]
    if ungated:
        raise ValueError(f"non-empty slots with candidates must be gated: {ungated}")

    gates: list[GateSpec] = []
    unaligned: list[tuple[str, int, str]] = []
    orig: list[int] = list(range(n_prompt))  # logical order; -1 for candidates
    segment: list[int] = [0] * n_prompt  # gate id of own segment (0 = ungated)
    cand_block: list[str | None] = [None] * n_prompt
    input_ids = list(base_ids)
    position_ids = list(range(n_prompt))

    for block_id in gated_blocks:
        lo, hi = spans[block_id]
        gate = GateSpec(len(gates) + 1, "incumbent", block_id, None, hi - lo, 1.0)
        gates.append(gate)
        for index in range(lo, hi):
            segment[index] = gate.gate_id

    for block_id, texts in candidates.items():
        lo, hi = spans[block_id]
        for candidate_index, text in enumerate(texts):
            if text is None:  # placeholder: keeps indices stable when passes are chunked
                continue
            ids = candidate_token_ids(prompt, tokenizer, values, block_id, text)
            if ids is None:
                unaligned.append((block_id, candidate_index, text))
                continue
            gate = GateSpec(len(gates) + 1, "candidate", block_id, candidate_index, len(ids), 0.0)
            gates.append(gate)
            start = max(hi - len(ids), 0) if mode == "right" else lo
            for offset, token in enumerate(ids):
                input_ids.append(int(token))
                position_ids.append(start + offset)
                orig.append(-1)
                segment.append(gate.gate_id)
                cand_block.append(block_id)

    # Physical layout: prompt[:-1], candidates, prompt[-1], tail. The token physically
    # before the scored region must be its logical predecessor even when the tail is
    # empty; visibility is explicit, so physical order is otherwise irrelevant.
    for store in (input_ids, position_ids, orig, segment, cand_block):
        store.append(store.pop(n_prompt - 1))

    tail_start = len(input_ids)
    tail = list(tail_ids) + list(answer_ids)
    for offset, token in enumerate(tail):
        input_ids.append(int(token))
        position_ids.append(n_prompt + offset)
        orig.append(n_prompt + offset)
        segment.append(0)
        cand_block.append(None)
    answer_start = tail_start + len(tail_ids)
    total = len(input_ids)

    slots = sorted(spans, key=lambda block_id: (spans[block_id][0], spans[block_id][1]))
    order = {block_id: rank for rank, block_id in enumerate(slots)}
    offset_matrix = torch.zeros((total, len(slots)))
    if mode == "exact":
        for column, block_id in enumerate(slots):
            hi = spans[block_id][1]
            for index in range(total):
                owner = cand_block[index]
                # Prompt/tail tokens after the slot move with it; a candidate moves with
                # every *other* slot that precedes its own slot in document order.
                moved = orig[index] >= hi if owner is None else order[block_id] < order[owner]
                if moved:
                    offset_matrix[index, column] = 1.0

    orig_t = torch.tensor(orig)
    seg_t = torch.tensor(segment)
    is_cand = orig_t < 0
    block_lo = torch.full((total,), -1, dtype=torch.long)
    block_hi = torch.full((total,), -1, dtype=torch.long)
    slot_rank = torch.full((total,), -1, dtype=torch.long)
    for index, block_id in enumerate(cand_block):
        if block_id is not None:
            block_lo[index], block_hi[index] = spans[block_id]
            slot_rank[index] = order[block_id]

    q = torch.arange(total)[:, None]
    k = torch.arange(total)[None, :]
    q_orig, k_orig = orig_t[:, None], orig_t[None, :]
    q_cand, k_cand = is_cand[:, None], is_cand[None, :]
    # Rule 1: ordinary causal attention among prompt and tail tokens.
    visible = (~q_cand) & (~k_cand) & (k_orig <= q_orig)
    # Rule 2: a candidate sees the non-candidate prefix before its slot.
    visible |= q_cand & (~k_cand) & (k_orig < block_lo[:, None])
    # Rule 3a: causal attention inside one candidate.
    same_seg = seg_t[:, None] == seg_t[None, :]
    visible |= q_cand & k_cand & same_seg & (k <= q)
    # Rule 3b: a candidate sees candidates of strictly earlier slots (gated).
    visible |= q_cand & k_cand & (slot_rank[None, :] < slot_rank[:, None])
    # Rule 4: downstream non-candidates read candidates after the incumbent slot end.
    visible |= (~q_cand) & k_cand & (q_orig >= block_hi[None, :])
    gate_index = torch.where(
        (seg_t[None, :] > 0) & (~same_seg), seg_t[None, :].expand(total, total), 0
    )
    return SuperposedSequence(
        input_ids,
        position_ids,
        visible,
        gate_index,
        answer_start,
        [int(value) for value in answer_ids],
        gates,
        unaligned,
        spans,
        slots,
        offset_matrix,
        mode,
        orig,
        cand_block,
        None if answer_weights is None else [float(w) for w in answer_weights],
    )


def gated_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Eager attention under an explicit visibility matrix, gates and slot offsets."""
    import torch

    context = _ACTIVE.get()
    if context is None:
        # Transformers builds no mask for an unregistered mask name, so running this
        # function without our explicit visibility matrix would silently be non-causal.
        raise RuntimeError("gated attention used outside gated_mode()")
    out_dtype = query.dtype
    if context.cos is not None:
        # Extra RoPE rotation by each token's slot offset: R(p + w) = R(w) R(p).
        cos, sin = context.cos[None, None], context.sin[None, None]
        q32, k32 = query.float(), key.float()
        query = q32 * cos + _rotate_half(q32) * sin
        key = k32 * cos + _rotate_half(k32) * sin
        value = value.float()
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        batch, heads, length, dim = key.shape
        key = key[:, :, None].expand(batch, heads, groups, length, dim).reshape(
            batch, heads * groups, length, dim
        )
        value = value[:, :, None].expand(batch, heads, groups, length, dim).reshape(
            batch, heads * groups, length, dim
        )
    if context.patch is not None:
        context.patch.observe(module, query, key, value, scaling, kwargs.get("softcap"))
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    softcap = kwargs.get("softcap")
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    window = kwargs.get("sliding_window")
    if window is not None and max(context.visible.shape[-2:]) > window:
        raise ValueError("superposed sequence exceeds a sliding attention window")
    s = scores.float().masked_fill(~context.visible, float("-inf"))
    shifted = torch.exp(s - s.amax(dim=-1, keepdim=True))
    weighted = shifted * context.gate_matrix
    weights = (weighted / weighted.sum(dim=-1, keepdim=True)).to(value.dtype)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous().to(out_dtype)
    if context.patch is not None:
        context.patch.attach(module, output)
    return output, None


class gated_mode:  # noqa: N801 - used as a context manager
    """Temporarily route every attention layer of ``model`` through the gated kernel."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.previous: str | None = None

    def __enter__(self) -> None:
        from transformers import AttentionInterface

        AttentionInterface.register(ATTENTION_NAME, gated_attention_forward)
        self.previous = self.model.config._attn_implementation
        self.model.config._attn_implementation = ATTENTION_NAME

    def __exit__(self, *exc: object) -> None:
        self.model.config._attn_implementation = self.previous


def rotary_inv_freq(model: Any) -> Any:
    """The single RoPE frequency vector used by the model's attention layers."""
    import torch

    found = [
        module.inv_freq
        for module in model.modules()
        if isinstance(getattr(module, "inv_freq", None), torch.Tensor)
    ]
    if not found:
        raise ValueError("model has no rotary embedding with inv_freq")
    first = found[0]
    if any(item.shape != first.shape or not torch.equal(item, first) for item in found[1:]):
        raise ValueError("models with several RoPE frequency sets are not supported")
    return first


@dataclass(frozen=True, slots=True)
class GateGradient:
    loss: float
    gates: list[float]  # index 0 is the fixed ungated 1
    offsets: list[float]


class GateScorer:
    """Answer loss and gate/offset gradients over one superposed sequence."""

    def __init__(self, model: Any, *, checkpointing: bool = True) -> None:
        self.model = model
        self.checkpointing = checkpointing
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.inv_freq = rotary_inv_freq(model)

    def _device(self) -> Any:
        return next(self.model.parameters()).device

    def _loss(self, seq: SuperposedSequence) -> Any:
        """Answer CE; caller must hold an active gate context and gated_mode."""
        import torch
        import torch.nn.functional as functional

        device = self._device()
        ids = torch.tensor([seq.input_ids], device=device)
        positions = torch.tensor([seq.position_ids], device=device)
        keep = len(seq.answer_ids) + 1
        output = self.model(
            input_ids=ids, position_ids=positions, use_cache=False, logits_to_keep=keep
        )
        logits = output.logits[0, :-1].float()
        target = torch.tensor(seq.answer_ids, device=device)
        if seq.answer_weights is None:
            return functional.cross_entropy(logits, target)
        weights = torch.tensor(seq.answer_weights, device=device)
        per_token = functional.cross_entropy(logits, target, reduction="none")
        return (per_token * weights).sum() / weights.sum()

    def _context(self, seq: SuperposedSequence, gate_values: Any, offsets: Any) -> _GateContext:
        device = self._device()
        return _GateContext(
            seq.visible.to(device)[None, None],
            seq.gate_index.to(device),
            gate_values,
            offsets,
            seq.offset_matrix.to(device) if seq.offset_matrix is not None else None,
            self.inv_freq.to(device),
        )

    def _values(
        self,
        seq: SuperposedSequence,
        gate_overrides: Mapping[int, float] | None,
        offset_overrides: Mapping[int, float] | None,
    ) -> tuple[list[float], list[float]]:
        gates, offsets = seq.base_values(), seq.base_offsets()
        for gate_id, value in (gate_overrides or {}).items():
            gates[gate_id] = value
        for slot, value in (offset_overrides or {}).items():
            offsets[slot] = value
        return gates, offsets

    def gradients(
        self,
        seq: SuperposedSequence,
        gate_overrides: Mapping[int, float] | None = None,
        offset_overrides: Mapping[int, float] | None = None,
    ) -> GateGradient:
        """Loss and derivatives w.r.t. every gate and slot offset at the given point.

        The gate context and gated kernel stay active through backward because
        checkpointed layers are recomputed there.
        """
        import torch

        device = self._device()
        gate_list, offset_list = self._values(seq, gate_overrides, offset_overrides)
        gates = torch.tensor(gate_list, device=device, requires_grad=True)
        offsets = torch.tensor(offset_list, device=device, requires_grad=True)
        was_training = self.model.training
        if self.checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.train()
        previous = _ACTIVE.set(self._context(seq, gates, offsets))
        try:
            with gated_mode(self.model), torch.enable_grad():
                loss = self._loss(seq)
                inputs = [gates] + ([offsets] if offset_list else [])
                grads = torch.autograd.grad(loss, inputs, allow_unused=True)
        finally:
            _ACTIVE.reset(previous)
            if self.checkpointing:
                self.model.gradient_checkpointing_disable()
            self.model.train(was_training)
        gate_grad = grads[0].tolist()
        offset_grad = (
            grads[1].tolist() if offset_list and grads[1] is not None else [0.0] * len(offset_list)
        )
        return GateGradient(float(loss.item()), gate_grad, offset_grad)

    def value_at(
        self,
        seq: SuperposedSequence,
        gate_overrides: Mapping[int, float] | None = None,
        offset_overrides: Mapping[int, float] | None = None,
    ) -> float:
        import torch

        device = self._device()
        gate_list, offset_list = self._values(seq, gate_overrides, offset_overrides)
        gates = torch.tensor(gate_list, device=device)
        offsets = torch.tensor(offset_list, device=device)
        previous = _ACTIVE.set(self._context(seq, gates, offsets))
        try:
            with gated_mode(self.model), torch.no_grad():
                return float(self._loss(seq).item())
        finally:
            _ACTIVE.reset(previous)


def edit_scores(seq: SuperposedSequence, grad: GateGradient) -> dict[str, Any]:
    """First-order loss changes along straight lines from the base point to each vertex.

    Replacement ``b -> c``: ``dL/dg_c - dL/dg_b + (len c - len b) dL/dw_slot``;
    deletion of ``b``: ``-dL/dg_b - len(b) dL/dw_slot``; insertion: ``dL/dg_c``.
    Offset terms are zero outside ``mode="exact"``.
    """
    exact = seq.mode == "exact"
    incumbents = {g.block_id: g for g in seq.gates if g.kind == "incumbent"}

    def offset_grad(block_id: str) -> float:
        return grad.offsets[seq.slots.index(block_id)] if exact else 0.0

    deletion = {
        block: -grad.gates[g.gate_id] - g.token_count * offset_grad(block)
        for block, g in incumbents.items()
    }
    # Replacement for gated slots, pure insertion for empty slots.
    replacement: dict[str, dict[int, float]] = {}
    insertion: dict[str, dict[int, float]] = {}
    for gate in seq.gates:
        if gate.kind != "candidate" or gate.candidate_index is None:
            continue
        incumbent = incumbents.get(gate.block_id)
        insertion.setdefault(gate.block_id, {})[gate.candidate_index] = grad.gates[
            gate.gate_id
        ] + gate.token_count * offset_grad(gate.block_id)
        delta = gate.token_count - (incumbent.token_count if incumbent else 0)
        replacement.setdefault(gate.block_id, {})[gate.candidate_index] = (
            grad.gates[gate.gate_id]
            - (grad.gates[incumbent.gate_id] if incumbent else 0.0)
            + delta * offset_grad(gate.block_id)
        )
    return {"deletion": deletion, "insertion": insertion, "replacement": replacement}
