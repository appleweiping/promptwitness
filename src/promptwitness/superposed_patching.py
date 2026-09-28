"""Renormalized attention patching over superposed prompt blocks (GRAFT estimator).

A first-order gate derivative saturates: at ``g = 0`` it scales with the candidate's
raw attention mass relative to the existing mass, while the real edit is bounded by
softmax renormalization. This estimator keeps the attention step exact and
linearizes only the propagation through later layers:

    dL(b -> a) ~= sum_layers sum_{t downstream} < dL/do_t , o_t(a) - o_t(b) >,

where ``o_t(a) = (N_pre + N_a + N_post) / (Z_pre + Z_a + Z_post)`` is the attention
output of downstream query ``t`` with slot content ``a`` (a candidate, or nothing for
deletion), computed from the *exact* keys/values of the superposed pass. The query
is rotated by the edit's length change so RoPE distances to the prefix and to the
candidate equal those of the real edited prompt. One forward and one backward pass
yield estimates for every candidate of every slot (single-slot edits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .superposed_gates import (
    _ACTIVE,
    GateScorer,
    SuperposedSequence,
    _GateContext,
    _rotate_half,
    gated_mode,
)


@dataclass(slots=True)
class _Alternative:
    key: tuple[str, int | None]
    cand_idx: Any  # LongTensor of physical key indices, or None for deletion
    delta: int


@dataclass(slots=True)
class _SlotPlan:
    block_id: str
    pre: Any  # keys before the slot (non-candidates)
    inc: Any  # incumbent keys (may be empty)
    post: Any  # non-candidate keys after the slot; also the downstream queries
    post_orig: Any
    alternatives: list[_Alternative] = field(default_factory=list)


class _PatchCollector:
    """Saves per-layer q/k/v in forward and accumulates estimates in backward."""

    def __init__(self, plans: list[_SlotPlan], inv_freq: Any, *, check: bool = False) -> None:
        self.plans = plans
        self.inv_freq = inv_freq
        self.check = check
        self.saved: dict[int, tuple[Any, Any, Any, float, Any]] = {}
        self.outputs: dict[int, Any] = {}
        self.hooked: set[int] = set()
        self.scores: dict[tuple[str, int | None], float] = {}
        self.max_base_error = 0.0

    def observe(self, module: Any, query: Any, key: Any, value: Any, scaling: float,
                softcap: Any) -> None:
        # Checkpointed layers are recomputed in backward; keep the first (graph) call.
        if id(module) not in self.saved:
            self.saved[id(module)] = (query.detach(), key.detach(), value.detach(), scaling,
                                      softcap)

    def attach(self, module: Any, output: Any) -> None:
        if id(module) in self.hooked or not output.requires_grad:
            return
        self.hooked.add(id(module))
        layer = id(module)
        if self.check:
            self.outputs[layer] = output.detach()
        output.register_hook(lambda grad: self._accumulate(layer, grad))

    def _rotate(self, q: Any, delta: int) -> Any:
        import torch

        if delta == 0:
            return q
        angles = float(delta) * self.inv_freq.float().to(q.device)
        angles = torch.cat((angles, angles))
        return q * torch.cos(angles) + _rotate_half(q) * torch.sin(angles)

    def _accumulate(self, layer: int, grad: Any) -> None:
        import torch

        query, key, value, scaling, softcap = self.saved[layer]
        q_all, k_all, v_all = query[0].float(), key[0].float(), value[0].float()  # [H,T,D]
        g_all = grad[0].float().transpose(0, 1)  # [H,T,D]
        with torch.no_grad():
            for plan in self.plans:
                q = q_all[:, plan.post]
                g = g_all[:, plan.post]

                def score(qx: Any, idx: Any) -> Any:
                    s = torch.matmul(qx, k_all[:, idx].transpose(1, 2)) * scaling
                    if softcap is not None:
                        s = torch.tanh(s / softcap) * softcap
                    return s

                causal = plan.post_orig[None, :] <= plan.post_orig[:, None]
                s_post = score(q, plan.post).masked_fill(~causal[None], float("-inf"))
                s_pre = score(q, plan.pre) if plan.pre.numel() else None
                s_inc = score(q, plan.inc) if plan.inc.numel() else None
                parts = [x for x in (s_pre, s_inc, s_post) if x is not None]
                m = torch.cat([x.amax(-1, keepdim=True) for x in parts], -1).amax(-1, keepdim=True)

                def mass(s: Any, idx: Any) -> tuple[Any, Any]:
                    e = torch.exp(s - m)
                    return e.sum(-1), torch.matmul(e, v_all[:, idx])

                z_post, n_post = mass(s_post, plan.post)
                z_pre, n_pre = mass(s_pre, plan.pre) if s_pre is not None else (0.0, 0.0)
                z_inc, n_inc = mass(s_inc, plan.inc) if s_inc is not None else (0.0, 0.0)
                base = (n_pre + n_inc + n_post) / (z_pre + z_inc + z_post)[..., None]
                if self.check:
                    actual = self.outputs[layer][0].float().transpose(0, 1)[:, plan.post]
                    self.max_base_error = max(self.max_base_error,
                                              float((actual - base).abs().max()))
                base_dot = (g * base).sum()
                shifted: dict[int, tuple[Any, Any, Any]] = {}
                for alt in plan.alternatives:
                    if alt.delta not in shifted:
                        qd = self._rotate(q, alt.delta)
                        if s_pre is None:
                            shifted[alt.delta] = (qd, 0.0, 0.0)
                        else:
                            zp, np_ = mass(score(qd, plan.pre), plan.pre)
                            shifted[alt.delta] = (qd, zp, np_)
                    qd, zp, np_ = shifted[alt.delta]
                    if alt.cand_idx is None:
                        zc, nc = 0.0, 0.0
                    else:
                        zc, nc = mass(score(qd, alt.cand_idx), alt.cand_idx)
                    patched = (np_ + nc + n_post) / (zp + zc + z_post)[..., None]
                    value_ = float((g * patched).sum() - base_dot)
                    self.scores[alt.key] = self.scores.get(alt.key, 0.0) + value_


def patch_estimates(
    scorer: GateScorer,
    seq: SuperposedSequence,
    *,
    include_deletions: bool = True,
    check: bool = False,
    diagnostics: dict[str, float] | None = None,
) -> tuple[float, dict[tuple[str, int | None], float]]:
    """Base loss and renormalized first-order loss changes for all single-slot edits."""
    import torch

    if seq.mode != "exact":
        raise ValueError("renormalized patching needs mode='exact' positions")
    device = scorer._device()
    orig = torch.tensor(seq.orig)
    is_cand = orig < 0
    # Candidate tokens follow the prompt contiguously, in the order of their gates.
    cursor = next((i for i, o in enumerate(seq.orig) if o < 0), len(seq.orig))
    ranges: dict[int, Any] = {}
    for gate in seq.gates:
        if gate.kind == "candidate":
            ranges[gate.gate_id] = torch.arange(cursor, cursor + gate.token_count, device=device)
            cursor += gate.token_count
    plans: list[_SlotPlan] = []
    for block_id in seq.slots:
        lo, hi = seq.spans[block_id]
        pre = torch.nonzero((~is_cand) & (orig < lo)).flatten()
        inc = torch.nonzero((~is_cand) & (orig >= lo) & (orig < hi)).flatten()
        post = torch.nonzero((~is_cand) & (orig >= hi)).flatten()
        plan = _SlotPlan(block_id, pre.to(device), inc.to(device), post.to(device),
                         orig[post].to(device))
        incumbent = [g for g in seq.gates if g.block_id == block_id and g.kind == "incumbent"]
        base_len = incumbent[0].token_count if incumbent else 0
        for gate in seq.gates:
            if gate.block_id == block_id and gate.kind == "candidate":
                plan.alternatives.append(_Alternative(
                    (block_id, gate.candidate_index), ranges[gate.gate_id],
                    gate.token_count - base_len))
        if include_deletions and incumbent:
            plan.alternatives.append(_Alternative((block_id, None), None, -base_len))
        plans.append(plan)

    collector = _PatchCollector(plans, scorer.inv_freq, check=check)
    gates = torch.tensor(seq.base_values(), device=device, requires_grad=True)
    offsets = torch.tensor(seq.base_offsets(), device=device)
    context: _GateContext = scorer._context(seq, gates, offsets)
    context.patch = collector
    was_training = scorer.model.training
    if scorer.checkpointing:
        scorer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        scorer.model.train()
    previous = _ACTIVE.set(context)
    try:
        with gated_mode(scorer.model), torch.enable_grad():
            loss = scorer._loss(seq)
            loss.backward()
    finally:
        _ACTIVE.reset(previous)
        if scorer.checkpointing:
            scorer.model.gradient_checkpointing_disable()
        scorer.model.train(was_training)
    if diagnostics is not None:
        diagnostics["max_base_error"] = collector.max_base_error
        diagnostics["layers"] = float(len(collector.hooked))
    return float(loss.item()), dict(collector.scores)
