"""Exactness of superposed gates on tiny random fp32 models (needs torch+transformers).

Propositions checked, for Llama, Gemma-2 (softcapping) and OLMo-3 architectures:
  P1  the base point equals the stock forward of the incumbent prompt;
  P2  in ``mode="exact"`` every replacement vertex equals the stock forward of the
      actually rewritten prompt with ordinary positions, including length changes;
  P2d the deletion vertex equals the stock forward of the prompt without the block;
  P3  gate and slot-offset gradients match finite differences (one-sided at g=0).
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from promptwitness.models import Message, PromptDocument  # noqa: E402
from promptwitness.structured_prompt import BlockKind, PromptBlock, StructuredPrompt  # noqa: E402
from promptwitness.superposed_gates import GateScorer, build_superposed  # noqa: E402

TOKENIZER = os.environ.get("PW_TEST_TOKENIZER")
pytestmark = pytest.mark.skipif(TOKENIZER is None, reason="set PW_TEST_TOKENIZER to a local path")

VALUES = {"input": "Which is larger, 3 or 5?\nOptions:\n(A) 3\n(B) 5"}
CANDIDATES = {
    "task": ["Solve the question carefully and check each option.\n", "Solve it.\n"],
    "reasoning": ["Reason briefly.\n", "Think about every option one at a time, then decide.\n"],
}


def _prompt() -> StructuredPrompt:
    pieces = (
        ("input", BlockKind.INPUT_DATA, "Question:\n{{input}}\n\n", False),
        ("task", BlockKind.TASK_INSTRUCTION, "Solve the question carefully.\n", True),
        ("reasoning", BlockKind.REASONING_POLICY, "Think step by step before answering.\n", True),
        ("output", BlockKind.OUTPUT_INSTRUCTION, "End with Final answer: (X).\n", True),
    )
    blocks, content = [], ""
    for block_id, kind, text, editable in pieces:
        start = len(content)
        content += text
        blocks.append(PromptBlock(block_id, kind, text, editable, "q", start, len(content)))
    document = PromptDocument("t", (Message("user", content, message_id="q"),))
    return StructuredPrompt(document, tuple(blocks))


def _model(arch: str, vocab: int):
    torch.manual_seed(0)
    common = dict(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    if arch == "llama":
        config, cls = transformers.LlamaConfig(**common), transformers.LlamaForCausalLM
    elif arch == "gemma2":
        config = transformers.Gemma2Config(**common, head_dim=16, sliding_window=4096,
                                           attn_logit_softcapping=50.0,
                                           final_logit_softcapping=30.0)
        cls = transformers.Gemma2ForCausalLM
    else:
        config, cls = transformers.Olmo3Config(**common, sliding_window=4096), transformers.Olmo3ForCausalLM
    config._attn_implementation = "eager"
    return cls(config).float().eval()


def _stock(model, ids, answer) -> float:
    import torch.nn.functional as F

    tensor = torch.tensor([ids + answer])
    with torch.no_grad():
        logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                       use_cache=False).logits[0]
    pred = logits[len(ids) - 1 : len(ids) - 1 + len(answer)].float()
    return float(F.cross_entropy(pred, torch.tensor(answer)).item())


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.mark.parametrize("arch", ["llama", "gemma2", "olmo3"])
def test_exact_vertices_and_gradients(arch: str, tokenizer) -> None:
    model = _model(arch, len(tokenizer))
    prompt = _prompt()
    tail = tokenizer.encode("Five is larger.\nFinal answer: ", add_special_tokens=False)
    answer = tokenizer.encode("(B)", add_special_tokens=False)
    seq = build_superposed(prompt, tokenizer, VALUES, CANDIDATES, tail, answer, mode="exact")
    assert not seq.unaligned
    assert len({g.token_count for g in seq.gates if g.block_id == "task"}) == 3  # lengths differ
    scorer = GateScorer(model, checkpointing=False)
    base_ids = list(prompt.render_tokens(tokenizer, VALUES).input_ids)

    grad = scorer.gradients(seq)  # P1
    assert abs(grad.loss - _stock(model, base_ids + tail, answer)) < 1e-4

    for gate in [g for g in seq.gates if g.kind == "candidate"]:  # P2
        gates, offsets = seq.vertex(gate.block_id, gate.candidate_index)
        vertex = scorer.value_at(seq, gates, offsets)
        text = CANDIDATES[gate.block_id][gate.candidate_index]
        rewritten = list(prompt.replace_block(gate.block_id, text).render_tokens(tokenizer, VALUES).input_ids)
        assert abs(vertex - _stock(model, rewritten + tail, answer)) < 1e-4, gate

    for block_id in ("task", "reasoning", "output"):  # P2d
        gates, offsets = seq.vertex(block_id, None)
        lo, hi = seq.spans[block_id]
        without = base_ids[:lo] + base_ids[hi:]
        assert abs(scorer.value_at(seq, gates, offsets) - _stock(model, without + tail, answer)) < 1e-4

    eps = 1e-3  # P3
    for gate in seq.gates:
        if gate.kind == "incumbent":
            fd = (scorer.value_at(seq, {gate.gate_id: 1 + eps})
                  - scorer.value_at(seq, {gate.gate_id: 1 - eps})) / (2 * eps)
        else:
            fd = (scorer.value_at(seq, {gate.gate_id: eps}) - grad.loss) / eps
        assert abs(fd - grad.gates[gate.gate_id]) <= 2e-3 + 2e-2 * abs(fd), (gate, fd)
    for slot in range(len(seq.slots)):
        fd = (scorer.value_at(seq, None, {slot: eps})
              - scorer.value_at(seq, None, {slot: -eps})) / (2 * eps)
        assert abs(fd - grad.offsets[slot]) <= 2e-3 + 2e-2 * abs(fd), (slot, fd)


def test_surrogate_modes_differ_from_exact(tokenizer) -> None:
    """Without offsets a length-changing vertex is only a surrogate (ablation sanity)."""
    model = _model("llama", len(tokenizer))
    prompt = _prompt()
    tail = tokenizer.encode("Five is larger.\nFinal answer: ", add_special_tokens=False)
    answer = tokenizer.encode("(B)", add_special_tokens=False)
    scorer = GateScorer(model, checkpointing=False)
    text = CANDIDATES["task"][0]
    rewritten = list(prompt.replace_block("task", text).render_tokens(tokenizer, VALUES).input_ids)
    truth = _stock(model, rewritten + tail, answer)
    gaps = {}
    for mode in ("exact", "right", "left"):
        seq = build_superposed(prompt, tokenizer, VALUES, {"task": [text]}, tail, answer, mode=mode)
        gates, offsets = seq.vertex("task", 0)
        gaps[mode] = abs(scorer.value_at(seq, gates, offsets) - truth)
    # A tiny random model is weakly position-sensitive, so compare relative residuals.
    assert gaps["exact"] < 1e-5
    assert gaps["right"] > 10 * gaps["exact"] and gaps["left"] > 10 * gaps["exact"], gaps


def _prompt_with_slot() -> StructuredPrompt:
    pieces = (
        ("input", BlockKind.INPUT_DATA, "Question:\n{{input}}\n\n", False),
        ("task", BlockKind.TASK_INSTRUCTION, "Solve the question carefully.\n", True),
        ("reasoning", BlockKind.REASONING_POLICY, "Think step by step before answering.\n", True),
        ("procedure", BlockKind.REASONING_POLICY, "", True),
        ("output", BlockKind.OUTPUT_INSTRUCTION, "End with Final answer: (X).\n", True),
    )
    blocks, content = [], ""
    for block_id, kind, text, editable in pieces:
        start = len(content)
        content += text
        blocks.append(PromptBlock(block_id, kind, text, editable, "q", start, len(content)))
    document = PromptDocument("t", (Message("user", content, message_id="q"),))
    return StructuredPrompt(document, tuple(blocks))


@pytest.mark.parametrize("arch", ["llama", "gemma2"])
def test_insertion_and_combined_vertices(arch: str, tokenizer) -> None:
    model = _model(arch, len(tokenizer))
    prompt = _prompt_with_slot()
    candidates = dict(CANDIDATES)
    candidates["procedure"] = ["Check each option against the question.\n", "Double-check.\n"]
    tail = tokenizer.encode("Five is larger.\nFinal answer: ", add_special_tokens=False)
    answer = tokenizer.encode("(B)", add_special_tokens=False)
    seq = build_superposed(prompt, tokenizer, VALUES, candidates, tail, answer, mode="exact")
    assert not seq.unaligned and seq.spans["procedure"][0] == seq.spans["procedure"][1]
    scorer = GateScorer(model, checkpointing=False)
    grad = scorer.gradients(seq)
    base_ids = list(prompt.render_tokens(tokenizer, VALUES).input_ids)
    assert abs(grad.loss - _stock(model, base_ids + tail, answer)) < 1e-4

    for index, text in enumerate(candidates["procedure"]):  # insertion vertices
        gates, offsets = seq.vertex("procedure", index)
        real = list(prompt.replace_block("procedure", text).render_tokens(tokenizer, VALUES).input_ids)
        assert abs(scorer.value_at(seq, gates, offsets) - _stock(model, real + tail, answer)) < 1e-4

    # Combined vertex: replace task and reasoning and insert a procedure at once.
    choice = {"task": 0, "reasoning": 1, "procedure": 0}
    gates, offsets, edited = {}, {}, prompt
    for block_id, index in choice.items():
        g, o = seq.vertex(block_id, index)
        gates.update(g)
        offsets.update(o)
        edited = edited.replace_block(block_id, candidates[block_id][index])
    real = list(edited.render_tokens(tokenizer, VALUES).input_ids)
    assert abs(scorer.value_at(seq, gates, offsets) - _stock(model, real + tail, answer)) < 1e-4

    eps = 1e-3  # offset gradient of the empty slot
    slot = seq.slots.index("procedure")
    fd = (scorer.value_at(seq, None, {slot: eps}) - scorer.value_at(seq, None, {slot: -eps})) / (2 * eps)
    assert abs(fd - grad.offsets[slot]) <= 2e-3 + 2e-2 * abs(fd)


@pytest.mark.parametrize("arch", ["llama", "gemma2"])
def test_renormalized_patching_tracks_exact_edits(arch: str, tokenizer) -> None:
    from promptwitness.superposed_patching import patch_estimates

    model = _model(arch, len(tokenizer))
    prompt = _prompt_with_slot()
    candidates = dict(CANDIDATES)
    candidates["procedure"] = ["Check each option against the question.\n", "Double-check.\n"]
    tail = tokenizer.encode("Five is larger.\nFinal answer: ", add_special_tokens=False)
    answer = tokenizer.encode("(B)", add_special_tokens=False)
    seq = build_superposed(prompt, tokenizer, VALUES, candidates, tail, answer, mode="exact")
    scorer = GateScorer(model, checkpointing=True)
    diagnostics: dict[str, float] = {}
    base, estimates = patch_estimates(scorer, seq, check=True, diagnostics=diagnostics)
    assert diagnostics["max_base_error"] < 1e-4 and diagnostics["layers"] == 3
    exact = {}
    for (block_id, index) in estimates:
        gates, offsets = seq.vertex(block_id, index)
        exact[(block_id, index)] = scorer.value_at(seq, gates, offsets) - base
    keys = sorted(exact, key=str)
    est = torch.tensor([estimates[k] for k in keys])
    true = torch.tensor([exact[k] for k in keys])
    # First order in the residual stream: correlated and on the same scale (a raw gate
    # derivative instead saturates by orders of magnitude on real models).
    assert torch.corrcoef(torch.stack([est, true]))[0, 1] > 0.75, (est, true)
    assert 0.33 < float(est.abs().mean() / true.abs().mean()) < 3.0, (est, true)


def test_weighted_objective_matches_stock(tokenizer) -> None:
    """Verified objective: reasoning+answer scored, extractor weighted 0, empty tail."""
    import torch.nn.functional as F

    model = _model("llama", len(tokenizer))
    prompt = _prompt()
    reasoning = tokenizer.encode("Five is larger than three.", add_special_tokens=False)
    extractor = tokenizer.encode("\nFinal answer: ", add_special_tokens=False)
    answer = tokenizer.encode("(B)", add_special_tokens=False)
    scored = reasoning + extractor + answer
    weights = ([0.5 / len(reasoning)] * len(reasoning) + [0.0] * len(extractor)
               + [0.5 / len(answer)] * len(answer))
    seq = build_superposed(prompt, tokenizer, VALUES, CANDIDATES, [], scored,
                           mode="exact", answer_weights=weights)
    scorer = GateScorer(model, checkpointing=False)
    base_ids = list(prompt.render_tokens(tokenizer, VALUES).input_ids)
    tensor = torch.tensor([base_ids + scored])
    with torch.no_grad():
        logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).logits[0]
    per = F.cross_entropy(logits[len(base_ids) - 1 : -1].float(), torch.tensor(scored),
                          reduction="none")
    expected = float((per * torch.tensor(weights)).sum() / sum(weights))
    assert abs(scorer.gradients(seq).loss - expected) < 1e-4
