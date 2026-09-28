"""GRAFT structured prompt search under the GReaTer protocol (one task, model, seed).

Every scorer shares the same label-free proposals, fresh-reasoning verification,
acceptance rule, dev-set checkpoint selection and a single test evaluation:

  patch   GRAFT: renormalized superposed patching, one forward/backward per example
  gate    ablation: raw first-order gate derivative (saturates)
  exact   control: fixed-reasoning loss of every edited prompt (one forward per edit)
  random  control: random shortlist, no scoring
  textgrad baseline: section-local textual feedback from the same model on failed
          minibatch examples (labels visible to the critic, as in TextGrad/MPO), one
          feedback-driven rewrite per block, then the same verification

Only train examples drive search; dev selects among accepted prompts; test is read
once at the end. Writes a JSON record with the trajectory and a cost ledger.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness import graft_tasks
from promptwitness.graft_runtime import (
    Generation,
    ScoredTarget,
    answer_losses,
    answer_target,
    generate_batch,
    verified_targets,
)
from promptwitness.structured_prompt import StructuredPrompt
from promptwitness.superposed_gates import (
    GateScorer,
    build_superposed,
    candidate_token_ids,
    edit_scores,
)
from promptwitness.superposed_patching import patch_estimates

REWRITE_OPERATORS = (
    "Rephrase it with different wording.",
    "Make it more specific to the kind of problems shown in the examples.",
    "Add one concrete procedural step that helps solve such problems.",
    "Make it shorter while keeping every requirement.",
    "Warn about one common pitfall for such problems and how to avoid it.",
    "Turn it into a short numbered procedure.",
    "Make it more explicit about how to reach and state the final answer.",
    "Write an improved version of it.",
)
INSERT_OPERATORS = (
    "Write a short step-by-step procedure for solving such problems.",
    "Write one instruction that asks to verify the answer before finishing.",
    "Write one instruction about how to organize intermediate facts.",
    "Write one instruction warning about a common mistake in such problems.",
    "Write one instruction about how to compare the possible answers.",
    "Write one instruction that asks to restate the key facts first.",
    "Write one concise strategy hint for such problems.",
    "Write one instruction to double-check the final answer.",
)
Edit = tuple[str, int | None]  # (slot, candidate index) ; None = delete


class Ledger:
    def __init__(self) -> None:
        self.phases: dict[str, dict[str, float]] = {}

    def add(self, phase: str, seconds: float, **counts: float) -> None:
        bucket = self.phases.setdefault(phase, {"seconds": 0.0, "calls": 0})
        bucket["seconds"] += seconds
        bucket["calls"] += 1
        for name, value in counts.items():
            bucket[name] = bucket.get(name, 0) + value


class Runner:
    def __init__(self, model: Any, tokenizer: Any, spec: graft_tasks.TaskSpec, ledger: Ledger,
                 max_new_tokens: int) -> None:
        self.model, self.tokenizer, self.spec, self.ledger = model, tokenizer, spec, ledger
        self.max_new_tokens = max_new_tokens
        self.use_reasoning = True  # False: GReaTer's no-reasoning ablation for scoring
        self.candidates_per_pass = 32
        self.extractor = tokenizer.encode(spec.extractor, add_special_tokens=False)

    def ids(self, prompt: StructuredPrompt, example: graft_tasks.Example) -> list[int]:
        return list(prompt.render_tokens(self.tokenizer, {"input": example.question}).input_ids)

    def target(self, example: graft_tasks.Example) -> list[int]:
        return self.tokenizer.encode(self.spec.target(example.answer), add_special_tokens=False)

    def evaluate(self, prompt: StructuredPrompt, examples: Sequence[graft_tasks.Example],
                 phase: str, *, with_loss: bool = False, max_new_tokens: int | None = None
                 ) -> dict[str, Any]:
        """Two-stage reader: greedy reasoning, extractor, short greedy answer."""
        prompts = [self.ids(prompt, e) for e in examples]
        gens, seconds, tokens = generate_batch(self.model, self.tokenizer, prompts,
                                               max_new_tokens=max_new_tokens or self.max_new_tokens,
                                               batch_size=32)
        self.ledger.add(phase + "_reasoning", seconds, generated=tokens,
                        prompt_tokens=sum(map(len, prompts)))
        reads_in = [p + list(g.token_ids) + self.extractor for p, g in zip(prompts, gens)]
        reads, seconds, tokens = generate_batch(self.model, self.tokenizer, reads_in, max_new_tokens=8)
        self.ledger.add(phase + "_answer", seconds, generated=tokens)
        correct = [self.spec.correct(r.text, e.answer) for r, e in zip(reads, examples)]
        out: dict[str, Any] = {"accuracy": sum(correct) / len(correct), "correct": correct,
                               "reads": [r.text for r in reads],
                               "reasoning": [list(g.token_ids) for g in gens],
                               "truncated": sum(not g.ended for g in gens)}
        if with_loss:
            losses, seconds = answer_losses(self.model, [(r, self.target(e)) for r, e in zip(reads_in, examples)])
            self.ledger.add(phase + "_loss", seconds)
            out["loss"] = sum(losses) / len(losses)
        return out


def propose(runner: Runner, prompt: StructuredPrompt, slot: str, questions: list[str],
            seed: int, k: int) -> list[str]:
    """Label-free, type-conditioned proposals from the task model itself."""
    import torch

    tokenizer, model = runner.tokenizer, runner.model
    block = next(b for b in prompt.blocks if b.block_id == slot)
    examples = "\n\n".join(f"Example problem {i + 1}:\n{q}" for i, q in enumerate(questions))
    context = "".join(b.text for b in prompt.blocks if b.editable)
    requests = []
    if block.text:
        for operator in (REWRITE_OPERATORS[i % len(REWRITE_OPERATORS)] for i in range(k)):
            requests.append(
                "You are improving one block of an instruction prompt for a language model.\n"
                f"All current instructions:\n{context}\nBlock to edit ({block.kind.value}): "
                f"{block.text.strip()}\n{examples}\n\nEdit: {operator}\n"
                "Return only the new text of this block. Do not solve the examples, do not "
                "mention specific answers, and do not add placeholders.")
    else:
        for operator in (INSERT_OPERATORS[i % len(INSERT_OPERATORS)] for i in range(k)):
            requests.append(
                "You are adding one new block to an instruction prompt for a language model.\n"
                f"All current instructions:\n{context}\nNew block type: {block.kind.value}\n"
                f"{examples}\n\nTask: {operator}\n"
                "Return only the new block text. Do not solve the examples, do not mention "
                "specific answers, and do not add placeholders.")
    encoded = []
    for request in requests:
        enc = tokenizer.apply_chat_template([{"role": "user", "content": request}],
                                            tokenize=True, add_generation_prompt=True)
        encoded.append(list(enc["input_ids"] if hasattr(enc, "keys") else enc))
    started = perf_counter()
    outputs: list[str] = []
    device = next(model.parameters()).device
    width = max(map(len, encoded))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(encoded), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, seq in enumerate(encoded):
        ids[row, width - len(seq):] = torch.tensor(seq)
        mask[row, width - len(seq):] = 1
    torch.manual_seed(seed)
    with torch.no_grad():
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                             max_new_tokens=96, do_sample=True, temperature=0.8, top_p=0.95,
                             pad_token_id=pad)
    for row in gen[:, width:]:
        text = graft_tasks.clean_proposal(tokenizer.decode(row, skip_special_tokens=True))
        if text:
            outputs.append(text)
    runner.ledger.add("proposal", perf_counter() - started, generated=int(gen[:, width:].numel()))
    return outputs


def textual_gradient_pools(runner: Runner, prompt: StructuredPrompt, batch: list[graft_tasks.Example],
                           incumbent: dict[str, Any], seed: int) -> dict[str, list[str]]:
    """Section-local textual gradients: critique each block from failures, then rewrite it."""
    import torch

    tokenizer, model = runner.tokenizer, runner.model
    failures = [(e, r) for e, r, c in zip(batch, incumbent["reasoning"], incumbent["correct"]) if not c]
    if not failures:
        failures = list(zip(batch, incumbent["reasoning"]))[:2]
    shown = "\n\n".join(
        f"Problem:\n{e.question}\nModel reasoning:\n"
        f"{tokenizer.decode(r, skip_special_tokens=True)[-600:]}\n"
        f"Correct answer: {e.answer}" for e, r in failures[:3])
    context = "".join(b.text for b in prompt.blocks if b.editable)
    pools: dict[str, list[str]] = {}
    started = perf_counter()
    for index, block in enumerate(b for b in prompt.blocks if b.editable):
        role = f"the block '{block.text.strip()}'" if block.text else "a new optional block (currently empty)"
        request = (
            "An instruction prompt for a language model produced wrong answers.\n"
            f"Current instructions:\n{context}\nExamples of failures:\n{shown}\n\n"
            f"First, briefly explain what is wrong or missing in {role} of type {block.kind.value}. "
            "Then write an improved version of that block that would help on such problems. "
            "Do not mention the specific answers. End with the line 'NEW BLOCK:' followed by "
            "only the new block text.")
        enc = tokenizer.apply_chat_template([{"role": "user", "content": request}],
                                            tokenize=True, add_generation_prompt=True)
        ids = torch.tensor([list(enc["input_ids"] if hasattr(enc, "keys") else enc)], device=model.device)
        torch.manual_seed(seed + index)
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=320,
                                 do_sample=True, temperature=0.7, top_p=0.95,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        new = graft_tasks.clean_proposal(text) if "NEW BLOCK:" in text else ""
        pools[block.block_id] = [new] if new else []
    runner.ledger.add("textgrad_feedback", perf_counter() - started)
    return pools


def valid_edits(runner: Runner, prompt: StructuredPrompt, pools: dict[str, list[str]],
                probe: graft_tasks.Example) -> list[Edit]:
    values = {"input": probe.question}
    edits: list[Edit] = []
    for slot, texts in pools.items():
        block = next(b for b in prompt.blocks if b.block_id == slot)
        for index, text in enumerate(texts):
            if text.strip() == block.text.strip() or "{{" in text:
                continue
            ids = candidate_token_ids(prompt, runner.tokenizer, values, slot, text)
            if ids is not None and len(ids) <= 160:
                edits.append((slot, index))
        others = [b for b in prompt.blocks if b.editable and b.text and b.block_id != slot]
        if block.text and not block.required_literals and others:
            edits.append((slot, None))
    return edits


def apply(prompt: StructuredPrompt, pools: dict[str, list[str]], edit: Edit) -> StructuredPrompt:
    slot, index = edit
    if index is None:
        return delete_block(prompt, slot)
    return prompt.replace_block(slot, pools[slot][index])


def delete_block(prompt: StructuredPrompt, slot: str) -> StructuredPrompt:
    """Empty a block (it stays as an insertion slot)."""
    from dataclasses import replace

    old = next(b for b in prompt.blocks if b.block_id == slot)
    size = len(old.text)
    blocks = tuple(
        replace(b, text="", source_end=b.source_start) if b.block_id == slot else
        replace(b, source_start=b.source_start - size, source_end=b.source_end - size)
        if b.message_id == old.message_id and b.source_start >= old.source_end else b
        for b in prompt.blocks)
    messages = tuple(
        replace(m, content=m.content[: old.source_start] + m.content[old.source_end:])
        if m.message_id == old.message_id else m for m in prompt.document.messages)
    return StructuredPrompt(replace(prompt.document, messages=messages), blocks)


def score_edits(method: str, runner: Runner, scorer: GateScorer, prompt: StructuredPrompt,  # noqa: PLR0913
                pools: dict[str, list[str]], edits: list[Edit], batch: list[graft_tasks.Example],
                targets: list[ScoredTarget], rng: random.Random) -> dict[Edit, float]:
    """Estimated objective change per edit (lower is better), averaged over the minibatch.

    ``targets`` fix, per example, the unscored tail and the weighted scored tokens
    (GReaTer's answer objective or the verified-reasoning objective).
    """
    if method == "random":
        return {e: rng.random() for e in edits}
    totals = {e: 0.0 for e in edits}
    started = perf_counter()
    if method == "exact":
        pairs, owners, weights = [], [], []
        for example, target in zip(batch, targets):
            tail, scored = list(target.tail), list(target.scored)
            pairs.append((runner.ids(prompt, example) + tail, scored))
            owners.append(None)
            weights.append(target.weights)
            for edit in edits:
                pairs.append((runner.ids(apply(prompt, pools, edit), example) + tail, scored))
                owners.append(edit)
                weights.append(target.weights)
        losses, _ = answer_losses(runner.model, pairs, weights=weights)
        base_loss = 0.0
        for owner, loss in zip(owners, losses):
            if owner is None:
                base_loss = loss
            else:
                totals[owner] += (loss - base_loss) / len(batch)
        runner.ledger.add("score_exact", perf_counter() - started, forwards=len(pairs),
                          forward_tokens=sum(len(c) + len(a) for c, a in pairs))
        return totals
    # Superposed passes of bounded size; indices stay stable via None placeholders.
    replacements = [e for e in edits if e[1] is not None]
    size = max(1, runner.candidates_per_pass)
    chunks = [replacements[i : i + size] for i in range(0, len(replacements), size)] or [[]]
    tokens = passes = 0
    for chunk_index, chunk in enumerate(chunks):
        members = set(chunk)
        candidates = {slot: [text if (slot, i) in members else None for i, text in enumerate(texts)]
                      for slot, texts in pools.items()}
        for example, target in zip(batch, targets):
            seq = build_superposed(prompt, runner.tokenizer, {"input": example.question}, candidates,
                                   list(target.tail), list(target.scored), mode="exact",
                                   answer_weights=list(target.weights))
            tokens += len(seq.input_ids)
            passes += 1
            if method == "patch":
                _, est = patch_estimates(scorer, seq, include_deletions=chunk_index == 0)
            else:
                flat = edit_scores(seq, scorer.gradients(seq))
                est = {(s, i): v for s, d in flat["replacement"].items() for i, v in d.items()}
                if chunk_index == 0:
                    est.update({(s, None): v for s, v in flat["deletion"].items()})
            for edit in edits:
                if edit in est:
                    totals[edit] += est[edit] / len(batch)
    runner.ledger.add(f"score_{method}", perf_counter() - started, forwards=passes,
                      backwards=passes, forward_tokens=tokens)
    return totals


def build_targets(args: argparse.Namespace, runner: Runner, prompt: StructuredPrompt,
                  batch: list[graft_tasks.Example], incumbent: dict[str, Any]) -> list[ScoredTarget]:
    reasoning = incumbent["reasoning"]
    answers = [runner.target(e) for e in batch]
    if args.objective == "answer":
        return [answer_target(r if runner.use_reasoning else [], runner.extractor, a)
                for r, a in zip(reasoning, answers)]
    started = perf_counter()
    targets, _ = verified_targets(
        runner.model, runner.tokenizer, [runner.ids(prompt, e) for e in batch],
        [Generation(tuple(r), "", True) for r in reasoning], incumbent["correct"],
        runner.extractor, answers, lambda i, text: runner.spec.correct(text, batch[i].answer),
        samples=args.verify_samples, max_new_tokens=args.max_new_tokens, seed=args.seed)
    runner.ledger.add("verified_targets", perf_counter() - started)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("patch", "gate", "exact", "random", "textgrad"),
                        required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--dev-checkpoints", type=int, default=3)
    parser.add_argument("--candidates-per-pass", type=int, default=32)
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="keep attention activations instead of recomputing (faster, more memory)")
    parser.add_argument("--objective", choices=("answer", "verified"), default="answer",
                        help="GReaTer's answer loss, or verified reasoning plus answer")
    parser.add_argument("--verify-samples", type=int, default=4)
    parser.add_argument("--no-reasoning-scores", action="store_true",
                        help="ablation: score edits without reasoning in the tail")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    scorer = GateScorer(model, checkpointing=not args.no_checkpointing)
    spec = graft_tasks.spec(args.task)
    splits = graft_tasks.load_splits(args.data_dir, args.task)
    ledger = Ledger()
    runner = Runner(model, tokenizer, spec, ledger, args.max_new_tokens)
    runner.use_reasoning = not args.no_reasoning_scores
    runner.candidates_per_pass = args.candidates_per_pass
    rng = random.Random(args.seed)
    prompt = graft_tasks.initial_prompt()
    accepted: list[dict[str, Any]] = [{"round": 0, "text": prompt.document.messages[0].content}]
    prompts: dict[str, StructuredPrompt] = {accepted[0]["text"]: prompt}
    trajectory: list[dict[str, Any]] = []

    for round_index in range(1, args.rounds + 1):
        batch = rng.sample(splits["train"], args.batch)
        incumbent = runner.evaluate(prompt, batch, "incumbent", with_loss=True)
        slots = [b.block_id for b in prompt.blocks if b.editable]
        questions = [e.question for e in rng.sample(splits["train"], 2)]
        if args.method == "textgrad":
            pools = textual_gradient_pools(runner, prompt, batch, incumbent,
                                           args.seed * 1000 + round_index * 10)
        else:
            pools = {slot: propose(runner, prompt, slot, questions,
                                   args.seed * 1000 + round_index * 10 + j, args.k)
                     for j, slot in enumerate(slots)}
        edits = valid_edits(runner, prompt, pools, batch[0])
        record: dict[str, Any] = {"round": round_index, "incumbent_acc": incumbent["accuracy"],
                                  "incumbent_loss": incumbent["loss"], "edits": len(edits),
                                  "pools": pools}
        if not edits:
            trajectory.append({**record, "accepted": None})
            continue
        scoring = "random" if args.method == "textgrad" else args.method
        if args.method == "textgrad":
            edits = [e for e in edits if e[1] is not None]  # feedback rewrites only
        if not edits:
            trajectory.append({**record, "accepted": None})
            continue
        targets = (build_targets(args, runner, prompt, batch, incumbent)
                   if scoring != "random" else [])
        record["target_sources"] = [t.source for t in targets]
        estimates = score_edits(scoring, runner, scorer, prompt, pools, edits, batch, targets, rng)
        shortlist = sorted(edits, key=lambda e: estimates[e])[: args.mu]
        checks = []
        for edit in shortlist:
            candidate = apply(prompt, pools, edit)
            result = runner.evaluate(candidate, batch, "verify", with_loss=True)
            checks.append({"edit": list(edit), "estimate": estimates[edit],
                           "accuracy": result["accuracy"], "loss": result["loss"]})
        best = max(checks, key=lambda c: (c["accuracy"], -c["loss"]))
        take = (best["accuracy"] > incumbent["accuracy"] or
                (best["accuracy"] == incumbent["accuracy"] and best["loss"] < incumbent["loss"] - 1e-3))
        record.update({"estimates": {f"{s}:{'del' if i is None else i}": v for (s, i), v in estimates.items()},
                       "checks": checks, "accepted": best["edit"] if take else None})
        if take:
            prompt = apply(prompt, pools, tuple(best["edit"]))
            text = prompt.document.messages[0].content
            prompts[text] = prompt
            accepted.append({"round": round_index, "text": text})
        trajectory.append(record)
        print(json.dumps({"round": round_index, "inc_acc": incumbent["accuracy"],
                          "accepted": record["accepted"], "best_acc": best["accuracy"]}), flush=True)
    search_seconds = perf_counter() - started

    # Dev selection among the incumbents at evenly spaced checkpoint rounds
    # (0, R/k, ..., R), deduplicated; a fixed budget rule shared by all methods.
    marks = sorted({round(args.rounds * i / args.dev_checkpoints)
                    for i in range(args.dev_checkpoints + 1)})
    checkpoints: list[dict[str, Any]] = []
    for mark in marks:
        entry = max((a for a in accepted if a["round"] <= mark), key=lambda a: a["round"])
        if all(entry["text"] != c["text"] for c in checkpoints):
            checkpoints.append(entry)
    dev_scores = []
    for entry in checkpoints:
        dev = runner.evaluate(prompts[entry["text"]], splits["dev"], "dev",
                              max_new_tokens=args.eval_max_new_tokens)
        dev_scores.append(dev["accuracy"])
        entry["dev_accuracy"] = dev["accuracy"]
    best = max(range(len(checkpoints)), key=lambda i: (dev_scores[i], i))
    chosen = accepted.index(checkpoints[best])
    test = runner.evaluate(prompts[accepted[chosen]["text"]], splits["test"], "test",
                           max_new_tokens=args.eval_max_new_tokens)
    result = {"format": "promptwitness.graft-run/v1", "task": args.task, "method": args.method,
              "seed": args.seed, "model_path": args.model_path, "config": vars(args) | {
                  "data_dir": str(args.data_dir), "output": str(args.output)},
              "trajectory": trajectory, "accepted": accepted, "selected": chosen,
              "selected_text": accepted[chosen]["text"], "test_accuracy": test["accuracy"],
              "test_correct": test["correct"], "test_truncated": test["truncated"],
              "cost": ledger.phases, "search_seconds": search_seconds,
              "wall_seconds": perf_counter() - started,
              "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"test_accuracy": test["accuracy"], "selected_round": accepted[chosen]["round"],
                      "dev": dev_scores, "wall": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
