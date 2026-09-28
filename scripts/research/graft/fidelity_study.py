"""C1 fidelity study: do superposed gate derivatives rank real structural edits?

Fit rows only (never validation/holdout). For one task/model/seed:
  1. label-free candidate pools per editable block (same frozen model, unlabeled inputs);
  2. greedy reasoning under the incumbent for every row;
  3. per row, one exact-mode superposed sequence; estimators from gradients at
     (a) the incumbent point, with and without the slot-offset term,
     (b) each slot's hole (deletion vertex), (c) the joint slot-simplex centroid;
  4. exact vertex losses (= fixed-reasoning loss of each real edited prompt);
  5. fresh-reasoning outcomes of every edited prompt (batched generation): answer loss
     after the new reasoning and strict accuracy.
Writes one private JSON record with per-row values, timings and rank summaries.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness import graft_tasks
from promptwitness.graft_runtime import answer_losses, generate_batch
from promptwitness.structured_data import load_split
from promptwitness.structured_prompt import StructuredPrompt
from promptwitness.structured_search import initial_prompt as pilot_prompt
from promptwitness.superposed_gates import GateScorer, build_superposed, edit_scores
from promptwitness.superposed_patching import patch_estimates

OPERATORS = (
    "Rephrase it with different wording.",
    "Make it more specific to the kind of problems shown in the examples.",
    "Add one concrete procedural step that helps solve such problems.",
    "Make it shorter while keeping every requirement.",
    "Warn about one common pitfall for such problems and how to avoid it.",
    "Turn it into a short numbered procedure.",
    "Make it more explicit about how to use the answer options.",
    "Write an improved version of it.",
)
INSERT_OPERATORS = (
    "Write a short step-by-step procedure for solving such problems.",
    "Write one instruction that asks to verify the answer before finishing.",
    "Write one instruction about how to organize intermediate facts.",
    "Write one instruction warning about a common mistake in such problems.",
    "Write one instruction about how to compare the answer options.",
    "Write one instruction that asks to restate the key facts first.",
    "Write one concise strategy hint for such problems.",
    "Write one instruction to double-check the final choice.",
)


def propose(model: Any, tokenizer: Any, prompt: StructuredPrompt, block_id: str,
            questions: list[str], seed: int, k: int) -> list[str]:
    """Label-free, type-conditioned block proposals from the same frozen model."""
    import torch

    block = next(b for b in prompt.blocks if b.block_id == block_id)
    examples = "\n\n".join(f"Example problem {i + 1}:\n{q}" for i, q in enumerate(questions))
    context = "".join(b.text for b in prompt.blocks if b.editable)
    requests = []
    if not block.text:  # empty optional slot: insertion proposals
        for operator in INSERT_OPERATORS[:k]:
            requests.append(
                "You are adding one new block to an instruction prompt for a language model.\n"
                f"Current instructions:\n{context}\n"
                f"New block type: {block.kind.value}\n{examples}\n\nTask: {operator}\n"
                "Return only the new block text. Do not solve the examples, do not mention "
                "specific answers, and do not add placeholders."
            )
    for operator in OPERATORS[:k] if block.text else ():
        requests.append(
            "You are improving one block of an instruction prompt for a language model.\n"
            f"Block type: {block.kind.value}\n"
            f"Current block: {block.text.strip()}\n"
            f"Required literals that must appear verbatim: {list(block.required_literals)}\n"
            f"{examples}\n\nEdit: {operator}\n"
            "Return only the new block text. Do not solve the examples, do not mention "
            "specific answers, and do not add placeholders."
        )
    outputs: list[str] = []
    device = next(model.parameters()).device
    for index, request in enumerate(requests):
        enc = tokenizer.apply_chat_template([{"role": "user", "content": request}],
                                            tokenize=True, add_generation_prompt=True)
        ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
        tensor = torch.tensor([ids], device=device)
        torch.manual_seed(seed * 1000 + index)
        with torch.no_grad():
            gen = model.generate(tensor, attention_mask=torch.ones_like(tensor), max_new_tokens=96,
                                 do_sample=True, temperature=0.8, top_p=0.95,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0, len(ids):], skip_special_tokens=True).strip().strip('"').strip()
        if text:
            outputs.append(text if text.endswith("\n") else text + "\n")
    return outputs


def score_mc(response: str, answer: str) -> bool:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    match = re.fullmatch(r"Final answer:\s*\(([A-Z])\)", lines[-1]) if lines else None
    return bool(match) and f"({match.group(1)})" == answer


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--protocol", choices=("pilot", "greater"), default="pilot")
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--task", default="logical_deduction_three_objects")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--attn", default="sdpa", help="eager for Gemma-2 (softcapping)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to("cuda").eval()
    scorer = GateScorer(model)
    if args.protocol == "pilot":
        rows = list(load_split(args.shard_dir, args.manifest, args.task, "fit"))
        prompt = pilot_prompt()
        extractor_text = "\nFinal answer: "
        task_spec = None
    else:
        rows = [r for r in graft_tasks.load_splits(args.data_dir, args.task)["train"]]
        prompt = graft_tasks.initial_prompt()
        task_spec = graft_tasks.spec(args.task)
        extractor_text = task_spec.extractor
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    proposal_rows, rows = rows[:2], rows[2 : 2 + args.rows]
    editable = [b.block_id for b in prompt.blocks if b.editable]
    gold = (lambda a: a) if task_spec is None else task_spec.target  # noqa: E731
    t0 = perf_counter()
    candidates = {b: propose(model, tokenizer, prompt, b, [r.question for r in proposal_rows],
                             args.seed, args.k) for b in editable}
    proposal_seconds = perf_counter() - t0
    extractor = tokenizer.encode(extractor_text, add_special_tokens=False)

    # Edits: every policy-valid, cleanly tokenized replacement plus deletion of
    # non-required blocks. PromptWitness rejects rewrites that drop required literals.
    from promptwitness.superposed_gates import candidate_token_ids

    probe_values = {"input": rows[0].question}
    edits: list[tuple[str, int | None]] = []
    rejected: list[dict[str, Any]] = []
    for b in editable:
        for i, text in enumerate(candidates[b]):
            if candidate_token_ids(prompt, tokenizer, probe_values, b, text) is None:
                rejected.append({"block": b, "index": i})
            else:
                edits.append((b, i))
    edits += [(b.block_id, None) for b in prompt.blocks
              if b.editable and b.text and not b.required_literals]

    def edited(block: str, index: int | None) -> StructuredPrompt | None:
        if index is None:
            return None
        return prompt.replace_block(block, candidates[block][index])

    values = [{"input": r.question} for r in rows]
    answers = [tokenizer.encode(gold(r.answer), add_special_tokens=False) for r in rows]

    def correctness(prompts_ids: list[list[int]], outs: list[Any]) -> list[bool]:
        """Pilot: strict final line. GReaTer protocol: extractor then short greedy answer."""
        if task_spec is None:
            return [score_mc(o.text, r.answer) for o, r in zip(outs, rows)]
        reads, _, _ = generate_batch(model, tokenizer,
                                     [p + list(o.token_ids) + extractor for p, o in zip(prompts_ids, outs)],
                                     max_new_tokens=8)
        return [task_spec.correct(x.text, r.answer) for x, r in zip(reads, rows)]
    base_ids = [list(prompt.render_tokens(tokenizer, v).input_ids) for v in values]
    gens, gen_seconds, _ = generate_batch(model, tokenizer, base_ids, max_new_tokens=args.max_new_tokens)
    base_correct = correctness(base_ids, gens)

    per_row: list[dict[str, Any]] = []
    timing = {"grad_incumbent": 0.0, "grad_holes": 0.0, "grad_centroid": 0.0,
              "patch": 0.0, "exact_vertices": 0.0}
    gated = [b.block_id for b in prompt.blocks if b.editable and b.text]
    for row, v, ids, gen, ans in zip(rows, values, base_ids, gens, answers):
        tail = list(gen.token_ids) + extractor
        seq = build_superposed(prompt, tokenizer, v, candidates, tail, ans, mode="exact")
        record: dict[str, Any] = {"row_id": row.row_id, "unaligned": len(seq.unaligned)}
        t = perf_counter(); g0 = scorer.gradients(seq); timing["grad_incumbent"] += perf_counter() - t
        s0 = edit_scores(seq, g0)
        # Estimator without the offset term (ablation).
        def incumbent_grad(b: str) -> float:
            found = [g for g in seq.gates if g.block_id == b and g.kind == "incumbent"]
            return g0.gates[found[0].gate_id] if found else 0.0

        no_offset = {(b, i): g0.gates[seq.gate_for(b, i).gate_id] - incumbent_grad(b)
                     for (b, i) in edits if i is not None and (b, i, candidates[b][i]) not in seq.unaligned}
        # Hole point per slot: linearize from the deletion vertex.
        hole: dict[tuple[str, int | None], float] = {}
        t = perf_counter()
        for b in gated:
            gates, offsets = seq.vertex(b, None)
            gh = scorer.gradients(seq, gates, offsets)
            slot = seq.slots.index(b)
            inc = seq.gate_for(b, None)
            base_inc = gh.gates[inc.gate_id] + inc.token_count * gh.offsets[slot]
            for gate in seq.gates:
                if gate.kind == "candidate" and gate.block_id == b:
                    hole[(b, gate.candidate_index)] = (
                        gh.gates[gate.gate_id] + gate.token_count * gh.offsets[slot] - base_inc
                    )
            hole[(b, None)] = -base_inc
        timing["grad_holes"] += perf_counter() - t
        # Joint centroid of every slot simplex; offsets at the weighted mean length change.
        gates_c: dict[int, float] = {}
        offsets_c: dict[int, float] = {}
        for b in gated:
            members = [g for g in seq.gates if g.block_id == b]
            inc = seq.gate_for(b, None)
            for gate in members:
                gates_c[gate.gate_id] = 1.0 / len(members)
            offsets_c[seq.slots.index(b)] = sum(
                (gate.token_count - inc.token_count) / len(members) for gate in members)
        t = perf_counter(); gc = scorer.gradients(seq, gates_c, offsets_c); timing["grad_centroid"] += perf_counter() - t
        centroid: dict[tuple[str, int | None], float] = {}
        for b in gated:
            slot = seq.slots.index(b)
            inc = seq.gate_for(b, None)
            ref = gc.gates[inc.gate_id]
            for gate in seq.gates:
                if gate.kind == "candidate" and gate.block_id == b:
                    centroid[(b, gate.candidate_index)] = (
                        gc.gates[gate.gate_id] - ref + (gate.token_count - inc.token_count) * gc.offsets[slot])
        t = perf_counter()
        _, patched = patch_estimates(scorer, seq)
        timing["patch"] += perf_counter() - t
        # Exact vertices = fixed-reasoning loss of the real edited prompt.
        t = perf_counter()
        exact: dict[tuple[str, int | None], float] = {}
        for (b, i) in edits:
            if i is not None and all(u[:2] != (b, i) for u in seq.unaligned):
                gates, offsets = seq.vertex(b, i)
                exact[(b, i)] = scorer.value_at(seq, gates, offsets) - g0.loss
            elif i is None:
                gates, offsets = seq.vertex(b, None)
                exact[(b, None)] = scorer.value_at(seq, gates, offsets) - g0.loss
        timing["exact_vertices"] += perf_counter() - t
        key = lambda e: f"{e[0]}:{'del' if e[1] is None else e[1]}"  # noqa: E731
        record["base_loss"] = g0.loss
        record["estimates"] = {
            "incumbent": {key(e): (s0["replacement"][e[0]][e[1]] if e[1] is not None else s0["deletion"][e[0]]) for e in exact},
            "incumbent_no_offset": {key(e): no_offset[e] for e in exact if e in no_offset},
            "hole": {key(e): hole[e] for e in exact if e in hole},
            "centroid": {key(e): centroid[e] for e in exact if e in centroid},
            "patch": {key(e): patched[e] for e in exact if e in patched},
        }
        record["exact"] = {key(e): val for e, val in exact.items()}
        per_row.append(record)
        print(json.dumps({"row": row.row_id, "base": round(g0.loss, 3)}), flush=True)

    # Fresh reasoning for every edited prompt over all rows.
    fresh: dict[str, dict[str, Any]] = {}
    t = perf_counter()
    for (b, i) in edits:
        name = f"{b}:{'del' if i is None else i}"
        if i is None:
            spans: list[list[int]] = []
            for v in values:
                rendered = prompt.render_tokens(tokenizer, v)
                pos = rendered.block_positions[b]
                ids = list(rendered.input_ids)
                spans.append(ids[: pos[0]] + ids[pos[-1] + 1 :])
            prompts_ids = spans
        else:
            new_prompt = edited(b, i)
            prompts_ids = [list(new_prompt.render_tokens(tokenizer, v).input_ids) for v in values]
        outs, _, _ = generate_batch(model, tokenizer, prompts_ids, max_new_tokens=args.max_new_tokens)
        losses, _ = answer_losses(model, [(p + list(o.token_ids) + extractor, a)
                                          for p, o, a in zip(prompts_ids, outs, answers)])
        fresh[name] = {"accuracy": sum(correctness(prompts_ids, outs)) / len(rows),
                       "mean_loss": sum(losses) / len(losses)}
    base_losses, _ = answer_losses(model, [(p + list(o.token_ids) + extractor, a)
                                           for p, o, a in zip(base_ids, gens, answers)])
    fresh_seconds = perf_counter() - t
    base_fresh = {"accuracy": sum(base_correct) / len(rows), "mean_loss": sum(base_losses) / len(base_losses)}

    # Summaries: mean over rows per edit.
    names = sorted(set.intersection(*(set(r["exact"]) for r in per_row)))
    mean = lambda field, method=None: [  # noqa: E731
        sum((r["estimates"][method] if method else r[field])[n] for r in per_row) / len(per_row) for n in names]
    exact_mean = mean("exact")
    fresh_loss = [fresh[n]["mean_loss"] - base_fresh["mean_loss"] for n in names]
    fresh_acc = [fresh[n]["accuracy"] - base_fresh["accuracy"] for n in names]
    rng2 = random.Random(0)
    summary: dict[str, Any] = {"edits": len(names), "base_fresh": base_fresh,
                               "spearman_exact_vs_fresh_loss": spearman(exact_mean, fresh_loss),
                               "spearman_exact_vs_neg_fresh_acc": spearman(exact_mean, [-a for a in fresh_acc])}
    for method in ("incumbent", "incumbent_no_offset", "hole", "centroid", "patch"):
        usable = [j for j, n in enumerate(names) if all(n in r["estimates"][method] for r in per_row)]
        est = [sum(r["estimates"][method][names[j]] for r in per_row) / len(per_row) for j in usable]
        pooled_est = [r["estimates"][method][names[j]] for r in per_row for j in usable]
        pooled_exact = [r["exact"][names[j]] for r in per_row for j in usable]
        best = min(usable, key=lambda j: est[usable.index(j)]) if usable else None
        summary[method] = {
            "n": len(usable),
            "spearman_vs_exact_mean": spearman(est, [exact_mean[j] for j in usable]),
            "spearman_vs_exact_pooled_rows": spearman(pooled_est, pooled_exact),
            "spearman_vs_fresh_loss": spearman(est, [fresh_loss[j] for j in usable]),
            "top1": names[best] if best is not None else None,
            "top1_exact_delta": exact_mean[best] if best is not None else None,
            "top1_fresh_acc_delta": fresh_acc[best] if best is not None else None,
        }
    order = list(range(len(names)))
    rng2.shuffle(order)
    summary["random_top1_fresh_acc_delta_mean"] = sum(fresh_acc) / len(fresh_acc)
    summary["oracle_exact_top1"] = names[min(range(len(names)), key=lambda j: exact_mean[j])]
    summary["oracle_exact_top1_fresh_acc_delta"] = fresh_acc[min(range(len(names)), key=lambda j: exact_mean[j])]
    summary["best_fresh_acc_delta"] = max(fresh_acc)
    lengths = [len(tokenizer.encode(candidates[n.split(":")[0]][int(n.split(":")[1])], add_special_tokens=False))
               if not n.endswith("del") else 0 for n in names]
    summary["spearman_length_vs_exact"] = spearman(lengths, exact_mean)
    result = {"format": "promptwitness.graft-fidelity/v2", "protocol": args.protocol,
              "task": args.task, "model_path": args.model_path,
              "seed": args.seed, "rows": [r.row_id for r in rows], "candidates": candidates,
              "policy_rejected": rejected,
              "per_row": per_row, "fresh": fresh, "names": names, "summary": summary,
              "timing": {**timing, "proposal": proposal_seconds, "incumbent_generation": gen_seconds,
                         "fresh_all_edits": fresh_seconds, "wall": perf_counter() - started}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(json.dumps(result["timing"], indent=2))


if __name__ == "__main__":
    main()
