"""Do cheap fixed-reasoning objectives predict held-out accuracy of whole-block edits?

For one task/model/seed: label-free proposals (as in run_graft), edits = replacements +
deletions. On a few train rows, score every edit under three objectives, exactly (batched
forwards of the edited prompts) and with renormalized superposed patching:
  answer       GReaTer: CE(answer | prompt, greedy reasoning, extractor)
  verified     0.5 mean CE(r+) + 0.5 CE(answer | r+), r+ = gold-consistent self reasoning
  contrastive  -[log p(r+) - log p(r-)] (sums over reasoning tokens), r- an incorrect one
  fork         CE(r+_t) - CE(r-_t) at the first divergence t of r+ and r- (one is greedy)
  pg           policy-gradient score of expected accuracy over all judged samples
Then evaluate the base prompt and every edited prompt on N dev examples with fresh greedy
reasoning (the shared reader) and correlate objective changes with dev accuracy changes.
Train rows drive the objectives; dev rows are only the validity target; test is unused.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness import graft_tasks
from promptwitness.graft_runtime import answer_losses, generate_batch
from promptwitness.superposed_gates import GateScorer, build_superposed, candidate_token_ids
from promptwitness.superposed_patching import patch_estimates

from fidelity_study import spearman
from run_graft import Ledger, Runner, apply, propose


def sample_reasonings(model: Any, tokenizer: Any, prompts: list[list[int]], samples: int,
                      max_new_tokens: int, seed: int) -> list[list[list[int]]]:
    import torch

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    device = next(model.parameters()).device
    batch = [p for p in prompts for _ in range(samples)]
    out_all: list[list[int]] = []
    for begin in range(0, len(batch), 24):
        chunk = batch[begin:begin + 24]
        width = max(map(len, chunk))
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(chunk):
            ids[row, width - len(seq):] = torch.tensor(seq)
            mask[row, width - len(seq):] = 1
        torch.manual_seed(seed + begin)
        with torch.no_grad():
            gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                                 top_p=0.95, pad_token_id=pad)
        for row in gen[:, width:].tolist():
            tokens = []
            for token in row:
                if token in (eos, pad):
                    break
                tokens.append(token)
            out_all.append(tokens)
    return [out_all[i * samples:(i + 1) * samples] for i in range(len(prompts))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--dev", type=int, default=60)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn).to("cuda").eval()
    scorer = GateScorer(model)
    spec = graft_tasks.spec(args.task)
    splits = graft_tasks.load_splits(args.data_dir, args.task)
    rng = random.Random(args.seed)
    train = list(splits["train"])
    rng.shuffle(train)
    proposal_rows, rows = train[:2], train[2:2 + args.rows]
    dev = splits["dev"][: args.dev]
    runner = Runner(model, tokenizer, spec, Ledger(), args.max_new_tokens)
    prompt = graft_tasks.initial_prompt()
    slots = [b.block_id for b in prompt.blocks if b.editable]
    pools = {s: propose(runner, prompt, s, [r.question for r in proposal_rows], args.seed * 100 + j, args.k)
             for j, s in enumerate(slots)}
    probe = {"input": rows[0].question}
    edits = [(s, i) for s, texts in pools.items() for i, t in enumerate(texts)
             if candidate_token_ids(prompt, tokenizer, probe, s, t) is not None]
    edits += [(b.block_id, None) for b in prompt.blocks if b.editable and b.text]
    ext = runner.extractor

    # Reasoning material on train rows.
    base_prompts = [runner.ids(prompt, r) for r in rows]
    greedy = runner.evaluate(prompt, rows, "train_greedy")
    drafts = sample_reasonings(model, tokenizer, base_prompts, args.samples, args.max_new_tokens, args.seed)
    reads, _, _ = generate_batch(model, tokenizer, [p + d + ext for p, ds in zip(base_prompts, drafts) for d in ds],
                                 max_new_tokens=8)
    positive: list[list[int] | None] = []
    negative: list[list[int] | None] = []
    judged_rows: list[list[tuple[list[int], bool]]] = []
    for i, row in enumerate(rows):
        judged = [(list(greedy["reasoning"][i]), greedy["correct"][i])]
        judged += [(d, spec.correct(reads[i * args.samples + j].text, row.answer))
                   for j, d in enumerate(drafts[i]) if d]
        judged_rows.append(judged)
        positive.append(next((d for d, ok in judged if ok), None))
        negative.append(next((d for d, ok in judged if not ok), None))
    answers = [runner.tokenizer.encode(spec.target(r.answer), add_special_tokens=False) for r in rows]

    def objective_specs(i: int) -> dict[str, tuple[list[int], list[int], list[float], float]]:
        """name -> (tail, scored, weights, scale) where objective = scale * weighted-mean CE."""
        out = {"answer": (list(greedy["reasoning"][i]) + ext, answers[i], [1.0] * len(answers[i]), 1.0)}
        if positive[i] is not None:
            rp = positive[i]
            w = [0.5 / len(rp)] * len(rp) + [0.0] * len(ext) + [0.5 / len(answers[i])] * len(answers[i])
            out["verified"] = ([], rp + ext + answers[i], w, 1.0)
        if positive[i] is not None and negative[i] is not None:
            rp, rn = positive[i], negative[i]
            out["contrastive_pos"] = ([], rp, [1.0] * len(rp), float(len(rp)))
            out["contrastive_neg"] = ([], rn, [1.0] * len(rn), -float(len(rn)))
            # Fork margin: at the first token where the correct and incorrect reasoning
            # diverge (one of them is the greedy one), CE(correct) - CE(incorrect).
            t = next((k for k, (a, b) in enumerate(zip(rp, rn)) if a != b), None)
            if t is not None:
                out["fork_pos"] = (rp[:t], [rp[t]], [1.0], 1.0)
                out["fork_neg"] = (rp[:t], [rn[t]], [1.0], -1.0)
        # Policy-gradient score of expected accuracy: sum_s (acc_s - b) * dCE_sum(r_s),
        # b = mean correctness of the judged samples (greedy + sampled) of this row.
        judged = judged_rows[i]
        baseline = sum(ok for _, ok in judged) / len(judged)
        for k, (reasoning, ok) in enumerate(judged):
            if reasoning and ok != baseline:
                out[f"pg_{k}"] = ([], reasoning, [1.0] * len(reasoning),
                                  (float(ok) - baseline) * len(reasoning) / len(judged))
        return out

    exact: dict[str, dict[tuple, list[float]]] = {}
    patch: dict[str, dict[tuple, list[float]]] = {}
    contributing: dict[str, set[int]] = {}
    for i, row in enumerate(rows):
        values = {"input": row.question}
        for name, (tail, scored, weights, scale) in objective_specs(i).items():
            key = name.rsplit("_", 1)[0] if name.startswith(("contrastive", "fork", "pg_")) else name
            contributing.setdefault(key, set()).add(i)
            pairs = [(runner.ids(prompt, row) + tail, scored)]
            pairs += [(runner.ids(apply(prompt, pools, e), row) + tail, scored) for e in edits]
            losses, _ = answer_losses(model, pairs, weights=[weights] * len(pairs))
            seq = build_superposed(prompt, tokenizer, values, pools, tail, scored, mode="exact",
                                   answer_weights=weights)
            _, est = patch_estimates(scorer, seq)
            for j, e in enumerate(edits):
                exact.setdefault(key, {}).setdefault(e, [0.0] * len(rows))[i] += scale * (losses[j + 1] - losses[0])
                patch.setdefault(key, {}).setdefault(e, [0.0] * len(rows))[i] += scale * est.get(e, 0.0)
        print(json.dumps({"row": i, "positive": positive[i] is not None, "negative": negative[i] is not None}),
              flush=True)

    # Held-out validity target: dev accuracy of every edited prompt with fresh reasoning.
    base_dev = runner.evaluate(prompt, dev, "dev")["accuracy"]
    dev_acc: dict[tuple, float] = {}
    for e in edits:
        dev_acc[e] = runner.evaluate(apply(prompt, pools, e), dev, "dev")["accuracy"] - base_dev
    names = [f"{s}:{'del' if i is None else i}" for s, i in edits]
    delta = [dev_acc[e] for e in edits]
    summary: dict[str, Any] = {"base_dev_accuracy": base_dev, "edits": len(edits),
                               "dev_delta_mean": statistics.mean(delta), "dev_delta_best": max(delta),
                               "rows_with_positive": sum(p is not None for p in positive),
                               "rows_with_both": sum(p is not None and n is not None for p, n in zip(positive, negative))}
    for source, table in (("exact", exact), ("patch", patch)):
        for key, per_edit in table.items():
            usable_rows = sorted(contributing.get(key, set()))
            score = [statistics.mean(per_edit[e][i] for i in usable_rows) if usable_rows else 0.0 for e in edits]
            top = sorted(range(len(edits)), key=lambda j: score[j])[:3]
            summary[f"{source}_{key}"] = {
                "spearman_neg_objective_vs_dev_acc": spearman([-x for x in score], delta),
                "top3_dev_delta": statistics.mean(delta[j] for j in top),
                "rows": len(usable_rows)}
    result = {"format": "promptwitness.graft-validity/v1", "task": args.task, "model_path": args.model_path,
              "seed": args.seed, "pools": pools, "edits": names, "dev_delta": delta,
              "exact": {k: {n: v[e] for n, e in zip(names, edits)} for k, v in exact.items()},
              "patch": {k: {n: v[e] for n, e in zip(names, edits)} for k, v in patch.items()},
              "summary": summary, "wall_seconds": perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
