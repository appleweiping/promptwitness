"""How much does one prompt edit change greedy reasoning? (direct measurement)

For a validity-study record (token or block edits), regenerate greedy reasoning on the same
held-out questions under the incumbent and under every edit, with the study's own budget,
and measure per edit: the fraction of questions whose reasoning token sequence changes,
whose parsed answer changes, and whose correctness changes, and the position of the
first differing reasoning token. Generation goes through a vLLM generation server
(``--gen-server``) or the HF model.

Null control (2026-10-01): the incumbent is read ``--null-reads`` extra times under the same
engine; the same rates between its first read and each re-read measure how much of the
edit-induced change the engine produces on an unchanged prompt (numerical nondeterminism).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from promptwitness import graft_tasks
from promptwitness.graft_runtime import ANSWER_TOKENS, first_divergence, generate_batch, load_tokenizer, \
    use_remote_generation

from run_graft import apply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path, help="decision_study output (pools, edits, task, seed)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gen-server", help="vLLM generation server; without it the HF model generates")
    parser.add_argument("--null-reads", type=int, default=2, help="extra reads of the incumbent (null control)")
    parser.add_argument("--null-only", action="store_true", help="measure only the same-prompt null")
    parser.add_argument("--max-new-tokens", type=int, default=384, help="the validity studies' budget")
    parser.add_argument("--dev", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = None
    if args.gen_server:
        use_remote_generation(args.gen_server)
    else:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda").eval()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.model_path)
    spec = graft_tasks.spec(record["task"])
    offset = record.get("dev_offset", 0)
    dev = graft_tasks.load_splits(args.data_dir, record["task"])["dev"][offset: offset + args.dev]
    base_prompt = graft_tasks.initial_prompt()
    if record.get("state_edit"):
        base_prompt = base_prompt.replace_block("procedure", record["state_edit"])
    extractor = tokenizer.encode(spec.extractor, add_special_tokens=False)
    pools = record["pools"]

    def run(prompt) -> tuple[list[tuple[int, ...]], list[str | None], list[bool]]:
        ids = [list(prompt.render_tokens(tokenizer, {"input": q.question}).input_ids) for q in dev]
        gens, _, _ = generate_batch(model, tokenizer, ids, max_new_tokens=args.max_new_tokens)
        reads, _, _ = generate_batch(model, tokenizer, [p + list(g.token_ids) + extractor for p, g in zip(ids, gens)],
                                     max_new_tokens=ANSWER_TOKENS)
        parsed = [spec.parse(r.text) for r in reads]
        return [g.token_ids for g in gens], parsed, [spec.correct(r.text, q.answer) for r, q in zip(reads, dev)]

    base_reason, base_parsed, base_correct = run(base_prompt)
    null = []
    for _ in range(args.null_reads):
        reason, parsed, correct = run(base_prompt)
        null.append({"reasoning_changed": statistics.mean(a != b for a, b in zip(reason, base_reason)),
                     "answer_changed": statistics.mean(a != b for a, b in zip(parsed, base_parsed)),
                     "correctness_flipped": statistics.mean(a != b for a, b in zip(correct, base_correct)),
                     "accuracy_delta": statistics.mean(correct) - statistics.mean(base_correct)})
        print(json.dumps({"null_read": null[-1]}), flush=True)
    per_edit = {}
    for name in ([] if args.null_only else record["edits"]):
        slot, index = name.split(":")
        edit = (slot, None if index == "del" else int(index))
        reason, parsed, correct = run(apply(base_prompt, pools, edit))
        changed = [a != b for a, b in zip(reason, base_reason)]
        firsts = [first_divergence(list(a), list(b)) for a, b, c in zip(reason, base_reason, changed) if c]
        firsts = [f if f is not None else min(len(a), len(b)) for f, a, b in
                  zip(firsts, [r for r, c in zip(reason, changed) if c], [r for r, c in zip(base_reason, changed) if c])]
        per_edit[name] = {
            "reasoning_changed": statistics.mean(changed),
            "answer_changed": statistics.mean(a != b for a, b in zip(parsed, base_parsed)),
            "correctness_flipped": statistics.mean(a != b for a, b in zip(correct, base_correct)),
            "accuracy_delta": statistics.mean(correct) - statistics.mean(base_correct),
            "first_divergence_median": statistics.median(firsts) if firsts else None,
        }
        print(json.dumps({"edit": name, **per_edit[name]}), flush=True)
    summary: dict = {}
    if per_edit:
        summary = {k: statistics.mean(v[k] for v in per_edit.values())
                   for k in ("reasoning_changed", "answer_changed", "correctness_flipped")}
        medians = [v["first_divergence_median"] for v in per_edit.values() if v["first_divergence_median"] is not None]
        summary["first_divergence_median"] = statistics.median(medians) if medians else None
    summary["base_accuracy"] = statistics.mean(base_correct)
    if null:
        summary["null"] = {k: statistics.mean(n[k] for n in null)
                           for k in ("reasoning_changed", "answer_changed", "correctness_flipped")}
    out = {"record": str(args.record), "task": record["task"], "edit_kind": record.get("edit_kind", "block"),
           "model_path": args.model_path, "engine": "vllm-server" if args.gen_server else "hf",
           "per_edit": per_edit, "null_reads": null, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
