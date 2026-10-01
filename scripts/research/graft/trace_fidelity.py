"""One-pass likelihood ratios: can the superposed calculus replace per-candidate teacher forcing?

The H-dist estimators need, for every candidate prompt P' and every incumbent sample r_s,
the log-ratio log pi_P'^tau(r_s|x) - log pi_P^tau(r_s|x). ``is_study.py`` computes them
exactly with one teacher-forced pass per candidate. Here every candidate block of every
slot sits in one superposed sequence ahead of the sample, and renormalized patching
(``superposed_patching``) estimates all single-slot log-ratios from one forward and one
backward pass. Outputs: per-sample fidelity against the exact values of the is_study
record (Spearman across candidates, pooled correlation, absolute error) and a copy of the
is_study record whose ``logp`` holds the one-pass values, so ``analyze_dist.py`` evaluates
the same estimators with one-pass weights.

The sample's stop event is scored as the chat end-of-turn token (the exact record uses the
probability of any stop token); candidates whose blocks do not tokenize cleanly get no
estimate and keep their exact value (counted and reported).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

from promptwitness import graft_tasks
from promptwitness.graft_runtime import load_tokenizer
from promptwitness.superposed_gates import GateScorer, build_superposed
from promptwitness.superposed_patching import patch_estimates

from fidelity_study import spearman


def end_of_turn(tokenizer) -> int:  # noqa: ANN001
    for name in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>"):
        token = tokenizer.convert_tokens_to_ids(name)
        if isinstance(token, int) and token != tokenizer.unk_token_id:
            return token
    return int(tokenizer.eos_token_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path, help="decision_study record of the pool")
    parser.add_argument("dist", type=Path, help="is_study output with traces")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--output", type=Path, required=True, help="is_study-format copy with one-pass logp")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    record = json.loads(args.record.read_text(encoding="utf-8"))
    dist = json.loads(args.dist.read_text(encoding="utf-8"))
    if "traces" not in dist:
        raise SystemExit("is_study output has no traces (run before 2026-10-01 14:40 UTC)")
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn).to("cuda").eval()
    scorer = GateScorer(model)
    splits = graft_tasks.load_splits(args.data_dir, record["task"])
    by_id = {e.example_id: e for e in splits["train"]}
    rows = [by_id[i] for i in record["rows"]]
    base = graft_tasks.initial_prompt()
    if record.get("state_edit"):
        base = base.replace_block("procedure", record["state_edit"])
    pools = record["pools"]
    names = dist["edits"]
    keys = {(n.split(":")[0], None if n.split(":")[1] == "del" else int(n.split(":")[1])): n for n in names}
    tau = float(dist["tau"])
    eot = end_of_turn(tokenizer)

    one_pass = {n: [[None] * len(dist["traces"][x]) for x in range(len(rows))] for n in names}
    seconds, missing = 0.0, 0
    for x, row in enumerate(rows):
        for s, trace in enumerate(dist["traces"][x]):
            scored = list(trace) + ([eot] if dist["trace_ended"][x][s] else [])
            if not scored:
                continue
            seq = build_superposed(base, tokenizer, {"input": row.question}, pools, [], scored, mode="exact",
                                   answer_temperature=tau)
            started = perf_counter()
            loss, est = patch_estimates(scorer, seq)
            seconds += perf_counter() - started
            for key, name in keys.items():
                if key in est:
                    # mean CE over scored tokens -> change of the sum log-probability
                    one_pass[name][x][s] = dist["logp"]["base"][x][s] - len(scored) * est[key]
                else:
                    missing += 1
    exact_logw, patch_logw, per_sample_rho = [], [], []
    out_logp = {"base": dist["logp"]["base"]}
    for name in names:
        out_logp[name] = []
        for x in range(len(rows)):
            row_vals = []
            for s in range(len(dist["traces"][x])):
                exact = dist["logp"][name][x][s]
                value = one_pass[name][x][s]
                row_vals.append(exact if value is None else value)
                if value is not None:
                    exact_logw.append(exact - dist["logp"]["base"][x][s])
                    patch_logw.append(value - dist["logp"]["base"][x][s])
            out_logp[name].append(row_vals)
    for x in range(len(rows)):
        for s in range(len(dist["traces"][x])):
            pairs = [(dist["logp"][n][x][s], one_pass[n][x][s]) for n in names if one_pass[n][x][s] is not None]
            if len(pairs) >= 3:
                rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
                if rho is not None:
                    per_sample_rho.append(rho)
    pearson = statistics.correlation(exact_logw, patch_logw) if len(exact_logw) > 2 else None
    summary = {"samples": sum(len(t) for t in dist["traces"]), "candidates": len(names), "missing": missing,
               "one_pass_seconds": seconds, "exact_teacher_forcing_seconds": dist["cost_seconds"]["teacher_forcing"],
               "per_sample_spearman_median": statistics.median(per_sample_rho) if per_sample_rho else None,
               "per_sample_spearman_mean": statistics.mean(per_sample_rho) if per_sample_rho else None,
               "pooled_pearson_logw": pearson,
               "spearman_logw": spearman(exact_logw, patch_logw),
               "mean_abs_error_logw": statistics.mean(abs(a - b) for a, b in zip(exact_logw, patch_logw)),
               "sd_exact_logw": statistics.pstdev(exact_logw)}
    result = dict(dist) | {"logp": out_logp, "logp_source": "one-pass renormalized patching",
                           "fidelity": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
