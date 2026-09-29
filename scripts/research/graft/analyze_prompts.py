"""Non-brittleness analysis of final prompts (Stage E), identical for every method.

Input: a JSON list of entries {"name", "task", "text"} (collected from run records and
fixed-prompt evaluations). Modes:
  ppl       per-token perplexity of the instruction text under a reference model
  perturb   test accuracy under standardized perturbations with a task model:
            three single-word deletions (fixed seeds) and one paraphrase written by
            the reference model with a fixed instruction; plus the unperturbed prompt.
Transfer is measured separately with ``evaluate_prompts.py --sets file:<json>``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from promptwitness import graft_tasks
from promptwitness.graft_runtime import load_tokenizer

from run_graft import Ledger, Runner

PARAPHRASE = ("Paraphrase the following instruction for a language model. Keep its "
              "meaning and all requirements; change the wording. Return only the "
              "paraphrase.\n\nInstruction:\n{text}")


def instruction(entry: dict[str, Any]) -> str:
    text = entry["text"]
    return text.split("\n\n", 1)[1].strip() if text.startswith("{{input}}") else text.strip()


def perplexity(model: Any, tokenizer: Any, text: str) -> float:
    import torch

    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        loss = model(input_ids=ids, labels=ids).loss
    return math.exp(float(loss))


def word_deletions(text: str, count: int = 3) -> list[str]:
    words = text.split(" ")
    out = []
    for seed in range(count):
        if len(words) < 2:
            break
        index = random.Random(seed).randrange(len(words))
        out.append(" ".join(words[:index] + words[index + 1:]))
    return out


def paraphrase(model: Any, tokenizer: Any, text: str) -> str:
    import torch

    enc = tokenizer.apply_chat_template([{"role": "user", "content": PARAPHRASE.format(text=text)}],
                                        tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([list(enc["input_ids"] if hasattr(enc, "keys") else enc)], device=model.device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=160,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().strip('"')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("ppl", "perturb"))
    parser.add_argument("--entries", type=Path, required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--task-model")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    entries = json.loads(args.entries.read_text(encoding="utf-8"))
    results: dict[str, Any] = json.loads(args.output.read_text()) if args.output.exists() else {}
    ref_tok = load_tokenizer(args.reference_model)
    ref = AutoModelForCausalLM.from_pretrained(args.reference_model, dtype=torch.bfloat16).to("cuda").eval()
    if args.mode == "ppl":
        for entry in entries:
            text = instruction(entry)
            results[f"{entry['task']}/{entry['name']}"] = {
                "ppl": perplexity(ref, ref_tok, text), "words": len(text.split()), "text": text}
        args.output.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        return
    variants: dict[str, list[tuple[str, str]]] = {}
    for entry in entries:
        text = instruction(entry)
        variants[entry["name"] + "|" + entry["task"]] = (
            [("original", text)] + [(f"delete{i}", t) for i, t in enumerate(word_deletions(text))]
            + [("paraphrase", paraphrase(ref, ref_tok, text))])
    del ref
    torch.cuda.empty_cache()
    tok = load_tokenizer(args.task_model)
    model = AutoModelForCausalLM.from_pretrained(args.task_model, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn).to("cuda").eval()
    for key, versions in variants.items():
        name, task = key.split("|")
        spec = graft_tasks.spec(task)
        split = graft_tasks.load_splits(args.data_dir, task)["test"]
        runner = Runner(model, tok, spec, Ledger(), 512)
        for label, text in versions:
            result_key = f"{task}/{name}/{label}"
            if result_key in results:
                continue
            acc = runner.evaluate(graft_tasks.flat_prompt(text), split, "test")["accuracy"]
            results[result_key] = {"accuracy": acc, "text": text}
            args.output.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"key": result_key, "accuracy": acc}), flush=True)


if __name__ == "__main__":
    main()
