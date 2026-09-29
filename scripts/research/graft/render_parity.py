"""Hash the rendered token ids of fixed prompts, to check two environments render alike.

Run with each interpreter (e.g. the HF venv and the vLLM venv) and compare the printed
digests: the vLLM reader is only a drop-in replacement if prompts tokenize identically.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from promptwitness import graft_tasks
from promptwitness.graft_runtime import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["date_understanding", "gsm8k", "formal_fallacies"])
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model_path)
    prompts = {"greater_init": graft_tasks.initial_prompt(),
               "zs_cot": graft_tasks.flat_prompt("Let's think step by step.")}
    digest = hashlib.sha256()
    for task in args.tasks:
        spec = graft_tasks.spec(task)
        digest.update(str(tokenizer.encode(spec.extractor, add_special_tokens=False)).encode())
        for example in graft_tasks.load_splits(args.data_dir, task)["dev"][:20]:
            for prompt in prompts.values():
                ids = list(prompt.render_tokens(tokenizer, {"input": example.question}).input_ids)
                digest.update(str(ids).encode())
    print(Path(args.model_path).parts[-3], digest.hexdigest()[:16])


if __name__ == "__main__":
    main()
