"""Audit search-run records: does every checkpoint's serialized block list rebuild its prompt?

``select_and_test.py`` rebuilds checkpoint prompts with ``graft_tasks.prompt_from_blocks``,
which concatenates blocks in tuple order. Filling an empty slot can move an empty block
that shares its offset behind it, so tuple order and message order may differ; if two such
blocks later both hold text, the rebuilt prompt would swap them. This compares the rebuilt
user message with the recorded one (``text``) for every checkpoint and accepted prompt.

Usage: python audit_checkpoint_order.py RUN.json [RUN.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rebuilt(blocks: list[list[object]]) -> str:
    return "".join(str(text) for _, _, text, _ in blocks)


def main() -> None:
    bad = checked = 0
    for name in sys.argv[1:]:
        run = json.loads(Path(name).read_text(encoding="utf-8"))
        for entry in run.get("checkpoints", []):
            checked += 1
            if rebuilt(entry["blocks"]) != entry["text"]:
                bad += 1
                order = [b[0] for b in entry["blocks"]]
                print(f"MISMATCH {Path(name).name} round {entry['round']} tuple order {order}")
    print(f"checked {checked} checkpoints in {len(sys.argv) - 1} runs; mismatches {bad}")


if __name__ == "__main__":
    main()
