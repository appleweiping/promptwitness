"""Re-score a validity study against held-out targets re-read with the deterministic engine.

The validity records (Table 1) hold per-question held-out correctness of every edit
(``raw.dev``) and of the incumbent (``raw.base_dev``), read with the Hugging Face batched reader.
``reroll_study.py`` re-reads the same edits on the same held-out questions with the in-process
vLLM engine (``reroll3/<record>.json``: ``per_edit[edit].correct``, ``base_correct``). This script
writes copies of the records whose targets are the re-reads (predictors unchanged), so that
``analyze_decision.py`` reproduces Table 1 against the new targets, and prints per pool how well the
two target vectors agree (Spearman over edits of the held-out change; incumbent accuracies).

usage: reread_targets.py RECORD.json=REROLL.json [...] --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from fidelity_study import spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="RECORD.json=REROLL.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    agree = []
    print(f"{'pool':58s} {'base acc HF/vLLM':>17s} {'mean |edit effect| HF/vLLM':>27s} {'rho(targets)':>13s}")
    for pair in args.pairs:
        rec_path, reroll_path = (Path(p) for p in pair.split("="))
        record = json.loads(rec_path.read_text(encoding="utf-8"))
        reroll = json.loads(reroll_path.read_text(encoding="utf-8"))
        names = record["edits"]
        raw = record["raw"]
        new_base = reroll["base_correct"]
        if len(new_base) != len(raw["base_dev"]):
            raise SystemExit(f"{rec_path.name}: {len(new_base)} re-read questions vs {len(raw['base_dev'])} recorded")
        old_t = [statistics.mean(a - b for a, b in zip(raw["dev"][n], raw["base_dev"])) for n in names]
        new_dev = {n: reroll["per_edit"][n]["correct"] for n in names}
        new_t = [statistics.mean(a - b for a, b in zip(new_dev[n], new_base)) for n in names]
        rho = spearman(old_t, new_t)
        agree.append(rho)
        out = dict(record)
        out["raw"] = dict(raw, dev=new_dev, base_dev=new_base)
        out["targets"] = {"source": "deterministic in-process engine re-read", "reroll": str(reroll_path)}
        (args.out_dir / rec_path.name).write_text(json.dumps(out) + "\n", encoding="utf-8")
        label = f"{record['task'][:30]}/{Path(record['model_path']).parts[-3].split('--')[-1][:12]}" + (
            "*" if record.get("state_edit") else "")
        print(f"{label:58s} {statistics.mean(raw['base_dev']):7.3f}/{statistics.mean(new_base):.3f}"
              f"{statistics.mean(abs(x) for x in old_t):17.3f}/{statistics.mean(abs(x) for x in new_t):.3f}"
              f"{rho if rho is not None else float('nan'):13.2f}")
    print(f"mean Spearman between the two target vectors: {statistics.mean(r for r in agree if r is not None):+.2f}")


if __name__ == "__main__":
    main()
