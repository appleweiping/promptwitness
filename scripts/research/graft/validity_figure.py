"""Figure: how well each signal ranks prompt edits by held-out accuracy change.

Per predictor: one dot per run (Spearman rho with the dev-accuracy change), the pooled
mean with its paired-bootstrap interval, and the band of attainable rho (square root of
the dev target's split-half reliability, across runs). Reads the records and the
``analyze_decision.py --json`` summary; writes a PDF.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from decision_table import ROWS
from fidelity_study import spearman


def run_rho(run: dict, key: str) -> float:
    raw, names = run["raw"], run["edits"]
    base_dev = raw["base_dev"]
    target = [statistics.mean(raw["dev"][n][q] - base_dev[q] for q in range(len(base_dev))) for n in names]
    fork_index = {row: k for k, row in enumerate(raw["fork_rows"])}
    rows = list(range(len(raw["base_fresh"])))
    if key.startswith("answer"):
        pred = [-statistics.mean(raw[key][n]) for n in names]
    elif key.startswith("fork"):
        pred = [-statistics.mean(raw[key][n][fork_index[r]] for r in rows if r in fork_index) for n in names]
    else:
        subset = rows[:8] if key == "fresh8" else rows
        pred = [statistics.mean(raw["fresh"][n][r] - raw["base_fresh"][r] for r in subset) for n in names]
    return spearman(pred, target) or 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in (json.loads(f.read_text(encoding="utf-8")) for f in args.files) if "raw" in r]
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    ceilings = [d["rho_ceiling"] for d in summary["diagnostics"].values()]
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.axhspan(min(ceilings), max(ceilings), color="0.9", zorder=0, label="attainable (noise ceiling)")
    ax.axhline(0.0, color="0.5", lw=0.8, zorder=1)
    labels = []
    for x, (key, name) in enumerate(ROWS):
        values = [run_rho(run, key) for run in runs]
        ax.scatter([x + 0.12 * (i - (len(values) - 1) / 2) / max(1, len(values)) for i in range(len(values))],
                   values, s=10, color="0.45", zorder=2)
        pooled = summary["predictors"][key]
        lo, hi = pooled["rho_ci"]
        ax.errorbar([x + 0.22], [pooled["rho"]], yerr=[[pooled["rho"] - lo], [hi - pooled["rho"]]], fmt="o",
                    color="C3" if key.startswith("answer") else ("C0" if key.startswith("fork") else "C2"),
                    ms=4, capsize=2, zorder=3)
        labels.append(name.replace(" (\\greater{})", "").replace(", ", "\n"))
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Spearman with held-out\naccuracy change", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
