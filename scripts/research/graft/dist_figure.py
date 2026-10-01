"""Figure: how well each signal ranks block edits, as a function of the importance tempering beta.

Reads an ``analyze_dist.py --json`` summary. Left: pooled Spearman with the held-out change
(bias-corrected bootstrap interval, conditional on the pools) for the tempered estimators
with candidate reads, with the incumbent's own reads (reasoning-distribution term only) and
with soft reads, against GReaTer's fixed-reasoning objective and fresh greedy evaluation.
Right: best-3 regret (lower is better) with the uniformly random shortlist as reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BETAS = (0.0, 0.25, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads(args.summary.read_text(encoding="utf-8"))["predictors"]
    series = {
        "candidate reads": ("snis_hard_{}", "#1b6ca8", "o"),
        "incumbent reads (distribution term)": ("snis_incread_{}", "#2a9d5c", "s"),
        "soft reads": ("snis_soft_{}", "#9a6fb0", "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for label, (pattern, color, marker) in series.items():
        xs, rho, lo, hi, reg = [], [], [], [], []
        for b in BETAS:
            key = pattern.format(b)
            if key not in s:
                continue
            xs.append(b)
            rho.append(s[key]["rho"])
            lo.append(s[key]["rho_ci_bc"][0])
            hi.append(s[key]["rho_ci_bc"][1])
            reg.append(s[key]["regret"])
        # BC intervals need not contain the point (the bootstrap of a Spearman correlation with
        # a resampled noisy target is shifted), so draw them as ranges, not symmetric error bars.
        axes[0].plot(xs, rho, color=color, marker=marker, label=label)
        axes[0].vlines(xs, lo, hi, color=color, alpha=0.5, linewidth=2)
        axes[1].plot(xs, reg, color=color, marker=marker, label=label)
    refs = {"GReaTer objective (fixed greedy trace)": ("answer_exact", "#c0392b", "--"),
            "fresh greedy, 8 rows": ("fresh8", "#7f7f7f", ":"),
            "fresh greedy, 24 rows": ("fresh_all", "#333333", "-."),
            "random shortlist": ("random", "#bbbbbb", "-")}
    for label, (key, color, style) in refs.items():
        if key == "random":
            axes[1].axhline(s[key]["regret"], color=color, linestyle=style, label=label)
            continue
        axes[0].axhline(s[key]["rho"], color=color, linestyle=style, label=label)
        axes[1].axhline(s[key]["regret"], color=color, linestyle=style)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_xlabel(r"importance tempering $\beta$ ($\beta=0$: fixed reasoning)")
    axes[0].set_ylabel("Spearman with held-out change")
    axes[1].set_xlabel(r"importance tempering $\beta$")
    axes[1].set_ylabel("best-3 regret (lower is better)")
    axes[0].legend(fontsize=6.5, loc="lower right")
    axes[1].legend(fontsize=6.5, loc="upper right")
    for ax in axes:
        ax.set_xticks(BETAS)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
