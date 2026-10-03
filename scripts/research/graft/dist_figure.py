"""Figure: how well each signal ranks block edits, as a function of the importance tempering beta.

Reads an ``analyze_dist.py --json`` summary. Left: pooled Spearman with the held-out change
(bias-corrected bootstrap interval, conditional on the pools) for the tempered estimators
with candidate reads, with the incumbent's own reads (reasoning-distribution term only) and
with soft reads, against GReaTer's fixed-reasoning objective and fresh greedy evaluation.
Right: best-3 regret (lower is better) with the uniformly random shortlist as reference.
In the left panel the three estimator series are offset by a small horizontal dodge so
that their intervals do not overlap; the tick marks give the true beta values. Sized for
the 5.5 in ICLR text width (include at ``width=\\linewidth``); writes a vector PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BETAS = (0.0, 0.25, 0.5, 1.0)
BETA_LABELS = ("0", "1/4", "1/2", "1")
DODGE = 0.022


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    })

    s = json.loads(args.summary.read_text(encoding="utf-8"))["predictors"]
    series = {
        "tempered SNIS, candidate reads": ("snis_hard_{}", "#2a78d6", "o", -DODGE),
        "incumbent reads (distribution term)": ("snis_incread_{}", "#eb6834", "s", 0.0),
        "soft reads": ("snis_soft_{}", "#1baf7a", "^", DODGE),
    }
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45))
    handles, labels = [], []
    for label, (pattern, color, marker, dodge) in series.items():
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
        xd = [x + dodge for x in xs]
        (h,) = axes[0].plot(xd, rho, color=color, marker=marker, ms=4, lw=1.2, label=label, zorder=3)
        axes[0].vlines(xd, lo, hi, color=color, alpha=0.45, linewidth=2.0, zorder=2)
        axes[1].plot(xs, reg, color=color, marker=marker, ms=4, lw=1.2, zorder=3)
        handles.append(h)
        labels.append(label)
    refs = {"GReaTer objective (fixed greedy reasoning)": ("answer_exact", "#0b0b0b", (0, (4, 2))),
            "fresh greedy, 8 rows": ("fresh8", "#7a7975", (0, (1, 1.5))),
            "fresh greedy, 24 rows": ("fresh_all", "#52514e", (0, (5, 1.5, 1, 1.5))),
            "random shortlist (regret)": ("random", "#b5b4ae", "-")}
    for label, (key, color, style) in refs.items():
        h = axes[1].axhline(s[key]["regret"], color=color, linestyle=style, lw=1.0, zorder=1)
        if key != "random":
            axes[0].axhline(s[key]["rho"], color=color, linestyle=style, lw=1.0, zorder=1)
        handles.append(h)
        labels.append(label)
    axes[0].axhline(0, color="#0b0b0b", linewidth=0.5, zorder=0)
    axes[0].set_xlabel(r"importance tempering $\beta$ ($\beta=0$: fixed reasoning)")
    axes[0].set_ylabel("Spearman with held-out change")
    axes[1].set_xlabel(r"importance tempering $\beta$")
    axes[1].set_ylabel("best-3 regret (lower is better)")
    for ax in axes:
        ax.set_xticks(BETAS)
        ax.set_xticklabels(BETA_LABELS)
        ax.set_xlim(-0.08, 1.08)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, handlelength=2.6,
               columnspacing=1.2, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
