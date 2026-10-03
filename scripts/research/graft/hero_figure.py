"""Figure 1 candidate: what predicts the effect of a prompt edit?

Left: per-pool Spearman correlation between a signal and the held-out accuracy change of the
pool's LLM-proposed block edits (11 pools; marker = model), with the pooled mean. Signals: the
fixed-reasoning objective (GReaTer's, exact), the registered reweighting of the model's own
samples (tempered SNIS, beta chosen without labels), fresh greedy accuracy on 24 training
questions, and coupled sampled accuracy on 24 questions x 4 samples. Right: per pool, the mean
absolute read-off term |A| (what a fixed-reasoning signal can see) against the mean absolute
held-out effect of the edits (accuracy points).

Inputs: ``analyze_dist.py --json`` summary and the ``readoff_stats.py`` log (stdout table).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

SIGNALS = [("answer_exact", "fixed-reasoning objective\n(GReaTer, exact)"),
           ("primary", "reweighted own samples\n(registered; failed)"),
           ("fresh_all", "fresh greedy reasoning\n(24 questions)"),
           ("fresh_dist", "fresh sampled reasoning\n(24 questions x 4)")]
MODEL = {"Meta-Llama-3-8B-Instruct": ("Llama-3-8B", "o", "#0b5394"),
         "Qwen3-8B": ("Qwen3-8B", "s", "#e69138"),
         "gemma-2-9b-it": ("Gemma-2-9B", "^", "#38761d")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, help="analysis_11pools.json")
    parser.add_argument("readoff_log", type=Path, help="readoff_11pools.log")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    pools = summary["pools"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.15), gridspec_kw={"width_ratios": [1.55, 1]})
    for row, (key, label) in enumerate(SIGNALS):
        y = len(SIGNALS) - 1 - row
        values = []
        for i, pool in enumerate(pools):
            model = pool["pool"].split("/")[1]
            name, marker, color = MODEL[model]
            v = pool["rho"][key]
            values.append(v)
            jitter = ((i * 7) % 11 - 5) * 0.035
            ax.scatter(v, y + jitter, marker=marker, s=13, facecolor="none" if marker == "^" else color,
                       edgecolor=color, linewidth=0.7, zorder=3)
        mean = statistics.mean(values)
        ax.plot([mean, mean], [y - 0.32, y + 0.32], color="black", linewidth=1.6, zorder=4)
        ax.text(1.02, y, f"{mean:+.2f}", va="center", ha="left", fontsize=7, transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.set_yticks(range(len(SIGNALS)))
    ax.set_yticklabels([label for _, label in reversed(SIGNALS)], fontsize=6.8)
    ax.set_xlim(-0.95, 0.95)
    ax.set_xlabel("Spearman with held-out change (per pool; bar = mean)")
    handles = [plt.Line2D([], [], marker=m, linestyle="", markersize=4, markerfacecolor="none" if m == "^" else c,
                          markeredgecolor=c, label=n) for n, m, c in MODEL.values()]
    ax.legend(handles=handles, fontsize=6, loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    # right panel: read-off term vs held-out effect, per pool
    rows = []
    for line in args.readoff_log.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\w+)-(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows.append((m.group(1), float(m.group(4)) * 100, float(m.group(5)) * 100))
    order = sorted(range(len(rows)), key=lambda i: rows[i][2])
    for y, i in enumerate(order):
        model, a, held = rows[i]
        color = {"llama3": "#0b5394", "qwen3": "#e69138", "gemma2": "#38761d"}[model]
        bx.plot([0, held], [y, y], color=color, alpha=0.35, linewidth=3.2, solid_capstyle="butt")
        bx.scatter(a, y, color="black", s=8, zorder=3)
    bx.set_yticks([])
    bx.set_xlabel("accuracy points per edit")
    bx.set_title("dot: read-off term |A|; bar: |held-out effect|", fontsize=6.8, pad=3)
    bx.set_xlim(0, max(r[2] for r in rows) * 1.08)
    for a_ in (ax, bx):
        a_.spines["top"].set_visible(False)
        a_.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print("wrote", args.output, "| pooled means:",
          {k: round(statistics.mean(p["rho"][k] for p in pools), 3) for k, _ in SIGNALS})


if __name__ == "__main__":
    main()
