"""Figure: verification quality against the number of fresh generations per candidate.

Reads the ``verify_cost.json`` summary (11 pools; each cell is a verifier that spends
``rows x samples`` generations per candidate and is averaged over random row/sample
subsets). The x axis is the number of generations per candidate G on a log2 scale.
Left: pooled Spearman of the verifier's score with the held-out change. Right: realized
held-out change (accuracy points) of a verify-all step, i.e. adopting the verifier's top
edit in every pool. Series: greedy reads (rows x 1); sampled reads with common random
numbers (rows x 1 up to the largest row count, then more samples per row); sampled reads
with independent seeds (same layout). Sized for the 5.5 in ICLR text width (include at
``width=\\linewidth``); writes a vector PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TICKS = (4, 8, 12, 16, 24, 48, 96)
SERIES = (
    ("greedy", "greedy reads (rows $\\times$ 1)", "#2a78d6", "o", "-", True, 4.2),
    ("coupled", "sampled, common random numbers", "#eb6834", "s", "-", True, 4.2),
    ("indep", "sampled, independent seeds", "#1baf7a", "^", (0, (4, 2)), False, 6.0),
)


def select(cells: dict, kind: str) -> list[dict]:
    """Cells of one kind with one sample per row, plus the full-row cells with more samples."""
    rows_max = max(c["rows"] for c in cells.values())
    chosen = [c for c in cells.values() if c["kind"] == kind and (c["samples"] == 1 or c["rows"] == rows_max)]
    return sorted(chosen, key=lambda c: c["generations"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

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

    cells = json.loads(args.summary.read_text(encoding="utf-8"))["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    handles, labels = [], []
    for kind, label, color, marker, style, filled, size in SERIES:
        chosen = select(cells, kind)
        g = [c["generations"] for c in chosen]
        rho = [c["rho"] for c in chosen]
        gain = [100 * c["verify_all"] for c in chosen]
        face = color if filled else "white"
        kw = dict(color=color, marker=marker, linestyle=style, lw=1.1, ms=size, mfc=face, mec=color,
                  mew=1.0, zorder=3 if filled else 4)
        (h,) = axes[0].plot(g, rho, **kw)
        axes[1].plot(g, gain, **kw)
        handles.append(h)
        labels.append(label)
        print(kind, [(c["rows"], c["samples"], c["generations"], round(c["rho"], 3),
                      round(100 * c["verify_all"], 2)) for c in chosen])
        if kind == "coupled":
            for c in chosen:
                if c["samples"] > 1:
                    axes[0].annotate(f"{c['rows']}$\\times${c['samples']}", (c["generations"], c["rho"]),
                                     textcoords="offset points", xytext=(0, -11), ha="center",
                                     fontsize=6.5, color="#52514e")
    axes[1].axhline(0, color="#0b0b0b", linewidth=0.7, zorder=1)
    axes[0].set_ylabel("pooled Spearman with\nheld-out change")
    axes[1].set_ylabel("held-out change of\nverify-all step (points)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(TICKS))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(t) for t in TICKS])
        ax.set_xlim(3.4, 112)
        ax.set_xlabel("generations per candidate $G$")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, handlelength=2.6,
               columnspacing=1.5, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
