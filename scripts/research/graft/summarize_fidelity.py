"""Aggregate fidelity-study JSON files into a markdown table (no row text is printed)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from fidelity_study import spearman

METHODS = ("patch", "incumbent", "incumbent_no_offset", "hole", "centroid")


def topk_recall(est: list[float], exact: list[float], k: int) -> float:
    best = set(sorted(range(len(exact)), key=lambda i: exact[i])[:k])
    chosen = set(sorted(range(len(est)), key=lambda i: est[i])[:k])
    return len(best & chosen) / k


def normalized_regret(est: list[float], exact: list[float]) -> float:
    """(exact of estimator's top-1 - best exact) / (median exact - best exact); 0 is ideal."""
    chosen = exact[min(range(len(est)), key=lambda i: est[i])]
    best, median = min(exact), statistics.median(exact)
    return (chosen - best) / (median - best) if median > best else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    header = ("| run | edits | exact-fresh rho | " +
              " | ".join(f"{m} rho / top{args.k} / regret" for m in METHODS) +
              " | patch s/ex | exact s/ex |")
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    pooled: dict[str, list[float]] = {m: [] for m in METHODS}
    regrets: dict[str, list[float]] = {}
    for path in args.files:
        d = json.loads(path.read_text(encoding="utf-8"))
        names, rows = d["names"], d["per_row"]
        exact = [statistics.mean(r["exact"][n] for r in rows) for n in names]
        cells = []
        for method in METHODS:
            usable = [j for j, n in enumerate(names)
                      if all(n in r["estimates"].get(method, {}) for r in rows)]
            if len(usable) < 3:
                cells.append("–")
                continue
            est = [statistics.mean(r["estimates"][method][names[j]] for r in rows) for j in usable]
            sub = [exact[j] for j in usable]
            rho = spearman(est, sub)
            pooled[method].append(rho if rho is not None else 0.0)
            regret = normalized_regret(est, sub)
            regrets.setdefault(method, []).append(regret)
            cells.append(f"{rho:.2f} / {topk_recall(est, sub, args.k):.2f} / {regret:.2f}")
        t = d["timing"]
        n = len(rows)
        model = Path(d["model_path"]).parts[-3].split("--")[-1] if "snapshots" in d["model_path"] else d["model_path"]
        print(f"| {model} {d['task'][:18]} | {len(names)} | "
              f"{d['summary']['spearman_exact_vs_fresh_loss']:.2f} | " + " | ".join(cells) +
              f" | {t.get('patch', 0) / n:.1f} | {t['exact_vertices'] / n:.1f} |")
    print()
    for method, values in pooled.items():
        if values:
            print(f"{method}: mean rho {statistics.mean(values):.3f}, mean normalized regret "
                  f"{statistics.mean(regrets[method]):.3f} over {len(values)} runs")


if __name__ == "__main__":
    main()
