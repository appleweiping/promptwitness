"""LaTeX table of the registered H-dist analysis: every predictor against the held-out change.

Input: the --json summary written by analyze_dist.py (11 block pools). Output: a booktabs tabular
(no float, no caption) with predictors grouped by what they read:

  fixed reasoning         GReaTer's objective (exact); candidate reads of the incumbent's own
                          samples without reweighting (SNIS beta = 0)
  reweighted own samples  SNIS beta = 1/4, 1/2, 1; primary (beta* chosen without labels);
                          incumbent reads at beta* (distribution term only); first-order score
                          function; KL gate
  fresh reasoning         fresh greedy accuracy on 8 and 24 rows; coupled sampled 24 rows x 4
  random                  uniformly random shortlist (exact expectation)

Columns: pooled Spearman (mean over pools) with an interval; pooled best-3 regret; pools in which
the predictor's best-3 regret is strictly below the random expectation of that pool (taken from
the stored comparison when analyze_dist.py recorded one, otherwise counted from pools[*].regret
with the same strict rule); and, for the predictors the verify-all simulation covers, the mean
realized held-out change (accuracy points) and the number of pools improved / worsened.

Interval (--interval):
  pool (default)  percentile bootstrap over pools, the units of inference: resample the per-pool
                  Spearman values (pools[*].rho) with replacement, take the mean, repeat
                  --replicates times with a fixed --seed (one shared set of resampled pool
                  indices for all predictors), report the 2.5% and 97.5% quantiles (linear
                  interpolation between order statistics)
  bc              the bias-corrected bootstrap interval stored by analyze_dist.py
                  (predictors[*].rho_ci_bc; training and held-out questions resampled within
                  pools, conditional on the pools)
Both intervals are printed to stdout for every predictor; only the chosen one enters the table.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Fixed reasoning", (
        ("answer_exact", r"\greater{} objective (exact)"),
        ("snis_hard_0.0", r"Candidate reads, $\beta=0$"),
    )),
    ("Reweighted own samples", (
        ("snis_hard_0.25", r"SNIS, $\beta=1/4$"),
        ("snis_hard_0.5", r"SNIS, $\beta=1/2$"),
        ("snis_hard_1.0", r"SNIS, $\beta=1$"),
        ("primary", r"Primary, $\beta^\ast$"),
        ("inc_primary", r"Incumbent reads, $\beta^\ast$"),
        ("score_fn", r"First-order score function"),
        ("kl_gate", r"KL gate"),
    )),
    ("Fresh reasoning", (
        ("fresh8", r"Greedy, 8 rows"),
        ("fresh_all", r"Greedy, 24 rows"),
        ("fresh_dist", r"Sampled, $24\times4$ (coupled)"),
    )),
)
RANDOM = ("random", r"Random shortlist")
NCOLS = 7
HEADERS = {
    "pool": r"\makecell{Pooled Spearman\\{}[pool bootstrap]}",
    "bc": r"\makecell{Pooled Spearman\\{}[BC interval]}",
}


def signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def quantile(sorted_values: list[float], q: float) -> float:
    """Quantile with linear interpolation between order statistics (numpy's default rule)."""
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def pool_bootstrap(summary: dict, names: list[str], replicates: int, seed: int,
                   level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap of the mean per-pool Spearman, resampling pools with replacement.

    The same resampled pool indices are used for every predictor, so the intervals are paired.
    The mean of the per-pool values must reproduce the stored pooled Spearman.
    """
    pools = summary["pools"]
    n = len(pools)
    values: dict[str, list[float]] = {}
    for name in names:
        per_pool = [float(p["rho"][name]) for p in pools]
        if any(math.isnan(v) for v in per_pool):
            raise ValueError(f"{name}: undefined per-pool Spearman")
        pooled = summary["predictors"][name]["rho"]
        if abs(statistics.fmean(per_pool) - pooled) > 1e-9:
            raise ValueError(f"{name}: mean of pools[*].rho {statistics.fmean(per_pool)} != pooled {pooled}")
        values[name] = per_pool
    rng = random.Random(seed)
    means: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(replicates):
        idx = [math.floor(rng.random() * n) for _ in range(n)]
        for name in names:
            means[name].append(math.fsum(values[name][i] for i in idx) / n)
    alpha = (1.0 - level) / 2.0
    out = {}
    for name in names:
        ordered = sorted(means[name])
        out[name] = (quantile(ordered, alpha), quantile(ordered, 1.0 - alpha))
    return out


def lower_than_random(summary: dict, name: str) -> tuple[int, int, int]:
    """(pools with strictly lower regret than random, pools, near-ties) for one predictor."""
    stored = summary["comparisons"].get(f"{name}-random")
    pools = summary["pools"]
    counted = sum(p["regret"][name] < p["regret"]["random"] for p in pools)
    ties = sum(abs(p["regret"][name] - p["regret"]["random"]) < 1e-9 for p in pools)
    if stored is not None:
        if stored["pools_with_lower_regret"] != counted:
            raise ValueError(f"{name}: stored {stored['pools_with_lower_regret']} vs counted {counted}")
        return stored["pools_with_lower_regret"], stored["pools"], ties
    return counted, len(pools), ties


def row(summary: dict, name: str, label: str, pool_ci: dict[str, tuple[float, float]],
        interval_kind: str) -> tuple[str, str]:
    pred = summary["predictors"][name]
    bc_lo, bc_hi = pred["rho_ci_bc"]
    if name == "random":
        rho, interval, below = "0", "", "--"
        log_below, log_ci = "--", ""
    else:
        p_lo, p_hi = pool_ci[name]
        lo, hi = (p_lo, p_hi) if interval_kind == "pool" else (bc_lo, bc_hi)
        rho = signed(pred["rho"], 3)
        interval = rf"\scriptsize[{signed(lo, 3)}, {signed(hi, 3)}]"
        n_below, n_pools, ties = lower_than_random(summary, name)
        below = f"{n_below}/{n_pools}"
        log_below = below + (f" ({ties} ties)" if ties else "")
        log_ci = f"pool [{p_lo:+.3f}, {p_hi:+.3f}]  bc [{bc_lo:+.3f}, {bc_hi:+.3f}]"
    verify = summary["verify_all"].get(name)
    if verify is None:
        realized, counts, log_verify = "--", "--", "--"
    else:
        realized = signed(100 * verify["mean"], 2)
        counts = f"+{verify['improved']}/-{verify['worsened']}"
        log_verify = f"{realized} {counts}"
    tex = f"\\quad {label} & {rho} & {interval} & {pred['regret']:.3f} & {below} & {realized} & {counts} \\\\"
    log = (f"{name:16s} rho {pred['rho']:+.3f} {log_ci:44s} regret {pred['regret']:.3f}  "
           f"below random {log_below:12s} verify-all {log_verify}")
    return tex, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary", type=Path, help="analysis_11pools.json from analyze_dist.py --json")
    parser.add_argument("--latex", type=Path, required=True, help="output tabular (e.g. paper/tables/hdist.tex)")
    parser.add_argument("--interval", choices=("pool", "bc"), default="pool",
                        help="interval in the table: pool-level percentile bootstrap (default) or stored BC")
    parser.add_argument("--replicates", type=int, default=20000, help="pool bootstrap replicates")
    parser.add_argument("--seed", type=int, default=20261002, help="pool bootstrap seed")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    n_pools = len(summary["pools"])
    names = [name for _, members in GROUPS for name, _ in members]
    pool_ci = pool_bootstrap(summary, names, args.replicates, args.seed)
    lines = [
        r"\begin{tabular}{@{}lr@{\hspace{3pt}}lrcrc@{}}",
        r"\toprule",
        rf" & \multicolumn{{2}}{{c}}{{{HEADERS[args.interval]}}} & \makecell{{Best-3\\regret}}"
        r" & \makecell{Pools below\\random} & \makecell{Verify-all\\(points)} & \makecell{Pools\\$+$/$-$} \\",
    ]
    for title, members in GROUPS:
        lines += [r"\midrule", rf"\multicolumn{{{NCOLS}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for name, label in members:
            tex, log = row(summary, name, label, pool_ci, args.interval)
            lines.append(tex)
            print(log)
    tex, log = row(summary, *RANDOM, pool_ci, args.interval)
    lines += [r"\midrule", tex.replace(r"\quad ", "", 1), r"\bottomrule", r"\end{tabular}"]
    print(log)
    print(f"pools: {n_pools}; table interval: {args.interval}; pool bootstrap: {args.replicates} replicates, "
          f"seed {args.seed}, 95% percentile")
    print(f"H-dist supported: {summary['h_dist_supported']}; "
          f"H-inc supported (11-pool run): {summary['h_inc_supported']}")
    args.latex.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
