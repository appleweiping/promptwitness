"""Paired hierarchical bootstrap for decision-study records (H-fork validation).

Each bootstrap replicate resamples train rows (with replacement) for every predictor and
dev questions for the target, jointly across predictors (paired), recomputes each
predictor per edit, its Spearman rho with the dev-accuracy change and its best-5 regret
(mean dev change of the 5 truly best edits minus that of the predictor's top 5), and
records paired differences between predictors. Runs are pooled by averaging over runs
within a replicate (each run keeps its own proposal pool).

Predictors on all rows (the operational policy) and conditional on fork-eligible rows
(rows with at least one decision fork): ``*_elig`` restrict the answer loss and fresh
accuracy to those rows; the fork margin is defined only there.

Diagnostics per run: the split-half reliability of the dev target (Spearman-Brown
corrected; its square root bounds any predictor's attainable rho), and GReaTer's answer
loss split by whether the incumbent's greedy reasoning was right or wrong on the row.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from fidelity_study import spearman

KEYS = ("answer_exact", "answer_patch", "fork_exact", "fork_patch", "fresh8", "fresh_all",
        "answer_exact_elig", "fresh_all_elig", "answer_cond_exact", "answer_cond_patch", "random")
PAIRS = (("fork_patch", "answer_exact"), ("fork_exact", "answer_exact"), ("fork_patch", "fresh8"),
         ("fork_patch", "fresh_all"), ("fork_patch", "answer_exact_elig"),
         ("fork_patch", "fresh_all_elig"), ("fork_patch", "fork_exact"),
         # H-cond (preregistered 2026-09-29 before any TS3 data): answer-conditioned shortlist
         ("answer_cond_patch", "answer_patch"), ("answer_cond_patch", "random"),
         ("answer_cond_patch", "fresh8"), ("answer_cond_exact", "answer_exact"),
         # Same-incumbent, same-pool shortlist quality against a uniformly random shortlist
         ("answer_exact", "random"), ("answer_patch", "random"), ("fork_patch", "random"),
         ("fresh8", "random"), ("fresh_all", "random"))


def predictors_for(raw: dict, names: list[str], rows: list[int], questions: list[int]) -> tuple[dict, list[float]]:
    base_fresh = raw["base_fresh"]
    fork_rows = set(raw["fork_rows"])
    fork_index = {row: k for k, row in enumerate(raw["fork_rows"])}  # position in per-edit fork lists
    sampled_fork = [fork_index[r] for r in rows if r in fork_rows]
    eligible = [r for r in rows if r in fork_rows]
    first8 = [r for r in rows if r < 8]

    def mean_or_zero(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    out: dict[str, list[float]] = {}
    for key in ("answer_exact", "answer_patch"):
        out[key] = [-mean_or_zero([raw[key][n][r] for r in rows]) for n in names]
    out["answer_exact_elig"] = [-mean_or_zero([raw["answer_exact"][n][r] for r in eligible]) for n in names]
    for key in ("fork_exact", "fork_patch"):
        out[key] = [-mean_or_zero([raw[key][n][k] for k in sampled_fork]) for n in names]
    for key, subset in (("fresh8", first8), ("fresh_all", rows), ("fresh_all_elig", eligible)):
        out[key] = [mean_or_zero([raw["fresh"][n][r] - base_fresh[r] for r in subset]) for n in names]
    # H-cond: GReaTer's answer loss only on rows whose incumbent greedy answer is correct;
    # with no such row the ranking is uniformly random. None marks a uniformly random
    # ranking, scored by its exact expectation (rho 0; regret = best-k mean - mean).
    right = [r for r in rows if base_fresh[r] == 1]
    for key, source in (("answer_cond_exact", "answer_exact"), ("answer_cond_patch", "answer_patch")):
        out[key] = [-statistics.mean(raw[source][n][r] for r in right) for n in names] if right else None
    out["random"] = None
    base_dev = raw["base_dev"]
    target = [statistics.mean(raw["dev"][n][q] - base_dev[q] for q in questions) for n in names]
    return out, target


REGRET_K = 5


def regret5(pred: list[float], target: list[float]) -> float:
    """Best-k regret (k = REGRET_K; H-cond preregisters k = 3 via --regret-k 3)."""
    k = min(REGRET_K, len(target))
    best = statistics.mean(sorted(target, reverse=True)[:k])
    top = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k]
    return best - statistics.mean(target[i] for i in top)


def random_regret(target: list[float]) -> float:
    """Expected best-k regret of a uniformly random k-subset of edits."""
    k = min(REGRET_K, len(target))
    return statistics.mean(sorted(target, reverse=True)[:k]) - statistics.mean(target)


def diagnostics(run: dict, rng: random.Random, splits: int = 500) -> dict[str, float]:
    raw, names = run["raw"], run["edits"]
    base_dev, n = raw["base_dev"], len(raw["base_dev"])
    halves = []
    for _ in range(splits):
        order = list(range(n))
        rng.shuffle(order)
        a, b = order[: n // 2], order[n // 2:]
        da = [statistics.mean(raw["dev"][e][q] - base_dev[q] for q in a) for e in names]
        db = [statistics.mean(raw["dev"][e][q] - base_dev[q] for q in b) for e in names]
        halves.append(spearman(da, db) or 0.0)
    half = statistics.mean(halves)
    full = 2 * half / (1 + half) if half > -1 else float("nan")
    target = [statistics.mean(raw["dev"][e][q] - base_dev[q] for q in range(n)) for e in names]
    out = {"dev_reliability": full, "rho_ceiling": max(full, 0.0) ** 0.5,
           "fork_rows": len(raw["fork_rows"]) / len(raw["base_fresh"])}
    for label, value in (("right", 1), ("wrong", 0)):
        rows = [r for r, b in enumerate(raw["base_fresh"]) if b == value]
        if len(rows) >= 3:
            pred = [-statistics.mean(raw["answer_exact"][e][r] for r in rows) for e in names]
            out[f"answer_exact_{label}_rows"] = spearman(pred, target) or 0.0
    return out


def run_name(run: dict) -> str:
    state = "/state" if run.get("state_edit") else ""
    kind = "/token" if run.get("edit_kind") == "token" else ""
    return f"{run['task']}/{Path(run['model_path']).parts[-3].split('--')[-1]}/s{run['seed']}{state}{kind}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    parser.add_argument("--regret-k", type=int, default=5)
    args = parser.parse_args()
    global REGRET_K
    REGRET_K = args.regret_k
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in args.files]
    runs = [r for r in runs if "raw" in r]
    if not runs:
        raise SystemExit("no records with raw per-row data (decision_study v3)")
    rng = random.Random(args.seed)
    point: dict[str, dict[str, float]] = {}
    point_runs: dict[str, dict[str, list[float]]] = {}
    boot: dict[str, dict[str, list[float]]] = {m: {k: [] for k in KEYS} for m in ("rho", "regret5")}
    for replicate in range(args.replicates + 1):
        per: dict[str, dict[str, list[float]]] = {m: {k: [] for k in KEYS} for m in ("rho", "regret5")}
        for run in runs:
            raw, names = run["raw"], run["edits"]
            n_rows, n_q = len(raw["base_fresh"]), len(raw["base_dev"])
            if replicate == 0:
                rows, questions = list(range(n_rows)), list(range(n_q))
            else:
                rows = [rng.randrange(n_rows) for _ in range(n_rows)]
                questions = [rng.randrange(n_q) for _ in range(n_q)]
            preds, target = predictors_for(raw, names, rows, questions)
            for k in KEYS:
                if preds[k] is None:  # uniformly random shortlist: exact expectation
                    per["rho"][k].append(0.0)
                    per["regret5"][k].append(random_regret(target))
                else:
                    per["rho"][k].append(spearman(preds[k], target) or 0.0)
                    per["regret5"][k].append(regret5(preds[k], target))
        means = {m: {k: statistics.mean(v) for k, v in d.items()} for m, d in per.items()}
        if replicate == 0:
            point, point_runs = means, per
            continue
        for m in boot:
            for k in KEYS:
                boot[m][k].append(means[m][k])

    normal = statistics.NormalDist()

    def interval(values: list[float], point_value: float) -> tuple[float, float]:
        """Bias-corrected (BC) percentile interval. Plain percentiles sat at or outside their own
        point estimates here: resampling held-out questions attenuates Spearman and inflates
        the best-k winner, so the bootstrap distribution is shifted."""
        ordered = sorted(values)
        below = sum(v < point_value for v in values) + 0.5 * sum(v == point_value for v in values)
        z0 = normal.inv_cdf(min(max(below / len(values), 1e-4), 1 - 1e-4))

        def pick(q: float) -> float:
            return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered))))]

        return pick(normal.cdf(2 * z0 - 1.96)), pick(normal.cdf(2 * z0 + 1.96))

    def pool_test(values: list[float]) -> dict[str, float]:
        """Pool-level inference over runs: sign counts and an exact sign-flip permutation p-value
        (two-sided) for the mean; runs are the units, so no question/row resampling enters."""
        n = len(values)
        observed = abs(statistics.mean(values))
        flips = 0
        for mask in range(2 ** n):
            total = sum(-v if (mask >> i) & 1 else v for i, v in enumerate(values))
            flips += abs(total / n) >= observed - 1e-12
        return {"mean": statistics.mean(values), "positive": sum(v > 0 for v in values),
                "negative": sum(v < 0 for v in values), "n": n, "p_signflip": flips / 2 ** n}

    # JSON keys keep the historical name "regret5"; the k actually used is recorded here.
    summary: dict = {"runs": [run_name(r) for r in runs], "regret_k": REGRET_K, "diagnostics": {},
                     "predictors": {}, "differences": {}, "pool_level": {}}
    for run in runs:
        info = diagnostics(run, random.Random(args.seed))
        summary["diagnostics"][run_name(run)] = info
        print(run_name(run), " ".join(f"{k}={v:+.2f}" for k, v in info.items()))
    print(f"runs: {len(runs)}")
    for k in KEYS:
        lo, hi = interval(boot["rho"][k], point["rho"][k])
        rlo, rhi = interval(boot["regret5"][k], point["regret5"][k])
        summary["pool_level"][k] = {"rho": pool_test(point_runs["rho"][k])}
        summary["predictors"][k] = {"rho": point["rho"][k], "rho_ci": [lo, hi],
                                    "regret5": point["regret5"][k], "regret5_ci": [rlo, rhi]}
        print(f"{k:18s} rho {point['rho'][k]:+.3f} [{lo:+.2f}, {hi:+.2f}]   "
              f"regret@{REGRET_K} {point['regret5'][k]:.3f} [{rlo:.3f}, {rhi:.3f}]   pools rho "
              f"+{summary['pool_level'][k]['rho']['positive']}/-{summary['pool_level'][k]['rho']['negative']} "
              f"p={summary['pool_level'][k]['rho']['p_signflip']:.3f}")
    for a, b in PAIRS:
        d_rho = [x - y for x, y in zip(boot["rho"][a], boot["rho"][b])]
        d_reg = [y - x for x, y in zip(boot["regret5"][a], boot["regret5"][b])]  # >0: a has lower regret
        lo, hi = interval(d_rho, point["rho"][a] - point["rho"][b])
        rlo, rhi = interval(d_reg, point["regret5"][b] - point["regret5"][a])
        pool_rho = pool_test([x - y for x, y in zip(point_runs["rho"][a], point_runs["rho"][b])])
        pool_reg = pool_test([y - x for x, y in zip(point_runs["regret5"][a], point_runs["regret5"][b])])
        p_rho = sum(x > 0 for x in d_rho) / len(d_rho)
        p_reg = sum(x > 0 for x in d_reg) / len(d_reg)
        summary["differences"][f"{a}-{b}"] = {
            "rho": point["rho"][a] - point["rho"][b], "rho_ci": [lo, hi], "p_rho_gt0": p_rho,
            "regret5_gain": point["regret5"][b] - point["regret5"][a], "regret5_gain_ci": [rlo, rhi],
            "p_regret5_gain_gt0": p_reg, "pool_rho": pool_rho, "pool_regret_gain": pool_reg}
        print(f"{a + ' - ' + b:34s} d_rho {point['rho'][a] - point['rho'][b]:+.3f} [{lo:+.2f}, {hi:+.2f}] "
              f"P>0 {p_rho:.3f} | regret gain {point['regret5'][b] - point['regret5'][a]:+.3f} "
              f"[{rlo:+.3f}, {rhi:+.3f}] P>0 {p_reg:.3f} | pools: rho +{pool_rho['positive']}/-{pool_rho['negative']} "
              f"p={pool_rho['p_signflip']:.3f}, regret +{pool_reg['positive']}/-{pool_reg['negative']} p={pool_reg['p_signflip']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
