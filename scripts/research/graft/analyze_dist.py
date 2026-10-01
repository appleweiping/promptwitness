"""H-dist analysis (plan amendment 2026-10-01): distributional predictors vs held-out change.

Inputs: pairs of (decision_study record, is_study output) for the same pool. For every edit
and training row x the is_study output holds S incumbent samples with exact log-likelihood
ratios log w_s, hard reads under the candidate, soft (gold-probability) reads, and F coupled
fresh samples per prompt. Predictors (per edit; mean over rows of the per-row value):

  snis_hard_b   sum_s w^b hard'_s / sum_s w^b - mean_s hard_s(incumbent)      b in {0,1/4,1/2,1}
  snis_soft_b   the same with soft reads (b = 0: fixed-reasoning answer objective over samples)
  primary       snis_hard at b* = largest of {1, 1/2, 1/4} whose median (over edits) of the
                median (over rows) effective sample size is >= S/4 (else 1/4); no targets used
  score_fn      Cov_s(hard'_s, log w_s) + mean_s(hard'_s - hard_s)   (first-order version)
  fresh_dist    coupled fresh samples: mean correctness under P' minus under P
  answer_exact, fresh8, fresh_all   from the decision record (GReaTer objective; greedy fresh)
  random        uniformly random ranking (exact expectation)

Target: change in greedy accuracy on the record's held-out questions. Per pool: Spearman and
best-3 regret. Inference: pool-level counts (primary) and a paired bootstrap that resamples
training rows and held-out questions jointly for all predictors (conditional on the pools),
with bias-corrected (BC) percentile intervals. Kill criterion as preregistered.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from fidelity_study import spearman

BETAS = (0.0, 0.25, 0.5, 1.0)
K = 3


def logsumexp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def snis(logw: list[float], values: list[float], beta: float) -> float:
    if beta == 0.0:
        return statistics.mean(values)
    scaled = [beta * v for v in logw]
    norm = logsumexp(scaled)
    return sum(math.exp(s - norm) * x for s, x in zip(scaled, values))


def ess(logw: list[float], beta: float) -> float:
    if beta == 0.0:
        return float(len(logw))
    a = [beta * v for v in logw]
    return math.exp(2 * logsumexp(a) - logsumexp([2 * v for v in a]))


def per_row_values(dist: dict, name: str, beta_star: float) -> dict[str, list[float]]:
    """Per training row, every distributional predictor for one edit."""
    out: dict[str, list[float]] = {f"snis_hard_{b}": [] for b in BETAS}
    out |= {f"snis_soft_{b}": [] for b in BETAS}
    out |= {"primary": [], "score_fn": [], "fresh_dist": []}
    for x in range(len(dist["logp"]["base"])):
        logw = [a - b for a, b in zip(dist["logp"][name][x], dist["logp"]["base"][x])]
        hard_c, hard_b = dist["hard"][name][x], dist["hard"]["base"][x]
        soft_c, soft_b = dist["soft"][name][x], dist["soft"]["base"][x]
        base_hard, base_soft = statistics.mean(hard_b), statistics.mean(soft_b)
        for b in BETAS:
            out[f"snis_hard_{b}"].append(snis(logw, hard_c, b) - base_hard)
            out[f"snis_soft_{b}"].append(snis(logw, soft_c, b) - base_soft)
        out["primary"].append(snis(logw, hard_c, beta_star) - base_hard)
        mean_w, mean_c = statistics.mean(logw), statistics.mean(hard_c)
        cov = statistics.mean((c - mean_c) * (w - mean_w) for c, w in zip(hard_c, logw))
        out["score_fn"].append(cov + statistics.mean(c - b for c, b in zip(hard_c, hard_b)))
        out["fresh_dist"].append(statistics.mean(dist["fresh"][name][x]) - statistics.mean(dist["fresh"]["base"][x]))
    return out


def choose_beta(dist: dict, samples: int) -> tuple[float, dict[str, float]]:
    medians = {}
    for b in (1.0, 0.5, 0.25):
        per_edit = []
        for name in dist["edits"]:
            rows = [ess([a - c for a, c in zip(dist["logp"][name][x], dist["logp"]["base"][x])], b)
                    for x in range(len(dist["logp"]["base"]))]
            per_edit.append(statistics.median(rows))
        medians[str(b)] = statistics.median(per_edit)
    for b in (1.0, 0.5, 0.25):
        if medians[str(b)] >= samples / 4:
            return b, medians
    return 0.25, medians


def regret(pred: list[float], target: list[float]) -> float:
    k = min(K, len(target))
    best = statistics.mean(sorted(target, reverse=True)[:k])
    top = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k]
    return best - statistics.mean(target[i] for i in top)


def random_regret(target: list[float]) -> float:
    k = min(K, len(target))
    return statistics.mean(sorted(target, reverse=True)[:k]) - statistics.mean(target)


class Pool:
    def __init__(self, record: dict, dist: dict) -> None:
        if record["edits"] != dist["edits"]:
            raise ValueError("record and is_study output list different edits")
        self.record, self.dist = record, dist
        self.names = record["edits"]
        self.beta_star, self.ess_medians = choose_beta(dist, dist["samples"])
        self.values = {n: per_row_values(dist, n, self.beta_star) for n in self.names}
        raw = record["raw"]
        self.raw = raw
        self.n_rows, self.n_q = len(raw["base_fresh"]), len(raw["base_dev"])
        logw = [a - b for n in self.names for x in range(self.n_rows)
                for a, b in zip(dist["logp"][n][x], dist["logp"]["base"][x])]
        self.logw_sd = statistics.pstdev(logw)

    def name(self) -> str:
        state = "/state" if self.record.get("state_edit") else ""
        model = Path(self.record["model_path"]).parts[-3].split("--")[-1]
        return f"{self.record['task']}/{model}/s{self.record['seed']}{state}"

    def predictors(self, rows: list[int], questions: list[int]) -> tuple[dict, list[float]]:
        raw, names = self.raw, self.names
        out: dict[str, list[float] | None] = {}
        keys = list(next(iter(self.values.values())).keys())
        for key in keys:
            out[key] = [statistics.mean(self.values[n][key][x] for x in rows) for n in names]
        out["answer_exact"] = [-statistics.mean(raw["answer_exact"][n][x] for x in rows) for n in names]
        first8 = [x for x in rows if x < 8] or rows[:1]
        out["fresh8"] = [statistics.mean(raw["fresh"][n][x] - raw["base_fresh"][x] for x in first8) for n in names]
        out["fresh_all"] = [statistics.mean(raw["fresh"][n][x] - raw["base_fresh"][x] for x in rows) for n in names]
        out["random"] = None
        target = [statistics.mean(raw["dev"][n][q] - raw["base_dev"][q] for q in questions) for n in names]
        return out, target


def bc_interval(values: list[float], point: float) -> tuple[float, float]:
    """Bias-corrected percentile interval (95%)."""
    normal = statistics.NormalDist()
    below = sum(v < point for v in values) + 0.5 * sum(v == point for v in values)
    frac = min(max(below / len(values), 1e-4), 1 - 1e-4)
    z0 = normal.inv_cdf(frac)
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered))))]

    return pick(normal.cdf(2 * z0 - 1.96)), pick(normal.cdf(2 * z0 + 1.96))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="RECORD.json=IS_OUTPUT.json")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    pools = []
    for pair in args.pairs:
        rec, dist = pair.split("=")
        pools.append(Pool(json.loads(Path(rec).read_text(encoding="utf-8")),
                          json.loads(Path(dist).read_text(encoding="utf-8"))))
    rng = random.Random(args.seed)
    keys: list[str] | None = None
    point: dict[str, dict[str, list[float]]] = {}
    boot: dict[str, dict[str, list[float]]] = {}
    for replicate in range(args.replicates + 1):
        per: dict[str, dict[str, list[float]]] = {"rho": {}, "regret": {}}
        for pool in pools:
            if replicate == 0:
                rows, questions = list(range(pool.n_rows)), list(range(pool.n_q))
            else:
                rows = [rng.randrange(pool.n_rows) for _ in range(pool.n_rows)]
                questions = [rng.randrange(pool.n_q) for _ in range(pool.n_q)]
            preds, target = pool.predictors(rows, questions)
            keys = keys or list(preds)
            for k in keys:
                rho = 0.0 if preds[k] is None else (spearman(preds[k], target) or 0.0)
                reg = random_regret(target) if preds[k] is None else regret(preds[k], target)
                per["rho"].setdefault(k, []).append(rho)
                per["regret"].setdefault(k, []).append(reg)
        if replicate == 0:
            point = per
            continue
        for m in per:
            for k, v in per[m].items():
                boot.setdefault(m, {}).setdefault(k, []).append(statistics.mean(v))
    assert keys is not None
    summary: dict = {"pools": [], "predictors": {}, "comparisons": {}}
    for i, pool in enumerate(pools):
        info = {"pool": pool.name(), "beta_star": pool.beta_star, "ess_medians": pool.ess_medians,
                "logw_sd": pool.logw_sd,
                "rho": {k: point["rho"][k][i] for k in keys}, "regret": {k: point["regret"][k][i] for k in keys}}
        summary["pools"].append(info)
        print(f"{pool.name():50s} b*={pool.beta_star:<4} ESS(b=1,.5,.25)="
              + "/".join(f"{pool.ess_medians[b]:.1f}" for b in ("1.0", "0.5", "0.25"))
              + f" sd(logw)={pool.logw_sd:.2f}  rho primary {point['rho']['primary'][i]:+.2f}"
              f" fresh8 {point['rho']['fresh8'][i]:+.2f} answer {point['rho']['answer_exact'][i]:+.2f}")
    for k in keys:
        rho, reg = statistics.mean(point["rho"][k]), statistics.mean(point["regret"][k])
        summary["predictors"][k] = {"rho": rho, "rho_ci_bc": bc_interval(boot["rho"][k], rho),
                                    "regret": reg, "regret_ci_bc": bc_interval(boot["regret"][k], reg)}
        print(f"{k:16s} rho {rho:+.3f} {summary['predictors'][k]['rho_ci_bc']}  regret3 {reg:.3f}")
    for a, b in (("primary", "random"), ("primary", "fresh8"), ("primary", "answer_exact"),
                 ("primary", "fresh_all"), ("fresh_dist", "fresh8"), ("score_fn", "random"),
                 ("snis_soft_0.0", "answer_exact"), ("fresh_dist", "random"), ("fresh8", "random")):
        gain = [y - x for x, y in zip(boot["regret"][a], boot["regret"][b])]
        d_rho = [x - y for x, y in zip(boot["rho"][a], boot["rho"][b])]
        pools_better = sum(pb > pa for pa, pb in zip(point["regret"][a], point["regret"][b]))
        g0 = statistics.mean(point["regret"][b]) - statistics.mean(point["regret"][a])
        r0 = statistics.mean(point["rho"][a]) - statistics.mean(point["rho"][b])
        summary["comparisons"][f"{a}-{b}"] = {
            "regret_gain": g0, "regret_gain_ci_bc": bc_interval(gain, g0), "p_gain_gt0": sum(g > 0 for g in gain) / len(gain),
            "pools_with_lower_regret": pools_better, "pools": len(pools),
            "d_rho": r0, "d_rho_ci_bc": bc_interval(d_rho, r0), "p_drho_gt0": sum(d > 0 for d in d_rho) / len(d_rho)}
        c = summary["comparisons"][f"{a}-{b}"]
        print(f"{a + ' - ' + b:30s} regret gain {g0:+.3f} {c['regret_gain_ci_bc']} P>0 {c['p_gain_gt0']:.3f} "
              f"pools {pools_better}/{len(pools)} | d_rho {r0:+.3f} P>0 {c['p_drho_gt0']:.3f}")
    vs_random = summary["comparisons"]["primary-random"]
    vs_fresh8 = summary["comparisons"]["primary-fresh8"]
    supported = (vs_random["p_gain_gt0"] >= 0.9 and vs_random["pools_with_lower_regret"] >= math.ceil(7 * len(pools) / 11)
                 and vs_fresh8["d_rho"] >= -0.05)
    summary["h_dist_supported"] = supported
    print("H-dist (preregistered kill criterion):", "SUPPORTED" if supported else "NOT SUPPORTED")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
