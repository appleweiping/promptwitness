"""Verification at matched cost: greedy vs coupled sampled vs uncoupled sampled reasoning.

Exploratory (added 2026-10-02 after the registered H-dist analysis, which found the coupled
sampled evaluation `fresh_dist` to be the best predictor but at four times the generations of
greedy verification on the same rows). For each validity pool (decision record + is_study output)
and a budget of G generations per candidate:

  greedy-r      greedy correctness on r training rows (G = r)
  coupled-r-k   k sampled reasonings per row, tau 0.7, the same seed per (row, sample) for the
                candidate and the incumbent (common random numbers; G = r k)
  indep-r-k     as coupled, but the incumbent's samples use disjoint seeds (k <= F/2), so the two
                reads are independent; isolates the effect of coupling

Predictor of an edit: mean over rows of (candidate correctness - incumbent correctness). Target:
change of greedy accuracy on the pool's held-out questions. Each verifier is evaluated over
`--subsets` random draws of r rows (and of the k samples), so a cell is the expected quality at
that budget: Spearman with the target, best-3 regret, and the realized held-out change of a
verify-all step that accepts the best edit if its predicted change is positive (ties broken
uniformly). Pools are the units: pooled means and an exact sign-flip test over pools for
coupled vs greedy at equal G.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from pathlib import Path

from fidelity_study import spearman

K = 3


def regret(pred: list[float], target: list[float]) -> float:
    """Best-3 regret with ties in the prediction broken uniformly (expected value)."""
    k = min(K, len(target))
    best = statistics.mean(sorted(target, reverse=True)[:k])
    order = sorted(set(pred), reverse=True)
    picked, need = 0.0, k
    for value in order:
        group = [t for p, t in zip(pred, target) if p == value]
        take = min(need, len(group))
        picked += take * statistics.mean(group)
        need -= take
        if need == 0:
            break
    return best - picked / k


def verify_all(pred: list[float], target: list[float]) -> float:
    top = max(pred)
    if top <= 1e-12:
        return 0.0
    return statistics.mean(t for p, t in zip(pred, target) if p == top)


class Pool:
    def __init__(self, record: dict, dist: dict) -> None:
        assert record["edits"] == dist["edits"]
        self.names = record["edits"]
        raw = record["raw"]
        self.greedy = {n: raw["fresh"][n] for n in self.names}
        self.greedy_base = raw["base_fresh"]
        self.fresh = {n: dist["fresh"][n] for n in self.names}
        self.fresh_base = dist["fresh"]["base"]
        self.rows = len(self.greedy_base)
        self.samples = len(self.fresh_base[0])
        self.target = [statistics.mean(a - b for a, b in zip(raw["dev"][n], raw["base_dev"])) for n in self.names]
        state = "/state" if record.get("state_edit") else ""
        model = Path(record["model_path"]).parts[-3].split("--")[-1]
        self.label = f"{record['task']}/{model}{state}"

    def predict(self, kind: str, rows: list[int], cand: list[int], base: list[int]) -> list[float]:
        if kind == "greedy":
            return [statistics.mean(self.greedy[n][x] - self.greedy_base[x] for x in rows) for n in self.names]
        return [statistics.mean(statistics.mean(self.fresh[n][x][j] for j in cand)
                                - statistics.mean(self.fresh_base[x][j] for j in base) for x in rows)
                for n in self.names]


def cells(rows: int, samples: int) -> list[tuple[str, int, int]]:
    out = [("greedy", r, 1) for r in (4, 8, 12, 16, 24) if r <= rows]
    for k in (1, 2, 4):
        if k > samples:
            continue
        for r in (1, 2, 3, 4, 6, 8, 12, 16, 24):
            if r <= rows and r * k in (4, 8, 12, 16, 24, 48, 96):
                out.append(("coupled", r, k))
                if 2 * k <= samples:
                    out.append(("indep", r, k))
    return out


def evaluate(pool: Pool, kind: str, r: int, k: int, subsets: int, rng: random.Random) -> dict[str, float]:
    rhos, regrets, gains = [], [], []
    for _ in range(subsets):
        rows = rng.sample(range(pool.rows), r)
        perm = rng.sample(range(pool.samples), pool.samples)
        cand = perm[:k]
        base = cand if kind == "coupled" else perm[k:2 * k]
        pred = pool.predict(kind, rows, cand, base)
        rho = spearman(pred, pool.target)
        rhos.append(0.0 if rho is None else rho)
        regrets.append(regret(pred, pool.target))
        gains.append(verify_all(pred, pool.target))
    return {"rho": statistics.mean(rhos), "regret": statistics.mean(regrets), "verify_all": statistics.mean(gains)}


def sign_flip(diffs: list[float]) -> float:
    """Exact two-sided sign-flip p-value for the mean of per-pool differences."""
    observed = abs(statistics.mean(diffs))
    count = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        count += abs(statistics.mean(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
    return count / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="RECORD.json=IS_OUTPUT.json")
    parser.add_argument("--subsets", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    pools = []
    for pair in args.pairs:
        rec, dist = pair.split("=")
        pools.append(Pool(json.loads(Path(rec).read_text(encoding="utf-8")),
                          json.loads(Path(dist).read_text(encoding="utf-8"))))
    rows, samples = min(p.rows for p in pools), min(p.samples for p in pools)
    grid = cells(rows, samples)
    result: dict = {"pools": [p.label for p in pools], "subsets": args.subsets, "cells": {}}
    for kind, r, k in grid:
        key = f"{kind}-{r}x{k}"
        per = [evaluate(p, kind, r, k, args.subsets, random.Random(f"{args.seed}-{key}-{i}"))
               for i, p in enumerate(pools)]
        result["cells"][key] = {"kind": kind, "rows": r, "samples": k, "generations": r * k,
                                "per_pool": per,
                                **{m: statistics.mean(c[m] for c in per) for m in ("rho", "regret", "verify_all")}}
    random_regret = statistics.mean(
        statistics.mean(sorted(p.target, reverse=True)[:K]) - statistics.mean(p.target) for p in pools)
    result["random_regret"] = random_regret
    print(f"{len(pools)} pools, {args.subsets} subsets per cell; random best-3 regret {random_regret:.3f}")
    print(f"{'verifier':16s} {'G':>3s} {'rho':>7s} {'regret':>7s} {'verify-all':>10s}")
    for key, c in sorted(result["cells"].items(), key=lambda kv: (kv[1]["generations"], kv[1]["kind"])):
        print(f"{key:16s} {c['generations']:3d} {c['rho']:+7.3f} {c['regret']:7.3f} {100 * c['verify_all']:+9.2f}")
    # Coupled vs greedy (and coupled vs independent) at equal generations: per-pool differences.
    result["comparisons"] = {}
    for a, b in (("coupled", "greedy"), ("coupled", "indep")):
        for g in (4, 8, 12, 16, 24):
            ca = [c for c in result["cells"].values() if c["kind"] == a and c["generations"] == g]
            cb = [c for c in result["cells"].values() if c["kind"] == b and c["generations"] == g]
            for x, y in itertools.product(ca, cb):
                if b == "indep" and (x["rows"], x["samples"]) != (y["rows"], y["samples"]):
                    continue
                name = f"{a}-{x['rows']}x{x['samples']} vs {b}-{y['rows']}x{y['samples']}"
                comp = {}
                for m in ("rho", "regret", "verify_all"):
                    sign = -1 if m == "regret" else 1  # positive = a better
                    diffs = [sign * (pa[m] - pb[m]) for pa, pb in zip(x["per_pool"], y["per_pool"])]
                    comp[m] = {"mean": statistics.mean(diffs), "pools_a_better": sum(d > 0 for d in diffs),
                               "p_signflip": sign_flip(diffs)}
                result["comparisons"][name] = comp
                print(f"{name:34s} " + "  ".join(
                    f"{m} {comp[m]['mean']:+.3f} ({comp[m]['pools_a_better']}/{len(pools)}, p={comp[m]['p_signflip']:.3f})"
                    for m in ("rho", "regret", "verify_all")))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
