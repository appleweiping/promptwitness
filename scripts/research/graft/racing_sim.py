"""Adaptive verification at a fixed generation budget (exploratory, zero GPU).

Uses the validity records: for every pool, the greedy correctness of each of K LLM-proposed edits
and of the incumbent on 24 training rows (``raw.fresh``, ``raw.base_fresh``) and the edits'
held-out accuracy changes (``raw.dev`` vs ``raw.base_dev``). The incumbent's reads are shared by
all allocators and not counted. A verifier spends at most B candidate generations (one generation
= one candidate read on one row) and then accepts its best candidate if that candidate's paired
mean difference from the incumbent on the rows it was read on is positive. Score: realized
held-out change of the accepted edit (0 if none), averaged over pools and over random row orders.

Allocators at equal budget B:
  uniform     every candidate on the same floor(B/K) rows
  shortlist   3 random candidates on floor(B/3) rows (the Stage C pattern, random shortlist)
  halving     successive halving: all candidates on r0 rows, keep the better half by paired mean,
              add rows, repeat until one remains or the budget/rows run out (rows reused, so every
              survivor is compared on the same rows)
  racing      add one row at a time to all surviving candidates; drop a candidate once its paired
              mean is more than z * sqrt(flip rate / n) below the current leader's (normal
              approximation for paired 0/1 differences, flip rate estimated from the pool), stop at
              the budget or when one candidate is left
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


class Pool:
    def __init__(self, record: dict) -> None:
        raw = record["raw"]
        self.names = record["edits"]
        self.base = raw["base_fresh"]
        self.reads = [raw["fresh"][n] for n in self.names]
        self.target = [statistics.mean(a - b for a, b in zip(raw["dev"][n], raw["base_dev"])) for n in self.names]
        self.rows = len(self.base)
        diffs = [r - b for reads in self.reads for r, b in zip(reads, self.base)]
        self.flip = max(1e-3, statistics.mean(d != 0 for d in diffs))

    def paired(self, cand: int, rows: list[int]) -> float:
        return statistics.mean(self.reads[cand][x] - self.base[x] for x in rows)


def accept(pool: Pool, cand: int, rows: list[int]) -> float:
    return pool.target[cand] if rows and pool.paired(cand, rows) > 1e-12 else 0.0


def best_of(pool: Pool, cands: list[int], rows: list[int], rng: random.Random) -> int:
    scores = [pool.paired(c, rows) for c in cands]
    top = max(scores)
    return rng.choice([c for c, s in zip(cands, scores) if s == top])


def uniform(pool: Pool, budget: int, order: list[int], rng: random.Random) -> float:
    k = len(pool.names)
    r = min(pool.rows, budget // k)
    if r == 0:
        return 0.0
    rows = order[:r]
    return accept(pool, best_of(pool, list(range(k)), rows, rng), rows)


def shortlist(pool: Pool, budget: int, order: list[int], rng: random.Random) -> float:
    cands = rng.sample(range(len(pool.names)), 3)
    r = min(pool.rows, budget // 3)
    rows = order[:r]
    return accept(pool, best_of(pool, cands, rows, rng), rows) if r else 0.0


def halving(pool: Pool, budget: int, order: list[int], rng: random.Random) -> float:
    alive = list(range(len(pool.names)))
    rounds = max(1, math.ceil(math.log2(len(alive))))
    r = max(1, min(pool.rows, budget // (len(alive) * rounds)))
    spent, rows_used = 0, 0
    while True:
        need = len(alive) * (r - rows_used)
        if spent + need > budget or r > pool.rows:
            r = rows_used
            break
        spent += need
        rows_used = r
        if len(alive) == 1:
            break
        rows = order[:r]
        scores = sorted(((pool.paired(c, rows), rng.random(), c) for c in alive), reverse=True)
        alive = [c for _, _, c in scores[: max(1, len(alive) // 2)]]
        r = min(pool.rows, r * 2)
        if rows_used == pool.rows:
            break
    if rows_used == 0:
        return 0.0
    rows = order[:rows_used]
    return accept(pool, best_of(pool, alive, rows, rng), rows)


def racing(pool: Pool, budget: int, order: list[int], rng: random.Random, z: float = 1.0) -> float:
    alive = list(range(len(pool.names)))
    spent, n = 0, 0
    while n < pool.rows and spent + len(alive) <= budget and len(alive) > 1:
        spent += len(alive)
        n += 1
        rows = order[:n]
        scores = {c: pool.paired(c, rows) for c in alive}
        lead = max(scores.values())
        margin = z * math.sqrt(2 * pool.flip / n)
        alive = [c for c in alive if scores[c] >= lead - margin]
    if n == 0:
        return 0.0
    rows = order[:n]
    return accept(pool, best_of(pool, alive, rows, rng), rows)


def uniform_margin(pool: Pool, budget: int, order: list[int], rng: random.Random, c: float = 1.0) -> float:
    """Uniform allocation, but accept only if the paired gain exceeds c paired standard errors."""
    k = len(pool.names)
    r = min(pool.rows, budget // k)
    if r == 0:
        return 0.0
    rows = order[:r]
    cand = best_of(pool, list(range(k)), rows, rng)
    return pool.target[cand] if pool.paired(cand, rows) > c * math.sqrt(pool.flip / r) else 0.0


ALLOCATORS = {"uniform": uniform, "shortlist": shortlist, "halving": halving, "racing": racing,
              "unif+1se": uniform_margin,
              "unif+2se": lambda p, b, o, r: uniform_margin(p, b, o, r, 2.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--orders", type=int, default=400)
    parser.add_argument("--budgets", default="40,60,80,120,160,240,480")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    pools = [Pool(json.loads(p.read_text(encoding="utf-8"))) for p in args.records]
    budgets = [int(b) for b in args.budgets.split(",")]
    oracle = statistics.mean(max(p.target) for p in pools)
    print(f"{len(pools)} pools; oracle best edit {100 * oracle:+.2f} points; mean edit "
          f"{100 * statistics.mean(statistics.mean(p.target) for p in pools):+.2f}")
    result: dict = {"oracle": oracle, "cells": {}}
    print(f"{'budget':>6s} " + " ".join(f"{a:>10s}" for a in ALLOCATORS))
    for b in budgets:
        row = {}
        for name, fn in ALLOCATORS.items():
            per_pool = []
            for i, pool in enumerate(pools):
                rng = random.Random(f"{name}-{b}-{i}")
                gains = []
                for _ in range(args.orders):
                    order = rng.sample(range(pool.rows), pool.rows)
                    gains.append(fn(pool, b, order, rng))
                per_pool.append(statistics.mean(gains))
            row[name] = {"mean": statistics.mean(per_pool), "per_pool": per_pool}
        result["cells"][str(b)] = row
        print(f"{b:6d} " + " ".join(f"{100 * row[a]['mean']:+10.2f}" for a in ALLOCATORS))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
