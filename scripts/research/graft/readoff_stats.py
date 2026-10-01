"""How much of an edit's effect can the answer read-off carry? (zero-GPU, from is_study outputs)

For each pool: the fraction of (candidate, row, sample) triples whose hard read of the same
incumbent trace changes between the incumbent and the candidate prompt (read-off flips), its
signed mean (the read-off term A of the decomposition, estimated on incumbent samples), the
reasoning-distribution term D = mean_x [SNIS_1(w, c') - mean c'] and the spread of log w.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def lse(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", nargs="+", type=Path)
    args = parser.parse_args()
    print(f"{'pool':55s} {'flip':>6s} {'A':>7s} {'|A|':>6s} {'D':>7s} {'|D|':>6s} {'sd logw':>8s}")
    for path in args.dist:
        d = json.loads(path.read_text(encoding="utf-8"))
        flips, a_terms, d_terms, logws = [], [], [], []
        for name in d["edits"]:
            a_rows, d_rows = [], []
            for x in range(len(d["hard"]["base"])):
                hc, hb = d["hard"][name][x], d["hard"]["base"][x]
                flips += [int(u != v) for u, v in zip(hc, hb)]
                a_rows.append(statistics.mean(u - v for u, v in zip(hc, hb)))
                lw = [u - v for u, v in zip(d["logp"][name][x], d["logp"]["base"][x])]
                logws += lw
                norm = lse(lw)
                snis = sum(math.exp(w - norm) * c for w, c in zip(lw, hc))
                d_rows.append(snis - statistics.mean(hc))
            a_terms.append(statistics.mean(a_rows))
            d_terms.append(statistics.mean(d_rows))
        print(f"{path.stem:55s} {statistics.mean(flips):6.3f} {statistics.mean(a_terms):+7.3f} "
              f"{statistics.mean(abs(v) for v in a_terms):6.3f} {statistics.mean(d_terms):+7.3f} "
              f"{statistics.mean(abs(v) for v in d_terms):6.3f} {statistics.pstdev(logws):8.2f}")


if __name__ == "__main__":
    main()
