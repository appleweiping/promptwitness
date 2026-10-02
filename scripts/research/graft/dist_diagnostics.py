"""Exploratory diagnostics of the distributional scores on one pool (no hypothesis test).

Per pool: incumbent sample accuracy, fraction of samples that hit the token budget (truncated),
accuracy of truncated vs ended samples, mean length of correct vs incorrect samples, the
correlation across samples between log w and trace length (pooled over edits), and per edit the
correlation between its mean log w and its held-out change.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from fidelity_study import spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="RECORD.json=IS_OUTPUT.json")
    args = parser.parse_args()
    for pair in args.pairs:
        rec_path, dist_path = pair.split("=")
        rec = json.loads(Path(rec_path).read_text(encoding="utf-8"))
        d = json.loads(Path(dist_path).read_text(encoding="utf-8"))
        names = d["edits"]
        lengths = [l for row in d["trace_lengths"] for l in row]
        ended = [e for row in d["trace_ended"] for e in row]
        correct = [c for row in d["hard"]["base"] for c in row]
        trunc = [c for c, e in zip(correct, ended) if not e]
        fin = [c for c, e in zip(correct, ended) if e]
        print(f"== {Path(dist_path).stem}")
        print(f"   incumbent sample accuracy {statistics.mean(correct):.3f}; truncated {1 - statistics.mean(ended):.3f} "
              f"(acc {statistics.mean(trunc) if trunc else float('nan'):.3f}) vs ended acc "
              f"{statistics.mean(fin) if fin else float('nan'):.3f}")
        cl = [l for l, c in zip(lengths, correct) if c]
        il = [l for l, c in zip(lengths, correct) if not c]
        print(f"   mean length correct {statistics.mean(cl) if cl else float('nan'):.0f} vs incorrect "
              f"{statistics.mean(il) if il else float('nan'):.0f}")
        target = [statistics.mean(dv - b for dv, b in zip(rec["raw"]["dev"][n], rec["raw"]["base_dev"])) for n in names]
        mean_logw, len_corr = [], []
        for n in names:
            lw = [a - b for ra, rb in zip(d["logp"][n], d["logp"]["base"]) for a, b in zip(ra, rb)]
            mean_logw.append(statistics.mean(lw))
            rho = spearman(lw, lengths)
            if rho is not None:
                len_corr.append(rho)
        print(f"   per-sample Spearman(log w, length): median over edits {statistics.median(len_corr):+.2f}")
        print(f"   Spearman(edit mean log w, held-out change) {spearman(mean_logw, target):+.2f}")
        # Which edits does the score favour? mean change in expected length under the SNIS weights
        print(f"   held-out changes: mean {100 * statistics.mean(target):+.1f}, best {100 * max(target):+.1f}, "
              f"worst {100 * min(target):+.1f} points; base dev acc {statistics.mean(rec['raw']['base_dev']):.3f}")


if __name__ == "__main__":
    main()
