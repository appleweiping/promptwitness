"""Numerical re-read noise and acceptance regression inside structural search (exploratory).

With 50-row verification (``--batch 50``) every round reads the incumbent on the same training
rows, but in a request shared with that round's candidates, so batch composition differs between
rounds. From the run records (``trajectory``):

  re-read change   incumbent accuracy at round t+1 minus at round t when nothing was accepted at
                   t (same prompt, same rows: pure numerical re-roll inside the search loop)
  acceptance gap   accepted candidate's verification accuracy at round t minus the incumbent
                   accuracy at round t+1 (same prompt, same rows; selection plus re-roll)

A positive mean acceptance gap is a winner's curse: the accepted candidate's measured accuracy
was partly luck. Runs with minibatch verification (fewer than 50 rows) re-sample rows each round
and are reported separately (their changes mix row sampling with re-rolls).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def run_stats(record: dict) -> tuple[list[float], list[float]]:
    traj = record["trajectory"]
    rereads, gaps = [], []
    for now, nxt in zip(traj, traj[1:]):
        if now.get("accepted") is None:
            rereads.append(nxt["incumbent_acc"] - now["incumbent_acc"])
            continue
        edit = list(now["accepted"])
        chosen = [c for c in now["checks"] if list(c["edit"]) == edit]
        if chosen:
            gaps.append(chosen[0]["accuracy"] - nxt["incumbent_acc"])
    return rereads, gaps


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.mean(values),
            "sd": statistics.pstdev(values), "mean_abs": statistics.mean(abs(v) for v in values),
            "nonzero": sum(v != 0 for v in values) / len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path, help="Stage C run records (.json)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"reread": [], "gap": []})
    for path in args.runs:
        record = json.loads(path.read_text(encoding="utf-8"))
        if "trajectory" not in record:
            continue
        rows = record["config"].get("batch")
        key = f"{record['method']}-rows{rows}"
        rereads, gaps = run_stats(record)
        by_method[key]["reread"] += rereads
        by_method[key]["gap"] += gaps
    out = {}
    for key, v in sorted(by_method.items()):
        out[key] = {"reread_change": summarize(v["reread"]), "acceptance_gap": summarize(v["gap"])}
        r, g = out[key]["reread_change"], out[key]["acceptance_gap"]
        print(f"{key:22s} re-read: n {r['n']:3d}" + (f" mean {100 * r['mean']:+.1f} sd {100 * r['sd']:.1f} "
              f"nonzero {r['nonzero']:.2f}" if r["n"] else "")
              + f" | acceptance gap: n {g['n']:3d}" + (f" mean {100 * g['mean']:+.1f} sd {100 * g['sd']:.1f}" if g["n"] else ""))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
