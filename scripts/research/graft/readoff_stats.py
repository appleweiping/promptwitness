"""How large is the answer read-off term? (zero-GPU, from is_study outputs + decision records)

Decomposition (research/GRAFT_THEORY.md): J(P') - J(P) = D + A, with
A = E_x E_{r~pi_P}[c_P'(x, r) - c_P(x, r)] the change of the answer read on the incumbent's own
samples. A is estimated directly (on-policy, no weights) from the hard reads of the same samples
under the incumbent and under each candidate. It is compared with on-policy totals: the edit's
held-out greedy change (decision record) and the coupled fresh distributional change on the
training questions (fresh samples under each prompt). Also reported: read-off flip rate, the
estimated KL(pi_P || pi_P') = -mean log w per trace, and the within-question SD of log w.

(An earlier version compared |A| with a self-normalized importance estimate of D; that estimate
is at its permutation null on these pools, so it is not used.)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from fidelity_study import spearman


def label(stem: str) -> str:
    model = {"llama3": "L", "qwen3": "Q", "gemma2": "G"}[stem.split("-")[0]]
    task = stem.split("-")[1]
    short = {"logical_deduction_seven_objects": "LD7", "logical_deduction_three_objects": "LD3",
             "tracking_shuffled_objects_seven_objects": "TS7", "tracking_shuffled_objects_three_objects": "TS3"}
    star = "$^\\ast$" if stem.endswith("state") else ""  # outside the f-string (Python < 3.12)
    return f"{short.get(task, task)}, {model}{star}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="RECORD.json=IS_OUTPUT.json")
    parser.add_argument("--latex", type=Path, help="write a LaTeX tabular")
    args = parser.parse_args()
    rows_tex = []
    print(f"{'pool':50s} {'flip':>6s} {'|A|':>6s} {'|held|':>7s} {'|fresh|':>7s} {'rho(A,held)':>11s} "
          f"{'KL med':>7s} {'sd_row':>7s}")
    for pair in args.pairs:
        rec_path, dist_path = pair.split("=")
        rec = json.loads(Path(rec_path).read_text(encoding="utf-8"))
        d = json.loads(Path(dist_path).read_text(encoding="utf-8"))
        names = d["edits"]
        flips, a_terms, held, fresh, kls, sds = [], [], [], [], [], []
        for n in names:
            rows = range(len(d["hard"]["base"]))
            a_terms.append(statistics.mean(statistics.mean(u - v for u, v in zip(d["hard"][n][x], d["hard"]["base"][x]))
                                           for x in rows))
            flips += [int(u != v) for x in rows for u, v in zip(d["hard"][n][x], d["hard"]["base"][x])]
            held.append(statistics.mean(a - b for a, b in zip(rec["raw"]["dev"][n], rec["raw"]["base_dev"])))
            fresh.append(statistics.mean(statistics.mean(d["fresh"][n][x]) - statistics.mean(d["fresh"]["base"][x])
                                         for x in rows))
            lw = [[a - b for a, b in zip(d["logp"][n][x], d["logp"]["base"][x])] for x in rows]
            kls.append(-statistics.mean(v for row in lw for v in row))
            sds.append(statistics.mean(statistics.pstdev(row) for row in lw))
        stem = Path(dist_path).stem
        mean_abs = lambda xs: statistics.mean(abs(v) for v in xs)  # noqa: E731
        rho = spearman(a_terms, held)
        print(f"{stem:50s} {statistics.mean(flips):6.3f} {mean_abs(a_terms):6.3f} {mean_abs(held):7.3f} "
              f"{mean_abs(fresh):7.3f} {rho if rho is not None else float('nan'):+11.2f} "
              f"{statistics.median(kls):7.1f} {statistics.mean(sds):7.2f}")
        rows_tex.append((label(stem), statistics.mean(flips), mean_abs(a_terms), mean_abs(held), mean_abs(fresh),
                         rho, statistics.median(kls), statistics.mean(sds)))
    if args.latex:
        lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                 r"Pool & Read-off flips & $|A|$ & $|\Delta_{\text{held-out}}|$ & $|\Delta_{\text{fresh}}|$ & "
                 r"$\rho(A,\Delta_{\text{held-out}})$ & KL & sd$_x(\log w)$ \\", r"\midrule"]
        for name, f, a, h, fr, rho, kl, sd in rows_tex:
            lines.append(f"{name} & {100 * f:.1f}\\% & {100 * a:.1f} & {100 * h:.1f} & {100 * fr:.1f} & "
                         f"{'--' if rho is None else f'{rho:+.2f}'} & {kl:.1f} & {sd:.1f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        args.latex.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
