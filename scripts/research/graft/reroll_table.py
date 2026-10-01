"""Summarize re-roll studies with the same-prompt null (deterministic reader).

For each reroll_study output: edit kind, mean fraction of questions whose greedy reasoning /
parsed answer / correctness changes under an edit, the same rates for re-reads of the unchanged
incumbent (null), the median first divergence, and the spread of net accuracy changes.
Optionally writes a LaTeX tabular.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--latex", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.files:
        d = json.loads(path.read_text(encoding="utf-8"))
        if not d.get("per_edit"):
            continue
        s, null = d["summary"], d["summary"].get("null", {})
        deltas = [v["accuracy_delta"] for v in d["per_edit"].values()]
        model = {"llama": "L", "qwen": "Q", "gemma": "G"}[next(k for k in ("llama", "qwen", "gemma")
                                                              if k in d["model_path"].lower())]
        task = {"logical_deduction_seven_objects": "LD7",
                "tracking_shuffled_objects_seven_objects": "TS7"}.get(d["task"], d["task"])
        rows.append((f"{task}, {model}", d["edit_kind"], s["reasoning_changed"], s["answer_changed"],
                     s["correctness_flipped"], s.get("first_divergence_median"), statistics.pstdev(deltas),
                     null.get("reasoning_changed"), null.get("correctness_flipped")))
        print(f"{rows[-1][0]:10s} {rows[-1][1]:6s} reasoning {s['reasoning_changed']:.3f} answer {s['answer_changed']:.3f} "
              f"correctness {s['correctness_flipped']:.3f} first-div {s.get('first_divergence_median')} "
              f"sd(acc) {statistics.pstdev(deltas):.3f} | null reasoning {null.get('reasoning_changed')} "
              f"correctness {null.get('correctness_flipped')}")
    if args.latex:
        out = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
               r"Pool & Edits & Reasoning & Answer & Correctness & First div. & sd(acc.) & Null reas./corr. \\",
               r"\midrule"]
        for name, kind, rc, ac, cf, fd, sd, nr, nc in rows:
            out.append(f"{name} & {kind} & {100 * rc:.0f}\\% & {100 * ac:.0f}\\% & {100 * cf:.1f}\\% & "
                       f"{fd if fd is not None else '--'} & {sd:.3f} & "
                       f"{100 * (nr or 0):.1f}\\%/{100 * (nc or 0):.1f}\\% \\\\")
        out += [r"\bottomrule", r"\end{tabular}"]
        args.latex.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
