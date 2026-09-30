"""LaTeX table for the validity (decision) study: rho per run, ceiling, pooled rho, cost.

Inputs: decision-study records (v3) and the JSON summary written by
``analyze_decision.py --json``. Every number in the table is read from these files.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from fidelity_study import spearman

ROWS = (("answer_exact", "Answer loss, exact (\\greater{})"), ("answer_patch", "Answer loss, patching"),
        ("fork_exact", "Fork margin, exact"), ("fork_patch", "Fork margin, patching"),
        ("fresh8", "Fresh accuracy, 8 rows"), ("fresh_all", "Fresh accuracy, 24 rows"))
MODELS = {"Meta-Llama-3-8B-Instruct": "L", "gemma-2-9b-it": "G", "Qwen3-8B": "Q"}
TASKS = {"logical_deduction_seven_objects": "LD7", "logical_deduction_three_objects": "LD3",
         "tracking_shuffled_objects_seven_objects": "TS7", "tracking_shuffled_objects_three_objects": "TS3"}
COST = {"answer_exact": ("answer_exact",), "answer_patch": ("answer_patch",),
        "fork_exact": ("reasoning_material", "fork_exact"), "fork_patch": ("reasoning_material", "fork_patch"),
        "fresh8": ("fresh8",), "fresh_all": ("fresh_all",)}


def label(run: dict) -> str:
    model = MODELS.get(Path(run["model_path"]).parts[-3].split("--")[-1], "?")
    state = ("$^\\ast$" if run.get("state_edit") else "") + ("$^t$" if run.get("edit_kind") == "token" else "")
    return f"{TASKS.get(run['task'], run['task'])}-{model}{state}"


def run_rho(run: dict, key: str) -> float:
    raw, names = run["raw"], run["edits"]
    base_dev = raw["base_dev"]
    target = [statistics.mean(raw["dev"][n][q] - base_dev[q] for q in range(len(base_dev))) for n in names]
    fork_index = {row: k for k, row in enumerate(raw["fork_rows"])}
    rows = list(range(len(raw["base_fresh"])))
    if key.startswith("answer"):
        pred = [-statistics.mean(raw[key][n]) for n in names]
    elif key.startswith("fork"):
        pred = [-statistics.mean(raw[key][n][fork_index[r]] for r in rows if r in fork_index) for n in names]
    else:
        subset = rows[:8] if key == "fresh8" else rows
        pred = [statistics.mean(raw["fresh"][n][r] - raw["base_fresh"][r] for r in subset) for n in names]
    return spearman(pred, target) or 0.0


def transposed(runs: list, summary: dict) -> str:
    short = ("Answer\\\\exact", "Answer\\\\patch", "Fork\\\\exact", "Fork\\\\patch", "Fresh\\\\8 rows", "Fresh\\\\24 rows")
    lines = ["\\begin{tabular}{l" + "r" * len(ROWS) + "r}", "\\toprule",
             "Run & " + " & ".join("\\makecell{" + s + "}" for s in short) + " & $\\sqrt{\\kappa}$ \\\\",
             "\\midrule"]
    for run in runs:
        state = ("/state" if run.get("state_edit") else "") + ("/token" if run.get("edit_kind") == "token" else "")
        key = f"{run['task']}/{Path(run['model_path']).parts[-3].split('--')[-1]}/s{run['seed']}{state}"
        ceiling = summary["diagnostics"][key]["rho_ceiling"]
        cells = [f"{run_rho(run, k):+.2f}" for k, _ in ROWS]
        lines.append(f"{label(run)} & " + " & ".join(cells) + f" & {ceiling:.2f} \\\\")
    lines.append("\\midrule")
    pooled = [summary["predictors"][k] for k, _ in ROWS]
    lines.append("Pooled & " + " & ".join(f"{p['rho']:+.2f}" for p in pooled) + " & \\\\")
    lines.append("95\\% int. & " + " & ".join(f"\\scriptsize[{p['rho_ci'][0]:+.2f},{p['rho_ci'][1]:+.2f}]"
                                            for p in pooled) + " & \\\\")
    cost = [statistics.mean(sum(r["summary"]["cost_seconds"].get(c, 0.0) for c in COST[k]) for r in runs)
            for k, _ in ROWS]
    lines.append("Cost (s) & " + " & ".join(f"{c:.0f}" for c in cost) + " & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--transpose", action="store_true", help="runs as rows, predictors as columns")
    args = parser.parse_args()
    runs = [r for r in (json.loads(f.read_text(encoding="utf-8")) for f in args.files) if "raw" in r]
    runs.sort(key=lambda r: (r["task"], r["model_path"], bool(r.get("state_edit"))))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if args.transpose:
        print(transposed(runs, summary))
        return
    cols = "l" + "r" * len(runs) + "rr"
    lines = ["\\begin{tabular}{" + cols + "}", "\\toprule",
             "Predictor & " + " & ".join(label(r) for r in runs) + " & Pooled & Cost (s) \\\\", "\\midrule"]
    for key, name in ROWS:
        cells = []
        for run in runs:
            raw, names = run["raw"], run["edits"]
            base_dev = raw["base_dev"]
            target = [statistics.mean(raw["dev"][n][q] - base_dev[q] for q in range(len(base_dev))) for n in names]
            fork_index = {row: k for k, row in enumerate(raw["fork_rows"])}
            rows = list(range(len(raw["base_fresh"])))
            if key.startswith("answer"):
                pred = [-statistics.mean(raw[key][n]) for n in names]
            elif key.startswith("fork"):
                pred = [-statistics.mean(raw[key][n][fork_index[r]] for r in rows if r in fork_index) for n in names]
            else:
                subset = rows[:8] if key == "fresh8" else rows
                pred = [statistics.mean(raw["fresh"][n][r] - raw["base_fresh"][r] for r in subset) for n in names]
            cells.append(f"{(spearman(pred, target) or 0.0):+.2f}")
        pooled = summary["predictors"][key]
        cost = statistics.mean(sum(r["summary"]["cost_seconds"].get(c, 0.0) for c in COST[key]) for r in runs)
        lines.append(f"{name} & " + " & ".join(cells) +
                     f" & {pooled['rho']:+.2f} [{pooled['rho_ci'][0]:+.2f}, {pooled['rho_ci'][1]:+.2f}] & {cost:.0f} \\\\")
    lines.append("\\midrule")
    ceil = [summary["diagnostics"][name]["rho_ceiling"] for name in summary["runs"]]
    order = {name: i for i, name in enumerate(summary["runs"])}
    by_run = []
    for run in runs:
        state = ("/state" if run.get("state_edit") else "") + ("/token" if run.get("edit_kind") == "token" else "")
        key =f"{run['task']}/{Path(run['model_path']).parts[-3].split('--')[-1]}/s{run['seed']}{state}"
        by_run.append(f"{ceil[order[key]]:.2f}")
    lines.append("Ceiling $\\sqrt{\\kappa}$ & " + " & ".join(by_run) + " & & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
