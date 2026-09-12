"""Pinned full Countdown/path input-only baselines; no model/provider execution.

Only aggregate measurements and hashes are published. Run from a checkout with
the matching package installed/editable, using an existing LongProc data root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

DATA_SHA256 = {
    "countdown/countdown_0.5k.json": (
        "2f0d1095158bcd87e955970cdabbff217dd1493b3e3d4a6439ffe631008965b8"
    ),
    "countdown/countdown_2k.json": (
        "5dbc0320e3fc55ba5627b26a90f8c64a6f0bc44e350b065c6c5da620f9675a43"
    ),
    "countdown/countdown_8k.json": (
        "d026d94ffdc3669396502c8586b812c2f9944365ca538c7ad5dfaad72613938f"
    ),
    "path_traversal/path_traversal_0.5k.json": (
        "8abd1e2f437b8799d697f33cddbbfdbb1eb80ebe9acbb4736bd5de5d8a28fcb7"
    ),
    "path_traversal/path_traversal_2k.json": (
        "100ca7c5215f275fd2d8686e95616af2a2edcfaa6e713f262b6f90f086322d61"
    ),
    "path_traversal/path_traversal_8k.json": (
        "39ae8ec2adc0cc3ab1e3cdd884a4c6783c88ed5b0cb0312382368ca55c39b0a4"
    ),
    "countdown/prompts.yaml": "ac6193f0168e5081343dc1fc8a79078281ddc33b27e7c5729826db1f6aefc396",
    "path_traversal/prompts.yaml": (
        "8625b1fe0c77b667b20bb2533b4116192f47aafb9069b0aa2295adb46cc7d685"
    ),
}
DATASETS = tuple(
    f"{family}_{bucket}"
    for family in ("countdown", "path_traversal")
    for bucket in ("0.5k", "2k", "8k")
)
# Prespecified before any full-dataset baseline run. No CLI tuning/sampling knobs.
LIMITS = {
    "max_states": 50_000,
    "max_transitions": 250_000,
    "max_input_bytes": 8 * 1024 * 1024,
    "max_edges": 100_000,
    "max_prediction_bytes": 1024 * 1024,
}


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    before = path.stat()
    if before.st_size > 64 * 1024 * 1024:
        raise ValueError("source exceeds benchmark file byte budget")
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            if size > 64 * 1024 * 1024:
                raise ValueError("source exceeds benchmark file byte budget")
            result.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed while hashing")
    return result.hexdigest()


def source_inventory(root: Path) -> dict[str, str]:
    files = list((root / "src/promptwitness").glob("*.py"))
    files += [root / "pyproject.toml", root / "benchmarks/benchmark_procedures.py"]
    return {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(files)}


def negative_controls(family: str, inputs: Any, result: Any) -> dict[str, str]:
    """Alter a produced solution without consulting any reference field."""
    tag = "Solution" if family == "countdown" else "Route"
    full = result.prediction or f"<{tag}>\n</{tag}>"
    if family == "countdown" and result.steps:
        lines = list(result.steps)
        equation, value = lines[-1].rsplit(" = ", 1)
        lines[-1] = f"{equation} = {int(value) + 1}"
        altered = "<Solution>\n" + "\n".join(lines) + "\n</Solution>"
    elif family == "path_traversal" and result.steps:
        outgoing = {edge["source"]: edge for edge in inputs["edges"]}
        current = inputs["source"]
        for _ in range(len(result.steps) - 1):
            current = outgoing[current]["target"]
        edge = outgoing[current]
        missing = "__authored_control_missing_destination__"
        cities = {name for row in inputs["edges"] for name in (row["source"], row["target"])}
        while missing in cities:
            missing += "_"
        lines = [*result.steps[:-1], f"From {current}, take a {edge['method']} to {missing}."]
        altered = "<Route>\n" + "\n".join(lines) + "\n</Route>"
    else:
        # Explicitly malformed fallback, rather than fabricating a successful answer.
        altered = f"<{tag}>\n[no complete baseline solution to alter]\n</{tag}>"
    return {"empty": "", "truncated": full.rsplit(f"</{tag}>", 1)[0], "altered": altered}


def evaluate_cases(family: str, cases: Any, limits: Any) -> dict[str, Any]:
    """Aggregate every case, including unresolved algorithms and failed scorings."""
    from promptwitness.procedure_baselines import solve_countdown, walk_graph_path
    from promptwitness.procedure_scores import score_procedure

    solver = solve_countdown if family == "countdown" else walk_graph_path
    if family not in ("countdown", "path_traversal"):
        raise ValueError("benchmark family is unsupported")
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    controls = {
        name: dict(attempted=0, scored=0, accepted=0, failed=0, unsupported=0)
        for name in ("empty", "truncated", "altered")
    }
    selected = scored = valid = reference_valid = reference_invalid = reference_unknown = 0
    score_unsupported = 0
    solved_invalid = state_total = transition_total = max_states = max_transitions = 0
    solver_seconds = score_seconds = 0.0
    inputs_hash, results_hash = hashlib.sha256(), hashlib.sha256()
    for case in cases:
        selected += 1
        result = None
        before = time.perf_counter()
        try:
            result = solver(
                case.input, limits=limits
            )  # The only data crossing the solver boundary.
        except (ValueError, TypeError, ArithmeticError) as exc:
            statuses["failed"] += 1
            error_types[type(exc).__name__] += 1
        finally:
            solver_seconds += time.perf_counter() - before
        if result is not None:
            statuses[result.status] += 1
            if result.reason is not None:
                reasons[result.reason] += 1
            inputs_hash.update(bytes.fromhex(result.input_sha256))
            results_hash.update(hashlib.sha256(encoded(result.to_dict())).digest())
            state_total += result.states
            transition_total += result.transitions
            max_states, max_transitions = (
                max(max_states, result.states),
                max(max_transitions, result.transitions),
            )
        else:
            # Retain an input identity and a failure marker even when the solver rejects.
            from promptwitness.procedure_json_data import canonical_bytes

            inputs_hash.update(hashlib.sha256(canonical_bytes(case.input)).digest())
            results_hash.update(hashlib.sha256(b"baseline-failed").digest())
        before = time.perf_counter()
        try:
            score = score_procedure(case, result.prediction if result and result.prediction else "")
            if score["primary_score"] is None:
                score_unsupported += 1
                reference_unknown += 1
            else:
                scored += 1
                valid += int(score["primary_score"] == 1.0)
                reference_valid += int(score["metrics"]["reference_valid"] == 1.0)
                reference_invalid += int(score["metrics"]["reference_valid"] != 1.0)
                solved_invalid += int(
                    result is not None
                    and result.status == "solved"
                    and score["primary_score"] != 1.0
                )
        except (ValueError, TypeError, ArithmeticError) as exc:
            reference_unknown += 1
            error_types["score/" + type(exc).__name__] += 1
        if result is None:
            predictions = {name: "" for name in controls}
        else:
            predictions = negative_controls(family, case.input, result)
        for name, prediction in predictions.items():
            controls[name]["attempted"] += 1
            try:
                score = score_procedure(case, prediction)
                if score["primary_score"] is None:
                    controls[name]["unsupported"] += 1
                else:
                    controls[name]["scored"] += 1
                    controls[name]["accepted"] += int(score["primary_score"] == 1.0)
            except (ValueError, TypeError, ArithmeticError) as exc:
                controls[name]["failed"] += 1
                error_types["control/" + type(exc).__name__] += 1
        score_seconds += time.perf_counter() - before
    return {
        "selected": selected,
        "statuses": dict(statuses),
        "unresolved_reasons": dict(reasons),
        "scored": scored,
        "scoring_failed": selected - scored - score_unsupported,
        "scoring_unsupported": score_unsupported,
        "valid": valid,
        "valid_fraction_all_selected": valid / selected if selected else None,
        "valid_fraction_scored": valid / scored if scored else None,
        "reference_valid": reference_valid,
        "reference_invalid": reference_invalid,
        "reference_diagnostic_unknown": reference_unknown,
        "solved_but_score_invalid": solved_invalid,
        "error_types": dict(error_types),
        "controls": controls,
        "work": {
            "states": state_total,
            "transitions": transition_total,
            "max_case_states": max_states,
            "max_case_transitions": max_transitions,
        },
        "timings_seconds": {"solver": solver_seconds, "scoring_and_controls": score_seconds},
        "input_inventory_sha256": inputs_hash.hexdigest(),
        "result_inventory_sha256": results_hash.hexdigest(),
    }


def run(data_root: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    data_root = data_root.resolve(strict=True)
    sources = source_inventory(root)
    dataset_files = {name: file_hash(data_root / name) for name in DATA_SHA256}
    if dataset_files != DATA_SHA256:
        raise ValueError("dataset differs from the preregistered source hashes")
    if tracemalloc.is_tracing():
        raise ValueError("benchmark requires its own Python allocation measurement")
    started = time.perf_counter()
    tracemalloc.start()
    try:
        # Hash before these imports so on-disk/runtime changes are caught at the end.
        from promptwitness.procedure_baselines import BASELINE_VERSION, ProcedureBaselineLimits
        from promptwitness.procedure_data import load_longproc_dataset
        from promptwitness.procedure_scores import PROCEDURE_SCORER_VERSION, ProcedureScoreLimits

        limits = ProcedureBaselineLimits(**LIMITS)
        protocol = {
            "datasets": list(DATASETS),
            "cases_per_dataset": 200,
            "selection": "every published row; no sampling",
            "solver_inputs": "ProcedureCase.input only; no reference/source/case identity",
            "limits": limits.to_dict(),
            "score_limits": ProcedureScoreLimits().to_dict(),
            "algorithms": ["countdown-integer-dfs/v1", "functional-directed-walk/v1"],
            "controls": [
                "empty",
                "missing closing tag",
                "altered final equation/edge; malformed fallback if unsolved",
            ],
            "failure_denominator": (
                "all selected cases; report scoring failures separately from invalid answers"
            ),
            "source_sha256": DATA_SHA256,
        }
        datasets = []
        for name in DATASETS:
            load_started = time.perf_counter()
            dataset = load_longproc_dataset(data_root, name)
            loading = time.perf_counter() - load_started
            if len(dataset.cases) != 200 or dataset.inventory["available"] != 200:
                raise ValueError("dataset count differs from preregistration")
            measurements = evaluate_cases(dataset.family, dataset.cases, limits)
            datasets.append(
                {
                    "name": name,
                    "loading_seconds": loading,
                    **measurements,
                    "case_inventory_sha256": dataset.inventory["case_digest_sha256"],
                }
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    imported = {}
    for name, module in list(sys.modules.items()):
        if name == "promptwitness" or name.startswith("promptwitness."):
            location = getattr(module, "__file__", None)
            if location is None:
                raise ValueError("imported runtime source cannot be verified")
            path = Path(location).resolve(strict=True)
            if not path.is_relative_to(root / "src"):
                raise ValueError("imported package is not the source-bound checkout")
            relative = path.relative_to(root).as_posix()
            if sources.get(relative) != file_hash(path):
                raise ValueError("imported runtime differs from source binding")
            imported[relative] = sources[relative]
    if sources != source_inventory(root) or dataset_files != {
        name: file_hash(data_root / name) for name in DATA_SHA256
    }:
        raise ValueError("source or dataset changed during benchmark; report not published")
    return {
        "format": "promptwitness.procedure-baseline-benchmark/v1",
        "completed": True,
        "kind": "authored-input-only-algorithms-on-published-procedural-data",
        "model_calls": 0,
        "network_calls": 0,
        "raw_dataset_redistributed": False,
        "protocol": protocol,
        "protocol_sha256": digest(encoded(protocol)),
        "datasets": datasets,
        "baseline_version": BASELINE_VERSION,
        "scorer_version": PROCEDURE_SCORER_VERSION,
        "source_sha256": sources,
        "imported_source_sha256": imported,
        "sources_and_data_unchanged": True,
        "tools": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "unicode_database": unicodedata.unidata_version,
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_python_tracemalloc_bytes": peak,
        "memory_scope": (
            "tracemalloc from delayed imports through loading/solving/scoring; excludes "
            "preexisting interpreter allocations, native allocations and process RSS; no children"
        ),
        "limitations": [
            "Output-difficulty buckets are not input token budgets.",
            "Published procedural/synthetic tasks; deterministic typed-input algorithms do not "
            "measure language-model quality or instruction following.",
            "Reference validity is a separate diagnostic; invalid reference answers do not "
            "change input-only predictions or exclude cases.",
            "LongProc data remains outside this repository; its Apache-2.0 project license and "
            "underlying dataset terms apply independently of this authored MIT implementation.",
            "No official evaluator execution, search-trace semantic equivalence, held-out "
            "learned-model evaluation, six-family completeness or whole-repository parity claim.",
        ],
    }


def publish(report: dict[str, Any], output: Path) -> dict[str, Any]:
    body = encoded(report)
    with output.open("xb") as stream:
        stream.write(body)
    return {"completed": True, "output": str(output), "sha256": digest(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output.exists()
        or not output.parent.is_dir()
        or output.is_relative_to(args.data_root.resolve())
    ):
        parser.error("output must be new, outside the dataset, with an existing parent directory")
    report = run(args.data_root)
    summary = publish(report, output)
    try:
        print(json.dumps(summary, sort_keys=True))
    except (OSError, UnicodeError, ValueError):
        # Publication succeeded; a closed/broken stdout must not claim otherwise.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
