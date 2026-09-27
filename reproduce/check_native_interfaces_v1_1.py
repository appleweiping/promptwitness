"""Exercise installed official interfaces on deterministic Python tasks, not NLP."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path


def run(gepa_source: Path, dspy_source: Path) -> dict:
    import dspy
    from dspy.evaluate import Evaluate
    from gepa.core.adapter import EvaluationBatch

    from promptwitness.incremental.optimizers import evaluate_dspy_native, evaluate_gepa_native

    source_hashes = {}
    for installed, pinned, label in (
        (
            Path(inspect.getfile(EvaluationBatch)),
            gepa_source / "src/gepa/core/adapter.py",
            "GEPA.adapter",
        ),
        (
            Path(inspect.getfile(Evaluate)),
            dspy_source / "dspy/evaluate/evaluate.py",
            "DSPy.Evaluate",
        ),
    ):
        if installed.read_bytes() != pinned.read_bytes():
            raise ValueError(f"installed {label} differs from the pinned source")
        source_hashes[label] = hashlib.sha256(installed.read_bytes()).hexdigest()

    class Adapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            outputs = [x * int(candidate["multiplier"]) for x in batch]
            return EvaluationBatch(
                outputs=outputs,
                scores=[float(output == x * 2) for x, output in zip(batch, outputs, strict=True)],
                trajectories=[{"x": x, "output": y} for x, y in zip(batch, outputs, strict=True)]
                if capture_traces
                else None,
                num_metric_calls=len(batch),
            )

    class Program(dspy.Module):
        def forward(self, x):
            return dspy.Prediction(answer=x * 2)

    data = [dspy.Example(x=x, answer=x * 2).with_inputs("x") for x in range(8)]
    evaluator = Evaluate(
        devset=data,
        metric=lambda example, pred: float(pred.answer == example.answer),
        num_threads=1,
        max_errors=1,
        failure_score=float("nan"),
        display_progress=False,
    )
    result = evaluate_dspy_native(evaluator, Program(), data)
    gepa = evaluate_gepa_native(Adapter(), list(range(8)), {"multiplier": "2"}, capture_traces=True)
    if result.score != 100.0 or [row[2] for row in result.results] != gepa.scores:
        raise ValueError("native actual score contracts changed")

    class BrokenProgram(dspy.Module):
        def forward(self, x):
            raise RuntimeError("mechanical failure sentinel test")

    blocked = False
    try:
        evaluate_dspy_native(evaluator, BrokenProgram(), data[:1])
    except Exception as error:
        # Pinned ParallelExecutor raises this generic Exception on max_errors;
        # distinguish that known cancellation from unrelated infrastructure errors.
        blocked = type(error) in (RuntimeError, ValueError) or (
            type(error) is Exception
            and str(error) == "Execution cancelled due to errors or interruption."
        )
        if not blocked:
            raise
    if not blocked:
        raise ValueError("native failure silently converted into an observed score")
    return {
        "format": "promptwitness.delta.native-interface-check/v1.1",
        "status": "MECHANICAL_INTERFACE_CHECK_PASSED",
        "versions": {"dspy": version("dspy"), "gepa": version("gepa")},
        "installed_pinned_sources_byte_identical": True,
        "source_sha256": source_hashes,
        "actual_python_task_scores_per_interface": 8,
        "failure_rejected": blocked,
        "real_model_calls": 0,
        "synthetic_mechanical_task": True,
        "native_optimizer_search_or_savings_measured": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gepa_source", type=Path)
    parser.add_argument("dspy_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run(args.gepa_source, args.dspy_source)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report))
