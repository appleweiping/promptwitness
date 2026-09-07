"""Benchmark prompt rendering and provider-boundary execution on examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

from promptwitness import (
    Scenario,
    ScenarioExecutor,
    evaluate_long_context,
    load_prompt,
    make_needle_cases,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "before.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "benchmarks/results/fixture.json")
    args = parser.parse_args()
    payload = source.read_bytes()
    document = load_prompt(source)
    scenarios = tuple(
        Scenario(
            f"fixture-{index}",
            {"customer_name": name, "order_id": f"ORD-{index:03d}"},
            ("fixture",),
        )
        for index, name in enumerate(("Ada", "Grace", "Lin"), start=1)
    )

    def provider(row):
        return {
            "message_count": len(row.messages),
            "characters": sum(len(item.content) for item in row.messages),
        }

    tracemalloc.start()
    started = time.perf_counter()
    report = ScenarioExecutor().run(document, scenarios, provider, workers=2)
    long_context = make_needle_cases(
        (("background " * 4, "methods " * 4, "results " * 4),),
        needle="Ada owns evaluation",
        query="Who owns evaluation?",
        expected="Ada",
        seed="fixture",
    )
    long_context_report = evaluate_long_context(long_context, lambda _case: "Ada")
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "kind": "fixture-real",
        "source": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload),
        "prompt_id": document.prompt_id,
        "scenarios": len(scenarios),
        "succeeded": report.succeeded,
        "long_context_cases": len(long_context),
        "long_context_accuracy": long_context_report.accuracy,
        "elapsed_seconds": elapsed,
        "peak_python_bytes": peak,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
