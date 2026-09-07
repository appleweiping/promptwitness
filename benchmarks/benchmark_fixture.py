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

from promptwitness import Scenario, ScenarioExecutor, load_prompt


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
