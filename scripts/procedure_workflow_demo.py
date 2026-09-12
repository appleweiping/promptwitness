"""Original six-family synthetic durable workflow; no model, network or subprocess.

The interruption is simulated by closing after a committed reservation. This is
not an operating-system kill test or evidence of language-model task quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from promptwitness.procedure_baselines import solve_countdown, walk_graph_path
from promptwitness.procedure_plan import load_procedure_suite
from promptwitness.task_runs import TaskRequest, TaskRunConflict, TaskRunStore

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/procedures"
FAMILIES = (
    "countdown",
    "path_traversal",
    "html_to_tsv",
    "tom_tracking",
    "travel_planning",
    "pseudo_to_code",
)
GENERATIONS = dict(zip(FAMILIES, (128, 160, 192, 224, 256, 288), strict=True))
DATABASE = "procedures.sqlite"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def envelope(text: str, *, finish: str = "stop") -> dict[str, Any]:
    return {
        "choices": [
            {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}
        ]
    }


class ScriptedProvider:
    """Hand-authored responses plus the two input-only algorithms; never reads gold."""

    def __init__(self) -> None:
        self.identity = {
            "provider": "authored-procedure-demo/v1",
            "model": "none",
            "network": "disabled",
        }
        self.calls: list[dict[str, Any]] = []
        self.partial_sent = False

    def __call__(self, request: TaskRequest) -> dict[str, Any]:
        require(dict(request.provider_identity) == self.identity, "unexpected scripted identity")
        inputs = json.loads(request.messages[1]["content"])
        if "numbers" in inputs:
            family = "countdown"
            text = solve_countdown(inputs).prediction
        elif "edges" in inputs:
            family = "path_traversal"
            text = walk_graph_path(inputs).prediction
        elif "html" in inputs:
            family = "html_to_tsv"
            text = "item\tmark\nMica\t😀\nMica\t😀\nQuartz\t{{literal}}"
        elif "story" in inputs:
            family = "tom_tracking"
            text = (
                "- Neri initially believes the pebble is in the green box.\n"
                "- After the unseen move, Neri still believes the pebble is in the green box."
            )
        elif "cities" in inputs:
            family = "travel_planning"
            text = (
                "<Plan>\n**Day 1-2:** Visit Aster for 2 days.\n"
                "**Day 2:** Fly from Aster to Beryl.\n"
                "**Day 2-4:** Visit Beryl for 3 days.\n</Plan>"
            )
        elif "pseudocode_lines" in inputs:
            family = "pseudo_to_code"
            text = "```cpp\n#include <iostream>\nint main() { std::cout << 7 << '\\n'; }\n```"
        else:
            raise ValueError("unexpected authored input shape")
        if not isinstance(text, str):
            raise RuntimeError("authored baseline did not find the expected solution")
        require(
            dict(request.generation) == {"max_tokens": GENERATIONS[family], "temperature": 0},
            "per-task generation settings were not delivered",
        )
        self.calls.append(
            {
                "family": family,
                "max_tokens": request.generation["max_tokens"],
                "request_sha256": request.digest,
            }
        )
        if family == "path_traversal" and not self.partial_sent:
            self.partial_sent = True
            # Even otherwise valid content is incomplete when the transport says length.
            return envelope(text, finish="length")
        return envelope(text)


def _source_hashes() -> dict[str, str]:
    paths = [
        EXAMPLES / "suite.json",
        *(EXAMPLES / f"cases-{name}.json" for name in FAMILIES),
        Path(__file__).resolve(),
    ]
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _statuses(report: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(row["status"] for row in report["results"]))


def run_demo(directory: Path) -> dict[str, Any]:
    """Create a new, caller-owned artifact directory; never reuse an existing DB."""
    directory = directory.resolve()
    if directory.exists() or not directory.parent.is_dir() or directory.is_relative_to(EXAMPLES):
        raise ValueError("demo requires a new directory outside example inputs")
    sources = _source_hashes()
    plan = load_procedure_suite(EXAMPLES / "suite.json")
    directory.mkdir(exist_ok=False)
    database = directory / DATABASE
    eligible = {
        item["family"]: item["id"] for item in plan.payload["items"] if item["skip_reason"] is None
    }
    require(
        len(plan.payload["items"]) == 12 and set(eligible) == set(FAMILIES),
        "authored suite inventory changed",
    )
    provider = ScriptedProvider()
    trace = []
    with TaskRunStore(database) as store:
        store.bind(plan, provider.identity)
        initial = store.snapshot()
        require(_statuses(initial) == {"skipped": 6, "ready": 6}, "initial task inventory differs")
        abandoned, old_revision = store.reserve(eligible["countdown"], expected_revision=0)
        require(old_revision == 1, "reservation revision differs")
        trace.append(
            {
                "stage": "reservation_committed_then_connection_closed",
                "calls": 0,
                "statuses": _statuses(store.snapshot()),
            }
        )
    # The old worker is known stopped: this demo closed before calling it.
    with TaskRunStore(database) as store:
        interrupted = store.run(plan, provider)
        require(len(provider.calls) == 5, "running item was automatically retried")
        require(
            _statuses(interrupted) == {"skipped": 6, "running": 1, "failed": 1, "succeeded": 4},
            "partial response was not rejected or interrupted state was lost",
        )
        route = next(
            row for row in interrupted["results"] if row["item_id"] == eligible["path_traversal"]
        )
        require(
            route["error"] == "ValueError" and route["result"] is None,
            "partial response was persisted as success",
        )
        trace.append(
            {
                "stage": "reopen_ready_only_run_rejects_length",
                "calls": 5,
                "statuses": _statuses(interrupted),
            }
        )
        store.retry(eligible["countdown"], expected_revision=old_revision)
        request, fresh_revision = store.reserve(eligible["countdown"], expected_revision=2)
        require(
            request.digest == abandoned.digest and fresh_revision == 3,
            "retry should retain request contents while replacing reservation revision",
        )
        try:
            store.complete(
                eligible["countdown"],
                envelope("late old worker output"),
                expected_revision=old_revision,
            )
        except TaskRunConflict:
            stale_rejected = True
        else:
            stale_rejected = False
        require(stale_rejected, "old reservation response overwrote the new attempt")
        store.complete(eligible["countdown"], provider(request), expected_revision=fresh_revision)
        store.retry(eligible["path_traversal"], expected_revision=2)
        final = store.run(plan, provider)
        require(
            len(provider.calls) == 7 and _statuses(final) == {"skipped": 6, "succeeded": 6},
            "explicit retry did not finish the expected six tasks",
        )
        trace.append(
            {"stage": "explicit_retries_complete", "calls": 7, "statuses": _statuses(final)}
        )
        require(
            store.run(plan, provider) == final and len(provider.calls) == 7,
            "successful items were re-invoked",
        )
    with TaskRunStore(database) as store:
        reopened = store.run(plan, provider)
        require(
            reopened == final and len(provider.calls) == 7,
            "reopen lost idempotent completed results",
        )
    trace.append(
        {"stage": "completed_reopen_no_new_calls", "calls": 7, "statuses": _statuses(reopened)}
    )
    completed = [row for row in final["results"] if row["status"] == "succeeded"]
    family_results = {
        row["family"]: {
            "revision": row["revision"],
            "attempts": row["attempts"],
            "primary_score": row["result"]["score"]["primary_score"],
            "score_status": row["result"]["score"]["score_status"],
        }
        for row in completed
    }
    require(
        all(
            family_results[name]["primary_score"] == 1.0
            for name in FAMILIES
            if name != "pseudo_to_code"
        ),
        "an independently authored expected answer did not satisfy its declared metric",
    )
    require(
        family_results["pseudo_to_code"]["primary_score"] is None
        and family_results["pseudo_to_code"]["score_status"] == "unsupported",
        "code-as-text incorrectly claimed executable correctness",
    )
    require(_source_hashes() == sources, "demo inputs changed during execution")
    return {
        "format": "promptwitness.procedure-workflow-demo/v1",
        "passed": True,
        "synthetic": True,
        "model_calls": 0,
        "network_calls": 0,
        "subprocesses": 0,
        "scripted_provider_calls": len(provider.calls),
        "reservation_attempts": sum(row["attempts"] for row in completed),
        "journal_events": sum(row["revision"] for row in final["results"]),
        "plan_sha256": plan.digest,
        "source_sha256": sources,
        "coverage": final["coverage"],
        "family_results": family_results,
        "trace": trace,
        "generation_max_tokens_by_family": GENERATIONS,
        "request_inventory_sha256": hashlib.sha256(
            json.dumps(provider.calls, sort_keys=True).encode()
        ).hexdigest(),
        "stale_response_rejected": stale_rejected,
        "successful_rerun_additional_calls": 0,
        "six_family_execution_complete": final["six_family_execution_complete"],
        "six_family_scoring_complete": final["six_family_scoring_complete"],
        "limitations": [
            "Interruption closes the connection after reservation; it is not an OS crash test.",
            "Hand-authored responses are fixture checks, not language-model quality evidence.",
            "Generation settings are delivered and identity-bound; token use is not measured.",
            "Pseudo-code executable correctness remains unsupported; code is never run.",
            "Six deliberately skipped byte-budget items remain in all coverage denominators.",
        ],
    }


def _check_output(output: Path | None, directory: Path | None) -> None:
    if output is None:
        return
    if output.exists() or output.is_symlink() or output.is_relative_to(EXAMPLES):
        raise ValueError("report output must be new and outside source examples")
    if directory is not None:
        reserved = {directory / (DATABASE + suffix) for suffix in ("", "-journal", "-wal", "-shm")}
        if output in reserved or output == directory:
            raise ValueError("report aliases the database or one of its sidecars")
    if not output.parent.is_dir() and (directory is None or output.parent != directory):
        raise ValueError("report parent directory does not exist")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, help="new artifact directory; default is temporary"
    )
    parser.add_argument("--output", type=Path, help="optional new aggregate JSON report")
    args = parser.parse_args()
    directory = args.directory.resolve() if args.directory else None
    output = args.output.resolve() if args.output else None
    _check_output(output, directory)
    if directory is None:
        with tempfile.TemporaryDirectory(prefix="promptwitness-procedure-demo-") as temporary:
            report = run_demo(Path(temporary) / "run")
    else:
        report = run_demo(directory)
    body = (json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    if output is not None:
        with output.open("xb") as stream:
            stream.write(body)
    try:
        print(body.decode(), end="")
    except (OSError, UnicodeError, ValueError):
        return 1  # Any successfully published file remains; never claim it was absent.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
