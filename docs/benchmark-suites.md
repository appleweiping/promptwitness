# Versioned benchmark suites

PromptWitness benchmark suites package several task families in one explicit,
reviewable JSON document. A suite keeps the task name next to every case,
requires globally unique case IDs, and exposes a stable `suite_digest` that
describes prompts and metadata without embedding expected answers. This makes
it possible to review a benchmark release independently from an answer file.

## Format

The root object must use `"format": "promptwitness.benchmark-suite.v1"`, a
non-empty `suite_id`, and a non-empty `tasks` array. Each task has a unique
`name`, optional `metadata`, and one or more cases. A case contains `case_id`,
`task`, `prompt`, and `expected`, plus optional metadata:

```json
{
  "format": "promptwitness.benchmark-suite.v1",
  "suite_id": "release-gate",
  "metadata": {"version": 1},
  "tasks": [
    {
      "name": "recall",
      "cases": [
        {
          "case_id": "recall-1",
          "task": "recall",
          "prompt": "Who?",
          "expected": "Ada"
        }
      ]
    }
  ]
}
```

Unknown fields are rejected so that accidental schema drift fails at the
boundary. `BenchmarkSuite.to_dict()` emits the same versioned shape and
`BenchmarkSuite.digest` excludes gold answers while retaining prompt metadata.

## Replay a suite

Predictions are a JSON object mapping case IDs to answer strings:

```console
promptwitness benchmark-suite suite.json predictions.json \
  --task recall --output report.json
```

Repeat `--task` to select several task families. Without a filter all tasks are
scored. Missing prediction keys are treated as empty answers, preserving the
same deterministic behavior as the single-file `benchmark` command. The report
contains per-task aggregates plus `suite_id`, `suite_digest`, and the selected
task names. The command returns zero when replay completes without provider
errors, two when a non-strict replay records a provider error, and one for
malformed input.

The Python API is equivalent:

```python
from promptwitness import evaluate_benchmark, load_benchmark_suite

suite = load_benchmark_suite("suite.json")
report = evaluate_benchmark(suite.cases, lambda case: answers[case.case_id])
```

