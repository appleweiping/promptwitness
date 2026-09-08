# Task-oriented benchmark replay

PromptWitness supports task-oriented benchmark fixtures suitable for long-
context suites such as recall, retrieval, reranking, citation, long-question
answering, summarization, and in-context learning:

```console
promptwitness benchmark cases.jsonl predictions.json \
  --task recall --task rag --output report.json
```

Each case contains `case_id`, `task`, `prompt`, `expected`, and optional
`metadata`. Cases may also declare an explicit dependency-free `scorer` and
`threshold`:

```json
{
  "case_id": "recall-1",
  "task": "recall",
  "prompt": "Which city is in the passage?",
  "expected": "Paris",
  "scorer": "contains",
  "threshold": 1.0
}
```

The built-ins are `exact` (the default), `contains`, `token_f1`, and `json`.
The continuous score is retained in each result; `correct` is derived by
comparing it with the case threshold. The scorer and threshold are included in
the prompt digest because they are part of the evaluation contract, while the
expected answer remains excluded.

The evaluator reports aggregate and per-task accuracy plus mean score,
preserves provider failures, and supports strict fail-fast mode. Provider
integrations can call `evaluate_benchmark()` directly with a callback that
receives a `BenchmarkCase`.
