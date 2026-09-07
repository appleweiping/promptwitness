# Task-oriented benchmark replay

PromptWitness supports task-oriented benchmark fixtures suitable for long-
context suites such as recall, retrieval, reranking, citation, long-question
answering, summarization, and in-context learning:

```console
promptwitness benchmark cases.jsonl predictions.json \
  --task recall --task rag --output report.json
```

Each case contains `case_id`, `task`, `prompt`, `expected`, and optional
`metadata`. The evaluator keeps expected answers out of the prompt digest,
reports aggregate and per-task accuracy, preserves provider failures, and
supports strict fail-fast mode. Provider integrations can call
`evaluate_benchmark()` directly with a callback that receives a
`BenchmarkCase`.
