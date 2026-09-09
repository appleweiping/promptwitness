# Durable long-context task suites

This is unreleased functionality on `feat/whole-repository-alignment`, not the
current `main` or published package. Use the [development installation](../README.md#install).

This workflow loads local records, prepares deterministic task prompts, checks
the complete input against explicit length budgets, calls a provider, scores
its output, and persists progress for later resumption. It is a substantive
execution path, separate from the older recorded-answer `long-context` command.

The seven task families remain explicit. Supporting a task family does **not**
mean reproducing an external benchmark's datasets, prompts, scores, or model
ecosystem. The checked-in example contains seven hand-authored tiny records and
canned answers: it validates integration, not model accuracy or long-context
performance. No datasets are downloaded automatically.

## Seven-family scope

| Family | Primary local metric | Other outputs and remaining work |
| --- | --- | --- |
| `recall` | Value recall | Exact case-folded values from a JSON string array; precision and format validity |
| `rag` | Normalized exact match | Best match over answer aliases; token F1 diagnostic |
| `rerank` | NDCG at configured k | MRR at k; requires a complete, duplicate-free JSON array of document IDs |
| `icl` | Case-sensitive label exact match | Allowed-label validity; deterministic ordering of local demonstrations |
| `cite` | **Unimplemented** | Citation-ID validity is a syntax diagnostic, not entailment or answer correctness |
| `longqa` | **Unimplemented** | EM/F1 diagnostics do not replace a factual-correctness model judge |
| `summ` | **Unimplemented** | EM/F1 diagnostics do not measure reference keypoint coverage or faithfulness |

`score_status: unsupported` and `unsupported_metrics` retain those missing
judges in each completed result. Reports list their capability gaps even if no
records in the family ran. There is no cross-family mean “benchmark score.”

For retrieval QA, normalization case-folds, removes ASCII punctuation and the
English articles a/an/the, then collapses whitespace. Token F1 uses multiset
overlap. Recall does not use substring matches: `Ada` is not `Adam`. Repeated
recall values invalidate the format and give zero primary recall. Ranking gain
is `2**relevance - 1` (computed with stable `expm1`), with logarithmic rank
discount; relevance grades are finite values from 0 to 4. Unknown, duplicate,
or missing document IDs give a zero ranking score. A task needs at least one
positive relevance grade. MRR regards any positive grade as relevant.

## Manifest and local adapters

See [the complete seven-family example](../examples/task_suite/suite.json).
A minimal manifest is:

```json
{
  "format": "promptwitness.task-suite/v1",
  "suite_id": "file-questions",
  "budgets": [4096, 16384],
  "seed": "experiment-v1",
  "generation": {"max_tokens": 128, "temperature": 0},
  "tasks": [{"id": "retrieval", "family": "rag", "path": "qa.jsonl"}]
}
```

Paths resolve relative to the manifest. Inputs are local JSON arrays or JSONL;
JSONL is delimited by LF, not Unicode characters embedded in strings. Duplicate
keys, non-finite JSON numbers, duplicate record/task IDs, unknown native fields,
and malformed task contracts fail before any provider call.

The default `native` record uses `id`, `question`, `answers` (nonempty strings),
and either `context` or `documents`. A document has string `id`, `text`, and an
optional `title`. `rerank` requires documents and a `relevance` object labeling
every document. `cite` requires documents; its answer references are optional.
`icl` uses `labels`, one expected answer, and nonempty `examples` with
`id`/`text`/`label`; a test cannot be its own example. Context for ICL is built
from the labeled examples, not a separate untyped context field.

Supported local export adapters:

- `kilt`, only `rag`: `id`, `question`, `answers`, and `ctxs` with text/title.
- `msmarco`, only `rerank`: `qid`, `query`, and `ctxs` with ID/text/label.
- `processed`, only recall/RAG/longQA/summary: `id`, `context`, `question`, and
  `answer` as a string or list of aliases/values.

These are schema adapters, not the original dataset generation/retrieval or
official evaluation pipelines. IDs must already be strings. An optional task
`limit` selects records by a stable SHA-256 order of seed/task/record ID;
`available`, `selected`, and `sampled_out` remain visible. `ranking_k` defaults
to 10 and is capped at the number of documents when scoring.

## Lengths and generation

The CLI/default API budget unit is **UTF-8 bytes of the entire canonical
messages JSON**, including system instructions, question, demonstrations, role
keys, and JSON framing. It is not a model token count. Budget bins are ceilings,
not achieved context lengths; actual measured input units are reported per
item and as min/max/mean. Small fixtures do not become long contexts by assigning
them a large ceiling.

There is no silent truncation or padding. Oversized inputs remain in the plan
and denominator as `skipped: input_budget_exceeded`, with zero provider calls.
The complete evidence and expected answer remain unchanged. Future context
construction/truncation policies need their own evaluation contract.

The API accepts a custom `counter(messages)` and `counter_identity` containing
`unit` and `implementation`. For model tokens, count the **complete chat
template**, including its generation prompt, and pin the tokenizer/template
revision in that identity. The library validates a positive integer count but
cannot inspect or prove a user callback's tokenizer. Changing the identity or
measured inputs prevents resumption against a previous run.

`generation.max_tokens` is a separate decoder token limit sent to the provider;
it is never subtracted from a byte count. Configure input token ceilings below
your model's total context window after reserving output tokens and any extra
provider framing. Only `max_tokens`, `temperature`, `top_p`, and `seed` are
accepted; these cannot replace messages, tools, or model settings.

## API and CLI

```python
from promptwitness import (
    OpenAICompatibleProvider,
    OpenAITaskProvider,
    TaskRunStore,
    load_task_suite,
)

plan = load_task_suite("suite.json")
provider = OpenAITaskProvider(
    OpenAICompatibleProvider("http://localhost:8000/v1/chat/completions", model="my-model")
)
with TaskRunStore("run.sqlite") as store:
    report = store.run(plan, provider, max_cases=10)
```

Credentials remain in the named environment variable, not the manifest or
database. Custom callables receive immutable `TaskRequest` objects containing
messages, generation settings, and model identity, **never gold labels or
relevance grades**. They must expose `identity` or supply an explicit non-secret
identity to `run`. Include the provider/model/configuration revision; arbitrary
callback implementations cannot be introspected for honesty.

```text
promptwitness task-suite plan suite.json
promptwitness task-suite run run.sqlite suite.json --endpoint http://localhost:8000/v1/chat/completions --model my-model --max-cases 10
promptwitness task-suite show run.sqlite
```

`plan` performs no network calls and omits messages/gold unless `--include-inputs`
is given. Run it again with the same configuration to continue only ready items.
The JSON report is printed to stdout. Exit 1 is an invalid invocation/data/I/O
error; exit 2 means invocations or primary scoring are incomplete, including
budget skips and unsupported judges. Exit 0 for a run means the configured
subset completed execution and scoring; explicit seven-family flags prevent
confusing a successful one-family subset with full seven-family coverage.

## Persistence and recovery

One database belongs to one exact plan and provider identity. The binding
includes raw input-file hashes (even whitespace changes matter), normalized
gold data, prompt/scoring implementation version, sampling, generation settings,
budgets, counter identity, and model/routing settings. Provider URLs and custom
headers are hashed, not stored verbatim. Completed outputs are reused; changed
configuration requires a new database. Database identity/version/schema are
checked before writes; existing unrelated SQLite files are rejected.

Each call first commits a `running` reservation. Completion or failure appends
an event; failures store only the exception type. Concurrent stale reservations
and late results cannot overwrite newer revisions. Ordinary SQL UPDATE/DELETE
of event rows is prohibited, and snapshots validate event transitions and
recompute metrics from pinned gold data. Checksums are integrity checks, not
signatures or protection against a database owner rewriting the whole artifact.

A failure remains `failed`; process interruption may leave `running`. Neither
is silently retried. After inspecting `show`, authorize a retry explicitly:

```text
promptwitness task-suite retry run.sqlite ITEM_ID --revision REVISION
```

For an interrupted running item, first stop/confirm the old worker is gone and
accept that its remote request might already have happened. `retry` makes the
item ready but makes no call itself. Alternatively, recover an already obtained
response with `respond run.sqlite ITEM_ID response.json --revision REVISION`.
Retry invalidates the old reservation; late writes then fail. External
exactly-once billing/effects cannot be guaranteed.

## Current limits and reproducible example

Local manifests are limited to 1 MiB, each dataset file to 64 MiB, and saved
prediction text to 1 MiB. Plans and datasets are materialized in memory. The
SQLite/metric validation is deliberately thorough and not a distributed-scale
executor: operations currently revalidate the materialized plan, so total work
can grow quadratically with the number of similarly sized cases. HTTP transport
currently reads a response before applying the saved
prediction limit. Custom callbacks have no forced timeout. Prompts, gold data,
and predictions may contain sensitive text: protect the database and reports.

```text
python examples/task_suite_demo.py new-run.sqlite new-report.json
```

This offline example stops after three calls, closes/reopens the database,
finishes seven calls, then confirms resumption adds none. It reports fourteen
planned items: seven budget skips, seven generated answers, four primary-scored
answers, and three unsupported judge-dependent families. It is intentionally
**not** an accuracy validation of a real language model or a published benchmark.
