# Durable procedural task suites

This functionality is on the unreleased `feat/whole-repository-alignment` branch.
Use the [development installation](../README.md#install). It is not a claim of
whole-repository parity, official evaluator compatibility, or model quality.

The original seven-family [`task-suite`](task-suites.md) remains unchanged.
Procedures use distinct formats: `promptwitness.procedure-suite/v1` manifests,
`promptwitness.procedure-plan/v1` plans and `promptwitness.procedure-report/v1`
reports. They share `TaskRunStore`'s existing SQLite schema and transactional
state machine, not a loosely extended seven-family record schema. An old reader
that does not know the procedure format must reject it.

## Supported properties

| Family | Primary metric | Explicit boundary |
| --- | --- | --- |
| Countdown | Full arithmetic solution validity | Exact integer steps and operand multiplicities; no search-trace judge |
| Directed path | Route validity | Supplied directed, typed edges; not natural-language map extraction |
| HTML to TSV | Duplicate-aware row F1 | Exact declared header/cells; no browser or JavaScript execution |
| Belief tracking | Declared trace exact match | Whitespace-normalized declared trace; not general belief entailment |
| Travel planning | Full plan validity | Inclusive dates, durations, fixed stays, direct flights; valid alternatives accepted |
| Pseudocode to C++ | Unsupported | Code-fence syntax only; no compiler or generated program is executed |

See [precise scoring rules](procedure-scoring.md), including cases in which an
invalid declared reference leaves no primary score. Arithmetic, path and travel
validity do not require matching the supplied answer. A false reference-validity
diagnostic does not by itself invalidate a correctly solved problem.

## Local data and a reproducible plan

The checked-in [example suite](../examples/procedures/suite.json) consists of six
small independently authored cases. Planning it makes no external calls:

```powershell
promptwitness procedure-suite plan examples/procedures/suite.json
```

A minimal local manifest is:

```json
{
  "format": "promptwitness.procedure-suite/v1",
  "suite_id": "authored-procedures",
  "seed": "experiment-1",
  "budgets": [4096, 16384],
  "generation": {"max_tokens": 1024, "temperature": 0},
  "tasks": [
    {"id": "arithmetic", "adapter": "cases", "path": "cases-countdown.json"}
  ]
}
```

`cases` reads a JSON array of closed `ProcedureCase.to_dict()` records, not JSONL
or arbitrary task dictionaries. `longproc` additionally requires a `dataset`
name and a `path` to an existing local data root. The
[data contract and pinned definitions](procedure-data.md) describe that adapter.
No download, reference Python import, YAML evaluation, HTML rendering or compiler
invocation occurs. Paths resolve relative to the manifest. Inputs and gold may
be private; source hashes are identifiers, not anonymization.

Task IDs are unique. Each task contains one family/output bucket and unique
record IDs. All eligible records are selected unless an explicit positive
`limit` is supplied. Selection is stable by the SHA-256 order of
`[seed, task_id, record_id]`, independent of source row order. Original source
row identity remains in each record's provenance. `available`, `excluded`,
`eligible`, `selected` and `sampled_out` distinguish adapter filtering from user
sampling. There is no silent sample to fit a budget. Each selected record has an
item at every requested input budget, including budgets that cause a skip.

Optional per-task `generation` replaces, rather than merges with, the suite
default. It must include `max_tokens`; only `temperature`, `top_p` and `seed` may
also be supplied. Every item retains its effective generation settings, and
every request/event binds those exact settings. The default is 1,024 output
tokens and temperature zero; actual provider/model limits remain external.

## Input isolation and length accounting

`procedure_messages(case)` uses only the typed `case.input` and a versioned
original system instruction. Reference answers, source metadata, record IDs,
demonstrations and search traces from reference fields are not included. The
durable request envelope still has its opaque item ID and provider identity for
replay. This is structural input/reference separation, not proof that a caller
has never placed an answer inside an input field, nor prompt-injection immunity.

The sole v1 counter is UTF-8 bytes of the **complete canonical messages JSON**,
including instructions, roles, field names and JSON escaping. It excludes the
provider envelope/generation fields and is not a tokenizer or HTTP request byte
counter. `budgets` are maximum input sizes, not demonstrated long-context lengths.
Published output buckets (`0.5k`, `2k`, `8k`) are an independent axis; never treat
them as input token budgets. `input_budget_exceeded` retains the item in all
coverage denominators and does not truncate a problem or invoke a provider.

A plan is immutable. Deserialization independently rederives prompts, byte
counts, skip states, IDs, sampling counts, cross-budget records and generation
agreement. Changing a JSON field and recomputing its outer digest cannot bypass
these invariants. The plan pins scorer/template versions and score limits as
well as source metadata. Source metadata is caller-supplied provenance plus
observed file hashes, not an independent signature or authenticity attestation.

## API execution and recovery

```python
from promptwitness import TaskRunStore, load_procedure_suite

plan = load_procedure_suite("examples/procedures/suite.json")


def offline_probe(request):
    # Deliberately incorrect, complete text: an integration probe, not a model.
    return "No solution provided."


with TaskRunStore("procedure-run.sqlite") as store:
    report = store.run(plan, offline_probe, identity={"provider": "offline-probe"})
```

`OpenAITaskProvider` is available for explicitly configured real model runs; the
[CLI guide](procedure-cli.md) explains credentials and local recovery commands.
Real provider calls can incur cost. Neither planning, loading, scoring nor the
offline demo calls a real provider.

Before an invocation, the runner commits a `running` reservation with request
digest and revision. `complete` accepts only a complete, unambiguous text answer.
Length-truncated responses and tool calls are rejected. A successful result,
recomputed score and completion event commit in one transaction. Exceptions
retain their type, not raw provider error text. Scoring resource exhaustion is a
failed execution with no score, not a zero correctness score.

Reopening the exact plan/provider identity reuses successful outputs. It never
automatically repeats failed or interrupted `running` calls. After confirming
the previous worker has stopped, `retry(item_id, expected_revision=...)` explicitly
authorizes a new attempt. Its new revision rejects late results from the old
worker. There is no remote exactly-once guarantee: an interrupted external call
may already have incurred cost. Snapshots replay stored transitions, verify
request/result digests and recompute scores from pinned cases. A checksum is
self-consistency evidence, not tamper-proof storage against an adversary who can
rewrite the complete database and all guard definitions.

The built-in task provider copies endpoint/model/header/timeout configuration
into a private base transport, verifies that snapshot against the reserved
identity, and sends with that same snapshot. Arbitrary callable providers with
an exposed identity are checked before and after returning. A changed or
unreadable post-call identity leaves `running` with no score and raises a
conflict for reconciliation; it does not label the answer as the original
model's output. A callable without an exposed identity relies on the caller's
explicit identity declaration. None of these checks authenticates an external
model server or proves what a dishonest custom callable executed internally.
Older stored results cannot retrospectively prove which configuration actually
ran. If an older run experienced concurrent provider changes, reconcile that
history and use a fresh run rather than assuming replay repairs attribution.

## Reports and operational bounds

Reports preserve every ready/running/failed/succeeded/skipped item. They expose
coverage overall and by family, input budget and output bucket, plus invalid
reference and unsupported-primary counts. Mean primary scores appear only
within a family; there is no average of unrelated metrics. Empty/unconfigured
groups have null coverage and cannot count as complete. Six-family completion
flags concern those six named families, **not** all 16 published definitions;
code execution being unsupported prevents six-family scoring completion.

The default CLI view omits problem/gold and prediction/processed text. IDs,
counts, source names and hashes remain private data. The API report and SQLite
file contain complete predictions, inputs and references. Use appropriate local
file permissions; this is not an encrypted multi-tenant service.

Defaults admit 64 tasks, 10,000 planned items, 16 input budgets, 64 MiB serialized
plan and four million JSON nodes. `limits`, `data_limits` and `score_limits`
accept documented bounded policy overrides. Data admission rejects oversized
inputs before provider calls; see the data guide for per-file/per-case limits
and the loader's aggregate accounting. Score defaults are 1 MiB prediction,
16 MiB case text, 64 KiB per line, 20,000 combined parsed lines and 500,000 work
items. Prediction policy can explicitly increase to 4 MiB. These quantities
measure serialized data/defined work, not RSS, disk size, elapsed time or tokens.

In a manifest, `data_limits.max_total_read_bytes`, `max_total_case_bytes` and
`max_records` apply across tasks, not afresh to each task. Repeated files read by
separate tasks are counted again. Records include primary source rows that are
later filtered or sampled out; ancillary travel demonstrations are separately
checked by the adapter and their file reads count in aggregate bytes. The
manifest itself has a separate 1 MiB ceiling. Normalized case bytes include input,
reference and provenance. Sampling cannot evade source-admission budgets.

Planning materializes selected records and messages. The existing SQLite runner
revalidates the complete plan on state changes and snapshots rescore completed
results and replay history. Cost therefore grows with plan, outputs and history;
it is not constant-time resume or a streaming million-case engine. There is no
overall run-history, database-file or multi-process memory quota. Start with an
explicit small sample and review the plan; choose process/host resource limits
separately for larger runs. No implicit data pruning, retry or deletion occurs.

The [offline lifecycle demonstration](procedure-demo.md) checks recovery and
coverage. The [input-only solver benchmark](procedure-baselines.md) measures
bounded arithmetic/path algorithms on fixed procedural data. Neither substitutes
for real language-model experiments, semantic judges, generated-code sandboxing,
or the full functionality of any reference repository.
