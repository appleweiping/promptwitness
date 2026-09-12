# Procedure-suite CLI

`promptwitness procedure-suite` plans and runs the six procedure families using
the existing durable `TaskRunStore`. Planning and reading do not call a model.
Only an explicit `run` with an endpoint and model can invoke a provider. No
background worker, polling, retry loop, model download or executable-code sandbox
is installed by these commands.

This is a development feature on `feat/whole-repository-alignment`, not a claim
that the same commands exist in a released package or on `main`. See
[procedure data](procedure-data.md) for case schemas, LongProc attribution,
output-difficulty definitions and the real local data audit.

## Plan a suite before execution

An independently authored typed-case suite can use this manifest:

```json
{
  "format": "promptwitness.procedure-suite/v1",
  "suite_id": "local-procedures",
  "budgets": [32000],
  "seed": "0",
  "generation": {"max_tokens": 1024},
  "tasks": [
    {
      "id": "authored-countdown",
      "adapter": "cases",
      "path": "cases.json"
    }
  ]
}
```

`cases.json` is a JSON array of complete `ProcedureCase.to_dict()` values. Each
task must contain cases with one family/output-difficulty bucket. To use an
already available local LongProc directory, replace the task with:

```json
{
  "id": "countdown-2k",
  "adapter": "longproc",
  "path": "../LongProc/data",
  "dataset": "countdown_2k",
  "limit": 10,
  "generation": {"max_tokens": 3072}
}
```

Task paths are relative to the manifest directory. `limit` is deterministic
seed-bound sampling **after** complete data validation and official bucket
selection, not a promise to skip validating unselected data. Per-task generation
settings are bound into each request; changing them requires a different run.
Optional `limits`, `score_limits` and `data_limits` tighten the declared local
resource policy. Input budgets count the **whole messages JSON in UTF-8 bytes**,
not model tokens. `2k` describes output difficulty; `max_tokens` is an independent
provider generation parameter.

```console
promptwitness procedure-suite plan suite.json
promptwitness procedure-suite plan suite.json --include-inputs
```

The default preview omits every item's `record` and `messages`. It includes the
plan digest, identifiers, source inventory, generation settings, measured input
byte counts and skip reasons. `--include-inputs` explicitly includes full prompts,
source metadata and **gold/reference data**. The original LongProc prompt files
are not executed or rendered; the procedure plan uses versioned local templates
that include typed problem inputs only.

## Run, reopen and inspect

Set a credential in the selected environment variable using your own secret
management. Never put the credential itself in `--endpoint`, `--model`, the
manifest, a response file or a command-line argument.

```console
promptwitness procedure-suite run run.sqlite suite.json --endpoint https://your-provider.example/v1/chat/completions --model your-model --api-key-env PROCEDURE_API_KEY --timeout 60 --max-cases 10
promptwitness procedure-suite show run.sqlite
```

Only `run` accepts/requires endpoint and model. `--api-key-env`, `--timeout`, and
`--max-cases` are optional. The example endpoint is a placeholder, not an available
service. Provider execution may incur costs; tests use only authored local HTTP
fixtures. Transport credentials are read from the environment and are not copied
into the saved provider identity. The model, routing/header digests, environment
variable name, timeout, transport version, exact plan and per-item generation are
pinned. See the provider transport documentation for its separate network limits.

Run the same command again to continue **ready** work. Successful records are
reused without invoking the provider again. `--max-cases` bounds new invocations
in that command; it does not change the pinned plan or silently reduce its
completion denominator. Failed and interrupted/running items are never retried
implicitly. Budget-skipped items remain visible in the report.

Default execution/read/recovery reports omit both the raw `prediction` and
`score.processed` (parsed rows, route, or belief trace can contain the same private
answer). They retain status, revision, attempt counts, metrics and fixed diagnostic
codes/counters. An explicit view includes saved predictions and parsed outputs:

```console
promptwitness procedure-suite show run.sqlite --include-inputs
```

For run reports, `--include-inputs` exposes saved response content; it does not
add full plan records to the report. Use `plan --include-inputs` for the complete
private plan. All views mark `private_data: true`: IDs, model names, inventories,
timing-independent counts and scores may still be sensitive. This is not an
anonymization mechanism or a guarantee that operator-chosen identifiers contain
no secrets. The SQLite database always contains full plan inputs, reference data
and saved predictions regardless of the console view.

## Explicit recovery

Inspect the item ID, status and revision with `show`. A `running` reservation means
the external request **may already have happened**. First ensure the old worker is
stopped. Choose deliberately between preserving an actual response and permitting
another call; neither option establishes remote exactly-once execution.

Authorize retry of a failed or abandoned running item:

```console
promptwitness procedure-suite retry run.sqlite ITEM_ID --revision 2
```

`retry` only moves that item to ready and invalidates the previous revision. It
does not call a provider. A subsequent explicit `run` consumes another reservation
and increments the attempt count. Late completion from the old attempt is rejected.
Wrong revisions, wrong states and old seven-family task databases are rejected;
the CLI does not migrate or repurpose them.

If the actual response was retained outside the process, supply a JSON response to
the still-running reservation:

```console
promptwitness procedure-suite respond run.sqlite ITEM_ID --revision 1 response.json
```

The file is parsed and validated before any state mutation. It can contain a JSON
string (an explicit operator-supplied completed text answer), or the completed
single-text provider envelope accepted by `task_scores.response_text`. An envelope
with truncated/absent completion status, tool calls, refusal, or an ambiguous text
choice is not converted into success. The CLI cannot authenticate that a manually
supplied response came from a provider or corresponds to a billable request; this
is a trusted-operator recovery action, not a provenance proof.

The response file limit is 16 MiB of encoded JSON, followed by a 4 MiB UTF-8
prediction ceiling and the plan's potentially smaller scoring limits. No generated
code is executed. A successful but unscorable code-generation response remains
explicitly unsupported, not silently counted as a passing primary score.

## Exit status and output delivery

| Exit | Meaning |
| --- | --- |
| `0` | A plan validated, or all planned execution and available primary scoring completed. A numerical primary score of zero is still a completed score, not a quality gate pass. |
| `1` | Invalid input/configuration, wrong database/revision, an I/O failure, or an undelivered read/plan preview. Diagnostics contain static text, not source paths, records or raw exceptions. |
| `2` | A valid execution report still has skipped, ready, running, failed or unsupported-primary work. |

The overall report does not average incomparable family metrics into a benchmark
score. `0` is not proof of high model quality or parity with the original LongProc
evaluator. In particular, pseudocode-to-code has no executable primary evaluator;
a suite containing it cannot claim complete primary scoring here.

Serialization is completed before writing stdout (64 MiB encoded-report cap).
If a `run`, `retry` or `respond` action completed but serialization/stdout delivery
then fails, the CLI preserves the already determined execution status and writes
a static warning when stderr is available. It does **not** roll back committed
work, retry a model call, or change exit status to CPython's shutdown-flush error
120. A short write is not retried because a prefix may already have reached the
consumer. Consumers must validate that they received complete JSON; exit `0`
alone does not guarantee report delivery. Reopen with `show` rather than blindly
retrying work. An earlier failure during a multi-item run may likewise occur after
some items committed; error diagnostics remind the operator to inspect revisions.

## Path and input boundaries

Before opening a writable store, `run` validates the complete plan and protects
the manifest, typed-case files, and referenced LongProc inventory from aliasing
the database or its `-wal`, `-shm`, and `-journal` sidecars. Checks cover both lexical
and resolved database locations and existing hard-link aliases. The database and
sidecars must be **outside the entire declared LongProc data directory**, even
if the prospective database file does not yet exist. This keeps execution state
separate from benchmark sources. A manifest content change between validation
reads is rejected.

`show`, `retry` and `respond` require an existing, nonempty, bound task database.
A read-only SQLite admission check prevents an unrelated pristine SQLite file
from being initialized by the store constructor. The procedure report format is
checked before recovery mutations. Response files cannot alias the database or
its sidecars. Source/response reads are bounded regular-file reads, with strict
UTF-8 and duplicate-key-rejecting JSON.

These controls assume a trusted local directory; they are not a sandbox against
concurrent hostile filesystem replacement. No output-file option is provided.
Shell redirection is operator-managed and can truncate a destination **before**
the CLI starts: never redirect stdout onto a database, sidecar, manifest, cases,
or benchmark source file. Keep response files, database backups and expanded
reports private.

## Verification scope

`tests/test_procedure_cli.py` uses independent authored cases and real subprocesses
for plan/run/show/retry/respond. Local ephemeral HTTP tests check exact per-item
generation, request credential placement, gold exclusion, resume without another
call, incomplete/lost responses, explicit retry counts, unsupported primary scores,
and TSV/ToM parsed-output privacy. Additional checks cover stale manual completion,
legacy/pristine SQLite preservation, source/sidecar hard links, source-directory
containment, strict response limits, and genuine closed-consumer pipes. Mocked
publication tests isolate short writes and unavailable embedded host streams;
they are not represented as real-provider or model-quality evidence.
