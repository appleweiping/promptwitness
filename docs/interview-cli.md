# Interview CLI: explicit execution and private evidence

These commands belong to the development branch, not a claim of released feature
availability or interview-quality validation. Install the same checkout/ref as
the [development README](../README.md). `promptwitness interview --help` lists
the commands. Nothing runs in the background; there is no automatic model,
endpoint, credential, deployment, or paid-call selection.

## Configure and start

Prepare a strict `InterviewPlan` JSON file. It must name the currently supported
`analysis/v1` and `question/v1` prompt versions. A role configuration contains
exactly `analyst` and `questioner`, each with these five required fields:

```json
{
  "analyst": {
    "endpoint": "http://127.0.0.1:8080/chat",
    "model": "your-explicit-model",
    "api_key_env": "INTERVIEW_ANALYST_KEY",
    "timeout": 30,
    "headers": {}
  },
  "questioner": {
    "endpoint": "http://127.0.0.1:8080/chat",
    "model": "your-explicit-model",
    "api_key_env": "INTERVIEW_QUESTIONER_KEY",
    "timeout": 30,
    "headers": {}
  }
}
```

This example does not start a local provider. You must explicitly configure a
working compatible server. HTTP is not encrypted; use HTTPS for appropriate
remote deployments. Configured remote execution can incur provider charges.
`api_key_env` may be `null` for an explicitly unauthenticated endpoint; otherwise
provide an environment-variable **name**, not a key. Timeouts are positive finite
numbers up to 300 seconds and use the documented [transport bounds](provider-traces.md).

Header names containing common credential markers are rejected, and raw
`api_key`/unknown configuration fields are not accepted. This is a limited
name-based check, **not a universal secret detector**. Arbitrary nonstandard
header values or URL parameters may still contain secrets; keep configuration
private and use the named credential environment variable for bearer credentials.
The journal pins derived endpoint/header hashes and nonsecret model/transport/
environment-name settings, not the raw configuration file or resolved key.

```text
promptwitness interview start interviews.db first plan.json --participant-id person-1 --roles roles.json --command-id create-first
```

`start` initializes a new supported database or adds an interview to an existing
supported database. It does not migrate an unrelated database or call a provider.
It validates plan/configuration files before initialization. To import explicitly
selected prior memory from this **same database**, supply `--prior-heads heads.json`:

```json
[
  {
    "interview_id": "previous-interview",
    "revision": 12,
    "digest": "replace-with-the-actual-64-character-lowercase-sha256-head"
  }
]
```

The digest placeholder is not valid executable input. Obtain actual heads from
verified previous output. At most 127 prior heads are permitted; participant and
source-membership checks occur in the journal. No prior coverage is inherited.

## Ask, answer and continue

```text
promptwitness interview run interviews.db first --roles roles.json --operation-id first-question
```

`run` stops at participant input, proposal review, explicit recovery or a valid
termination. Both roles must be configured explicitly, but an unused role does
not require its environment variable to exist. Configuration is checked against
the pinned identity before each actual stage. Secrets are resolved only when
that stage invokes the provider; configuration never comes from an answer.

Normal stdout is UTF-8 JSON containing `head`, `next_action`, counts, stable
error codes, pending proposal descriptions, and the current question when waiting
for an answer. It does not include hidden provider prompts, original responses or
past participant answer text. All output is marked `private_data: true`; even a
question or identifier may be sensitive.

Save the participant's exact answer in a UTF-8 file, then use the returned
question ID and exact head. The following symbolic values must be replaced:

```text
promptwitness interview answer interviews.db first answer.txt --question-id QUESTION_ID --answer-id answer-1 --revision REVISION --digest HEAD_DIGEST --command-id answer-first
promptwitness interview run interviews.db first --roles roles.json --operation-id analyze-first-answer
```

Files are decoded without newline translation or Unicode normalization. The
answer's revision is its committed event sequence. The answer command does not
invoke analysis itself; the subsequent explicit `run` does. An operation may
commit analysis and then publish a next question in separate transactions. A
later failure does not undo already committed steps.

## Explicit decisions and recovery

All participant/domain changes require `--revision`, `--digest` and a stable
`--command-id`. Exact command retries return the historical receipt; a changed
input under the same ID conflicts. Use the current head for a new decision.

```text
promptwitness interview skip interviews.db first --question-id QUESTION_ID --scope criterion --revision REVISION --digest HEAD_DIGEST --command-id skip-one
promptwitness interview proposal interviews.db first --proposal-id PROPOSAL_ID --decision accepted --revision REVISION --digest HEAD_DIGEST --command-id review-one
promptwitness interview finish interviews.db first --reason participant_stopped --revision REVISION --digest HEAD_DIGEST --command-id stop-first
```

Skip scope is `criterion` or `topic`; the literal answer `skip` is ordinary text.
Proposal decisions are `accepted` or `rejected`. Automatic finish reasons cannot
be used to force success: the reducer checks them against actual coverage,
eligible work and budgets. `failed` requires a recorded stage failure.

After an interrupted/failed stage, ordinary `run` does not repeat the pending
provider call. Inspect the state and explicitly authorize a replacement:

```text
promptwitness interview retry interviews.db first --roles roles.json --operation-id retry-first-attempt
```

Use a new retry operation ID for a genuinely new authorization. A repeated ID
cannot authorize another call. Retrying can repeat external work or charges if
the original response was lost; it is not exactly-once execution. Old attempt
results are fenced. No scheduled monitoring or automatic retry is created.

## Inspect, recall, export and replay

```text
promptwitness interview show interviews.db first
promptwitness interview show interviews.db first --content
promptwitness interview report interviews.db first
promptwitness interview recall interviews.db first --query-file query.txt
promptwitness interview export interviews.db first private-export.json
promptwitness interview replay private-export.json --trusted-head independently-trusted-head.json
```

`show --content` includes the full private derived state. Reports include
source-linked assessments and gaps. Recall is scoped lexical BM25 over the
current verified snapshot, not vector/semantic retrieval. Query input is exact
UTF-8; no hidden model invocation occurs. These commands open existing databases
only, although supported SQLite recovery/sidecar activity can occur on opening.

Exports include the original private transcript, provider completion envelopes
and evidence. They are **not redacted or safe-to-publish artifacts**. Replay is
offline and does not create/import a database. It requires a separate strict
`SourceHead` JSON file containing `interview_id`, `revision`, and `digest`, obtained
through an independently trusted channel. The export's own embedded head is not
an independent anchor. The CLI rejects identical or hardlink/symlink-aliased
export/head input files, but cannot establish where a user obtained a copied
head. Hashes do not authenticate an attacker replacing both history and anchor.

## File safety, limits and exit status

Before reading inputs or opening SQLite, the CLI checks resolved paths and
existing-file identity. Inputs must be regular files and independent of each
other and the database plus `-wal`, `-shm`, `-journal` sidecars, including lexical
and resolved spellings. Export output must be new, distinct from all protected
paths, and located in an existing directory. Files are never silently overwritten.

Export is rendered and bounded before writing, flushed to a private temporary
file in the destination directory, then installed with an exclusive hard link.
A competing existing destination makes installation fail without replacing it;
cleanup then removes the temporary file. If cleanup itself fails, the CLI returns
1 and warns that an artifact or private temporary file may already exist. Successful
installation is not rolled back: inspect the destination and remove any private
temporary alias before retrying. A filesystem without hard-link support produces a
controlled error rather than falling back to overwrite. These checks are not a
defense against an adversary concurrently replacing directories or changing
filesystem ownership; use a trusted local directory.

Configuration input is capped at 128 KiB; answer/query input at 1 MiB before
plan-specific checks; trusted-head input at 16 KiB; other strict JSON input and
exports at the shared 16 MiB contract limit. Duplicate JSON keys, malformed UTF-8,
unknown fields and oversized files are rejected. Plan turn budgets remain maxima,
not guarantees against earlier history/context/snapshot limits. Nothing is
silently truncated.

- Exit **0**: the command completed; inspect the report's reason rather than
  interpreting every completed command as full agenda coverage.
- Exit **1**: input, file, database or command failure. For mutating commands,
  inspect the current head before retrying: earlier runner stages may have committed.
- Exit **2**: a valid returned state needs explicit recovery (`await_retry` or
  `await_completion`). Argument-parser usage errors also use its conventional 2.

If a command completed but stdout cannot be delivered (closed pipe, encoding or
other output error), the CLI reports that distinction on stderr and keeps the
completed command's status. It does not falsely report rollback. If both output
streams are closed, inspect the durable head or export file directly. Credential
and raw exception strings are not printed by this command handler; participant
and provider content in successful private outputs is not generically sanitized.
