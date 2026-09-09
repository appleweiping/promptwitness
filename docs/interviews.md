# Durable evidence-linked interviews

Development-branch capability. The sequential runner connects the
[pure interview contracts](interview-contracts.md),
[stage requests and responses](interview-stages.md) and
[deterministic state policy](interview-state.md) to one SQLite journal. This is
an engineering workflow, not evidence of interview usefulness or factual model
accuracy. It does not implement concurrent exploration agents, semantic vector
memory, speech, a multi-user web service or a research/user-study suite.

## One domain history

`InterviewJournal` owns creation, published questions, participant answers,
reservations, original provider completions, interpreted assessments, emergent
proposals and source-linked memories in a single database. A successful analysis
appends its event, replaces the derived head and inserts all new memory-index
rows in one transaction. SQLite failure at any of those writes rolls back the
whole analysis, including its command receipt. Memory-index rows are not another
authoritative store: reads reconstruct them from verified source events.

Normal progression is:

```text
create → reserve question → commit question → wait for participant
       → commit answer → reserve analysis → commit analysis
       → explicit proposal review, next question, or explicit finish
```

A skip is an explicit criterion/topic command, not an answer containing a magic
word. Coverage is explicitly model-assessed and cumulative under the documented
state policy. Prior-interview memories may inform questions but never implicitly
mark a new interview's required criteria covered. Accepted emergent criteria
remain separate from the original required denominator.

## Public Python entry points

- `InterviewJournal(path, create=False)` opens an existing identified database;
  `create=True` permits initializing an empty or new regular file.
- `create_interview(journal, interview_id, participant_id, plan, command_id=...,
  analyst_identity=..., questioner_identity=..., prior_heads=())` pins the plan,
  real template/schema hashes, role configurations and explicit prior history.
- `InterviewRunner(journal, analyst=..., questioner=...).run_until_input(id,
  operation_id=...)` advances until participant input, proposal review, explicit
  recovery or termination is required.
- `runner.retry_pending(id, operation_id=...)` is an explicit replacement of one
  uncertain or failed reservation, with a new request/attempt identity. Retry
  operation IDs are single-use across source heads; reusing one with different
  bound inputs conflicts before another provider call.
- `journal.execute(InterviewCommand(...))` commits an explicit answer, skip,
  proposal decision or finish and returns that command's exact historical state
  on a matching replay. Every command binds ID, expected revision and head digest.
- `journal.submit(command)` additionally returns `InterviewReceipt.applied`.
  Only `applied=True` authorizes the runner's first invocation of newly reserved
  work. Two connections replaying the same reservation do not both invoke it.
- `journal.read(id, revision=None, expected_head=None)` validates history and
  derived caches; an optional `SourceHead` provides an independent expected anchor.
- `journal.events(id)`, `interview_report(state)`,
  `journal.memory_snapshot(participant_id, heads)` and `journal.export(id)` expose
  checked evidence. `verify_interview_export(value, expected_head=...)` performs
  pure offline replay against a required separately trusted head.

Role providers implement `identity` and `__call__(InterviewStageRequest)`.
`OpenAIInterviewProvider(OpenAICompatibleProvider(...))` derives the actual
transport identity, checks it again before invocation, and verifies that the
complete HTTP JSON body equals the reserved bytes. Configure a named model and
endpoint explicitly; no destination, credential or tool is inferred from an answer.
Invoking a configured remote model can incur its normal external costs. The
project's scripted and loopback tests make no paid model calls.

The sequential context policy includes up to the most recent 64 published
questions and their complete associated participant answers. It does not truncate
an answer's text. Requests record the exact included inventory; older transcript
events remain in the journal, not automatically in the model's request. Question
recall uses the scheduled topic and criterion descriptions as its lexical query,
the current verified memory snapshot, and the plan's BM25/budget policy. Analysis
may assess only its scheduled criterion and cite its current actual answer. These
are explicit first-runner policies, not semantic dialogue-quality claims.

## Crash, concurrency and completion boundaries

Reservations commit before provider invocation; the SQLite write lock is released
before network or scripted provider work. Ordinary resume observes pending work
and makes **zero repeated calls**. A process can fail after a provider executes but
before completion commits. An explicit retry may therefore repeat external work
or cost; this is not an external exactly-once guarantee. The original attempt is
fenced by request/attempt identity and compare-and-swap head checks.

Provider exceptions record stable error codes, not exception strings. A process
interrupt leaves the reservation pending. Failed or truncated completions do not
publish a question, assessment, proposal or memory. Successful events preserve the
original bounded positive-completion envelope and the normalized domain result;
replay parses the envelope again and requires an exact result match. A caller
cannot omit a finish reason and have the runner fabricate `stop`.

A matching historical receipt returns its original state, not a later current
state. This distinction matters when displaying old command results. The runner
uses the `applied` decision from the same database transaction and does not treat
an old reservation receipt as new authorization to invoke a provider.

## Database ownership and source verification

Before connecting, the journal rejects symbolic/hardlink aliases for its file or
known SQLite sidecars, orphan sidecars beside an empty/new target, nonregular files
and foreign nonempty database headers. Raw main-file application/version markers
are checked before the first SQLite query: otherwise SQLite can recover a foreign
hot rollback journal before returning an application-ID query. A regression uses
an actual child-process crash to verify rejection preserves the foreign main file
and its rollback journal.

The supported database's complete table/index/trigger DDL is compared to a
separately generated application-owned schema, not SQL executed from the candidate
database. Initializers recheck ownership under the writer lock; DDL and markers
are transactional. Existing supported databases may undergo normal SQLite recovery
or sidecar activity when opened; `create=False` is not a no-filesystem-writes mode.
Concurrent external file replacement is outside these checks: use a trusted local
directory, not an adversarial multi-user filesystem or untrusted writable share.

Prior memories must exactly match histories at the explicitly selected heads in
this same journal. Source event ancestry, participant identity, answer bytes and
owning memory revisions are replayed before constructing the snapshot. A declared
head or self-consistent memory object alone is insufficient for import. Snapshots
remain pinned if their source interview advances later.

Hash chains are consistency checks against a trusted anchor, not signatures,
encryption or authentication against a database owner who can rewrite both history
and the anchor. An export's embedded prior snapshots preserve declared external
anchors; verifying that one export alone is not independent authentication of all
its source databases. Keep separately trusted source heads/exports when that
provenance requirement matters.

## Resource and privacy limits

An interview's stored event JSON plus repeated command/digest metadata is limited
to 64 MiB and 100,000 events. Individual closed contracts, full state snapshots
and complete exports have the shared 16 MiB envelope cap. A valid bounded history
can exceed the smaller single-export cap; export then fails, never silently
truncates. There is no automatic retention deletion or event compaction.

A raw positive-completion envelope can fit its individual limit but exceed the
combined command/event/state limit once interpreted results and journal metadata
are included. Admission then fails atomically after invocation and the reservation
remains pending; ordinary resume still makes zero calls. Inspect the error and use
an explicit recovery/stop decision. A model's positive finish status does not
override storage bounds or imply that its result was committed.

Reads replay all events and revalidate requests, completions and source bindings.
Writes do that work again under the writer lock before appending. Cost therefore
scales with complete retained history and its domain validation/retrieval work,
not constant-time indexed access. Memory retains the event inventory, current
state and at most one selected historical state, not every historical snapshot.
SQLite/native buffers and Python object overhead are additional to serialized-byte
limits. These bounds are admission controls, not a hard total-RSS or time guarantee.

Journals, exports, contexts and provider completions contain private participant
text and model output. They are **not safe-to-publish redacted artifacts**. Only
provider configuration credentials and error strings are excluded at the described
boundaries; a participant or model can still place sensitive text in successful
content. Participant scoping is not user authentication, secure multi-tenancy,
encryption or a personal-data deletion policy. Applications own access control,
consent, storage protection and retention.
