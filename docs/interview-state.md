# Pure interview state and scheduling

This development API reduces an append-only interview event sequence into an
immutable state. It does not call a model, write files or a database, choose IDs,
read the clock, or perform random sampling. Pure reducer tests establish causal
and structural behavior, not interview usefulness or semantic accuracy.

The public functions in `promptwitness.interview_state` are `make_event`,
`apply_event`, `replay_interview`, `schedule`, `interview_report`, and
`current_memory_snapshot`. The journal must authenticate imported source heads,
atomically persist events and memory-index additions, and return historical
command receipts. This module never writes a second memory store.

## Commands and event heads

`InterviewCommand` has `interview_id`, `command_id`, `expected_revision`,
`expected_digest`, `kind`, and a closed kind-specific JSON `payload`. Creation
requires revision `0` and digest `None`; subsequent commands name the exact
current event head. `make_event(state, command)` validates the transition without
changing the input. `apply_event(state, event)` returns an `InterviewTransition`
with a new `state`, its derived `action`, and a tuple of `memory_additions`.

The event stores its sequence, previous digest, command ID and complete command
digest, kind, payload, and its own canonical SHA-256 digest. An answer's revision
is the sequence of its `answer_committed` event, not a separate turn counter.
`state.digest` is the event-head digest; it is deliberately not a hash of the
derived state cache. `state.to_dict()` is a complete exported view, not a trusted
state-import format. Recover by replaying the events.

Every command/event/state JSON envelope has the existing **16 MiB encoded
canonical UTF-8 limit**, including envelope overhead. Values are closed JSON,
deeply snapshotted, and finite; Boolean values cannot stand in for revisions.
Exceeding a limit rejects the whole transition. This is not a Python heap/RSS
limit. Replay accepts a bounded list/tuple of at most 100,000 events.

```python
from promptwitness.interview_state import apply_event, make_event, replay_interview

event = make_event(None, creation_command)
transition = apply_event(None, event)
state = transition.state
restored = replay_interview([event.to_dict()], expected_head=state.head)
assert restored.to_dict() == state.to_dict()
```

`expected_head` is a `SourceHead` or its exact
`{interview_id, revision, digest}` dictionary. A trusted expected head detects
valid suffix truncation in addition to internal corruption. Without an external
trusted head, an internally valid prefix cannot be distinguished from a complete
history. Hashes do not authenticate a writer or detect an attacker replacing the
whole history and its trusted anchor.

A repeated command ID in a replay log is invalid, even when the input matches.
Historical idempotence belongs to the journal: the same command returns its
original receipt/state only when its full input matches, before current-head CAS.
Changing the input under the same ID must conflict.

## Creation and exact payloads

The `created` payload has exactly:

- `participant_id`, `plan` (the complete `InterviewPlan.to_dict()`);
- `provider_identities`: nonempty identity objects for `analyst` and `questioner`;
- `template_hashes`: the actual `interview_stages.TEMPLATE_HASHES`;
- `renderer_version`: `interview_stages.RENDERER_VERSION`;
- `scheduler_version`: `scheduler/v1`;
- `memory_snapshot`: the original same-participant `MemorySnapshot.to_dict()`.

Creation allows at most 127 prior source heads, leaving one of the snapshot's
128 head slots for this interview. The source snapshot cannot include the
interview being created. Imported memory
never silently marks a criterion covered. Identity objects are caller-supplied
nonsecret configuration; the reducer does not discover credentials or authenticate
their owners. Template hashes and supported versions must match the actual
request renderer, not arbitrary caller declarations.

| Event kind | Exact payload fields |
| --- | --- |
| `question_reserved`, `analysis_reserved` | `request` (complete stage request) |
| `question_committed` | `request_id`, `attempt_id`, `request_digest`, `completion`, `result`, `question_id` |
| `analysis_committed` | `request_id`, `attempt_id`, `request_digest`, `completion`, `result` |
| `answer_committed` | `question_id`, `answer_id`, `text` |
| `question_skipped` | `question_id`, `scope` (`criterion` or `topic`) |
| `stage_failed` | `request_id`, `attempt_id`, `request_digest`, `error_code` |
| `retry_authorized` | `request` (a new complete request for the same stage) |
| `proposal_decided` | `proposal_id`, `decision` (`accepted` or `rejected`) |
| `finished` | `reason` |

`completion` is the original complete provider response envelope, not a bare text
string, reconstructed response, or completion-record wrapper. The reducer reruns
the stage completion parser, including positive transport-completion validation,
and compares its normalized output to `result` with exact canonical JSON equality.
Both raw completion and result remain in the event so replay repeats this check.
Production callers must not synthesize a successful finish reason. Scripted test
responses are explicitly authored contract fixtures, not actual model evidence.

Error codes are `provider_error`, `invalid_response`, `deadline_exceeded`,
`interrupted`, and `response_incomplete`. Raw exception strings are not accepted
as error codes. Full user/provider content in successful events, including
provider metadata and raw completion, is not redacted. These exports are private
data, not safe public reports; callers must handle retention and access appropriately.

## Reservation, completion and explicit retry

A request binds the head **before reservation**, target, original plan, accepted
agenda, actual source answers and questions, provider identity, rendered template,
generation settings, selected memory, and unique request/attempt IDs. A completion
must CAS against the head **after reservation** and match all three request
identity fields. Only one stage may be pending. Participant input and agenda
decisions cannot interleave with an active stage.

On uncertain execution or a recorded stage failure, `retry_authorized` is one
atomic event: it replaces the old reservation with a new request at the current
head and consumes another reservation. It is **not** a ready-state event followed
by another reservation. Both the request ID and attempt ID must be new. Any late
completion from the old attempt is rejected. The pure result is `await_completion`,
not an instruction to automatically call a provider.

The runner may invoke only a newly committed reservation that it owns. Reading
an existing pending reservation after a restart must perform zero automatic
calls; explicit retry authorization is required. These rules cannot establish
external exactly-once execution, determine billing, or undo already performed
provider effects.

`stage_reservations` conservatively consumes the plan's `provider_calls` budget.
`stage_completions` counts validated committed responses. Reports deliberately
return `provider_call_count: null`: a reservation can precede a crash before any
call, and a failed/uncertain call may have incurred a charge.

## Deterministic scheduler/v1

Each topic permits at most `1 + followups_per_topic` **published questions**.
Failed attempts use provider budget but do not publish questions or consume a
topic question slot. Prefer the current unresolved, unskipped criterion, then
remaining criteria of that topic; next use original topic/criterion order,
then accepted emergent topics in acceptance order. Question modes cannot bypass
eligibility or budgets. Recall is indicated by actually selected memory IDs,
not a separate scheduling override.

An answer or explicit skip consumes exactly one participant turn. A topic skip
can make several criteria ineligible but does not erase assessments or change
the original denominator. The literal answer `skip` is ordinary participant
text. The final allowed answer still reaches analysis before the participant
budget stopping rule, provided one provider reservation remains.

Pending proposals yield `await_review` before any next question is reserved.
Accepted emergent topics use a separate coverage denominator. Pending/rejected
proposals never count toward coverage. Proposal IDs, topic/criterion IDs and
whitespace-collapsed/casefolded topic descriptions cannot duplicate existing
items. Published questions use the same normalized-exact text comparison while
retaining their original text. Semantic novelty and semantic deduplication are
not implemented.

Analysis parsing and all actual-answer evidence revalidation complete before any
assessment, memory or proposal becomes visible. New positive evidence is limited
to the actual current answer included in the request. Prior memory, summaries,
hidden answers and hypothetical exchanges cannot become observed answer evidence.
Model analysis cannot create memory corrections (`supersedes` is not accepted by
the stage contract). Imported correction ancestry still needs journal
authentication; a reviewed correction command is outside this slice.

Before returning new memories, the reducer also validates the complete combined
prior-plus-current recall snapshot. Its existing 10,000-entry and 16 MiB envelope
bounds can reject an analysis before the plan's turn budget is reached. Event and
state audit bounds likewise mean a configured turn count is a maximum, not a
promise that arbitrary answer sizes can execute that many turns. Failure returns
no partial transition; there is no silent deletion of old memory, answer
compression, truncation, or automatic provider retry.

Cumulative assessment ranks are `covered > partial > unanswered`. A weaker new
proposal does not silently revoke a stronger existing assessment; same-rank or
stronger evidence may replace it. The full new proposal remains in the event for
review. This is an explicit accumulation policy, not a truth test. Reviewed
retraction/downgrade commands are not implemented in this slice.

## Termination and honest reports

- `agenda_completed`: all original required criteria are assessed covered, with
  every accepted emergent gap explicitly skipped, and no pending answer,
  analysis, stage, or proposal.
- `unresolved`: original required gaps remain but skips/topic caps leave no
  eligible work.
- `budget_exhausted`: work remains but the relevant turn/reservation budget is
  exhausted. An unanalysed final answer remains in the report.
- `participant_stopped`: explicit stop, including cancelling uncertain work.
- `failed`: explicit termination after a recorded stage failure.

Required coverage being complete does not bypass accepted emergent work. A topic
question cap leaving an unskipped emergent gap yields `budget_exhausted`, not
completion. Explicitly skipped emergent gaps may remain visible when the original
agenda qualifies as completed. Reports show assessed coverage rather
than factual accuracy, skips, deferrals, proposals, pending/failed work, and
unanalysed answer IDs. A finished state accepts no further events.

This pure component is not a deployed service, a completed autonomous interview
product, semantic memory retrieval, or evidence of reference-repository parity.
