# Durable conversation sessions

`SessionJournal` records a conversation as append-only SQLite events. The public
`SessionRunner` advances that conversation through provider responses and local
tool calls until the next user message, an unavailable handler, an interrupted
operation, an explicit finish, or the configured provider-turn limit. Each new
request includes the complete accumulated history, including function call IDs
and their corresponding tool results.

## Run an offline example

```bash
python examples/session_demo.py /tmp/support-session.sqlite
```

The example asks about an order, receives a declared function call, supplies an
order result, closes and reopens the database, and produces an assistant answer
from the persisted conversation. It uses a deterministic local provider so it
needs no credentials. Its final replay assertion verifies the journal without
invoking a provider or tool.

## Python workflow

```python
from promptwitness import (
    OpenAICompatibleProvider,
    OpenAISessionProvider,
    SessionJournal,
    SessionRunner,
    load_prompt,
)

transport = OpenAICompatibleProvider(
    "https://your-provider.example/v1/chat/completions",
    model="your-model",
    api_key_env="MODEL_API_KEY",
)
with SessionJournal("sessions.sqlite") as journal:
    state = journal.create(
        "support-17",
        load_prompt("prompt.json"),
        values={"customer": "Ada"},
        max_turns=8,
    )
    # A prompt ending with a user message is immediately ready for a provider.
    # A prompt ending with a system or assistant message waits for user input.
    if state.status == "waiting_user":
        state = journal.add_user(
            "support-17",
            "Where is my order?",
            expected_revision=state.revision,
        )
    state = SessionRunner(journal).run(
        "support-17",
        OpenAISessionProvider(transport),
        handlers={"lookup_order": lambda arguments: {"status": "shipped"}},
    )
    print(state.to_dict())
```

The prompt's `ToolSpec` parameters and required fields are validated before any
tool from a provider response executes. Every call must identify a declared tool,
have a unique nonempty call ID, and contain valid JSON-object arguments. Validation
checks the whole assistant response first, so an invalid second call cannot leave
the first call partially executed. Call IDs are unique across the session.

Handlers are explicit Python callables provided by the application. The session
never imports arbitrary code or dispatches a command based on a model response.
It executes multiple calls in declared order, records each result, and sends all
correlated results back in the next request. If a handler is unavailable, `run`
returns `ready_tool` before reserving or invoking it; the caller can attach a result
through the public reservation/completion API or resume with that handler later.

The first provider reservation pins a provider identity into the journal and
every request digest. `OpenAISessionProvider` declares the model, endpoint and
custom-header configuration digests, timeout, and API-key environment-variable
name. It checks these settings again immediately before sending. Credential values
are never recorded; request content and accepted responses are recorded. To switch
models or transport settings, create a new session so the provenance remains clear.

A plain Python provider callback must supply an explicit, JSON-compatible
`identity={"provider": "my-local-provider", "version": "2"}` argument to
`SessionRunner.run` or `SessionJournal.reserve_provider`. Include every model and
generation setting that changes its behavior, without placing credentials in the
identity. The journal cannot inspect arbitrary application callables to verify this
declaration. Recorded replay providers obtain their identity from the event log.
Manual CLI `respond` pins `{"provider": "manual"}` on an unbound session; subsequent
manual responses to a pending request retain that request's pinned identity.

## States, turns, and interrupted work

The usual lifecycle is:

```text
waiting_user → ready_provider → pending_provider
                              → waiting_user (assistant text)
                              → ready_tool → pending_tool → ready_provider
```

Each external request or tool operation has a reservation event committed **before**
the external call begins. A process crash leaves `pending_provider` or
`pending_tool`. The runner returns this state when reopened; it cannot infer whether
an interrupted tool already changed an external system. It therefore does not
repeat the operation. An exception records the error's class name and retains the
pending reservation; exception messages and request headers are not journaled.

Recovery uses `complete_provider(session_id, request_id, response,
expected_revision=...)` or `complete_tool(session_id, call_id, result,
expected_revision=...)` after the caller has obtained the result. A failed tool can
be resolved with an explicit JSON error result such as `{"error": "unavailable"}`
when appropriate for the application. `finish` stops further execution at any
nonterminal state. This design provides durable operation reservations, **not** an
exactly-once guarantee for external side effects.

`max_turns` counts accepted assistant responses, including responses requesting
tools. It is a lifetime bound across restarts and user messages. At that limit the
session ends with `reason="turn_limit"`; any tool calls in the final response are
retained in the transcript but are not executed. A turn-limited session is not
evidence that the user's task has been answered. Explicit `finish` events retain
their own caller-provided reason. Existing events are never replaced or deleted.

Each write requires the revision observed by its caller. The SQLite transaction
rejects a stale revision with `SessionConflict` before committing an event. Two
workers cannot reserve the same ready operation. Use one journal connection per
thread or process. A worker receiving a conflict must inspect the new state; do
not retry external effects automatically.

## CLI and explicit results

```bash
promptwitness session create sessions.sqlite support prompt.json --values values.json
promptwitness session show sessions.sqlite support
promptwitness session user sessions.sqlite support message.txt --revision 1
promptwitness session run sessions.sqlite support \
  --endpoint http://localhost:8000/v1/chat/completions --model local-model
```

CLI commands do not run local function handlers. When a provider requests a tool,
inspect the journal, obtain the tool result through your application, then run:

```bash
promptwitness session tool-result sessions.sqlite support call-17 result.json --revision 4
promptwitness session run sessions.sqlite support \
  --endpoint http://localhost:8000/v1/chat/completions --model local-model
```

Use the actual current revision and call ID from your session; the numbers above
illustrate one possible event sequence. `session respond ... response.json
--revision N` supplies an offline assistant response or resolves a pending provider
reservation. It accepts either a direct assistant message or a single-choice
OpenAI chat-completion response. Malformed responses leave the reservation pending.

`show` prints state and metadata; add `--messages` to print content. The pending
metadata includes request/call identifiers and any failure type, not tool arguments.
Exit code `2` means an external operation is pending and needs recovery, `1` means
invalid input or I/O failure, and `0` means the command completed. A status of
`ready_tool` requires a tool result even though no operation is yet pending.

```bash
promptwitness session export sessions.sqlite support support.events.json
promptwitness session replay support.events.json --messages
promptwitness session finish sessions.sqlite support --revision N --reason reviewed
```

Export creates a new file exclusively and refuses an existing destination. The
export includes the full conversation, prompt, tool arguments, results and raw
accepted provider responses: keep it under the same access controls as source
conversation data. It also includes the final checksum, enabling replay to detect
truncation against that recorded head. `SessionReplayProvider` and `session run
--replay-events ...` reuse a response only for an exact request digest, including
session ID, request ID, provider/model settings, history and tool contracts.
Reproducing a run requires the
same session identity, creation settings, user messages and supplied tool results;
the replay provider does not execute tools.

## Integrity, resources, and scope

Each event checksum covers its session, sequence, kind, payload and previous
checksum. Reads verify the chain, event transitions and SQLite head before returning
state. SQLite triggers reject ordinary updates or deletes to event rows. This is
integrity checking, not a cryptographic signature: a database owner who rewrites
both events and the head can manufacture a new history. Keep a trusted exported
head digest independently when provenance across an untrusted boundary matters.

`max_tool_calls` limits each assistant response; `max_event_bytes` bounds canonical
JSON payload size; `max_context_bytes` rejects an oversized complete request before
reserving it. Defaults are 8 calls, 1 MiB per event and 4 MiB per request. These are
byte limits, not model tokenizer limits. Validation and replay run in time linear
in the event log plus its JSON content; reopening/committing reconstructs state
from the log, so long histories have cumulative replay cost. Runtime memory holds
the history and provider response; HTTP transport currently reads a response before
the journal applies its payload limit. Handler timeouts remain the application's
responsibility; only the supplied HTTP transport has a request timeout.

The session protocol currently supports text assistant messages and OpenAI-style
function calls. Existing multimodal initial prompt blocks remain intact when sent
to the provider. Hosted batch APIs, streaming tool-call assembly, parallel agent
agendas, persistent vector memory, speech, and a multi-user web application remain
outside this session feature. It does not establish whole-repository equivalence
with an interview-agent research system.
