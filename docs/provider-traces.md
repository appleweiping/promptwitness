# Provider traces and replay

`TraceRecorder` wraps any callable provider accepted by `ScenarioExecutor`. It
records request, response, and conventionally declared `tool_calls` events using
digests rather than headers or raw variable assignments. `ReplayProvider` can
then replay a response by rendered prompt digest for deterministic regression
tests.

```python
from promptwitness import ReplayProvider, TraceRecorder

recorded = TraceRecorder(provider)
report = executor.run(document, scenarios, recorded)
recorded.save("provider-traces.json")
replay = ReplayProvider({trace.prompt_digest: trace.output for trace in recorded.traces})
```

`load_traces()` authenticates the saved output and event digests before making
the traces available for review or replay.

The trace file is an audit inventory, not a cryptographic signature from an
external provider. Keep outputs and trace files under the same access controls as
the original provider responses.

For local OpenAI-compatible gateways, `OpenAICompatibleProvider` sends rendered
messages using `urllib`, reads an optional bearer token from an environment
variable, and never returns that token in the response object or trace.

For gateways that support server-sent events, use
`OpenAICompatibleStreamingProvider`. Its `stream(row)` iterator parses only
`data:` JSON events, ignores keep-alives, stops at `[DONE]`, and rejects
malformed UTF-8/JSON or non-object chunks. Calling the provider aggregates text
content deltas and includes the decoded `stream_chunks` so a replay fixture can
audit the exact chunk sequence without retaining request headers or API keys.

`ToolDispatcher` handles a provider response's `tool_calls` array under explicit
call-count and result-size limits. Arguments are parsed as JSON objects, unknown
tools and handler exceptions become per-call outcomes, and successful results
can be converted to follow-up tool messages through `ToolBatch.messages()`.
Only argument/result digests are required for audit storage; callers should
apply their own authorization before registering side-effecting handlers.
