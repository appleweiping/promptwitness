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

The trace file is an audit inventory, not a cryptographic signature from an
external provider. Keep outputs and trace files under the same access controls as
the original provider responses.
