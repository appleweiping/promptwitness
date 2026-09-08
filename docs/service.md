# PromptWitness local service

`PromptService` exposes prompt validation, compatibility diffs, and concrete
tool-call checks through a small JSON dispatch API. `create_server()` provides
a loopback-first `POST /v1/dispatch` endpoint and returns deterministic JSON.

```python
from promptwitness import PromptService

result = PromptService().dispatch(
    {
        "operation": "check_call",
        "prompt": "examples/before.json",
        "tool": "lookup_order",
        "arguments": {"order_id": "A-1"},
    }
)
```

The service never calls a model or provider. Keep the default loopback binding
when using it as a local CI or editor integration.

Set `from_format` for validation or `before_format`/`after_format` for diffs to
`openai`, `anthropic`, `gemini`, or `langchain` when the files are not native
schema-v1 documents. Conversion warnings remain explicit in the adapter API; the
service returns the same strict document findings.

`convert` exposes the same provider-to-native conversion boundary for automation.
It accepts `prompt`, a required `from_format`, and an optional `prompt_id`, then
returns the strict schema-v1 document plus explicit loss warnings.

`matrix` renders a JSON scenario array with deterministic variable substitution,
optionally persists an authenticated artifact, and returns row digests and
rendered messages. `matrix_diff` verifies two artifacts and returns added,
removed, and changed scenario digests without re-rendering a provider prompt.
