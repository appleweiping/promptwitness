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
