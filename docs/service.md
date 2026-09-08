# PromptWitness local service

`PromptService` exposes prompt validation, compatibility diffs, and concrete
tool-call checks through a small JSON dispatch API. `create_server()` provides
a loopback-first `POST /v1/dispatch` endpoint and returns deterministic JSON.

```python
from promptwitness import PromptService

result = PromptService().dispatch({
    "operation": "check_call",
    "prompt": "examples/before.json",
    "tool": "lookup_order",
    "arguments": {"order_id": "A-1"},
})
```

The service never calls a model or provider. Keep the default loopback binding
when using it as a local CI or editor integration.
