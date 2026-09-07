# Prompt version registry and replay

`PromptRegistry` is a local SQLite store for native PromptWitness documents. It
appends immutable integer versions, stores a canonical JSON digest, records
optional human-readable approvals, compares stored versions with the normal diff
engine, and renders message variables without calling a model or evaluating
expressions.

```python
from promptwitness import PromptRegistry, load_prompt

with PromptRegistry("prompts.sqlite") as registry:
    version = registry.put(load_prompt("prompt.json"))
    registry.approve(version.prompt_id, version.version, "reviewer@example.test")
    rendered = registry.replay(version.prompt_id, {"customer_name": "Ada"})
```

`put` uses optimistic concurrency when `expected_previous_digest` is supplied.
The check and insert happen in one SQLite transaction. Versions never overwrite
one another. `get` returns a newly parsed document, not a mutable reference to a
stored object. SQLite parameter binding is used for all IDs and content.

Replay is deliberately small: it substitutes supported `{{ variable }}` names in
message text and preserves the stored tool contracts. Strict mode rejects missing
variables; loose mode leaves them untouched. It does not execute Jinja, Python,
tools, provider calls, or model inference. Variables in tool schemas are not
rendered. Approval is an audit annotation, not a cryptographic signature or
authorization decision; the actor string must be independently authenticated by
the surrounding workflow.

The registry rejects existing SQLite databases with an unknown schema version and
does not migrate them implicitly. Back up the database before concurrent external
maintenance. This module does not provide a remote registry, RBAC, secrets store,
prompt quality score, or experiment/evaluation runner; those remain separate
capabilities in larger prompt research systems.
