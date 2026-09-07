# Long-context provider evaluation

PromptWitness includes deterministic needle cases and reports accuracy by
context depth and needle position. `evaluate_long_context()` accepts a simple
answerer callback, while `make_provider_answerer()` adapts an
OpenAI-compatible provider callback to the same interface:

```python
from promptwitness import (
    OpenAICompatibleProvider,
    evaluate_long_context,
    make_needle_cases,
    make_provider_answerer,
)

cases = make_needle_cases(
    (("background A", "background B"),),
    needle="Ada",
    query="Who is named?",
    expected="Ada",
)
provider = OpenAICompatibleProvider("http://127.0.0.1:8000/v1/chat/completions")
report = evaluate_long_context(cases, make_provider_answerer(provider))
```

The adapter sends only the rendered context/question as a user message. It
extracts `choices[0].message.content` or `output_text`, rejects ambiguous
responses, and never puts the expected answer into the provider payload or
prompt digest. Failures remain visible in the report unless `strict=True` is
requested.
