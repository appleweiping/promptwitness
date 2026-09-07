# Variable scenario matrices

`render_matrix` applies one immutable `PromptDocument` to a named sequence of
variable scenarios. Each row records rendered messages, observed variable names,
tags, and a SHA-256 digest that excludes the variable values themselves. This
makes the result suitable for regression evidence without copying secrets into
reports.

```python
from promptwitness import Scenario, render_matrix

rows = render_matrix(
    prompt,
    [Scenario("baseline", {"name": "Ada"}), Scenario("empty", {"name": ""})],
)
```

`compare_matrices` reports added, removed, and changed scenario digests. Rendering
is local and side-effect free; it never calls a provider or evaluates template
expressions.
