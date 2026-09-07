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

Use `save_matrix(rows, path)` to write an authenticated artifact for CI or
review. The artifact includes rendered messages, row digests, tags, and an
inventory digest; `MatrixArtifact.load(path)` verifies both row content and the
inventory before returning it. The CLI accepts `matrix ... --artifact PATH` for
the same workflow.

Authenticated artifacts can be compared without re-rendering prompts:

```bash
promptwitness matrix-diff before-matrix.json after-matrix.json \
  --output matrix-diff.json
```

The command verifies both artifacts, rejects different prompt IDs, reports
added/removed/changed scenario digests, and exits with status 2 when any
scenario changed. This makes matrix rendering a CI-compatible regression gate.

`ScenarioExecutor` adds an explicit provider boundary for local or remote LLM
adapters. It executes rendered rows in stable input order, supports bounded
retries and workers, stores only output digests plus provider outputs, and
reports failures per scenario without leaking the original variable map.

For long-context retrieval checks, `make_needle_cases()` inserts a known needle
at a deterministic position in each context and `evaluate_long_context()` runs
an answerer without exposing the expected answer in the rendered prompt. The
report retains per-case failures and aggregates accuracy by context depth and
front/middle/back position, which makes lost-in-the-middle regressions visible.
The same report can be reproduced in CI with
`promptwitness long-context cases.json predictions.json --output report.json`,
where predictions is a case-ID-to-answer JSON object.
