# PromptWitness

PromptWitness makes chat-prompt changes reviewable. It parses a small, versioned
JSON format, validates operational hazards, compares template variables and
tool contracts, and emits deterministic JSON, Markdown, or standalone HTML for
CI and code review.

It does **not** score prompt quality, call an LLM, or claim that a structurally
valid prompt is safe. The narrow goal is to catch changes such as “a new render
variable is now required” or “a tool parameter disappeared” before deployment.

[![CI](https://github.com/appleweiping/promptwitness/actions/workflows/ci.yml/badge.svg)](https://github.com/appleweiping/promptwitness/actions/workflows/ci.yml)
[![CodeQL](https://github.com/appleweiping/promptwitness/actions/workflows/codeql.yml/badge.svg)](https://github.com/appleweiping/promptwitness/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/appleweiping/promptwitness/badge)](https://scorecard.dev/viewer/?uri=github.com/appleweiping/promptwitness)
[![Release](https://img.shields.io/github/v/release/appleweiping/promptwitness?sort=semver)](https://github.com/appleweiping/promptwitness/releases)
[![Python](https://img.shields.io/badge/python-3.10--3.14-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What it catches

| Change | Default impact | Why |
|---|---|---|
| New `{{ variable }}` | breaking | Existing render callers do not provide it |
| Removed variable | warning | Callers may still send an unused value |
| Message role/name change or removal | breaking | Conversation structure changed |
| Message content change/addition | warning | Behavior may change, but the render API may not |
| Tool removal or parameter schema change | breaking | The executor contract can no longer match |
| Optional tool parameter addition | info | Existing calls remain representable |
| Required tool parameter addition | breaking | Existing calls become incomplete |
| Metadata or tool-description change | info | No structural invocation change |

The severities are an explicit compatibility policy, not universal truth.
`DiffOptions` lets a caller tighten message-change handling, and the CLI's
`--fail-on` chooses the CI threshold.

Provider adapters convert documented subsets of OpenAI, Anthropic, and LangChain
payloads into the same native model. Conversion is loss-aware: unsupported fields
produce warnings, while multimodal or otherwise unrepresentable content fails.

## Install

PromptWitness has no runtime dependencies.

Install the latest source from GitHub:

```bash
python -m pip install "git+https://github.com/appleweiping/promptwitness.git"
```

For a source checkout:

```bash
python -m pip install -e ".[dev]"
```

## Sixty-second demo

Validate a prompt, then compare two revisions:

```bash
promptwitness validate examples/before.json
promptwitness diff examples/before.json examples/after.json
```

The diff exits with status `2` because it finds breaking changes and prints:

![PromptWitness HTML report generated from the included example](docs/demo-report.png)

```text
# PromptWitness diff

`order-assistant-v1` → `order-assistant-v2` · **breaking** · 5 change(s), 2 breaking

| Severity | Kind | Path | Summary |
|---|---|---|---|
| warning | `message_content_changed` | `/messages/0/content` | message content changed |
| breaking | `variable_added` | `/variables/locale` | new render input 'locale' is required |
| info | `tool_description_changed` | `/tools/lookup_order/description` | tool description changed |
| breaking | `tool_parameter_removed` | `/tools/lookup_order/parameters/include_history` | tool parameter 'include_history' was removed |
| info | `metadata_changed` | `/metadata` | prompt metadata changed |
```

Generate a review artifact that opens without a server or CDN:

```bash
promptwitness diff examples/before.json examples/after.json \
  --format html --output promptwitness-report.html --fail-on never
```

For prompts with inserted or removed messages, opt into structural alignment to
avoid positional cascade noise:

```bash
promptwitness diff examples/before.json examples/after.json \
  --message-alignment smart
```

## Prompt format

Schema version 1 is intentionally small and portable:

```json
{
  "schema_version": 1,
  "id": "order-assistant-v1",
  "messages": [
    {"role": "system", "content": "Help {{ customer_name }}."},
    {"role": "user", "content": "Check {{ order_id }}."}
  ],
  "tools": [
    {
      "name": "lookup_order",
      "description": "Look up an order.",
      "parameters": {"order_id": {"type": "string"}},
      "required": ["order_id"]
    }
  ],
  "metadata": {"owner": "support"}
}
```

Unknown fields, duplicate object keys, and non-finite JSON numbers are rejected
so misspellings or parser-specific values cannot silently change meaning.
Tool parameter schemas support an intentionally conservative JSON-Schema-like
subset during validation: each parameter is an object with a recognized
`type`, and may include a non-empty `enum`. PromptWitness preserves additional
schema keys for comparison without claiming to implement full JSON Schema.
See [the format reference](docs/prompt-format.md).

## CLI

```text
promptwitness validate PROMPT [--require-system-first|--no-require-system-first]
                            [--allow-empty-content|--no-allow-empty-content]
                            [--secret-scan|--no-secret-scan] [--policy POLICY]
                            [--from-format native|openai|anthropic|langchain|auto]
                            [--format json|markdown|html|sarif] [--output PATH]
                            [--fail-on never|warning|breaking]

promptwitness diff BEFORE AFTER [--include-metadata|--ignore-metadata]
                              [--before-format native|openai|anthropic|langchain|auto]
                              [--after-format native|openai|anthropic|langchain|auto]
                              [--context-lines N]
                              [--message-alignment positional|smart]
                              [--policy POLICY]
                              [--format json|markdown|html|sarif] [--output PATH]
                              [--fail-on never|warning|breaking]

promptwitness convert PROVIDER.json [--from-format auto|native|openai|anthropic|langchain]
                                  [--id PROMPT_ID] [--output native.json]
```

Exit codes are stable and automation-friendly:

- `0`: the report did not reach the configured threshold;
- `1`: the input or output could not be processed;
- `2`: a valid report reached `--fail-on` (default: `breaking`).

Use JSON for machines, Markdown for pull requests, HTML for a portable visual
review, and SARIF 2.1.0 for code-scanning interfaces. All formats are generated
from the same typed report.

## Provider adapters

`promptwitness convert` produces deterministic native schema-version-1 JSON.
`validate` and `diff` can also consume provider formats directly. Adapter warnings
go to stderr and enumerate source fields that were not represented; they are never
hidden inside a successful report. See [provider adapters](docs/provider-adapters.md)
for the exact supported subsets and loss rules.

## Policy files

A versioned JSON policy can declare allowed roles, validation toggles, alignment,
context lines, and a severity override for any stable `ChangeKind`. CLI flags take
precedence where both are supplied.

```bash
promptwitness diff released.json current.json \
  --policy examples/strict-policy.json --format sarif --output prompt.sarif
```

See [policy files](docs/policies.md). Keep policies in source control and review
their changes together with prompt changes.

## Python API

```python
from promptwitness import DiffOptions, Severity, compare_prompts, load_prompt
from promptwitness.reporting import render_json

before = load_prompt("examples/before.json")
after = load_prompt("examples/after.json")
report = compare_prompts(
    before,
    after,
    DiffOptions(content_change_severity=Severity.BREAKING),
)

print(render_json(report))
if not report.compatible:
    raise SystemExit("prompt contract changed")
```

Template rendering is deliberately literal—there are no expressions, filters,
attribute access, or code execution:

```python
from promptwitness.variables import render_template

rendered = render_template("Hello {{ name }}", {"name": "Ada"})
```

## Architecture

```mermaid
flowchart LR
    A[Versioned JSON] --> B[Strict parser]
    B --> C[Typed prompt model]
    C --> D[Static validator]
    C --> K[Deterministic message alignment]
    K --> E[Structural diff]
    L[Provider JSON] --> M[Loss-aware adapters]
    M --> C
    N[Versioned policy] --> D
    N --> E
    E --> F[Compatibility policy]
    D --> G[Typed findings]
    F --> H[Typed changes]
    G --> I[JSON / Markdown / HTML / SARIF]
    H --> I
    I --> J[Human review or CI gate]
```

The parser owns format errors; validation owns suspicious-but-parseable
content; diffing owns compatibility semantics; reporting has no policy logic.
Read [the architecture notes](docs/architecture.md) for determinism and trust
boundaries.

## CI example

```yaml
- name: Check prompt compatibility
  run: |
    promptwitness validate prompts/current.json --format json
    promptwitness diff prompts/released.json prompts/current.json \
      --policy prompt-policy.json --format sarif --output prompt-diff.sarif
```

If warning-level review is mandatory, add `--fail-on warning`. For report-only
jobs, use `--fail-on never`.

## Security and limitations

- The credential detector is a small local heuristic. It can miss secrets and
  can flag harmless examples; it is not a secret scanner.
- Variable parsing supports only `{{ identifier }}` with letters, digits,
  `_`, `.`, and `-`. It intentionally does not evaluate a template language.
- Positional message comparison remains available and is the default for backward
  compatibility. Smart alignment uses content/role/name similarity but remains a
  heuristic; review the explicit added, removed, and changed records.
- Smart alignment evaluates a quadratic grid of message pairs and uses approximately
  one byte per pair for backtracking. Each pair also runs text similarity, so very
  long message contents add their own cost; use positional mode for large transcripts.
- Tool schemas are compared structurally, not resolved through `$ref`, coercion,
  or model-provider extensions.
- Compatibility does not predict model behavior. Evaluation data and human
  review remain necessary.
- HTML output escapes report text and has no external resources, but should
  still be treated as an artifact derived from untrusted input.
- `--output` refuses to overwrite any prompt or policy input, including through
  resolved path aliases or existing hard links.

Please report vulnerabilities privately using [SECURITY.md](SECURITY.md).

## Development

```bash
ruff check .
ruff format --check .
mypy src
pytest
python -m build
```

The test suite uses branch coverage and enforces a 90% minimum. A reproducible
synthetic workload is available in [`benchmarks/`](benchmarks/README.md).

## Roadmap

- stable message identifiers in a future native schema;
- optional JSON Schema resolution for tool contracts;
- richer multimodal prompt modeling without silently flattening non-text content.

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
please include tests for every new compatibility rule.

## License

MIT. See [LICENSE](LICENSE).
