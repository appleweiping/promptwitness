## Summary

Describe the user-visible change and why it is needed.

## Verification

- [ ] Tests cover lossy conversion, validation, alignment, policy, report, and unsafe path boundaries.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest` pass.
- [ ] SARIF validates and wheel/sdist isolated-install smoke checks pass.
- [ ] Documentation, benchmark evidence, and changelog are updated when behavior changes.
- [ ] No private prompts, credentials, tool results, or generated build artifacts are included.

## Compatibility and risk

Describe prompt schema, provider conversion loss, alignment, policy, SARIF, performance, and migration implications.
