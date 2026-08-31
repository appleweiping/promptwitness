# Contributing

Thank you for improving PromptWitness. Open an issue before a large format or
compatibility-policy change so the behavior can be agreed independently of its
implementation.

## Local workflow

1. Create a focused branch and virtual environment.
2. Install with `python -m pip install -e ".[dev]"`.
3. Add tests that fail without the intended change.
4. Run `ruff check .`, `ruff format --check .`, `mypy src`, `pytest`, and
   `python -m build`.
5. Update the format or architecture documentation when a public contract
   changes, and add an entry under `Unreleased` in `CHANGELOG.md`.

Keep the runtime dependency-free unless a proposal demonstrates that a new
dependency is essential. Never add real credentials or confidential prompts to
tests, examples, issues, or commit history.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
