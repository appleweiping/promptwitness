# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- Add optional stable native message identifiers and explicit ID-based diff
  alignment for reorder-safe prompt compatibility checks.

- Add overall and per-task accuracy gates to the versioned benchmark-suite CLI.
- Add replayable provider wrappers and secret-safe request/tool/response event traces.
- Authenticate provider-trace loading and reject tampered outputs or event inventories.
- Add a dependency-free OpenAI-compatible HTTP provider with environment-based key handling.
- Add an OpenAI-compatible SSE streaming provider with deterministic text aggregation
  and retained non-secret chunk audit data.
- Add bounded local tool-call dispatch with per-call digests, isolated failures,
  and provider-compatible tool messages.
- Add deterministic needle-in-context long-context evaluation with depth and
  front/middle/back diagnostics.
- Add a `promptwitness long-context` CLI gate for replaying recorded answers
  against serialized needle cases.
- Add task-oriented benchmark case loading, per-task replay evaluation, and a
  `promptwitness benchmark` CLI with expected-answer-safe digests.
- Add strict versioned multi-task benchmark suites with globally unique case IDs,
  suite digests, task selection, and a `promptwitness benchmark-suite` CLI gate.
- Add replay-provider matrices with ordered concurrent execution, output-digest
  agreement diagnostics, strict provider manifests, and a `provider-matrix` CLI.
- Add an OpenAI-compatible long-context provider adapter that extracts common
  response shapes without leaking expected answers into prompts or digests.

### Added

- Transactional SQLite prompt version registry with content digests, optimistic concurrency,
  review annotations, stored-version diffs, and deterministic variable replay.
- Variable scenario rendering matrices with secret-safe digests and regression diffs.
- Authenticated, persistable rendering-matrix artifacts with optional CLI emission.
- Ordered scenario execution with provider boundaries, bounded retries, and per-row failure reports.

## [0.2.0] - 2026-08-31

### Added

- Add loss-aware OpenAI, Anthropic, and LangChain prompt adapters and a `convert` command.
- Add deterministic smart message alignment with memory-efficient backtracking.
- Add strict versioned validation/diff policy files and per-change severity overrides.
- Add SARIF 2.1.0 output with stable rule IDs and fingerprints.
- Add provider/policy examples, documentation, and a reproducible synthetic benchmark.

## [0.1.0] - 2026-08-31

### Added

- Strict prompt schema version 1 parser and immutable models.
- Static validation for roles, templates, tool schemas, and secret-like literals.
- Structural message, variable, tool, and metadata compatibility reports.
- Deterministic JSON, Markdown, and standalone HTML output.
- Dependency-free Python API and CI-oriented command line interface.
