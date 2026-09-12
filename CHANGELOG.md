# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- Add closed, immutable six-family procedure cases, all 16 local LongProc data
  definitions, bounded strict JSON ingestion, source inventories and explicit
  filtering/sampling accounting. Keep reference fields out of provider messages.
- Add original bounded arithmetic/path/travel validators, multiplicity-aware TSV
  scoring and declared belief-trace comparison. Preserve invalid references as
  diagnostics; report code execution and semantic/search judges as unsupported.
- Extend durable task execution with a separately versioned procedure plan and
  report, per-task generation, input-byte budgets and pinned scoring limits.
  Preserve the historical seven-family format and explicit retry/CAS semantics.
- Freeze each built-in task provider's transport configuration for its call and
  reject unverifiable or changed post-call provider identity without attributing
  a result to the old model. Uncertain reservations require explicit recovery.
- Add input-only integer-search and directed-walk baselines, a fixed full-data
  benchmark protocol, private-by-default procedure CLI summaries and an authored
  offline workflow demonstration. These are not real-model accuracy claims.
- Extend closed-stdout-safe Python 3.14 parser/formatter setup to the procedural
  benchmark and demonstration scripts. Retain publication tests on every platform
  and rerun the unchanged full-data protocol against the corrected script.

- Keep parser and formatter help plain on Python 3.14 as on earlier versions,
  avoiding its startup color probe against already-closed host output streams.
- Preserve completed interview command status when an embedding host closes its
  text output streams, including `ValueError` from closed Python stream objects.

- Connect evidence-linked interviews through a deterministic state reducer,
  single-database event/head/memory transactions and a sequential provider runner.
  Preserve complete raw responses for replay, scope historical recall, distinguish
  newly applied reservations from historical receipts, and require explicit retry.
- Add interview CLI workflows and an original scripted multi-interview demo;
  these verify engineering behavior, not real-model interview quality.
- Reject foreign hot SQLite journals before connection, validate complete owned
  schema/guards, and prevent known database/sidecar aliases before opening.
- Share exact provider JSON preparation with interview request budgets and pin
  a fresh transport configuration snapshot for each interview invocation.

- Bound direct provider HTTP(S) requests/responses and framing, disable automatic
  redirects/environment proxies, enforce exchange deadlines after DNS, and retain
  normal TLS verification with redacted transport exceptions.
- Preserve actual streamed finish reasons instead of fabricating `stop`; reject
  non-text/multi-choice aggregates and pin the corrected v2 transport identity.
- Add strict interview plan, answer/citation, assessed-coverage and source-linked
  memory contracts plus scoped deterministic lexical BM25 retrieval. These are
  usable independently of the durable runner, not a semantic index or research
  quality claim. Keep original and explicitly accepted emergent coverage separate.
- Include public localhost TLS test fixtures and the existing aggregate fixture
  benchmark report in source distributions.

- Add integrated seven-family task-suite planning, local dataset adapters,
  complete-prompt byte/custom-token budgets, real provider execution, specific
  retrieval/classification metrics, and explicit unsupported model-judge metrics.
- Add model/input/configuration-bound SQLite task resumption, committed call
  reservations, explicit retries, stale-writer protection, score/event integrity
  checks, per-length/family coverage, and a tested offline multi-stage example.

- Add durable multi-turn conversation sessions with pinned rendered prompts,
  hash-linked SQLite events, optimistic concurrency, validated function tools,
  correlated result feedback, bounded provider turns, and explicit crash recovery.
- Add `session` create/user/run/respond/tool-result/show/export/replay/finish CLI
  commands, exact-request provider replay, and complete OpenAI-compatible tool
  history transport.
- Enforce references inside frozen schema combinators, recursive JSON equality
  for `enum`/`const`, and conjunctive `$ref` sibling assertions before tool dispatch.
  This corrects the previously documented override semantics for `$ref` siblings.

- Add explicit benchmark scorers (`exact`, `contains`, `token_f1`, and `json`)
  with per-case thresholds, continuous score reporting, and scorer-aware
  prompt digests.
- Add strict Gemini/Vertex `generateContent` prompt adaptation, including
  system instructions, multimodal parts, model-role normalization, and
  function declarations.
- Preserve provider-native string message identifiers across OpenAI, Anthropic,
  Gemini, and LangChain adapters for ID-based alignment.
- Preserve Anthropic image, document, tool-use, and tool-result content blocks
  through the common structured multimodal representation.
- Add strict OpenAI Responses API adaptation for `input`/`instructions`,
  message identifiers, function-call events, and top-level function tools.
- Preserve structured multimodal content blocks across adapters, native JSON,
  rendering, diffs, registry replay, and provider requests.
- Add a strict loopback JSON/HTTP service for validation, diff, and tool-call checks.
- Allow the service boundary to validate and diff provider-native adapter formats.
- Expose provider-to-native conversion and adapter warnings through the loopback service.
- Expose authenticated scenario matrix rendering and matrix-diff regression
  reports through the loopback service.
- Expose replayable long-context needle evaluation through the loopback service.
- Expose deterministic replay-provider matrix execution and pairwise agreement
  reporting through the loopback service.

- Add bounded pre-dispatch tool-call argument validation through `check-call` and `validate_tool_arguments`.

- Add opt-in local JSON Schema `$ref` resolution for tool-parameter diffs,
  with cycle and external-reference rejection.

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
