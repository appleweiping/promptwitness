# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
