# Architecture and guarantees

PromptWitness is organized around one rule: parsing, validation, compatibility,
and presentation must remain separable. A consumer can adopt one layer without
accepting hidden network calls or model-specific behavior.

## Data flow

1. `parser.py` accepts UTF-8 JSON, rejects unknown fields, and creates immutable
   `PromptDocument`, `Message`, and `ToolSpec` values.
2. `variables.py` recognizes the deliberately small `{{ identifier }}` syntax.
   It extracts or substitutes text without evaluating expressions.
3. `validation.py` produces ordered `Finding` values. Policies choose accepted
   roles and optional checks; they do not mutate the document.
4. `diff.py` compares ordered messages, global variable contracts, named tools,
   and optionally metadata. It produces ordered `Change` values.
5. `reporting.py` serializes typed reports. JSON is the machine contract;
   Markdown and HTML are review views over exactly the same facts.
6. `cli.py` maps report severities to stable process exit codes.

## Determinism

Given equal Python values and options, validation and comparison preserve input
order where it is meaningful and sort mapping-derived findings by key. Reports
contain no timestamp, hostname, absolute path, random value, or network result.
JSON keys are sorted. This makes generated artifacts suitable for source review
and byte-for-byte comparison.

Every reported path is an RFC 6901 JSON Pointer. Tool and parameter names that
contain `/` or `~` are escaped, so distinct fields cannot collapse into the same
machine-readable location.

Structural equality follows JSON types rather than Python coercion: booleans are
never equal to numbers. Integer and floating representations of the same finite
JSON number (for example `1` and `1.0`) are treated as equal.

## Compatibility model

PromptWitness treats requirements imposed on existing callers or executors as
breaking. A newly referenced template variable, removed tool, changed parameter
schema, newly required parameter, role change, or removed message falls into
this category. Text changes are warnings by default because they may alter
behavior without making rendering impossible.

This is a local policy, not semantic versioning for every prompt system. Library
callers can set message severities with `DiffOptions`; the CLI separately picks
which severity fails a job.

## Trust boundaries

Input documents are untrusted. Parsing is strict, HTML is escaped, and no
template content is executed. The library does not fetch references or invoke
tools. Loading a very large JSON document can still consume memory because the
standard JSON parser materializes it; callers handling untrusted uploads should
enforce file-size limits before PromptWitness.

The secret-like literal rule is a review hint only. A dedicated secret scanner,
access control, and removal from Git history are separate responsibilities.

## Extension boundaries

New format versions should be explicit branches in the parser, not permissive
acceptance of unknown shapes. New findings and changes receive stable enum
values. Report schema changes require a new `schema_version`. Provider-specific
translation belongs in optional adapters so the core remains dependency-free.
