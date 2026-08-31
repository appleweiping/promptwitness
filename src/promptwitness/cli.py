"""Command-line interface for validation and compatibility gates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

from .adapters import AdapterError, AdapterFormat, load_adapted_prompt, render_prompt_json
from .diff import MessageAlignment, compare_prompts
from .models import DiffReport, Severity, ValidationReport
from .parser import PromptFormatError
from .policies import PolicyBundle, PolicyFormatError, load_policy
from .reporting import render_html, render_json, render_markdown, render_sarif
from .validation import validate_prompt

_FAIL_CHOICES = ("never", "warning", "breaking")
_FORMAT_CHOICES = ("json", "markdown", "html", "sarif")


class _PromptWitnessParser(argparse.ArgumentParser):
    """Use exit 1 for malformed invocations, reserving exit 2 for report gates."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the public argument parser for documentation and tests."""

    parser = _PromptWitnessParser(
        prog="promptwitness",
        description="Validate and compare versioned chat prompt specifications.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate one prompt document")
    validate.add_argument("prompt", type=Path)
    validate.add_argument(
        "--from-format",
        choices=[source.value for source in AdapterFormat],
        default=AdapterFormat.NATIVE.value,
    )
    validate.add_argument("--policy", type=Path)
    system_position = validate.add_mutually_exclusive_group()
    system_position.add_argument(
        "--require-system-first", action="store_true", dest="require_system_first"
    )
    system_position.add_argument(
        "--no-require-system-first", action="store_false", dest="require_system_first"
    )
    validate.set_defaults(require_system_first=None)
    empty_content = validate.add_mutually_exclusive_group()
    empty_content.add_argument("--allow-empty-content", action="store_true")
    empty_content.add_argument(
        "--no-allow-empty-content", action="store_false", dest="allow_empty_content"
    )
    validate.set_defaults(allow_empty_content=None)
    secret_scan = validate.add_mutually_exclusive_group()
    secret_scan.add_argument("--secret-scan", action="store_true", dest="scan_literal_secrets")
    secret_scan.add_argument("--no-secret-scan", action="store_false", dest="scan_literal_secrets")
    validate.set_defaults(scan_literal_secrets=None)
    _add_output_arguments(validate)
    validate.set_defaults(handler=_run_validate)

    diff = subparsers.add_parser("diff", help="compare two prompt documents")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument(
        "--before-format",
        choices=[source.value for source in AdapterFormat],
        default=AdapterFormat.NATIVE.value,
    )
    diff.add_argument(
        "--after-format",
        choices=[source.value for source in AdapterFormat],
        default=AdapterFormat.NATIVE.value,
    )
    diff.add_argument("--policy", type=Path)
    metadata = diff.add_mutually_exclusive_group()
    metadata.add_argument("--include-metadata", action="store_true", dest="include_metadata")
    metadata.add_argument("--ignore-metadata", action="store_false", dest="include_metadata")
    diff.set_defaults(include_metadata=None)
    diff.add_argument("--context-lines", type=_non_negative_int)
    diff.add_argument("--message-alignment", choices=[mode.value for mode in MessageAlignment])
    _add_output_arguments(diff)
    diff.set_defaults(handler=_run_diff)

    convert = subparsers.add_parser(
        "convert", help="convert a provider prompt/request to native schema version 1"
    )
    convert.add_argument("prompt", type=Path)
    convert.add_argument(
        "--from-format",
        choices=[source.value for source in AdapterFormat],
        default=AdapterFormat.AUTO.value,
    )
    convert.add_argument("--id", dest="prompt_id")
    convert.add_argument("--output", type=Path)
    convert.set_defaults(handler=_run_convert)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Exit 0 means the selected gate passed, 1 means input/output failure, and 2
    means a valid report reached the configured finding threshold.
    """

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (AdapterError, PolicyFormatError, PromptFormatError, OSError, ValueError) as error:
        print(f"promptwitness: {error}", file=sys.stderr)
        return 1


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=_FORMAT_CHOICES, default="markdown")
    parser.add_argument("--output", type=Path, help="write the report instead of stdout")
    parser.add_argument(
        "--fail-on",
        choices=_FAIL_CHOICES,
        default="breaking",
        help="return exit 2 at this severity (default: breaking)",
    )


def _run_validate(arguments: argparse.Namespace) -> int:
    _ensure_distinct_output(arguments.output, arguments.prompt, arguments.policy)
    result = load_adapted_prompt(arguments.prompt, AdapterFormat(arguments.from_format))
    _emit_adapter_warnings(result.warnings)
    base = _policy(arguments.policy).validation
    policy = replace(
        base,
        require_system_first=(
            base.require_system_first
            if arguments.require_system_first is None
            else arguments.require_system_first
        ),
        allow_empty_content=(
            base.allow_empty_content
            if arguments.allow_empty_content is None
            else arguments.allow_empty_content
        ),
        scan_literal_secrets=(
            base.scan_literal_secrets
            if arguments.scan_literal_secrets is None
            else arguments.scan_literal_secrets
        ),
    )
    report = validate_prompt(result.document, policy)
    _emit(_render(report, arguments.format, artifact_uri=arguments.prompt.name), arguments.output)
    return 2 if _reaches_threshold(report, arguments.fail_on) else 0


def _run_diff(arguments: argparse.Namespace) -> int:
    _ensure_distinct_output(
        arguments.output,
        arguments.before,
        arguments.after,
        arguments.policy,
    )
    before = load_adapted_prompt(arguments.before, AdapterFormat(arguments.before_format))
    after = load_adapted_prompt(arguments.after, AdapterFormat(arguments.after_format))
    _emit_adapter_warnings(before.warnings + after.warnings)
    base = _policy(arguments.policy).diff
    options = replace(
        base,
        include_metadata=(
            base.include_metadata
            if arguments.include_metadata is None
            else arguments.include_metadata
        ),
        context_lines=(
            base.context_lines if arguments.context_lines is None else arguments.context_lines
        ),
        message_alignment=(
            base.message_alignment
            if arguments.message_alignment is None
            else MessageAlignment(arguments.message_alignment)
        ),
    )
    report = compare_prompts(before.document, after.document, options)
    _emit(_render(report, arguments.format, artifact_uri=arguments.after.name), arguments.output)
    return 2 if _reaches_threshold(report, arguments.fail_on) else 0


def _run_convert(arguments: argparse.Namespace) -> int:
    _ensure_distinct_output(arguments.output, arguments.prompt)
    result = load_adapted_prompt(
        arguments.prompt,
        AdapterFormat(arguments.from_format),
        prompt_id=arguments.prompt_id,
    )
    _emit_adapter_warnings(result.warnings)
    _emit(render_prompt_json(result.document), arguments.output)
    return 0


def _render(
    report: DiffReport | ValidationReport,
    output_format: str,
    *,
    artifact_uri: str = "prompt.json",
) -> str:
    if output_format == "json":
        return render_json(report)
    if output_format == "html":
        return render_html(report)
    if output_format == "sarif":
        return render_sarif(report, artifact_uri=artifact_uri)
    return render_markdown(report)


def _emit(content: str, output: Path | None) -> None:
    if output is None:
        print(content, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def _ensure_distinct_output(output: Path | None, *inputs: Path | None) -> None:
    """Refuse to overwrite a prompt or policy, including through aliases."""

    if output is None:
        return
    resolved_output = output.resolve(strict=False)
    for source in inputs:
        if source is None:
            continue
        same_resolved_path = resolved_output == source.resolve(strict=False)
        try:
            same_existing_file = output.samefile(source)
        except OSError:
            same_existing_file = False
        if same_resolved_path or same_existing_file:
            raise ValueError(f"output path {output} must not overwrite input path {source}")


def _reaches_threshold(report: DiffReport | ValidationReport, fail_on: str) -> bool:
    if fail_on == "never":
        return False
    severities = (
        (change.severity for change in report.changes)
        if isinstance(report, DiffReport)
        else (finding.severity for finding in report.findings)
    )
    threshold = 1 if fail_on == "warning" else 2
    ranks = {Severity.INFO: 0, Severity.WARNING: 1, Severity.BREAKING: 2}
    return any(ranks[severity] >= threshold for severity in severities)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _policy(path: Path | None) -> PolicyBundle:
    return load_policy(path) if path is not None else PolicyBundle()


def _emit_adapter_warnings(warnings: tuple[str, ...]) -> None:
    for warning in warnings:
        print(f"promptwitness: adapter warning: {warning}", file=sys.stderr)
