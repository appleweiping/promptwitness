"""Command-line interface for validation and compatibility gates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from .diff import DiffOptions, compare_prompts
from .models import DiffReport, Severity, ValidationReport
from .parser import PromptFormatError, load_prompt
from .reporting import render_html, render_json, render_markdown
from .validation import ValidationPolicy, validate_prompt

_FAIL_CHOICES = ("never", "warning", "breaking")
_FORMAT_CHOICES = ("json", "markdown", "html")


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
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate one prompt document")
    validate.add_argument("prompt", type=Path)
    validate.add_argument("--require-system-first", action="store_true")
    validate.add_argument("--allow-empty-content", action="store_true")
    validate.add_argument("--no-secret-scan", action="store_true")
    _add_output_arguments(validate)
    validate.set_defaults(handler=_run_validate)

    diff = subparsers.add_parser("diff", help="compare two prompt documents")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--ignore-metadata", action="store_true")
    diff.add_argument("--context-lines", type=_non_negative_int, default=2)
    _add_output_arguments(diff)
    diff.set_defaults(handler=_run_diff)
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
    except (PromptFormatError, OSError, ValueError) as error:
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
    document = load_prompt(arguments.prompt)
    policy = ValidationPolicy(
        require_system_first=arguments.require_system_first,
        allow_empty_content=arguments.allow_empty_content,
        scan_literal_secrets=not arguments.no_secret_scan,
    )
    report = validate_prompt(document, policy)
    _emit(_render(report, arguments.format), arguments.output)
    return 2 if _reaches_threshold(report, arguments.fail_on) else 0


def _run_diff(arguments: argparse.Namespace) -> int:
    before = load_prompt(arguments.before)
    after = load_prompt(arguments.after)
    options = DiffOptions(
        include_metadata=not arguments.ignore_metadata,
        context_lines=arguments.context_lines,
    )
    report = compare_prompts(before, after, options)
    _emit(_render(report, arguments.format), arguments.output)
    return 2 if _reaches_threshold(report, arguments.fail_on) else 0


def _render(report: DiffReport | ValidationReport, output_format: str) -> str:
    if output_format == "json":
        return render_json(report)
    if output_format == "html":
        return render_html(report)
    return render_markdown(report)


def _emit(content: str, output: Path | None) -> None:
    if output is None:
        print(content, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


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
