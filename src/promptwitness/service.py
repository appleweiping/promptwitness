"""Loopback JSON service for PromptWitness validation and compatibility checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .adapters import AdapterFormat, load_adapted_prompt, prompt_to_dict
from .diff import compare_prompts
from .invocations import validate_tool_arguments
from .models import PromptDocument
from .parser import load_prompt
from .validation import ValidationPolicy, validate_prompt


class PromptService:
    """Dispatch deterministic prompt validation, diff, and call checks."""

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be an object")
        operation = request.get("operation")
        if operation == "validate":
            document = _load_document(request, "prompt")
            validation = validate_prompt(document, _policy(request))
            return {
                "operation": operation,
                "prompt_id": validation.prompt_id,
                "valid": validation.valid,
                "findings": [_finding(item) for item in validation.findings],
            }
        if operation == "diff":
            before = _load_document(request, "before", "before_format")
            after = _load_document(request, "after", "after_format")
            diff_report = compare_prompts(before, after)
            return {
                "operation": operation,
                "before_id": diff_report.before_id,
                "after_id": diff_report.after_id,
                "compatible": diff_report.compatible,
                "breaking_count": diff_report.breaking_count,
                "warning_count": diff_report.warning_count,
                "changes": [_change(item) for item in diff_report.changes],
            }
        if operation == "check_call":
            document = _load_document(request, "prompt")
            name = request.get("tool")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool must be a non-empty string")
            tool = document.tool_map().get(name)
            if tool is None:
                raise ValueError(f"unknown tool {name!r}")
            arguments = request.get("arguments")
            if not isinstance(arguments, Mapping):
                raise ValueError("arguments must be an object")
            call_report = validate_tool_arguments(tool, arguments)
            return {"operation": operation, **call_report.to_dict()}
        if operation == "convert":
            source_format = request.get("from_format")
            try:
                selected = AdapterFormat(source_format)
            except (TypeError, ValueError) as error:
                raise ValueError("from_format must be a supported adapter format") from error
            prompt_id = request.get("prompt_id")
            if prompt_id is not None and (not isinstance(prompt_id, str) or not prompt_id.strip()):
                raise ValueError("prompt_id must be a non-empty string when supplied")
            result = load_adapted_prompt(_path(request, "prompt"), selected, prompt_id=prompt_id)
            return {
                "operation": operation,
                "prompt": prompt_to_dict(result.document),
                "warnings": list(result.warnings),
            }
        raise ValueError("operation must be validate, diff, check_call, or convert")


def create_server(
    service: PromptService | None = None, *, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Create a loopback-first JSON server; call ``serve_forever`` to run it."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    target = service or PromptService()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/dispatch":
                self._write(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 0 or size > 4 * 1024 * 1024:
                    raise ValueError("Content-Length must be between 0 and 4194304")
                payload = target.dispatch(json.loads(self.rfile.read(size).decode("utf-8")))
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OSError) as error:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._write(HTTPStatus.OK, payload)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _path(request: Mapping[str, Any], name: str) -> Path:
    value = request.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    return Path(value)


def _load_document(
    request: Mapping[str, Any], path_name: str, format_name: str = "from_format"
) -> PromptDocument:
    source_format = request.get(format_name, AdapterFormat.NATIVE.value)
    try:
        selected = AdapterFormat(source_format)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{format_name} must be a supported adapter format") from error
    if selected is AdapterFormat.NATIVE:
        return load_prompt(_path(request, path_name))
    return load_adapted_prompt(_path(request, path_name), selected).document


def _policy(request: Mapping[str, Any]) -> ValidationPolicy:
    values: dict[str, Any] = {}
    for name in (
        "require_system_first",
        "allow_empty_content",
        "report_repeated_variables",
        "scan_literal_secrets",
    ):
        value = request.get(name)
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            values[name] = value
    return ValidationPolicy(**values)


def _finding(finding: Any) -> dict[str, str]:
    return {
        "code": finding.code.value,
        "path": finding.path,
        "severity": finding.severity.value,
        "message": finding.message,
    }


def _change(change: Any) -> dict[str, Any]:
    return {
        "kind": change.kind.value,
        "path": change.path,
        "severity": change.severity.value,
        "summary": change.summary,
    }


__all__ = ["PromptService", "create_server"]
