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
from .long_context import LongContextCase, evaluate_long_context
from .matrix import MatrixArtifact, Scenario, compare_matrices, render_matrix, save_matrix
from .models import PromptDocument, message_content_to_wire
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
        if operation == "matrix":
            document = _load_document(request, "prompt")
            scenarios_path = _path(request, "scenarios")
            try:
                raw = json.loads(scenarios_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot read scenarios: {error}") from error
            if not isinstance(raw, list):
                raise ValueError("scenario file must contain a JSON array")
            scenarios: list[Scenario] = []
            for index, item in enumerate(raw, start=1):
                if not isinstance(item, Mapping):
                    raise ValueError(f"scenario entry {index} must be an object")
                values = item.get("values", {})
                tags = item.get("tags", [])
                if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                    raise ValueError(f"scenario entry {index} tags must be an array of strings")
                scenario_id = item.get("id")
                if not isinstance(scenario_id, str) or not scenario_id.strip():
                    raise ValueError(
                        f"invalid scenario entry {index}: id must be a non-empty string"
                    )
                try:
                    scenarios.append(Scenario(scenario_id, values, tuple(tags)))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid scenario entry {index}: {error}") from error
            strict = request.get("strict", True)
            if not isinstance(strict, bool):
                raise ValueError("strict must be a boolean")
            rows = render_matrix(document, scenarios, strict=strict)
            payload: dict[str, Any] = {
                "operation": operation,
                "prompt_id": document.prompt_id,
                "rows": [_rendered_row(row) for row in rows],
            }
            artifact_path = request.get("artifact")
            if artifact_path is not None:
                if not isinstance(artifact_path, str) or not artifact_path.strip():
                    raise ValueError("artifact must be a non-empty path string when supplied")
                payload["artifact"] = save_matrix(rows, artifact_path).to_dict()
            return payload
        if operation == "matrix_diff":
            before_artifact = MatrixArtifact.load(str(_path(request, "before")))
            after_artifact = MatrixArtifact.load(str(_path(request, "after")))
            if before_artifact.prompt_id != after_artifact.prompt_id:
                raise ValueError("matrix artifacts must use the same prompt ID")
            differences = compare_matrices(before_artifact.rows, after_artifact.rows)
            return {
                "operation": operation,
                "schema_version": 1,
                "prompt_id": before_artifact.prompt_id,
                "before_digest": before_artifact.digest,
                "after_digest": after_artifact.digest,
                "changed": any(item.changed for item in differences),
                "scenarios": [
                    {
                        "scenario_id": item.scenario_id,
                        "before_digest": item.before_digest,
                        "after_digest": item.after_digest,
                        "changed": item.changed,
                    }
                    for item in differences
                ],
            }
        if operation == "long_context":
            cases_path = _path(request, "cases")
            try:
                raw = json.loads(cases_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot read long-context cases: {error}") from error
            if not isinstance(raw, list) or not raw:
                raise ValueError("long-context cases must be a non-empty JSON array")
            cases: list[LongContextCase] = []
            for index, item in enumerate(raw, start=1):
                if not isinstance(item, Mapping):
                    raise ValueError(f"long-context case {index} must be an object")
                try:
                    context = item["context"]
                    if not isinstance(context, list):
                        raise ValueError("context must be an array")
                    cases.append(
                        LongContextCase(
                            item["case_id"],
                            tuple(context),
                            item["needle"],
                            item["query"],
                            item["expected"],
                            item["needle_index"],
                        )
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid long-context case {index}: {error}") from error
            predictions = request.get("predictions")
            if not isinstance(predictions, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in predictions.items()
            ):
                raise ValueError("predictions must map case IDs to strings")
            strict = request.get("strict", False)
            if not isinstance(strict, bool):
                raise ValueError("strict must be a boolean")
            report = evaluate_long_context(
                cases,
                lambda case: predictions.get(case.case_id, ""),
                strict=strict,
            )
            return {"operation": operation, "report": report.to_dict()}
        raise ValueError(
            "operation must be validate, diff, check_call, convert, matrix, matrix_diff, "
            "or long_context"
        )


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


def _rendered_row(row: Any) -> dict[str, Any]:
    return {
        "id": row.scenario_id,
        "digest": row.digest,
        "variables": list(row.variables),
        "tags": list(row.tags),
        "messages": [
            {
                "id": message.message_id,
                "role": message.role,
                "name": message.name,
                "content": message_content_to_wire(message),
            }
            for message in row.messages
        ],
    }


__all__ = ["PromptService", "create_server"]
