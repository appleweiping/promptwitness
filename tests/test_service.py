import json
import threading
import urllib.request

import pytest

from promptwitness import PromptService, create_server


def _prompt(tmp_path):
    path = tmp_path / "prompt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "demo",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"id": {"type": "string"}},
                        "required": ["id"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prompt_service_dispatches_validate_diff_and_call(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt = _prompt(tmp_path)
    service = PromptService()
    assert service.dispatch({"operation": "validate", "prompt": str(prompt)})["valid"]
    assert service.dispatch({"operation": "diff", "before": str(prompt), "after": str(prompt)})[
        "compatible"
    ]
    call = service.dispatch(
        {"operation": "check_call", "prompt": str(prompt), "tool": "lookup", "arguments": {}}
    )
    assert call["valid"] is False
    assert call["issues"][0]["path"] == "/id"


def test_prompt_service_http_dispatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    server = create_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/dispatch",
            data=json.dumps({"operation": "validate", "prompt": str(_prompt(tmp_path))}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read())["valid"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_prompt_service_validates_request_options(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt = _prompt(tmp_path)
    service = PromptService()
    assert (
        service.dispatch(
            {"operation": "validate", "prompt": str(prompt), "require_system_first": True}
        )["valid"]
        is False
    )
    with pytest.raises(ValueError):
        service.dispatch({"operation": "other", "prompt": str(prompt)})
    with pytest.raises(ValueError):
        service.dispatch(
            {"operation": "validate", "prompt": str(prompt), "allow_empty_content": "yes"}
        )
    with pytest.raises(ValueError):
        service.dispatch(
            {"operation": "check_call", "prompt": str(prompt), "tool": "missing", "arguments": {}}
        )
    with pytest.raises(ValueError):
        service.dispatch(
            {"operation": "check_call", "prompt": str(prompt), "tool": "lookup", "arguments": []}
        )
    with pytest.raises(ValueError):
        create_server(port=65536)


def test_prompt_service_accepts_openai_adapter_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "openai.json"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
        encoding="utf-8",
    )
    result = PromptService().dispatch(
        {"operation": "validate", "prompt": str(path), "from_format": "openai"}
    )
    assert result["valid"] is True


def test_prompt_service_adapts_openai_diff_and_rejects_format(tmp_path) -> None:  # type: ignore[no-untyped-def]
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps({"messages": [{"role": "user", "content": "goodbye"}]}),
        encoding="utf-8",
    )
    diff = PromptService().dispatch(
        {
            "operation": "diff",
            "before": str(before),
            "after": str(after),
            "before_format": "openai",
            "after_format": "openai",
        }
    )
    assert diff["compatible"] is True
    with pytest.raises(ValueError):
        PromptService().dispatch(
            {"operation": "validate", "prompt": str(before), "from_format": "unknown"}
        )


def test_prompt_service_converts_provider_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "openai.json"
    source.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.2,
            }
        ),
        encoding="utf-8",
    )
    response = PromptService().dispatch(
        {
            "operation": "convert",
            "prompt": str(source),
            "from_format": "openai",
            "prompt_id": "converted",
        }
    )
    assert response["prompt"]["id"] == "converted"
    assert response["prompt"]["messages"][0]["content"] == "hello"
    assert response["warnings"]


def test_prompt_service_convert_validates_format_and_id(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "openai.json"
    source.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        PromptService().dispatch(
            {"operation": "convert", "prompt": str(source), "from_format": "unknown"}
        )
    with pytest.raises(ValueError):
        PromptService().dispatch(
            {
                "operation": "convert",
                "prompt": str(source),
                "from_format": "openai",
                "prompt_id": "",
            }
        )


def test_prompt_service_renders_and_diffs_matrix(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt = tmp_path / "prompt.json"
    prompt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "matrix-prompt",
                "messages": [{"role": "user", "content": "Hello {{name}}"}],
            }
        ),
        encoding="utf-8",
    )
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        json.dumps([{"id": "ada", "values": {"name": "Ada"}, "tags": ["demo"]}]),
        encoding="utf-8",
    )
    artifact = tmp_path / "matrix.json"
    response = PromptService().dispatch(
        {
            "operation": "matrix",
            "prompt": str(prompt),
            "scenarios": str(scenarios),
            "artifact": str(artifact),
        }
    )
    assert response["prompt_id"] == "matrix-prompt"
    assert response["rows"][0]["messages"][0]["content"] == "Hello Ada"
    assert response["artifact"]["digest"]
    diff = PromptService().dispatch(
        {"operation": "matrix_diff", "before": str(artifact), "after": str(artifact)}
    )
    assert diff["changed"] is False
    assert diff["scenarios"][0]["changed"] is False


def test_prompt_service_matrix_validates_requests(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt = _prompt(tmp_path)
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps([{"id": "one", "values": {}}]), encoding="utf-8")
    with pytest.raises(ValueError, match="strict"):
        PromptService().dispatch(
            {
                "operation": "matrix",
                "prompt": str(prompt),
                "scenarios": str(scenarios),
                "strict": "yes",
            }
        )
    invalid_payloads = [
        {},
        [1],
        [{"id": "one", "tags": "bad"}],
        [{"id": ""}],
        [{"id": "one", "values": []}],
    ]
    for index, raw in enumerate(invalid_payloads):
        invalid = tmp_path / f"invalid-{index}.json"
        invalid.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError):
            PromptService().dispatch(
                {"operation": "matrix", "prompt": str(prompt), "scenarios": str(invalid)}
            )
    with pytest.raises(ValueError, match="cannot read scenarios"):
        PromptService().dispatch(
            {
                "operation": "matrix",
                "prompt": str(prompt),
                "scenarios": str(tmp_path / "missing.json"),
            }
        )
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps([{"id": "one", "values": {}}]), encoding="utf-8")
    without_artifact = PromptService().dispatch(
        {
            "operation": "matrix",
            "prompt": str(prompt),
            "scenarios": str(valid),
            "strict": False,
        }
    )
    assert "artifact" not in without_artifact
    with pytest.raises(ValueError, match="artifact"):
        PromptService().dispatch(
            {
                "operation": "matrix",
                "prompt": str(prompt),
                "scenarios": str(valid),
                "artifact": "",
            }
        )
