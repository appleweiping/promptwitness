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
