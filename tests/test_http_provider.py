from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from promptwitness import (
    OpenAICompatibleProvider,
    OpenAICompatibleStreamingProvider,
    Scenario,
    load_prompt,
    render_matrix,
)


class _Handler(BaseHTTPRequestHandler):
    seen: ClassVar[dict[str, object]] = {}

    def do_POST(self) -> None:
        size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(size))
        self.__class__.seen = {
            "authorization": self.headers.get("Authorization"),
            "payload": payload,
        }
        if payload.get("stream"):
            events = [
                b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            body = b"".join(events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps({"id": "mock", "choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_openai_compatible_provider_sends_prompt_without_variables(monkeypatch) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("PROMPTWITNESS_TEST_KEY", "secret")
        document = load_prompt("examples/before.json")
        (row,) = render_matrix(
            document, (Scenario("one", {"customer_name": "Ada", "order_id": "1"}),)
        )
        result = OpenAICompatibleProvider(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            api_key_env="PROMPTWITNESS_TEST_KEY",
            model="mock",
        )(row)
        assert result["id"] == "mock"
        assert _Handler.seen["authorization"] == "Bearer secret"
        assert "variables" not in _Handler.seen["payload"]
    finally:
        server.shutdown()
        server.server_close()


def test_http_provider_requires_key_when_configured() -> None:
    provider = OpenAICompatibleProvider("http://127.0.0.1:1", api_key_env="MISSING_PROMPT_KEY")
    document = load_prompt("examples/before.json")
    (row,) = render_matrix(document, (Scenario("one", {"customer_name": "Ada", "order_id": "1"}),))
    with pytest.raises(ValueError, match="is not set"):
        provider(row)


def test_streaming_provider_aggregates_sse_and_exposes_chunks() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        document = load_prompt("examples/before.json")
        (row,) = render_matrix(
            document, (Scenario("one", {"customer_name": "Ada", "order_id": "1"}),)
        )
        provider = OpenAICompatibleStreamingProvider(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        )
        result = provider(row)
        assert result["choices"][0]["message"]["content"] == "hello"
        assert len(result["stream_chunks"]) == 3
        assert _Handler.seen["payload"]["stream"] is True
        assert list(provider.stream(row))[1]["choices"][0]["delta"]["content"] == "hel"
    finally:
        server.shutdown()
        server.server_close()
