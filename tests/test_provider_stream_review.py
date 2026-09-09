"""Independent real-wire completion-status regressions; no hosted calls."""

from __future__ import annotations

import json
import socket
import threading
import traceback
from contextlib import contextmanager

import pytest

from promptwitness import OpenAICompatibleStreamingProvider
from promptwitness.matrix import RenderedScenario
from promptwitness.models import Message
from promptwitness.task_scores import response_text

ROW = RenderedScenario(
    "stream-review", "local", (Message("user", "local request"),), (), "0" * 64, ()
)
PRIVATE = "PRIVATE_UPSTREAM_DETAIL"


@contextmanager
def stream_peer(chunks, *, done=True):
    """Serve one independently framed SSE reply over a literal loopback socket."""
    body = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
    if done:
        body += b"data: [DONE]\n\n"
    payload = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    failures = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)

        def serve():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(3)
                    received = bytearray()
                    while b"\r\n\r\n" not in received:
                        incoming = connection.recv(4096)
                        if not incoming:
                            raise AssertionError("request headers ended early")
                        received.extend(incoming)
                        if len(received) > 65536:
                            raise AssertionError("unexpected request size")
                    head, initial = bytes(received).split(b"\r\n\r\n", 1)
                    lengths = [
                        int(line.split(b":", 1)[1])
                        for line in head.split(b"\r\n")[1:]
                        if line.lower().startswith(b"content-length:")
                    ]
                    assert len(lengths) == 1
                    remaining = lengths[0] - len(initial)
                    while remaining > 0:
                        incoming = connection.recv(min(4096, remaining))
                        if not incoming:
                            raise AssertionError("request body ended early")
                        remaining -= len(incoming)
                    connection.sendall(payload)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            yield f"http://127.0.0.1:{listener.getsockname()[1]}/chat"
        finally:
            thread.join(timeout=5)
            assert not thread.is_alive(), "review server did not stop"
            assert not failures, "review server failed"


def choice(delta, *, reason=None, index=0):
    return {"choices": [{"index": index, "delta": delta, "finish_reason": reason}]}


@pytest.mark.parametrize(("reason", "done"), [(None, False), (None, True), ("length", True)])
def test_unfinished_stream_is_not_relabelled_complete(reason, done):
    chunks = [choice({"content": "partial"}, reason=reason)]
    with stream_peer(chunks, done=done) as endpoint:
        result = OpenAICompatibleStreamingProvider(endpoint, timeout=2)(ROW)
    assert result["choices"][0]["finish_reason"] == reason
    assert result["choices"][0]["message"]["content"] == "partial"
    assert result["stream_chunks"] == chunks
    with pytest.raises(ValueError, match="complete"):
        response_text(result)


@pytest.mark.parametrize(
    "chunks",
    [
        [
            choice(
                {"tool_calls": [{"function": {"name": PRIVATE, "arguments": "{}"}}]},
                reason="tool_calls",
            )
        ],
        [choice({"function_call": {"name": PRIVATE, "arguments": "{}"}}, reason="function_call")],
        [choice({"refusal": PRIVATE}, reason="stop")],
        [{"error": {"message": PRIVATE}}],
        [choice({"role": "tool", "content": PRIVATE}, reason="stop")],
        [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "A"}, "finish_reason": "stop"},
                    {"index": 1, "delta": {"content": "B"}, "finish_reason": "stop"},
                ]
            }
        ],
        [choice({"content": "A"}, index=0), choice({"content": "B"}, reason="stop", index=1)],
        [choice({"content": "A"}, reason="stop"), choice({"content": PRIVATE})],
        [choice({"content": "A"}, reason="stop"), choice({}, reason="length")],
    ],
    ids=[
        "tool",
        "function",
        "refusal",
        "error",
        "nonassistant",
        "multi",
        "changed-index",
        "late-text",
        "conflict",
    ],
)
def test_ambiguous_or_nontext_stream_never_becomes_completed_text(chunks):
    with stream_peer(chunks) as endpoint, pytest.raises(ValueError) as caught:
        OpenAICompatibleStreamingProvider(endpoint, timeout=2)(ROW)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))


def test_completed_text_and_usage_are_preserved():
    chunks = [
        choice({"role": "assistant"}),
        choice({"content": "A😀"}),
        choice({"content": " e\u0301"}),
        choice({}, reason="stop"),
        {"choices": [], "usage": {"completion_tokens": 4}},
    ]
    with stream_peer(chunks) as endpoint:
        result = OpenAICompatibleStreamingProvider(endpoint, timeout=2)(ROW)
    assert response_text(result) == "A😀 e\u0301"
    assert result["stream_chunks"] == chunks


def test_raw_stream_still_retains_noncompletion_payloads():
    chunks = [choice({"content": "partial"}, reason="length"), {"error": {"message": PRIVATE}}]
    with stream_peer(chunks) as endpoint:
        result = list(OpenAICompatibleStreamingProvider(endpoint, timeout=2).stream(ROW))
    assert result == chunks
