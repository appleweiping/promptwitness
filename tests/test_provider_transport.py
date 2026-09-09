"""Fault injection over real local HTTP/TLS sockets; no model or external calls."""

from __future__ import annotations

import json
import socket
import ssl
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import pytest

from promptwitness import (
    OpenAICompatibleProvider,
    OpenAICompatibleStreamingProvider,
    Scenario,
    load_prompt,
    render_matrix,
)
from promptwitness import _provider_http as transport
from promptwitness.sessions import OpenAISessionProvider
from promptwitness.task_scores import response_text

KEY = "private-test-only-provider-token"
KEY_ENV = "PROMPTWITNESS_TRANSPORT_TEST_KEY"
MESSAGES = [{"role": "user", "content": "authored local request"}]


@pytest.fixture(autouse=True)
def credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)


@contextmanager
def peer(
    callback: Callable[[socket.socket], None], *, tls: ssl.SSLContext | None = None
) -> Iterator[str]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    connections: list[socket.socket] = []

    def run() -> None:
        try:
            connection, _ = listener.accept()
            connections.append(connection)
            connection.settimeout(2)
            if tls is not None:
                connection = tls.wrap_socket(connection, server_side=True)
                connections.append(connection)
            with connection:
                callback(connection)
        except OSError:
            return

    thread = threading.Thread(target=run)
    thread.start()
    try:
        yield f"{'https' if tls else 'http'}://127.0.0.1:{listener.getsockname()[1]}/chat"
    finally:
        listener.close()
        for connection in connections:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def receive(connection: socket.socket) -> tuple[dict[str, str], Any]:
    with connection.makefile("rb") as stream:
        first = stream.readline(8192)
        assert first.startswith(b"POST ")
        headers = {}
        while line := stream.readline(65536).strip():
            name, value = line.decode("ascii").split(":", 1)
            headers[name.lower()] = value.strip()
        body = stream.read(int(headers["content-length"]))
    return headers, json.loads(body)


def wire(
    body: bytes = b'{"choices":[]}', *, status: int = 200, headers: bytes = b"", length: bool = True
) -> bytes:
    framing = f"Content-Length: {len(body)}\r\n".encode() if length else b""
    return (
        f"HTTP/1.1 {status} Result\r\n".encode()
        + framing
        + headers
        + b"Connection: close\r\n\r\n"
        + body
    )


def reply(data: bytes, seen: list[Any] | None = None) -> Callable[[socket.socket], None]:
    def callback(connection: socket.socket) -> None:
        incoming = receive(connection)
        if seen is not None:
            seen.append(incoming)
        connection.sendall(data)

    return callback


def provider(endpoint: str, **options: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(endpoint, api_key_env=KEY_ENV, **options)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"status": "incomplete", "arbitrary_vendor_metadata": {"bool": True, "int": 1}},
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "missing finish reason"}}]},
        {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]},
    ],
)
def test_preserves_full_object_and_completion_status(value: Any) -> None:
    with peer(reply(wire(json.dumps(value).encode()))) as endpoint:
        assert provider(endpoint).complete(MESSAGES) == value
    if value.get("choices", [{}])[0].get("finish_reason") != "stop":
        with pytest.raises(ValueError):
            response_text(value)


@pytest.mark.parametrize("mode", ["length", "close", "chunked"])
def test_valid_framing_and_exact_request_metadata(mode: str) -> None:
    data = b'{"reply":"\xe9\x9b\xaa","usage":{"tokens":3}}'
    body = data
    headers = b"Content-Type: application/json; charset=UTF-8\r\n"
    if mode == "chunked":
        body = (
            b"4\r\n"
            + data[:4]
            + b"\r\n"
            + f"{len(data) - 4:x}\r\n".encode()
            + data[4:]
            + b"\r\n0\r\n\r\n"
        )
        headers += b"Transfer-Encoding: chunked\r\n"
    seen: list[Any] = []
    with peer(reply(wire(body, headers=headers, length=mode == "length"), seen)) as endpoint:
        result = provider(
            endpoint, model="test", headers={"x-tenant": "local", "authorization": "old"}
        ).complete(
            MESSAGES,
            generation={"temperature": 0.2, "max_tokens": 32},
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
    assert result["reply"] == "雪"
    assert seen[0][0]["authorization"] == f"Bearer {KEY}"
    assert seen[0][0]["accept-encoding"] == "identity"
    assert seen[0][1]["temperature"] == 0.2 and seen[0][1]["model"] == "test"
    assert seen[0][1]["tools"][0]["function"]["name"] == "f"


@pytest.mark.parametrize(
    "body",
    [
        b"NaN",
        b"[]",
        b"null",
        b"\xff",
        b'{"SECRET":1,"SECRET":2}',
        b'{"value":NaN}',
        b'{"value":1e9999}',
        b'{"value":"\\ud800SECRET"}',
        b'{"value":' + b"[" * 65 + b"0" + b"]" * 65 + b"}",
        b'{"SECRET":"unfinished',
    ],
)
def test_strict_json_and_redacted_exception_chains(body: bytes) -> None:
    with peer(reply(wire(body))) as endpoint, pytest.raises(ValueError) as caught:
        provider(endpoint).complete(MESSAGES)
    formatted = "".join(traceback.format_exception(caught.value))
    assert "SECRET" not in formatted and KEY not in formatted
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None or caught.value.__suppress_context__


@pytest.mark.parametrize(
    "headers",
    [
        b"Content-Length: 1\r\n",
        b"Transfer-Encoding: chunked\r\n",
        b"Content-Encoding: gzip\r\n",
        b"Content-Type: text/html\r\n",
        b"Content-Type: application /json\r\n",
        b"Content-Type: application/json\r\nContent-Type: application/json\r\n",
        b"SECRET malformed\r\n",
        b"X: " + b"a" * 8192 + b"\r\n",
        b"".join(f"X-{i}: ".encode() + b"x" * 4000 + b"\r\n" for i in range(17)),
    ],
    ids=[f"framing-{index}" for index in range(9)],
)
def test_header_limits_and_ambiguous_framing(headers: bytes) -> None:
    with peer(reply(wire(headers=headers))) as endpoint, pytest.raises(ValueError) as caught:
        provider(endpoint).complete(MESSAGES)
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "data",
    [
        b"SECRET invalid HTTP status\r\n\r\n",
        wire(b"{}", status=201),
        wire(b"{}", status=204),
        wire(b"{}")[:-1],
        wire(b"", length=False, headers=b"Content-Length: +1\r\n"),
        wire(b"", length=False, headers=b"Content-Length: 9999999999\r\n"),
        wire(b"", length=False, headers=b"Transfer-Encoding: gzip\r\n"),
        wire(b"Z\r\n{}\r\n0\r\n\r\n", length=False, headers=b"Transfer-Encoding: chunked\r\n"),
        wire(b"2\r\n{}\r\n", length=False, headers=b"Transfer-Encoding: chunked\r\n"),
    ],
)
def test_invalid_status_lengths_and_chunks(data: bytes) -> None:
    with peer(reply(data)) as endpoint, pytest.raises(ValueError) as caught:
        provider(endpoint).complete(MESSAGES)
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))


def test_error_body_not_read_and_redirect_does_not_forward_credentials() -> None:
    with socket.socket() as target:
        target.bind(("127.0.0.1", 0))
        target.listen(1)
        target.settimeout(0.15)
        redirect = f"Location: http://127.0.0.1:{target.getsockname()[1]}/secret\r\n".encode()
        for code in (302, 307, 308, 401, 500):

            def answer(connection: socket.socket, code: int = code) -> None:
                receive(connection)
                connection.sendall(wire(b"", status=code, headers=redirect, length=False))
                time.sleep(0.2)  # No body or EOF until after client should reject.

            with peer(answer) as endpoint, pytest.raises(ValueError, match=f"HTTP {code}"):
                provider(endpoint, timeout=0.1).complete(MESSAGES)
        with pytest.raises(TimeoutError):
            target.accept()


def test_environment_proxies_never_receive_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as proxy:
        proxy.bind(("127.0.0.1", 0))
        proxy.listen(1)
        proxy.settimeout(0.1)
        for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(name, f"http://127.0.0.1:{proxy.getsockname()[1]}")
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        with peer(reply(wire(b"{}"))) as endpoint:
            assert provider(endpoint).complete(MESSAGES) == {}
        with pytest.raises(TimeoutError):
            proxy.accept()


@pytest.mark.parametrize("mode", ["declared", "close", "chunked", "framing"])
def test_body_and_wire_limits(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport, "MAX_BODY_BYTES", 1024)
    monkeypatch.setattr(transport, "MAX_WIRE_BYTES", 2048)
    data = b'{"value":"' + b"x" * 1024 + b'"}'
    if mode in ("declared", "close"):
        reply_bytes = wire(data, length=mode == "declared")
    elif mode == "chunked":
        reply_bytes = wire(
            f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n",
            length=False,
            headers=b"Transfer-Encoding: chunked\r\n",
        )
    else:
        reply_bytes = wire(
            b"1;SECRET=" + b"x" * 2000 + b"\r\n{\r\n1\r\n}\r\n0\r\n\r\n",
            length=False,
            headers=b"Transfer-Encoding: chunked\r\n",
        )
    with peer(reply(reply_bytes)) as endpoint, pytest.raises(ValueError):
        provider(endpoint).complete(MESSAGES)


@pytest.mark.parametrize("stage", ["header", "body", "send", "tls"])
def test_whole_exchange_deadline_under_trickle_and_blocked_io(stage: str) -> None:
    def slow(connection: socket.socket) -> None:
        if stage in ("send", "tls"):
            time.sleep(0.4)
            return
        receive(connection)
        if stage == "body":
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n")
        for _ in range(15):
            connection.sendall(b" ")
            time.sleep(0.02)
        time.sleep(0.1)

    with peer(slow) as endpoint:
        if stage == "tls":
            endpoint = endpoint.replace("http:", "https:")
        started = time.monotonic()
        with pytest.raises(ValueError):
            provider(endpoint, timeout=0.15).complete(
                [{"role": "user", "content": "x" * (8 * 1024 * 1024)}]
                if stage == "send"
                else MESSAGES
            )
        assert time.monotonic() - started < 0.8
    assert not any(t.name == "promptwitness-provider-deadline" for t in threading.enumerate())


def test_dns_resolution_gap_does_not_start_late_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = socket.getaddrinfo

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.08)
        return original(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow)
    with pytest.raises(ValueError, match="deadline"):
        provider("http://127.0.0.1:1", timeout=0.02).complete(MESSAGES)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), 301, 10**400])
def test_timeout_is_finite_bounded_number(timeout: Any) -> None:
    with pytest.raises(ValueError, match="timeout"):
        provider("http://127.0.0.1:1", timeout=timeout)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://localhost",
        "http://user:SECRET@localhost",
        "http://localhost#SECRET",
        "http://localhost:0",
        "http://localhost:65536",
        "http://localhost/\r\nSECRET",
        "http:///missing",
        "http://localhost\\evil",
        "http://[::1%lo]/",
        1,
    ],
)
def test_endpoint_rejection_is_secret_safe(url: Any) -> None:
    with pytest.raises(ValueError) as caught:
        provider(url)
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "headers",
    [
        [],
        {"Host": "elsewhere"},
        {"Content-Length": "9"},
        {"Authorization": "PRIVATE\r\nSECRET: x"},
        {"X": "\ud800"},
        {"X": "1", "x": "2"},
        {"Proxy-Authorization": "secret"},
        {"X": "x" * 65536},
        {"X": 1},
    ],
)
def test_custom_header_contract(headers: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        provider("http://127.0.0.1:1", headers=headers)


def test_outbound_combined_header_limit_and_mutated_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # The custom field fits alone, but defaults and auth make the whole header too large.
    with (
        peer(lambda connection: connection.recv(1)) as endpoint,
        pytest.raises(ValueError, match=r"headers.*limit"),
    ):
        provider(endpoint, headers={"X": "x" * (65536 - 10)}).complete(MESSAGES)
    changed = provider("http://127.0.0.1:1")
    changed.timeout = float("nan")
    with pytest.raises(ValueError, match="timeout"):
        changed.complete(MESSAGES)
    monkeypatch.setenv(KEY_ENV, "PRIVATE\r\nSECRET")
    with pytest.raises(ValueError, match="credential"):
        provider("http://127.0.0.1:1").complete(MESSAGES)


def test_https_validates_ca_and_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(
        fixtures / "provider-localhost-cert.pem", fixtures / "provider-localhost-key.pem"
    )
    # Untrusted self-signed certificate must fail under the actual default context.
    with peer(reply(wire(b"{}")), tls=server_context) as endpoint, pytest.raises(ValueError):
        provider(endpoint, timeout=2).complete(MESSAGES)
    real_default = ssl.create_default_context

    def trust_fixture() -> ssl.SSLContext:
        context = real_default(cafile=str(fixtures / "provider-localhost-cert.pem"))
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        return context

    monkeypatch.setattr(transport.ssl, "create_default_context", trust_fixture)
    actual_lookup = socket.getaddrinfo

    def fixture_lookup(host: str, *args: Any, **kwargs: Any) -> Any:
        # Route only this authored hostname to our IPv4 loopback fixture. TLS
        # still receives 'localhost' and performs actual SAN verification.
        return actual_lookup("127.0.0.1" if host == "localhost" else host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fixture_lookup)
    with peer(reply(wire(b"{}")), tls=server_context) as endpoint:
        assert (
            provider(endpoint.replace("127.0.0.1", "localhost"), timeout=2).complete(MESSAGES) == {}
        )
    # Trust does not bypass hostname verification: this certificate has no IP SAN.
    with peer(reply(wire(b"{}")), tls=server_context) as endpoint, pytest.raises(ValueError):
        provider(endpoint, timeout=2).complete(MESSAGES)


def test_stream_shares_bounds_and_preserves_actual_finish_reason() -> None:
    document = load_prompt("examples/before.json")
    (row,) = render_matrix(document, (Scenario("one", {"customer_name": "Ada", "order_id": "1"}),))
    body = (
        b':keepalive\n\ndata: {"choices":[{"delta":{"content":"partial"},'
        b'"finish_reason":"length"}]}\n'
    )
    with peer(reply(wire(body, headers=b"Content-Type: text/event-stream\r\n"))) as endpoint:
        result = OpenAICompatibleStreamingProvider(endpoint)(row)
        assert result["choices"][0]["finish_reason"] == "length"
        with pytest.raises(ValueError):
            response_text(result)
        assert result["stream_chunks"][0]["choices"][0]["finish_reason"] == "length"
    for broken in (b'data: {"SECRET":1,"SECRET":2}\n', b"data: \xff\n"):
        with peer(reply(wire(broken))) as endpoint, pytest.raises(ValueError) as caught:
            list(OpenAICompatibleStreamingProvider(endpoint).stream(row))
        assert "SECRET" not in "".join(traceback.format_exception(caught.value))


def test_transport_version_is_in_durable_provider_identity() -> None:
    identity = OpenAISessionProvider(provider("http://127.0.0.1:1")).identity
    assert identity["transport"] == "promptwitness.direct-http/v2"
    assert KEY not in json.dumps(identity)


@pytest.mark.parametrize(
    "body",
    [
        b"2\r\n{}XX0\r\n\r\n",
        b"2\r\n{}\r\n0\r\n",
        b"+2\r\n{}\r\n0\r\n\r\n",
        b"2\n{}\r\n0\r\n\r\n",
        b"2;bad=\x00\r\n{}\r\n0\r\n\r\n",
        b"2\r\n{}\r\n0\r\nContent-Length: 2\r\n\r\n",
        b"2\r\n{}\r\n0\r\nX: 1\r\nx: 2\r\n\r\n",
        b"2\r\n{}\r\n0\r\nSECRET malformed\r\n\r\n",
    ],
)
def test_strict_chunk_termination_and_trailers(body: bytes) -> None:
    with (
        peer(
            reply(wire(body, length=False, headers=b"Transfer-Encoding: chunked\r\n"))
        ) as endpoint,
        pytest.raises(ValueError) as caught,
    ):
        provider(endpoint).complete(MESSAGES)
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))


def test_valid_bounded_chunk_trailers() -> None:
    data = b"00000000002 ;\textension=yes\r\n{}\r\n0\r\nX-Checksum: local\r\n\r\n"
    with peer(
        reply(wire(data, length=False, headers=b"Transfer-Encoding: chunked\r\n"))
    ) as endpoint:
        assert provider(endpoint).complete(MESSAGES) == {}


@pytest.mark.parametrize("delta", [{"role": None}, {"content": []}, {"content": 7}])
def test_aggregate_rejects_invalid_delta_field_types(delta: Any) -> None:
    document = load_prompt("examples/before.json")
    (row,) = render_matrix(document, (Scenario("one", {"customer_name": "Ada", "order_id": "1"}),))
    data = (
        b"data: "
        + json.dumps({"choices": [{"delta": delta, "finish_reason": "stop"}]}).encode()
        + b"\n"
    )
    with peer(reply(wire(data))) as endpoint, pytest.raises(ValueError):
        OpenAICompatibleStreamingProvider(endpoint)(row)


def test_done_sentinel_does_not_hide_truncated_http_framing() -> None:
    document = load_prompt("examples/before.json")
    (row,) = render_matrix(document, (Scenario("one", {"customer_name": "Ada", "order_id": "1"}),))
    data = b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\ndata:[DONE]\n'
    with peer(reply(wire(data))) as endpoint:
        assert response_text(OpenAICompatibleStreamingProvider(endpoint)(row)) == "ok"
    bad = wire(data).replace(str(len(data)).encode(), str(len(data) + 10).encode(), 1)
    with peer(reply(bad)) as endpoint, pytest.raises(ValueError, match="incomplete"):
        OpenAICompatibleStreamingProvider(endpoint)(row)
