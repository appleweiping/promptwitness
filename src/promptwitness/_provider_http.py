"""Private direct HTTP(S) transport with fixed, versioned resource limits."""

from __future__ import annotations

import http.client
import json
import re
import socket
import ssl
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

from .parser import _finite_float, _reject_non_finite, _unique_object

TRANSPORT_VERSION = "promptwitness.direct-http/v2"
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_WIRE_BYTES = 32 * 1024 * 1024
MAX_LINE_BYTES = 8192


def endpoint_parts(value: Any) -> tuple[str, str, int, str, str]:
    try:
        if not isinstance(value, str) or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
            raise ValueError
        parsed = urlsplit(value)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or "%" in parsed.hostname
            or "\\" in value
        ):
            raise ValueError
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError
        host = parsed.hostname
        authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        return parsed.scheme, host, port, authority, target
    except ValueError:
        raise ValueError(
            "endpoint must be an absolute HTTP(S) URL without credentials or fragment"
        ) from None


def timeout_value(value: Any) -> float:
    if type(value) not in (int, float) or not 0 < value <= 300:
        raise ValueError("timeout must be a finite number in (0, 300]")
    return float(value)


def checked_headers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("headers must map strings to strings")
    result: dict[str, str] = {}
    seen = set()
    size = 0
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError("headers must map strings to strings")
        if (
            re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) is None
            or key.lower() in seen
            or key.lower()
            in {
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
                "proxy-authorization",
                "proxy-connection",
                "accept-encoding",
            }
            or any(not (32 <= ord(char) <= 126 or char == "\t") for char in item)
        ):
            raise ValueError("provider headers contain invalid, duplicate or reserved fields")
        size += len(key) + len(item) + 4
        if size > MAX_HEADER_BYTES:
            raise ValueError("provider headers exceed byte limit")
        seen.add(key.lower())
        result[key] = item
    return result


def json_object(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_finite,
            parse_float=_finite_float,
        )
        if not isinstance(payload, dict):
            raise ValueError
        pending = [(payload, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 64:
                raise ValueError
            if isinstance(item, dict):
                for key, nested in item.items():
                    key.encode("utf-8")
                    pending.append((nested, depth + 1))
            elif isinstance(item, list):
                pending.extend((nested, depth + 1) for nested in item)
            elif isinstance(item, str):
                item.encode("utf-8")
        return payload
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("provider returned invalid JSON object") from None


class _Reader:
    """Count headers and all body framing before http.client parses/allocates it."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.header_phase = True
        self.count = 0

    def _read(self, method: str, count: int = -1) -> bytes:
        limit = MAX_HEADER_BYTES if self.header_phase else MAX_WIRE_BYTES
        remaining = limit + 1 - self.count
        size = remaining if count < 0 else min(count, remaining)
        if method == "readline":
            size = min(size, MAX_LINE_BYTES + 1)
        result: bytes = getattr(self.stream, method)(size)
        self.count += len(result)
        if self.count > limit or (method == "readline" and len(result) > MAX_LINE_BYTES):
            raise ValueError("provider response exceeds framing byte limit")
        return result

    def readline(self, count: int = -1) -> bytes:
        return self._read("readline", count)

    def read(self, count: int = -1) -> bytes:
        return self._read("read", count)

    def read1(self, count: int = -1) -> bytes:
        return self._read("read1", count)

    def close(self) -> None:
        self.stream.close()

    def flush(self) -> None:
        return


class _SocketStream:
    """Recoverable short raw-socket polls; makefile timeouts are not recoverable."""

    def __init__(self, sock: socket.socket, deadline: float, owner: Any) -> None:
        self.sock = sock
        self.owner = owner
        self.deadline = deadline
        self.buffer = bytearray()
        self.closed = False

    def _receive(self, count: int) -> bytes:
        while True:
            remaining = self.deadline - time.monotonic()
            if self.closed or remaining <= 0:
                raise ValueError("provider request deadline exceeded")
            self.sock.settimeout(min(0.05, remaining))
            try:
                return self.sock.recv(min(count, 65536))
            except TimeoutError:
                continue

    def read1(self, count: int) -> bytes:
        if self.buffer:
            result = bytes(self.buffer[:count])
            del self.buffer[:count]
            return result
        return self._receive(count) if count else b""

    def read(self, count: int) -> bytes:
        value = bytearray()
        while len(value) < count:
            chunk = self.read1(count - len(value))
            if not chunk:
                break
            value.extend(chunk)
        return bytes(value)

    def readline(self, count: int) -> bytes:
        while True:
            end = self.buffer.find(b"\n", 0, count)
            if end >= 0 or len(self.buffer) >= count:
                size = end + 1 if end >= 0 else count
                result = bytes(self.buffer[:size])
                del self.buffer[:size]
                return result
            chunk = self._receive(count - len(self.buffer))
            if not chunk:
                result = bytes(self.buffer)
                self.buffer.clear()
                return result
            self.buffer.extend(chunk)

    def close(self) -> None:
        self.closed = True
        self.buffer.clear()
        self.owner.close()


class _Response(http.client.HTTPResponse):
    def __init__(self, sock: socket.socket, *, deadline: float, **kwargs: Any) -> None:
        super().__init__(sock, **kwargs)
        # HTTPConnection closes its socket as soon as a close-delimited response
        # begins. Keep the original makefile reference alive to retain that fd,
        # but read only through recoverable raw-socket polls.
        self.fp = _Reader(_SocketStream(sock, deadline, self.fp))  # type: ignore[assignment]

    def begin(self) -> None:
        if self.headers is not None:
            return
        super().begin()
        self.fp.header_phase = False

    def _get_chunk_left(self) -> int | None:
        if self.chunk_left:
            return int(self.chunk_left)
        if self.chunk_left == 0 and self.fp.read(2) != b"\r\n":
            raise ValueError("provider chunk delimiter is invalid")
        line = self.fp.readline(MAX_LINE_BYTES + 1)
        if not line.endswith(b"\r\n"):
            raise ValueError("provider chunk size is incomplete")
        hexadecimal, _, extension = line[:-2].partition(b";")
        hexadecimal = hexadecimal.strip(b" \t")
        if re.fullmatch(rb"[0-9A-Fa-f]+", hexadecimal) is None or any(
            (byte < 32 and byte != 9) or byte == 127 for byte in extension
        ):
            raise ValueError("provider chunk size is invalid")
        count = int(hexadecimal, 16)
        if count > MAX_BODY_BYTES:
            raise ValueError("provider chunk exceeds body byte limit")
        if count:
            self.chunk_left = count
            return count
        trailer_bytes = 0
        names = set()
        while True:
            trailer = self.fp.readline(MAX_LINE_BYTES + 1)
            trailer_bytes += len(trailer)
            if not trailer.endswith(b"\r\n") or trailer_bytes > MAX_HEADER_BYTES:
                raise ValueError("provider trailers are incomplete or exceed byte limit")
            if trailer == b"\r\n":
                finished_stream = self.fp
                self.fp = None  # type: ignore[assignment] # HTTPResponse's documented closed state
                finished_stream.close()
                self.chunk_left = None
                return None
            name, separator, value = trailer[:-2].partition(b":")
            if (
                not separator
                or re.fullmatch(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name) is None
                or name.lower() in names
                or name.lower()
                in {
                    b"content-length",
                    b"transfer-encoding",
                    b"content-type",
                    b"content-encoding",
                    b"host",
                    b"authorization",
                }
                or any((byte < 32 and byte != 9) or byte == 127 for byte in value)
            ):
                raise ValueError("provider trailers are invalid")
            names.add(name.lower())


class _Connection(http.client.HTTPConnection):
    deadline: float

    def send(self, data: Any) -> None:
        if self.sock is None or not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("provider request transport is unavailable")
        pending = memoryview(data)
        while pending:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("provider request deadline exceeded")
            self.sock.settimeout(min(0.05, remaining))
            try:
                sent = self.sock.send(pending[:65536])
            except TimeoutError:
                continue
            if not sent:
                raise ValueError("provider request transport is unavailable")
            pending = pending[sent:]


def _framing(response: http.client.HTTPResponse, *, streaming: bool) -> int | None:
    if response.msg.defects:
        raise ValueError("provider returned malformed response headers")
    lengths = response.headers.get_all("Content-Length", [])
    transfers = response.headers.get_all("Transfer-Encoding", [])
    types = response.headers.get_all("Content-Type", [])
    if (
        len(lengths) > 1
        or len(transfers) > 1
        or (transfers and lengths)
        or (transfers and transfers[0].lower() != "chunked")
        or response.headers.get_all("Content-Encoding", [])
        or len(types) > 1
    ):
        raise ValueError("provider returned ambiguous or unsupported response framing")
    if types:
        media = "text/event-stream" if streaming else "application/json"
        if (
            re.fullmatch(
                re.escape(media) + r"(?:[ \t]*;[ \t]*charset[ \t]*=[ \t]*(?:utf-8|\"utf-8\"))?",
                types[0].strip(" \t"),
                re.IGNORECASE,
            )
            is None
        ):
            raise ValueError("provider returned an unsupported content type")
    if lengths:
        if re.fullmatch(r"[0-9]{1,10}", lengths[0]) is None or int(lengths[0]) > MAX_BODY_BYTES:
            raise ValueError("provider response length is invalid or exceeds byte limit")
        return int(lengths[0])
    return None


@contextmanager
def exchange(
    endpoint: str,
    timeout: float,
    headers: Mapping[str, str],
    body: bytes,
    *,
    streaming: bool = False,
) -> Iterator[http.client.HTTPResponse]:
    scheme, host, port, authority, target = endpoint_parts(endpoint)
    timeout = timeout_value(timeout)
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("provider request exceeds byte limit")
    started = time.monotonic()
    active: list[socket.socket | None] = [None]
    connection = _Connection(host, port, timeout=timeout)
    connection.deadline = started + timeout

    class TimedResponse(_Response):
        def __init__(self, sock: socket.socket, **kwargs: Any) -> None:
            super().__init__(sock, deadline=started + timeout, **kwargs)

    connection.response_class = TimedResponse

    def remaining() -> float:
        value = timeout - (time.monotonic() - started)
        if value <= 0:
            raise ValueError("provider request deadline exceeded")
        return value

    try:
        # getaddrinfo itself cannot be interrupted by the stdlib socket timeout.
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        connected = False
        for family, kind, proto, _, address in addresses:
            remaining()
            sock = socket.socket(family, kind, proto)
            active[0] = sock
            sock.settimeout(remaining())
            try:
                sock.connect(address)
            except OSError:
                sock.close()
                continue
            connected = True
            break
        if not connected or active[0] is None:
            raise ValueError("provider connection failed")
        current = active[0]
        if scheme == "https":
            context = ssl.create_default_context()
            current = context.wrap_socket(
                current, server_hostname=host, do_handshake_on_connect=False
            )
            active[0] = current
            current.settimeout(remaining())
            current.do_handshake()
        current.settimeout(remaining())
        connection.sock = current
        request_headers = {
            **headers,
            "Host": authority,
            "Connection": "close",
            "Accept-Encoding": "identity",
            "Content-Length": str(len(body)),
        }
        header_bytes = len(f"POST {target} HTTP/1.1\r\n\r\n".encode("ascii")) + sum(
            len(f"{key}: {value}\r\n".encode("ascii")) for key, value in request_headers.items()
        )
        if header_bytes > MAX_HEADER_BYTES:
            raise ValueError("provider request headers exceed byte limit")
        connection.request(
            "POST",
            target,
            body,
            request_headers,
        )
        with connection.getresponse() as response:
            if response.status != 200:
                # Do not read or retain an error body; close it immediately.
                raise ValueError(f"provider returned HTTP {response.status}")
            expected = _framing(response, streaming=streaming)
            # Keep expected byte count after HTTPResponse decrements its length.
            response._promptwitness_expected = expected  # type: ignore[attr-defined]
            yield response
            remaining()
    except (OSError, http.client.HTTPException):
        raise ValueError("provider request failed or exceeded its deadline") from None
    finally:
        connection.close()
        if active[0] is not None:
            active[0].close()


def response_chunks(response: http.client.HTTPResponse) -> Iterator[bytes]:
    size = 0
    while chunk := response.read1(min(65536, MAX_BODY_BYTES + 1 - size)):
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise ValueError("provider response exceeds body byte limit")
        yield chunk
    if response._promptwitness_expected is not None and size != response._promptwitness_expected:  # type: ignore[attr-defined]
        raise ValueError("provider response body is incomplete")
