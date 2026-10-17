"""Bounded HTTP/1.x request reading and response writing over a raw socket.

Hand-rolled on purpose: ``http.server`` buffers 64 KiB request lines and 100 headers and has
no total read deadline. Here the whole header block is capped at 8 KiB and must arrive within
five seconds TOTAL (a slow-drip client cannot keep re-arming a per-recv timeout). Anything not
a well-formed GET/HEAD-shaped request is rejected with a fixed machine-readable code; untrusted
input is never echoed. Every response closes the connection, so bytes after a rejected request
(smuggled bodies, pipelined requests) are never parsed as another request.
"""

from __future__ import annotations

import re
import socket
import time
from dataclasses import dataclass
from email.utils import formatdate

MAX_HEADER_BYTES = 8 * 1024
MAX_HEADERS = 64
READ_TIMEOUT_SECONDS = 5.0
WRITE_TIMEOUT_SECONDS = 5.0

_REASONS = {
    200: "OK",
    304: "Not Modified",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    410: "Gone",
    421: "Misdirected Request",
    431: "Request Header Fields Too Large",
    500: "Internal Server Error",
    503: "Service Unavailable",
}
_METHOD = re.compile(rb"[A-Z]{1,16}")
_TARGET = re.compile(rb"/[\x21-\x7e]{0,2000}")
_HEADER = re.compile(rb"([!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}):[ \t]*([\x20-\x7e\t]*?)[ \t]*")
_UNIQUE = frozenset(
    {"host", "authorization", "content-length", "transfer-encoding", "if-none-match"}
)


class WireError(Exception):
    """A request that must be answered with a fixed status and machine-readable code."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


class ClientGone(Exception):
    """The peer closed before sending a request; nothing to answer."""


@dataclass(frozen=True)
class Request:
    method: str
    target: str
    version: str
    headers: dict[str, str]
    has_body: bool


def _read_head(sock: socket.socket, timeout: float, max_bytes: int) -> bytes:
    deadline = time.monotonic() + timeout
    buffer = b""
    while True:
        end = buffer.find(b"\r\n\r\n")
        if end >= 0:
            if end + 4 > max_bytes:
                raise WireError(431, "headers_too_large")
            return buffer[:end]
        if len(buffer) >= max_bytes:
            raise WireError(431, "headers_too_large")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WireError(408, "request_timeout")
        sock.settimeout(remaining)
        try:
            chunk = sock.recv(max_bytes + 1 - len(buffer))
        except TimeoutError:
            raise WireError(408, "request_timeout") from None
        except OSError:
            raise ClientGone from None
        if not chunk:
            if buffer:
                raise WireError(400, "bad_request")
            raise ClientGone
        buffer += chunk


def _parse_line(line: bytes) -> tuple[str, str, str]:
    parts = line.split(b" ")
    if len(parts) != 3 or not _METHOD.fullmatch(parts[0]) or not _TARGET.fullmatch(parts[1]):
        raise WireError(400, "bad_request")
    version = parts[2]
    if version not in (b"HTTP/1.1", b"HTTP/1.0"):
        raise WireError(400, "bad_request")
    return parts[0].decode("ascii"), parts[1].decode("ascii"), version.decode("ascii")


def _parse_headers(lines: list[bytes]) -> dict[str, str]:
    if len(lines) > MAX_HEADERS:
        raise WireError(431, "headers_too_large")
    headers: dict[str, str] = {}
    for line in lines:
        match = _HEADER.fullmatch(line)
        if match is None:
            raise WireError(400, "bad_request")
        name = match.group(1).decode("ascii").lower()
        if name in headers and name in _UNIQUE:
            raise WireError(400, "bad_request")
        headers[name] = match.group(2).decode("ascii")
    return headers


def _has_body(headers: dict[str, str]) -> bool:
    if "transfer-encoding" in headers:
        return True
    length = headers.get("content-length")
    if length is None:
        return False
    if not length.isascii() or not length.isdigit():
        raise WireError(400, "bad_request")
    return any(char != "0" for char in length)


def read_request(
    sock: socket.socket,
    *,
    timeout: float = READ_TIMEOUT_SECONDS,
    max_bytes: int = MAX_HEADER_BYTES,
) -> Request:
    """One bounded request head; raises ``WireError`` (answer it) or ``ClientGone`` (drop it)."""
    head = _read_head(sock, timeout, max_bytes)
    first, _, rest = head.partition(b"\r\n")
    method, target, version = _parse_line(first)
    headers = _parse_headers(rest.split(b"\r\n") if rest else [])
    return Request(method, target, version, headers, _has_body(headers))


def write_response(
    sock: socket.socket,
    status: int,
    headers: list[tuple[str, str]],
    body: bytes,
    *,
    head_only: bool = False,
    declared_length: int | None = None,
) -> None:
    """Send one response and nothing else: fixed header set, ``Connection: close``.

    ``declared_length`` lets HEAD report the GET body's length; 304 sends no length at all.
    """
    lines = [f"HTTP/1.1 {status} {_REASONS.get(status, 'Error')}"]
    lines.append(f"Date: {formatdate(usegmt=True)}")
    lines.extend(f"{name}: {value}" for name, value in headers)
    if status != 304:
        length = len(body) if declared_length is None else declared_length
        lines.append(f"Content-Length: {length}")
    lines.append("Connection: close")
    payload = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
    if not head_only and status != 304:
        payload += body
    sock.settimeout(WRITE_TIMEOUT_SECONDS)
    try:
        sock.sendall(payload)
    except OSError:
        return
