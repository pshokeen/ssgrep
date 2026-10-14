"""Bounded OTLP/HTTP POST: transport only, no business logic.

``post`` never raises for transport problems; every result is an :class:`Outcome` whose
``kind``/``category`` are enumerated safe values. No exception text, response body (other
than a bounded HTTP 200 body), request body, or header value ever leaves this module.

Transport policy (checked BEFORE any socket is opened)
    * ``https`` to any host; ``http`` only to a loopback host (``127.0.0.1``, ``::1``,
      ``localhost``). URL userinfo, query, fragment, whitespace/control characters and
      other schemes are refused.
    * The request body is at most ``max_request_bytes`` (1 MiB).
    * An explicit ``Target.proxy`` (from policy) must be ``http://host[:port]`` without
      userinfo, and is used only for HTTPS targets via a CONNECT tunnel. ``https://``
      proxies are refused (TLS to the proxy itself is not implemented) and a proxy with an
      ``http://`` loopback target is refused (loopback development traffic never needs a
      proxy). Ambient proxy configuration (``*_PROXY`` env, system settings) is never read:
      only :mod:`http.client` is used and ``proxy=None`` means a direct connection.
    * TLS always verifies certificate and hostname (``ssl.create_default_context`` or the
      injected context); a handshake failure is ``tls_error`` and no request byte is sent.
    * Redirects (any 3xx) are never followed: ``redirect_refused``, no second connection.

Bounds
    * Responses: at most ``max_response_bytes`` of body are *counted as read*; one more
      byte ends the read as ``oversize_response``. A declared ``Content-Length`` over the
      cap is refused without reading the body. Header line/count limits are those of
      :mod:`http.client` (64 KiB per line, 100 headers) => ``malformed_response``.
    * Time: ``connect_timeout`` bounds connect, each socket operation is bounded by
      ``read_timeout``, and a wall-clock watchdog enforces ``connect_timeout +
      read_timeout`` across the whole exchange (slow-drip defence). The watchdog shuts the
      live socket down, so even a blocked read ends. DNS resolution itself is not
      interruptible; policy destinations are expected to be resolvable or IP literals.

Status mapping (``kind="response"``): 200 ``ok`` (body kept only here); 401 ``unauthorized``;
403 ``forbidden``; 429 ``rate_limited``; other 4xx ``http_4xx``; 5xx ``http_5xx`` (retryability
is the caller's decision, see ``export_retry``); any other 2xx or 1xx (OTLP defines only 200)
=> kind ``malformed_response``, category ``malformed``.
"""

from __future__ import annotations

import http.client
import re
import socket
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

USER_AGENT = "ssgrep-export"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
MAX_RETRY_AFTER = 3600
READ_CHUNK = 8192
MAX_URL_CHARS = 2048
MAX_HEADER_VALUE = 4096
RESERVED_HEADERS = frozenset(
    {"host", "content-length", "content-type", "connection", "user-agent", "transfer-encoding"}
)
_HEADER_NAME = re.compile(r"[A-Za-z0-9-]{1,64}")
_SAFE_CHARS = re.compile(r"[\x21-\x7e]+")
_VALUE_CHARS = re.compile(r"[\x20-\x7e]*")
_DELTA_SECONDS = re.compile(r"[0-9]{1,12}")

KINDS = (
    "response",
    "network_error",
    "timeout",
    "tls_error",
    "redirect_refused",
    "oversize_response",
    "transport_refused",
    "malformed_response",
)


def _safe_url(url: str) -> str:
    """The URL without userinfo/query/fragment, for repr only."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{host}{port}{parts.path}"
    except ValueError:
        return "<invalid-url>"


@dataclass(frozen=True)
class Target:
    url: str
    proxy: str | None = None
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def __repr__(self) -> str:
        proxy = _safe_url(self.proxy) if self.proxy else None
        count = len(self.headers)
        return f"Target(url={_safe_url(self.url)!r}, proxy={proxy!r}, headers=<{count} redacted>)"


@dataclass(frozen=True)
class Outcome:
    kind: str
    status: int | None = None
    body: bytes | None = field(default=None, repr=False)
    retry_after_seconds: int | None = None
    category: str = "network"


class _Refused(Exception):
    """Policy refusal raised before any socket exists (carries no data)."""


class _Truncated(Exception):
    """The peer closed before the declared body was complete."""


class _Oversize(Exception):
    pass


@dataclass(frozen=True)
class _Plan:
    host: str
    port: int
    https: bool
    path: str
    proxy: tuple[str, int] | None
    headers: tuple[tuple[str, str], ...]


def _check_text(value: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise _Refused


def _hostport(url: str, schemes: tuple[str, ...]) -> tuple[str, str, int, str]:
    _check_text(url, _SAFE_CHARS)
    if len(url) > MAX_URL_CHARS:
        raise _Refused
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise _Refused from error
    if parts.scheme not in schemes or not parts.hostname or "@" in parts.netloc:
        raise _Refused
    if parts.query or parts.fragment or "?" in url or "#" in url or port == 0:
        raise _Refused
    default = 443 if parts.scheme == "https" else 80
    return parts.scheme, parts.hostname, port or default, parts.path or "/"


def _plan(target: Target, body: bytes, content_type: str, max_request: int) -> _Plan:
    if len(body) > max_request:
        raise _Refused
    _check_text(content_type, _VALUE_CHARS)
    scheme, host, port, path = _hostport(target.url, ("http", "https"))
    https = scheme == "https"
    if not https and host not in LOOPBACK_HOSTS:
        raise _Refused
    proxy: tuple[str, int] | None = None
    if target.proxy is not None:
        proxy_scheme, proxy_host, proxy_port, proxy_path = _hostport(target.proxy, ("http",))
        if not https or proxy_path != "/":
            raise _Refused
        proxy = (proxy_host, proxy_port)
    seen: set[str] = set()
    for name, value in target.headers:
        _check_text(name, _HEADER_NAME)
        _check_text(value, _VALUE_CHARS)
        if len(value) > MAX_HEADER_VALUE or name.lower() in RESERVED_HEADERS | seen:
            raise _Refused
        seen.add(name.lower())
    return _Plan(host, port, https, path, proxy, target.headers)


def parse_retry_after(value: str | None, *, now: Callable[[], float] = time.time) -> int | None:
    """Delta-seconds or HTTP-date as whole seconds clamped to 0..3600; None if unparseable."""
    if value is None:
        return None
    text = value.strip()
    if _DELTA_SECONDS.fullmatch(text):
        return min(int(text), MAX_RETRY_AFTER)
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if moment.tzinfo is None:
        return None
    return max(0, min(int(moment.timestamp() - now()), MAX_RETRY_AFTER))


def read_bounded(read: Callable[[int], bytes], cap: int) -> bytes:
    """Read until EOF, counting bytes; raises ``_Oversize`` once ``cap + 1`` bytes were read."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = read(min(READ_CHUNK, cap + 1 - total))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > cap:
            raise _Oversize
        chunks.append(chunk)


class _Connection(http.client.HTTPSConnection):
    """HTTPS connection whose handshake happens in ``connect`` (before any request byte)."""

    def __init__(self, host: str, port: int, context: ssl.SSLContext, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout, context=context)
        self.tls_context = context

    def connect(self) -> None:
        http.client.HTTPConnection.connect(self)  # TCP (+ CONNECT tunnel when set)
        server_name = getattr(self, "_tunnel_host", None) or self.host
        self.sock = self.tls_context.wrap_socket(
            self.sock, server_hostname=server_name, do_handshake_on_connect=False
        )
        self.sock.do_handshake()


class _Watchdog:
    """Wall-clock deadline: shuts the live socket down when time is up."""

    def __init__(self, seconds: float) -> None:
        self.fired = threading.Event()
        self.connection: http.client.HTTPConnection | None = None
        self.sock: socket.socket | None = None  # kept: http.client drops its own on close
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True

    def start(self, connection: http.client.HTTPConnection) -> None:
        self.connection = connection
        self._timer.start()

    def _fire(self) -> None:
        self.fired.set()
        sock = self.sock or getattr(self.connection, "sock", None)
        if sock is not None:
            try:  # bypass SSLSocket.shutdown(): it mutates state another thread is reading
                socket.socket.shutdown(sock, socket.SHUT_RDWR)
            except OSError:
                pass

    def stop(self) -> None:
        self._timer.cancel()


def _category(status: int) -> str:
    if status == 200:
        return "ok"
    named = {401: "unauthorized", 403: "forbidden", 429: "rate_limited"}
    if status in named:
        return named[status]
    return "http_4xx" if 400 <= status < 500 else "http_5xx"


def _refused(category: str = "refused_transport") -> Outcome:
    return Outcome("transport_refused", category=category)


def _exchange(
    plan: _Plan,
    body: bytes,
    content_type: str,
    context: ssl.SSLContext,
    limits: tuple[float, float, int],
    dog: _Watchdog,
) -> Outcome:
    connect_timeout, read_timeout, cap = limits
    if plan.https:
        connection: http.client.HTTPConnection = _Connection(
            *(plan.proxy or (plan.host, plan.port)), context, connect_timeout
        )
        if plan.proxy:
            connection.set_tunnel(plan.host, plan.port)
    else:
        connection = http.client.HTTPConnection(plan.host, plan.port, timeout=connect_timeout)
    dog.start(connection)
    try:
        connection.connect()
        if dog.fired.is_set():
            raise TimeoutError
        assert connection.sock is not None
        dog.sock = connection.sock
        connection.sock.settimeout(read_timeout)
        connection.putrequest("POST", plan.path, skip_accept_encoding=True)
        connection.putheader("User-Agent", USER_AGENT)
        connection.putheader("Content-Type", content_type)
        connection.putheader("Content-Length", str(len(body)))
        connection.putheader("Connection", "close")
        for name, value in plan.headers:
            connection.putheader(name, value)
        connection.endheaders(body)
        return _read_response(connection, cap)
    finally:
        connection.close()


def _read_response(connection: http.client.HTTPConnection, cap: int) -> Outcome:
    response = connection.getresponse()
    status = response.status
    retry_after = parse_retry_after(response.getheader("Retry-After"))
    if 300 <= status < 400:
        return Outcome("redirect_refused", status, None, retry_after, "redirect")
    declared = response.getheader("Content-Length")
    if declared is not None and _DELTA_SECONDS.fullmatch(declared.strip()):
        if int(declared) > cap:
            return Outcome("oversize_response", status, None, retry_after, "oversize")
    data = read_bounded(response.read, cap)
    if response.length:  # connection closed before the declared body was complete
        raise _Truncated
    if status != 200 and not (400 <= status < 600):
        return Outcome("malformed_response", status, None, retry_after, "malformed")
    return Outcome(
        "response", status, data if status == 200 else None, retry_after, _category(status)
    )


def post(
    target: Target,
    body: bytes,
    *,
    content_type: str = "application/json",
    context: ssl.SSLContext | None = None,
    connect_timeout: float = 10.0,
    read_timeout: float = 10.0,
    max_response_bytes: int = 64 * 1024,
    max_request_bytes: int = 1024 * 1024,
) -> Outcome:
    """POST ``body`` once; always returns an :class:`Outcome` (see the module docstring)."""
    try:
        plan = _plan(target, body, content_type, max_request_bytes)
    except _Refused:
        return _refused()
    limits = (connect_timeout, read_timeout, max_response_bytes)
    dog = _Watchdog(connect_timeout + read_timeout)
    try:
        ctx = context or ssl.create_default_context()
        return _exchange(plan, body, content_type, ctx, limits, dog)
    except Exception as error:  # noqa: BLE001 - mapped to an enumerated category, text dropped
        if dog.fired.is_set():
            return Outcome("timeout", category="timeout")
        return _failure(error)
    finally:
        dog.stop()


def _failure(error: Exception) -> Outcome:
    """Map any transport exception to an enumerated outcome (message text is discarded)."""
    if isinstance(error, TimeoutError):
        return Outcome("timeout", category="timeout")
    if isinstance(error, _Oversize):
        return Outcome("oversize_response", category="oversize")
    if isinstance(error, ssl.SSLEOFError | ssl.SSLZeroReturnError):
        return Outcome("network_error", category="network")
    if isinstance(error, ssl.SSLError):
        return Outcome("tls_error", category="tls")
    if isinstance(error, http.client.RemoteDisconnected | http.client.IncompleteRead | _Truncated):
        return Outcome("network_error", category="network")
    if isinstance(error, http.client.InvalidURL | ValueError):
        return Outcome("malformed_response", category="malformed")
    if isinstance(error, http.client.HTTPException):
        return Outcome("malformed_response", category="malformed")
    return Outcome("network_error", category="network")
