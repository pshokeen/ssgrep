"""Deterministic OTLP/HTTP JSON request bodies from the neutral export model (design D10).

Follows the opentelemetry-proto JSON mapping (https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding):
lowerCamelCase names, enums as integers, 64-bit integers as decimal strings. Histogram bounds and
sums are exact decimal literals that OTLP wants as JSON numbers; they are written as raw numeric
tokens so no Python float is ever created. Identical inputs always produce identical bytes.
"""

from __future__ import annotations

import http.client
import json
import random
import re
import socket
import ssl
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

from ssgrep.accounting.ledger.ledger_contracts import Signal

MAX_BATCH_BYTES = 1024 * 1024
SERVICE_NAME = "ssgrep"
SCOPE_NAME = "ssgrep.accounting"


@dataclass(frozen=True)
class Resource:
    """Approved pseudonymous resource attributes (design D10).

    service name/version, a destination-scoped installation pseudonym, the policy version, and
    (metrics only) the persisted counter epoch. No organization subject, tenant, or device name
    is ever asserted by the client: the collector derives authoritative identity from the
    enrolled credential and overwrites or rejects anything the payload claims.
    """

    service_version: str
    installation_id: str
    policy_version: int
    counter_epoch: int | None = None


@dataclass(frozen=True)
class Hist:
    """Explicit-bounds histogram data point. ``counts`` are PER-BUCKET (not cumulative)."""

    bounds: tuple[str, ...]  # ascending finite upper bounds as decimal literals (seconds)
    counts: tuple[int, ...]  # len(bounds) + 1; the last bucket is +Inf
    count: int
    sum: str  # exact decimal literal in the metric's unit (seconds)


@dataclass(frozen=True)
class Point:
    """One OTLP data point (sum, gauge or histogram) with its metric metadata."""

    name: str
    kind: str  # "sum" (monotonic, cumulative) | "gauge" | "histogram" (cumulative)
    unit: str  # UCUM: "s", "By", "1", "{execution}" ...
    description: str
    attributes: tuple[tuple[str, str], ...]
    time_unix_nano: int
    value: int | None = None  # sum / gauge
    hist: Hist | None = None
    start_unix_nano: int | None = None  # REQUIRED for sum/histogram: persisted epoch start


@dataclass(frozen=True)
class LogRecord:
    """One OTLP log record; ``body`` is the canonical JSON text of a ledger record body."""

    time_unix_nano: int
    body: str
    attributes: tuple[tuple[str, str | int], ...] = ()  # int -> AnyValue.intValue (string)
    severity_number: int = 9  # INFO


@dataclass(frozen=True)
class Batch:
    """One immutable request: exact bytes plus the identities needed to track it."""

    batch_id: str
    signal: Signal
    generation: int
    snapshot_id: str
    index: int
    total: int
    kind: str  # "data" | "tombstone"
    body: bytes
    items: int  # data points or log records in the request
    record_ids: tuple[str, ...] = ()


FIELDS_VERSION = {Signal.METRICS.value: "metrics-v2", Signal.LOGS.value: "logs-v2"}

_CUMULATIVE = 2
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]{1,9})?")
_HOLE = b"\x00"  # never occurs in rendered output: control characters are escaped


class ExportTooLarge(ValueError):
    """A single data point or log record cannot fit in one request."""


@dataclass(frozen=True)
class Raw:
    """A validated numeric JSON token written verbatim (never a Python float)."""

    token: str

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or _NUMBER.fullmatch(self.token) is None:
            raise ValueError(f"invalid decimal literal: {self.token!r}")


@dataclass(frozen=True)
class _Pre:
    """Already-rendered bytes spliced in as-is (internal)."""

    data: bytes


def _write(value: Any, out: list[bytes]) -> None:
    if isinstance(value, _Pre):
        out.append(value.data)
    elif isinstance(value, Raw):
        out.append(value.token.encode("ascii"))
    elif isinstance(value, bool):
        out.append(b"true" if value else b"false")
    elif value is None:
        out.append(b"null")
    elif isinstance(value, int):
        out.append(str(value).encode("ascii"))
    elif isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=True).encode("ascii"))
    elif isinstance(value, dict):
        out.append(b"{")
        for i, (key, item) in enumerate(value.items()):
            if i:
                out.append(b",")
            out.append(json.dumps(str(key), ensure_ascii=True).encode("ascii") + b":")
            _write(item, out)
        out.append(b"}")
    elif isinstance(value, (list, tuple)):
        out.append(b"[")
        for i, item in enumerate(value):
            if i:
                out.append(b",")
            _write(item, out)
        out.append(b"]")
    else:
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def dumps(value: Any) -> bytes:
    """Compact, ASCII-only JSON bytes; ``Raw`` tokens are emitted verbatim."""
    out: list[bytes] = []
    _write(value, out)
    return b"".join(out)


def _i64(value: object, what: str, *, non_negative: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an int")
    if not _INT64_MIN <= value <= _INT64_MAX or (non_negative and value < 0):
        raise ValueError(f"{what} out of range")
    return str(value)


def _attrs(items: Sequence[tuple[str, str | int]]) -> list[dict[str, Any]]:
    keys = [k for k, _ in items]
    if len(set(keys)) != len(keys) or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("attribute keys must be unique non-empty strings")
    out: list[dict[str, Any]] = []
    for key, val in sorted(items, key=lambda kv: kv[0]):
        if isinstance(val, bool):
            raise ValueError("bool attribute values are not allowed")
        if isinstance(val, int):
            any_value: dict[str, Any] = {"intValue": _i64(val, f"attribute {key}")}
        elif isinstance(val, str):
            any_value = {"stringValue": val}
        else:
            raise ValueError(f"attribute {key} has unsupported type")
        out.append({"key": key, "value": any_value})
    return out


def _resource(resource: Resource) -> dict[str, Any]:
    attrs: list[tuple[str, str | int]] = [
        ("service.name", SERVICE_NAME),
        ("service.version", resource.service_version),
        ("ssgrep.installation.id", resource.installation_id),
        ("ssgrep.policy.version", resource.policy_version),
    ]
    if resource.counter_epoch is not None:
        attrs.append(("ssgrep.counter.epoch", resource.counter_epoch))
    return {"attributes": _attrs(attrs)}


def _scope(resource: Resource) -> dict[str, Any]:
    return {"name": SCOPE_NAME, "version": resource.service_version}


def empty_request(signal: Signal) -> bytes:
    """Request with no data, used by connectivity checks."""
    return b'{"resourceMetrics":[]}' if signal == Signal.METRICS else b'{"resourceLogs":[]}'


def _envelope(resource: Resource, outer: str, scope_key: str, inner: str) -> tuple[bytes, bytes]:
    doc = {
        outer: [
            {
                "resource": _resource(resource),
                scope_key: [{"scope": _scope(resource), inner: [_Pre(_HOLE)]}],
            }
        ]
    }
    head, tail = dumps(doc).split(_HOLE)
    return head, tail


# ---- metrics --------------------------------------------------------------------------------


def _check_point(p: Point) -> None:
    for text in (p.name, p.unit, p.description):
        if not isinstance(text, str):
            raise ValueError("metric name/unit/description must be strings")
    if not p.name:
        raise ValueError("metric name required")
    _i64(p.time_unix_nano, "timeUnixNano", non_negative=True)
    if p.kind in ("sum", "gauge"):
        if p.hist is not None or p.value is None:
            raise ValueError(f"{p.kind} point needs value and no histogram")
        _i64(p.value, "asInt", non_negative=p.kind == "sum")
        if p.kind == "sum":
            if p.start_unix_nano is None:
                raise ValueError("sum point needs start_unix_nano")
            _i64(p.start_unix_nano, "startTimeUnixNano", non_negative=True)
        elif p.start_unix_nano is not None:
            raise ValueError("gauge point must not carry start_unix_nano")
    elif p.kind == "histogram":
        h = p.hist
        if h is None or p.value is not None or p.start_unix_nano is None:
            raise ValueError("histogram point needs hist and start, no value")
        _i64(p.start_unix_nano, "startTimeUnixNano", non_negative=True)
        if len(h.counts) != len(h.bounds) + 1:
            raise ValueError("histogram counts must be len(bounds)+1")
        for c in h.counts:
            _i64(c, "bucketCount", non_negative=True)
        _i64(h.count, "count", non_negative=True)
        if sum(h.counts) != h.count:
            raise ValueError("histogram bucket counts must sum to count")
        Raw(h.sum)
        bounds = [Decimal(Raw(b).token) for b in h.bounds]
        if any(a >= b for a, b in zip(bounds, bounds[1:], strict=False)):
            raise ValueError("histogram bounds must be strictly ascending")
    else:
        raise ValueError(f"unknown point kind: {p.kind!r}")


def _data_point(p: Point) -> bytes:
    attrs = _attrs(p.attributes)
    if p.kind == "gauge":
        dp: dict[str, Any] = {
            "attributes": attrs,
            "timeUnixNano": _i64(p.time_unix_nano, "t"),
            "asInt": _i64(p.value, "v"),
        }
    elif p.kind == "sum":
        dp = {
            "attributes": attrs,
            "startTimeUnixNano": _i64(p.start_unix_nano, "s"),
            "timeUnixNano": _i64(p.time_unix_nano, "t"),
            "asInt": _i64(p.value, "v"),
        }
    else:
        h = p.hist
        assert h is not None
        dp = {
            "attributes": attrs,
            "startTimeUnixNano": _i64(p.start_unix_nano, "s"),
            "timeUnixNano": _i64(p.time_unix_nano, "t"),
            "count": _i64(h.count, "c"),
            "sum": Raw(h.sum),
            "bucketCounts": [_i64(c, "b") for c in h.counts],
            "explicitBounds": [Raw(b) for b in h.bounds],
        }
    return dumps(dp)


def _metric_key(p: Point) -> tuple[str, str, str, str]:
    return (p.name, p.unit, p.kind, p.description)


def _metric_wrapper(key: tuple[str, str, str, str]) -> tuple[bytes, bytes]:
    name, unit, kind, description = key
    hole = [_Pre(_HOLE)]
    if kind == "sum":
        body: dict[str, Any] = {
            "dataPoints": hole,
            "aggregationTemporality": _CUMULATIVE,
            "isMonotonic": True,
        }
    elif kind == "gauge":
        body = {"dataPoints": hole}
    else:
        body = {"dataPoints": hole, "aggregationTemporality": _CUMULATIVE}
    head, tail = dumps({"name": name, "description": description, "unit": unit, kind: body}).split(
        _HOLE
    )
    return head, tail


def _sorted_points(points: Sequence[Point]) -> list[tuple[Point, bytes]]:
    unique: dict[tuple, Point] = {}
    for point in points:
        _check_point(point)
        attributes = tuple(sorted(point.attributes))
        normalized = (
            point if point.attributes == attributes else replace(point, attributes=attributes)
        )
        key = (*_metric_key(normalized), attributes, normalized.time_unix_nano)
        held = unique.setdefault(key, normalized)
        if held != normalized:
            raise ValueError("conflicting duplicate metric point")
    ordered = sorted(
        unique.values(), key=lambda p: (*_metric_key(p), p.attributes, p.time_unix_nano)
    )
    return [(point, _data_point(point)) for point in ordered]


def _assemble_metrics(
    env: tuple[bytes, bytes], groups: list[tuple[tuple[str, str, str, str], list[bytes]]]
) -> bytes:
    metrics = []
    for key, dps in groups:
        head, tail = _metric_wrapper(key)
        metrics.append(head + b",".join(dps) + tail)
    return env[0] + b",".join(metrics) + env[1]


def _group(
    items: Sequence[tuple[Point, bytes]],
) -> list[tuple[tuple[str, str, str, str], list[bytes]]]:
    groups: list[tuple[tuple[str, str, str, str], list[bytes]]] = []
    for p, dp in items:
        key = _metric_key(p)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(dp)
        else:
            groups.append((key, [dp]))
    return groups


def serialize_metrics(resource: Resource, points: Sequence[Point]) -> bytes:
    if not points:
        return empty_request(Signal.METRICS)
    env = _envelope(resource, "resourceMetrics", "scopeMetrics", "metrics")
    return _assemble_metrics(env, _group(_sorted_points(points)))


def split_metrics(
    resource: Resource, points: Sequence[Point], *, max_bytes: int = MAX_BATCH_BYTES
) -> list[tuple[bytes, int]]:
    """Greedy, deterministic chunking over the sorted points; every body is <= ``max_bytes``."""
    if not points:
        return []
    env = _envelope(resource, "resourceMetrics", "scopeMetrics", "metrics")
    base = len(env[0]) + len(env[1])
    overhead: dict[tuple[str, str, str, str], int] = {}
    chunks: list[list[tuple[Point, bytes]]] = [[]]
    size = base
    last: tuple[str, str, str, str] | None = None
    for p, dp in _sorted_points(points):
        key = _metric_key(p)
        if key not in overhead:
            head, tail = _metric_wrapper(key)
            overhead[key] = len(head) + len(tail)
        alone = base + overhead[key] + len(dp)
        if alone > max_bytes:
            raise ExportTooLarge(f"data point {p.name} needs {alone} bytes > {max_bytes}")
        cur = chunks[-1]
        add = len(dp) + (1 if key == last else (overhead[key] + (1 if cur else 0)))
        if cur and size + add > max_bytes:
            chunks.append([])
            cur = chunks[-1]
            add = overhead[key] + len(dp)
            size = base
        cur.append((p, dp))
        size += add
        last = key
    out = [(_assemble_metrics(env, _group(c)), len(c)) for c in chunks]
    if any(len(b) > max_bytes for b, _ in out):
        raise AssertionError("chunk size accounting drifted")  # pragma: no cover
    return out


# ---- logs -----------------------------------------------------------------------------------


def _log_bytes(r: LogRecord) -> tuple[int, bytes]:
    t = _i64(r.time_unix_nano, "timeUnixNano", non_negative=True)
    if isinstance(r.severity_number, bool) or not isinstance(r.severity_number, int):
        raise ValueError("severity_number must be an int")
    if not 1 <= r.severity_number <= 24:
        raise ValueError("severity_number out of range")
    if not isinstance(r.body, str):
        raise ValueError("log body must be a string")
    doc = {
        "timeUnixNano": t,
        "observedTimeUnixNano": t,
        "severityNumber": r.severity_number,
        "severityText": "INFO",
        "body": {"stringValue": r.body},
        "attributes": _attrs(r.attributes),
    }
    return r.time_unix_nano, dumps(doc)


def _sorted_logs(records: Sequence[LogRecord]) -> list[bytes]:
    rendered = [_log_bytes(r) for r in records]
    rendered.sort()
    return [b for _, b in rendered]


def serialize_logs(resource: Resource, records: Sequence[LogRecord]) -> bytes:
    if not records:
        return empty_request(Signal.LOGS)
    head, tail = _envelope(resource, "resourceLogs", "scopeLogs", "logRecords")
    return head + b",".join(_sorted_logs(records)) + tail


def split_logs(
    resource: Resource, records: Sequence[LogRecord], *, max_bytes: int = MAX_BATCH_BYTES
) -> list[tuple[bytes, int]]:
    if not records:
        return []
    head, tail = _envelope(resource, "resourceLogs", "scopeLogs", "logRecords")
    base = len(head) + len(tail)
    chunks: list[list[bytes]] = [[]]
    size = base
    for rec in _sorted_logs(records):
        if base + len(rec) > max_bytes:
            raise ExportTooLarge(f"log record needs {base + len(rec)} bytes > {max_bytes}")
        add = len(rec) + (1 if chunks[-1] else 0)
        if chunks[-1] and size + add > max_bytes:
            chunks.append([])
            add = len(rec)
            size = base
        chunks[-1].append(rec)
        size += add
    return [(head + b",".join(c) + tail, len(c)) for c in chunks]


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
is the caller's decision; the caller applies this module's retry policy); other 2xx or 1xx
(``OTLP`` defines only 200) => kind ``malformed_response``, category ``malformed``.
"""


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


BASE_SECONDS = 1
MAX_BACKOFF_SECONDS = 300
MAX_DELAY_SECONDS = 3600
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
BLOCKING_STATUSES = frozenset({401, 403})
RETRYABLE_KINDS = frozenset({"network_error", "timeout"})

ACK_CANDIDATE = "ack_candidate"
RETRY = "retry"
BLOCKED = "blocked"
PERMANENT = "permanent"
REFUSED = "refused"


def backoff_seconds(
    attempt: int, retry_after: int | None, *, rand: Callable[[], float] = random.random
) -> int:
    """Seconds to wait before retry number ``attempt`` (1-based); see the module docstring."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be an integer >= 1")
    exponent = min(attempt - 1, 20)  # 2**20 s is far beyond the cap; avoids huge shifts
    delay = min(BASE_SECONDS << exponent, MAX_BACKOFF_SECONDS)
    fraction = min(max(rand(), 0.0), 1.0)
    low = delay / 2
    jittered = int(low + (delay - low) * fraction)
    result = max(jittered, 1)
    if retry_after is not None and retry_after > 0:
        result = max(result, retry_after)
    return min(result, MAX_DELAY_SECONDS)


def is_retryable(kind: str, status: int | None) -> bool:
    """True for transient failures: disconnects, timeouts, and HTTP 429/502/503/504."""
    if kind in RETRYABLE_KINDS:
        return True
    return kind == "response" and status in RETRYABLE_STATUSES


def classify(outcome: Outcome) -> str:
    """One of ``ack_candidate | retry | blocked | permanent | refused``.

    ``ack_candidate``: HTTP 200 (the body still has to be parsed for partial success).
    ``retry``: transient, same immutable batch. ``blocked``: 401/403, the destination stays
    blocked until credentials/config change. ``permanent``: other 4xx/5xx, quarantine.
    ``refused``: local policy/TLS/redirect/size/format failures that need a config change.
    """
    if outcome.kind == "response":
        status = outcome.status
        if status == 200:
            return ACK_CANDIDATE
        if status in BLOCKING_STATUSES:
            return BLOCKED
        if is_retryable(outcome.kind, status):
            return RETRY
        return PERMANENT
    if is_retryable(outcome.kind, outcome.status):
        return RETRY
    return REFUSED
