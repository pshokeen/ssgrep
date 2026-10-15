"""Deterministic OTLP/HTTP JSON request bodies from the neutral export model (design D10).

Follows the opentelemetry-proto JSON mapping (https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding):
lowerCamelCase names, enums as integers, 64-bit integers as decimal strings. Histogram bounds and
sums are exact decimal literals that OTLP wants as JSON numbers; they are written as raw numeric
tokens so no Python float is ever created. Identical inputs always produce identical bytes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from ssgrep.accounting.export_model import (
    MAX_BATCH_BYTES,
    SCOPE_NAME,
    SERVICE_NAME,
    LogRecord,
    Point,
    Resource,
)
from ssgrep.accounting.ledger_contracts import Signal

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
