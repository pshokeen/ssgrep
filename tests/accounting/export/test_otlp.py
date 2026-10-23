"""OTLP/HTTP JSON serialization: hand-written expected documents + generic conformance walker.

Spec rules cited (https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding):
  R1 lowerCamelCase field names; R2 enums are integers; R3 64-bit integers are decimal strings;
  R4 attributes are [{key, value:{stringValue|intValue}}]; R5 doubles are JSON numbers.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest

from ssgrep.accounting.export import otlp, receiver as receiver_contract
from ssgrep.accounting.export.otlp import (
    MAX_BATCH_BYTES,
    ExportTooLarge,
    Hist,
    LogRecord,
    Point,
    Raw,
    Resource,
    dumps,
    empty_request,
    serialize_logs,
    serialize_metrics,
    split_logs,
    split_metrics,
)
from ssgrep.accounting.ledger.ledger_contracts import Signal
from tests.accounting.export import delivery_fixtures as fx
from tests.accounting.export.delivery_fixtures import L
from tests.accounting.export.lab_receiver import LabReceiver

RES_M = Resource("1.2.3", "inst-abc", 4, counter_epoch=7)
RES_L = Resource("1.2.3", "inst-abc", 4)
ENUM_FIELDS = {"aggregationTemporality", "severityNumber"}
CAMEL = re.compile(r"^[a-z][a-zA-Z0-9]*$")
INT64_STR = re.compile(r"^-?(0|[1-9][0-9]*)$")
DATA_KEYS = {
    "attributes",
    "timeUnixNano",
    "startTimeUnixNano",
    "asInt",
    "count",
    "sum",
    "bucketCounts",
    "explicitBounds",
}
STRING_INT_FIELDS = {
    "startTimeUnixNano",
    "timeUnixNano",
    "observedTimeUnixNano",
    "asInt",
    "count",
    "intValue",
}


def _no_constants(token: str) -> None:
    raise AssertionError(f"non-finite JSON constant {token}")


def _decode(body: bytes) -> dict:
    assert body.isascii()
    return json.loads(body, parse_float=Decimal, parse_constant=_no_constants)


def _walk(node: object, path: str = "") -> None:
    """Generic OTLP JSON rules over a decoded tree (floats decoded as Decimal)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "key":  # attribute key: free-form data, not a field name
                assert isinstance(v, str)
                continue
            assert CAMEL.match(k), f"R1 non-lowerCamel field {path}/{k}"
            assert k not in {"asDouble"}, "asInt only"
            assert type(v) is not int or k in ENUM_FIELDS, f"R3 bare int {path}/{k}"
            if k in STRING_INT_FIELDS:
                assert isinstance(v, str) and INT64_STR.match(v), f"R3 {path}/{k}={v!r}"
                assert -(2**63) <= int(v) <= 2**63 - 1
            elif k == "bucketCounts":
                assert all(isinstance(c, str) and INT64_STR.match(c) for c in v)
            elif k in {"sum", "explicitBounds"} and path.endswith("dataPoints[]"):
                vals = v if isinstance(v, list) else [v]
                assert all(isinstance(x, Decimal) for x in vals), "R5 numbers"
            elif k == "isMonotonic":
                assert isinstance(v, bool)
            elif k in ENUM_FIELDS:
                assert type(v) is int, "R2"
            elif k == "value":
                assert set(v) in ({"stringValue"}, {"intValue"})
            if isinstance(v, (dict, list)):
                _walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for item in node:
            _walk(item, f"{path}[]")
    else:
        if isinstance(node, bool) or isinstance(node, (str, Decimal)):
            return
        leaf = path.rsplit("/", 1)[-1]
        raise AssertionError(f"bare JSON int {node!r} at {path} ({leaf})")


def _check(body: bytes) -> dict:
    doc = _decode(body)
    _walk(doc)
    assert b"NaN" not in body and b"Infinity" not in body
    return doc


def _sum(
    name="ssgrep.executions", attrs=(("agent", "claude"),), t=2_000, start=1_000, v=5
) -> Point:
    return Point(name, "sum", "{execution}", "Executions", attrs, t, value=v, start_unix_nano=start)


GAUGE = Point(
    "ssgrep.net", "gauge", "By", "Net bytes", (("scenario", "central"),), 3_000, value=-42
)
HIST = Point(
    "ssgrep.latency",
    "histogram",
    "s",
    "Latency",
    (("op", "search"),),
    4_000,
    hist=Hist(("0.001", "0.5", "12.345678"), (1, 2, 0, 3), 6, "12.345678"),
    start_unix_nano=1_000,
)


def _attr(k, v):
    return {"key": k, "value": v}


EXPECTED_RESOURCE_M = {
    "attributes": [
        _attr("service.name", {"stringValue": "ssgrep"}),
        _attr("service.version", {"stringValue": "1.2.3"}),
        _attr("ssgrep.counter.epoch", {"intValue": "7"}),
        _attr("ssgrep.installation.id", {"stringValue": "inst-abc"}),
        _attr("ssgrep.policy.version", {"intValue": "4"}),
    ]
}


def test_counter_canonical_exact_bytes() -> None:
    body = serialize_metrics(RES_M, [_sum()])
    expected = (
        '{"resourceMetrics":[{"resource":{"attributes":['
        '{"key":"service.name","value":{"stringValue":"ssgrep"}},'
        '{"key":"service.version","value":{"stringValue":"1.2.3"}},'
        '{"key":"ssgrep.counter.epoch","value":{"intValue":"7"}},'
        '{"key":"ssgrep.installation.id","value":{"stringValue":"inst-abc"}},'
        '{"key":"ssgrep.policy.version","value":{"intValue":"4"}}]},'
        '"scopeMetrics":[{"scope":{"name":"ssgrep.accounting","version":"1.2.3"},"metrics":['
        '{"name":"ssgrep.executions","description":"Executions","unit":"{execution}","sum":{"dataPoints":['
        '{"attributes":[{"key":"agent","value":{"stringValue":"claude"}}],'
        '"startTimeUnixNano":"1000","timeUnixNano":"2000","asInt":"5"}],'
        '"aggregationTemporality":2,"isMonotonic":true}}]}]}]}'
    )
    assert body == expected.encode()
    _check(body)


def test_gauge_negative_has_no_start_and_asint() -> None:
    body = serialize_metrics(RES_M, [GAUGE])
    doc = _check(body)
    metric = doc["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
    assert metric["gauge"] == {
        "dataPoints": [
            {
                "attributes": [_attr("scenario", {"stringValue": "central"})],
                "timeUnixNano": "3000",
                "asInt": "-42",
            }
        ]
    }
    assert b"asDouble" not in body and b"startTimeUnixNano" not in body


def test_histogram_exact_literals_and_per_bucket_strings() -> None:
    body = serialize_metrics(RES_M, [HIST])
    doc = _check(body)
    h = doc["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]["histogram"]
    assert h["aggregationTemporality"] == 2 and "isMonotonic" not in h
    dp = h["dataPoints"][0]
    assert dp["bucketCounts"] == ["1", "2", "0", "3"]
    assert dp["count"] == "6"
    assert dp["explicitBounds"] == [Decimal("0.001"), Decimal("0.5"), Decimal("12.345678")]
    assert dp["sum"] == Decimal("12.345678")
    assert b'"explicitBounds":[0.001,0.5,12.345678]' in body
    assert b'"sum":12.345678' in body
    assert b"e-" not in body and b"0.0010000000000000002" not in body
    assert dp["startTimeUnixNano"] == "1000"
    zero = Point(
        "h", "histogram", "s", "d", (), 5, hist=Hist(("1",), (0, 0), 0, "0"), start_unix_nano=1
    )
    assert b'"sum":0,' in serialize_metrics(RES_M, [zero])


def test_one_metric_per_name_unit_kind_description_sorted() -> None:
    pts = [
        _sum("b.metric", (("a", "2"),), v=1),
        _sum("a.metric", (("a", "2"),), v=2),
        _sum("a.metric", (("a", "1"),), v=3),
        GAUGE,
    ]
    doc = _check(serialize_metrics(RES_M, pts))
    metrics = doc["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    assert [m["name"] for m in metrics] == ["a.metric", "b.metric", "ssgrep.net"]
    first = metrics[0]["sum"]["dataPoints"]
    assert [d["asInt"] for d in first] == ["3", "2"]


def test_same_name_different_kind_are_separate_metrics() -> None:
    a = _sum("x")
    b = Point("x", "gauge", "{execution}", "Executions", (), 1, value=1)
    doc = _check(serialize_metrics(RES_M, [a, b]))
    assert len(doc["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]) == 2


def test_resource_attribute_key_sets_exact() -> None:
    m = _check(serialize_metrics(RES_M, [_sum()]))
    keys = {a["key"] for a in m["resourceMetrics"][0]["resource"]["attributes"]}
    assert keys == {
        "service.name",
        "service.version",
        "ssgrep.installation.id",
        "ssgrep.policy.version",
        "ssgrep.counter.epoch",
    }
    assert m["resourceMetrics"][0]["resource"] == EXPECTED_RESOURCE_M
    rec = LogRecord(10, "{}")
    lg = _check(serialize_logs(RES_L, [rec]))
    keys = {a["key"] for a in lg["resourceLogs"][0]["resource"]["attributes"]}
    assert keys == {
        "service.name",
        "service.version",
        "ssgrep.installation.id",
        "ssgrep.policy.version",
    }


def test_logs_exact_document_and_hostile_body() -> None:
    text = 'q"uote\\ new\nline\ttab \x01\x1f \u00e9 \u4e2d \U0001f600 </script>'
    rec = LogRecord(123, text, (("ssgrep.record.revision", 2), ("ssgrep.record.id", "ab")))
    body = serialize_logs(RES_L, [rec])
    doc = _check(body)
    recs = doc["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
    assert recs == [
        {
            "timeUnixNano": "123",
            "observedTimeUnixNano": "123",
            "severityNumber": 9,
            "severityText": "INFO",
            "body": {"stringValue": text},
            "attributes": [
                _attr("ssgrep.record.id", {"stringValue": "ab"}),
                _attr("ssgrep.record.revision", {"intValue": "2"}),
            ],
        }
    ]
    assert doc["resourceLogs"][0]["scopeLogs"][0]["scope"] == {
        "name": "ssgrep.accounting",
        "version": "1.2.3",
    }
    assert b"\\ud83d\\ude00" in body  # surrogate pair escape
    assert b"\\u00e9" in body and b"\\n" in body and b"\\u0001" in body
    assert b"\n" not in body and b"\t" not in body


def test_determinism_across_attribute_and_point_order() -> None:
    attrs = (("z", "1"), ("a", "2"), ("m", "3"))
    ref = serialize_metrics(RES_M, [_sum(attrs=tuple(sorted(attrs)))])
    for perm in itertools.permutations(attrs):
        assert serialize_metrics(RES_M, [_sum(attrs=perm)]) == ref
    pts = [_sum("n1"), GAUGE, HIST, _sum("n0")]
    base = serialize_metrics(RES_M, pts)
    for perm in itertools.permutations(pts):
        assert serialize_metrics(RES_M, list(perm)) == base
    assert base == serialize_metrics(RES_M, pts)
    recs = [LogRecord(2, "b"), LogRecord(1, "a")]
    assert serialize_logs(RES_L, recs) == serialize_logs(RES_L, recs[::-1])


def test_no_bare_ints_except_enums_and_bool() -> None:
    body = serialize_metrics(RES_M, [_sum(), GAUGE, HIST]) + serialize_logs(
        RES_L, [LogRecord(1, "x", (("n", 5),))]
    )
    ints: set[str] = set()

    def walk(n: object, key: str = "") -> None:
        if isinstance(n, dict):
            for k, v in n.items():
                walk(v, k)
        elif isinstance(n, list):
            for v in n:
                walk(v, key)
        elif isinstance(n, int) and not isinstance(n, bool):
            ints.add(key)

    for chunk in (
        serialize_metrics(RES_M, [_sum(), GAUGE, HIST]),
        serialize_logs(RES_L, [LogRecord(1, "x", (("n", 5),))]),
    ):
        walk(json.loads(chunk))
    assert body
    assert ints == {"aggregationTemporality", "severityNumber"}


def test_int64_bounds() -> None:
    big = _sum(v=2**63 - 1, t=2**63 - 1, start=0)
    assert b'"asInt":"9223372036854775807"' in serialize_metrics(RES_M, [big])
    for bad in (_sum(v=2**63), _sum(v=-(2**63) - 1), _sum(t=2**63), _sum(start=-1)):
        with pytest.raises(ValueError):
            serialize_metrics(RES_M, [bad])
    assert b'"asInt":"-9223372036854775808"' in serialize_metrics(
        RES_M, [Point("g", "gauge", "1", "d", (), 1, value=-(2**63))]
    )
    with pytest.raises(ValueError):
        serialize_logs(RES_L, [LogRecord(2**63, "x")])
    with pytest.raises(ValueError):
        serialize_logs(RES_L, [LogRecord(1, "x", (("n", 2**63),))])


def test_invalid_points_rejected() -> None:
    h = HIST.hist
    bad = [
        Point("s", "sum", "1", "d", (), 1, value=1),  # no start
        Point("s", "sum", "1", "d", (), 1, start_unix_nano=1),  # no value
        Point("g", "gauge", "1", "d", (), 1, value=1, start_unix_nano=1),  # gauge with start
        Point("g", "gauge", "1", "d", (), 1),  # gauge without value
        Point("h", "histogram", "s", "d", (), 1, start_unix_nano=1),  # no hist
        Point("h", "histogram", "s", "d", (), 1, hist=h),  # no start
        Point("h", "histogram", "s", "d", (), 1, hist=h, start_unix_nano=1, value=1),
        Point(
            "h", "histogram", "s", "d", (), 1, hist=Hist(("1",), (1,), 1, "1"), start_unix_nano=1
        ),  # counts len
        Point(
            "h", "histogram", "s", "d", (), 1, hist=Hist(("1",), (1, 1), 3, "1"), start_unix_nano=1
        ),  # sum != count
        Point(
            "h",
            "histogram",
            "s",
            "d",
            (),
            1,
            hist=Hist(("2", "1"), (0, 0, 0), 0, "0"),
            start_unix_nano=1,
        ),
        Point("x", "summary", "1", "d", (), 1, value=1),
        Point(
            "s", "sum", "1", "d", (("a", "1"), ("a", "2")), 1, value=1, start_unix_nano=1
        ),  # dup attr key
    ]
    for p in bad:
        with pytest.raises(ValueError):
            serialize_metrics(RES_M, [p])
        with pytest.raises(ValueError):
            split_metrics(RES_M, [p])


@pytest.mark.parametrize(
    "lit",
    [
        "1e-3",
        "0.0010000000000000002x",
        "NaN",
        "Infinity",
        "-",
        "",
        "01",
        "1.",
        ".5",
        "0.1234567890",
        "1,2",
        "1 ",
        "+1",
    ],
)
def test_raw_rejects_injection_and_non_decimals(lit: str) -> None:
    assert Raw("-0.5").token == "-0.5"  # positive control: a negative decimal is a valid literal
    with pytest.raises(ValueError):
        Raw(lit)
    p = Point(
        "h", "histogram", "s", "d", (), 1, hist=Hist(("1",), (0, 0), 0, lit), start_unix_nano=1
    )
    with pytest.raises(ValueError):
        serialize_metrics(RES_M, [p])
    p = Point(
        "h", "histogram", "s", "d", (), 1, hist=Hist((lit,), (0, 0), 0, "0"), start_unix_nano=1
    )
    with pytest.raises(ValueError):
        serialize_metrics(RES_M, [p])


def test_empty_request_bytes_and_empty_inputs() -> None:
    assert empty_request(Signal.METRICS) == b'{"resourceMetrics":[]}'
    assert empty_request(Signal.LOGS) == b'{"resourceLogs":[]}'
    assert split_metrics(RES_M, []) == []
    assert split_logs(RES_L, []) == []
    assert serialize_metrics(RES_M, []) == b'{"resourceMetrics":[]}'
    assert serialize_logs(RES_L, []) == b'{"resourceLogs":[]}'


def _points(n: int) -> list[Point]:
    return [_sum("m", (("i", f"{i:06d}"),), t=2_000, start=1_000, v=i) for i in range(n)]


def _all_asints(bodies: list[tuple[bytes, int]]) -> list[str]:
    out = []
    for b, n in bodies:
        doc = _check(b)
        dps = [
            d["asInt"]
            for m in doc["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
            for d in m["sum"]["dataPoints"]
        ]
        assert len(dps) == n
        out += dps
    return out


def test_split_metrics_small_limit_preserves_points() -> None:
    pts = _points(40)
    limit = len(serialize_metrics(RES_M, pts[:4]))
    chunks = split_metrics(RES_M, pts, max_bytes=limit)
    assert len(chunks) > 5
    assert all(len(b) <= limit for b, _ in chunks)
    assert _all_asints(chunks) == [str(i) for i in range(40)]
    assert split_metrics(RES_M, pts, max_bytes=limit) == chunks
    # exact size: each chunk is exactly what serialize_metrics would produce for its points
    start = 0
    for b, n in chunks:
        assert b == serialize_metrics(RES_M, pts[start : start + n])
        start += n


def test_split_metrics_multiple_metrics_across_chunks() -> None:
    pts = [_sum(f"m{k}", (("i", str(i)),), v=i) for k in range(3) for i in range(6)]
    limit = len(serialize_metrics(RES_M, pts[:3]))
    chunks = split_metrics(RES_M, pts, max_bytes=limit)
    assert all(len(b) <= limit for b, _ in chunks)
    seen = []
    for b, _ in chunks:
        for m in _check(b)["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]:
            seen += [(m["name"], d["asInt"]) for d in m["sum"]["dataPoints"]]
    assert seen == [(f"m{k}", str(i)) for k in range(3) for i in range(6)]


def test_split_metrics_single_point_too_large() -> None:
    p = _sum("m", (("big", "x" * 500),))
    with pytest.raises(ExportTooLarge):
        split_metrics(RES_M, [p], max_bytes=300)
    assert issubclass(ExportTooLarge, ValueError)


def test_split_metrics_over_one_mib_real_limit() -> None:
    pts = [
        Point(
            "ssgrep.big",
            "sum",
            "By",
            "d",
            (("k", "v" * 200 + f"{i:07d}"),),
            5,
            value=i,
            start_unix_nano=1,
        )
        for i in range(7000)
    ]
    whole = serialize_metrics(RES_M, pts)
    assert len(whole) > MAX_BATCH_BYTES
    chunks = split_metrics(RES_M, pts)
    assert len(chunks) >= 2
    assert all(len(b) <= MAX_BATCH_BYTES for b, _ in chunks)
    assert sum(n for _, n in chunks) == 7000
    assert _all_asints(chunks) == [str(i) for i in range(7000)]


def test_split_logs() -> None:
    recs = [LogRecord(i + 1, "payload-" + "x" * 50 + str(i)) for i in range(30)]
    limit = len(serialize_logs(RES_L, recs[:3]))
    chunks = split_logs(RES_L, recs, max_bytes=limit)
    assert len(chunks) >= 10 and all(len(b) <= limit for b, _ in chunks)
    times = []
    for b, n in chunks:
        lr = _check(b)["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
        assert len(lr) == n
        times += [int(r["timeUnixNano"]) for r in lr]
    assert times == list(range(1, 31))
    with pytest.raises(ExportTooLarge):
        split_logs(RES_L, [LogRecord(1, "y" * 2000)], max_bytes=500)


def test_split_logs_over_one_mib() -> None:
    recs = [LogRecord(i + 1, "z" * 1000 + str(i)) for i in range(2500)]
    chunks = split_logs(RES_L, recs)
    assert len(chunks) >= 2 and all(len(b) <= MAX_BATCH_BYTES for b, _ in chunks)
    assert sum(n for _, n in chunks) == 2500


def test_invalid_log_inputs() -> None:
    for rec in (
        LogRecord(1, "x", severity_number=0),
        LogRecord(1, "x", severity_number=25),
        LogRecord(1, "x", (("a", "1"), ("a", "2"))),
        LogRecord(-1, "x"),
    ):
        with pytest.raises(ValueError):
            serialize_logs(RES_L, [rec])


def test_otlp_json_preserves_null_and_decimal_tokens_but_refuses_python_float() -> None:
    value = {"missing": None, "duration": Raw("0.001"), "flags": [True, False]}
    assert dumps(value) == b'{"missing":null,"duration":0.001,"flags":[true,false]}'
    with pytest.raises(TypeError):
        dumps({**value, "duration": 0.001})
    assert json.loads(dumps(value))["missing"] is None


RESOURCE = Resource("1.2.3", "installation-control", 1, counter_epoch=1)
POINT = Point(
    "ssgrep.control", "sum", "{execution}", "Executions", (), 7, value=3, start_unix_nano=0
)
RECORD = LogRecord(7, '{"revision":1}', (("ssgrep.record.revision", 1),))


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"description": 17},
        {"value": True},
        {"value": "3"},
        {"time_unix_nano": False},
        {"start_unix_nano": "0"},
        {"attributes": (("flag", True),)},
        {"attributes": (("fraction", 1.5),)},
        {"attributes": (("missing", None),)},
    ],
    ids=[
        "empty-name",
        "nontext-description",
        "bool-value",
        "string-value",
        "bool-time",
        "string-start",
        "bool-attr",
        "float-attr",
        "null-attr",
    ],
)
def test_metrics_reject_coercible_types_without_losing_integer_wire_encoding(
    changes: dict,
) -> None:
    invalid = replace(POINT, **changes)
    for render in (otlp.serialize_metrics, otlp.split_metrics):
        with pytest.raises(ValueError):
            render(RESOURCE, [invalid])
    body = otlp.serialize_metrics(RESOURCE, [POINT])
    metric = json.loads(body)["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
    assert metric["sum"]["dataPoints"][0] == {
        "attributes": [],
        "timeUnixNano": "7",
        "startTimeUnixNano": "0",
        "asInt": "3",
    }
    assert metric["sum"]["aggregationTemporality"] == 2
    assert metric["sum"]["isMonotonic"] is True
    assert otlp.split_metrics(RESOURCE, [POINT], max_bytes=len(body)) == [(body, 1)]
    with pytest.raises(otlp.ExportTooLarge):
        otlp.split_metrics(RESOURCE, [POINT], max_bytes=len(body) - 1)


def test_metric_serializer_deduplicates_equal_points_but_rejects_conflicting_payloads() -> None:
    assert otlp.split_metrics(RESOURCE, [POINT, POINT]) == otlp.split_metrics(RESOURCE, [POINT])

    conflict = replace(POINT, value=4)
    with pytest.raises(ValueError, match="conflicting duplicate metric point"):
        otlp.split_metrics(RESOURCE, [POINT, conflict])


@pytest.mark.parametrize(
    "changes",
    [
        {"severity_number": True},
        {"severity_number": "9"},
        {"body": {"revision": 1}},
        {"time_unix_nano": "7"},
        {"attributes": (("ssgrep.record.revision", True),)},
        {"attributes": (("ssgrep.record.revision", 1.0),)},
    ],
    ids=[
        "bool-severity",
        "string-severity",
        "object-body",
        "string-time",
        "bool-revision",
        "float-revision",
    ],
)
def test_logs_reject_coercion_and_preserve_string_body_and_int64_attributes(
    changes: dict,
) -> None:
    invalid = replace(RECORD, **changes)
    for render in (otlp.serialize_logs, otlp.split_logs):
        with pytest.raises(ValueError):
            render(RESOURCE, [invalid])
    body = otlp.serialize_logs(RESOURCE, [RECORD])
    record = json.loads(body)["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert record["body"] == {"stringValue": '{"revision":1}'}
    assert record["attributes"] == [{"key": "ssgrep.record.revision", "value": {"intValue": "1"}}]
    assert record["severityNumber"] == 9 and type(record["severityNumber"]) is int
    assert record["timeUnixNano"] == record["observedTimeUnixNano"] == "7"
    assert otlp.split_logs(RESOURCE, [RECORD], max_bytes=len(body)) == [(body, 1)]
    with pytest.raises(otlp.ExportTooLarge):
        otlp.split_logs(RESOURCE, [RECORD], max_bytes=len(body) - 1)


@pytest.mark.parametrize(
    "damage",
    [
        "unknown-basis",
        "boolean-cost",
        "empty-scenarios",
        "wrong-revision-kind",
        "non-date-period",
        "too-many-costs",
        "too-many-missing-reasons",
        "wrong-scenario-shape",
        "nonboolean-complete",
        "scenario-unit-mismatch",
    ],
)
def test_receiver_rejects_invalid_adjustments_without_poisoning_the_next_valid_revision(
    damage: str,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(L,), logs_permitted=True, enabled=(L,))
        fx.commit_ledger(1)
        fx.publish(1)
        assert fx.cycle(L).acked == 1
        records = json.loads(lab.requests[0].body)["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
        assert len(records) == 1
        body = json.loads(records[0]["body"]["stringValue"])
        key = receiver_contract.Key(
            "authenticated-tenant", "authenticated-device", body["record_id"]
        )
        accepted, outcome = receiver_contract.apply(receiver_contract.ReceiverState(), key, body)
        assert outcome is receiver_contract.Outcome.APPLIED_NEW
        next_revision: dict[str, Any] = {
            **copy.deepcopy(body),
            "revision": 2,
            "supersedes_revision": 1,
            "record_type": "adjustment",
        }
        invalid = copy.deepcopy(next_revision)
        if damage == "unknown-basis":
            invalid["scenarios"]["central"]["net"]["basis"] = "guessed"
        elif damage == "boolean-cost":
            invalid["known_costs"] = {"response_chars": True}
        elif damage == "empty-scenarios":
            invalid["scenarios"] = {}
        elif damage == "wrong-revision-kind":
            invalid["record_type"] = "contribution"
        elif damage == "non-date-period":
            invalid["period"] = "yesterday"
        elif damage == "too-many-missing-reasons":
            invalid["scenarios"]["central"]["credit_missing"] = {
                f"reason-{i}": 1 for i in range(receiver_contract.MAX_ENTRIES + 1)
            }
        elif damage == "wrong-scenario-shape":
            invalid["scenarios"]["central"] = {}
        elif damage == "nonboolean-complete":
            invalid["scenarios"]["central"]["complete"] = 1
        elif damage == "scenario-unit-mismatch":
            invalid["unit"]["unit"] = "work_seconds"
        else:
            invalid["known_costs"] = {f"c{i}": 1 for i in range(receiver_contract.MAX_ENTRIES + 1)}

        rejected, outcome = receiver_contract.apply(accepted, key, invalid)
        assert outcome is receiver_contract.Outcome.REJECTED_INVALID
        assert rejected == accepted and rejected.current[key].revision == 1
        updated, outcome = receiver_contract.apply(rejected, key, next_revision)
        assert outcome is receiver_contract.Outcome.APPLIED_UPSERT
        assert list(updated.current) == [key] and updated.current[key].revision == 2
        if damage == "unknown-basis":
            without_bounds = copy.deepcopy(next_revision)
            for scenario in without_bounds["scenarios"].values():
                scenario["bounds"] = None
            no_bounds, outcome = receiver_contract.apply(accepted, key, without_bounds)
            assert outcome is receiver_contract.Outcome.APPLIED_UPSERT
            assert no_bounds.current[key].body["scenarios"]["central"]["bounds"] is None
        unit = body["unit"]
        totals = receiver_contract.current_totals(
            updated,
            methodology=body["methodology"],
            unit_key=(unit["unit"], unit["method"], unit["tokenizer"], unit["currency"]),
        )
        assert totals["rows"] == 1
        assert totals["scenarios"]["central"]["net"] == {
            "total": body["scenarios"]["central"]["net"]["value"],
            "known": 1,
            "unknown": 0,
        }
        with pytest.raises(ValueError, match="invalid authenticated tenant"):
            receiver_contract.current_totals(
                updated,
                methodology=body["methodology"],
                unit_key=(unit["unit"], unit["method"], unit["tokenizer"], unit["currency"]),
                tenant="",
            )
