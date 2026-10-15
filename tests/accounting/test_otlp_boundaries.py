"""Wire-type and receiver validation boundaries with valid neighboring controls."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import pytest

from ssgrep.accounting import otlp, receiver_contract
from ssgrep.accounting.export_model import LogRecord, Point, Resource
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import L
from tests.accounting.lab_receiver import LabReceiver

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
        else:
            invalid["known_costs"] = {f"c{i}": 1 for i in range(receiver_contract.MAX_ENTRIES + 1)}

        rejected, outcome = receiver_contract.apply(accepted, key, invalid)
        assert outcome is receiver_contract.Outcome.REJECTED_INVALID
        assert rejected == accepted and rejected.current[key].revision == 1
        updated, outcome = receiver_contract.apply(rejected, key, next_revision)
        assert outcome is receiver_contract.Outcome.APPLIED_UPSERT
        assert list(updated.current) == [key] and updated.current[key].revision == 2
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
