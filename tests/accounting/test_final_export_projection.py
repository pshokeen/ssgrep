"""Histogram input validation is owned by snapshot_samples, before OTLP projection."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from ssgrep.accounting import export, otlp
from ssgrep.accounting.export_model import Resource
from ssgrep.accounting.metrics import MetricsValidationError
from tests.accounting.http_builders import make_snapshot


@pytest.mark.parametrize(
    "damage", ["missing-infinity", "unexpected-bound", "decreasing", "count-mismatch"]
)
def test_malformed_histogram_is_rejected_before_export_and_valid_control_keeps_units(
    damage: str,
) -> None:
    envelope = make_snapshot(1, methodology_id="a" * 32)
    observations = copy.deepcopy(envelope.observations)
    counters = observations["counters"]
    assert isinstance(counters, dict)
    histogram = counters["duration"][0]
    if damage == "missing-infinity":
        histogram["buckets"][-1]["le_us"] = "90000000"
    elif damage == "unexpected-bound":
        histogram["buckets"][0]["le_us"] = "private-bound"
    elif damage == "decreasing":
        histogram["buckets"][0]["count"] = histogram["count"] + 1
    else:
        histogram["count"] += 1
    with pytest.raises(MetricsValidationError):
        export.project_metrics(replace(envelope, observations=observations))
    projection = export.project_metrics(envelope)
    histograms = [p for p in projection.points if p.kind == "histogram"]
    assert len(histograms) == 2
    resource = Resource("1.0", "installation-control", 1, counter_epoch=1)
    body = json.loads(otlp.serialize_metrics(resource, histograms))
    metrics = body["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    assert len(metrics) == 1 and metrics[0]["unit"] == "s"
    points = metrics[0]["histogram"]["dataPoints"]
    assert sorted(int(p["count"]) for p in points) == [1, 3]
    assert sorted(p["sum"] for p in points) == [0.006, 0.04]
    assert all(sum(int(n) for n in p["bucketCounts"]) == int(p["count"]) for p in points)
