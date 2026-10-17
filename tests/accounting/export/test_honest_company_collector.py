"""27.1 real OTLP Collector lane: current revisions, uncertainty and enrolled identity.

No network/container runs during authoring. Parent runs this lane after freeze;
it skips only for the existing lab's named missing Docker/pinned-image prerequisite.
"""

import json

import pytest

from ssgrep.accounting.export import (
    acks,
    otlp,
    otlp as export_http,
    records as export_records,
    registry as export_registry,
)
from ssgrep.accounting.export.otlp import Resource
from ssgrep.accounting.export.receiver import Key, Outcome, ReceiverState, apply, current_totals
from ssgrep.accounting.export.registry import queue
from ssgrep.accounting.views import snapshots
from tests.accounting.export import delivery_fixtures as fx, otel_collector_lab as lab
from tests.accounting.export.test_honest_company_projection import DEST, KEY, METHOD, envelope, rows


@pytest.fixture
def collector():
    reason = lab.unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    run = lab.new_run_id()
    instance = lab.Collector.start(run, "honest-company")
    try:
        yield instance
    finally:
        instance.stop()
        assert lab.labelled(run) == []


def target(collector, signal, token=lab.TOKEN_A):
    return export_http.Target(
        collector.base_url + f"/v1/{signal}", headers=(("Authorization", "Bearer " + token),)
    )


def work_candidate(missing):
    candidates, omitted = export_records.project_candidates(
        rows(missing), key=KEY, destination=DEST
    )
    assert omitted == 0
    return next(candidate for candidate in candidates if candidate.unit[0] == "work_seconds")


def test_real_collector_preserves_unknown_net_finite_bounds_and_measured_latency(collector):
    fx.configure(collector.base_url)
    snapshots.publish(fx.PROFILE, envelope())
    before = collector.count("metrics")
    result = fx.cycle(deps=fx.deps(environ={fx.ENV_NAME: lab.TOKEN_A}))
    assert result.sent == result.acked == 1
    state = fx.state()
    assert (
        state.last_collected_generation
        == state.last_attempted_generation
        == state.last_acked_generation
        == 1
    )
    documents = collector.wait_records("metrics", 1, since=before)
    points = [row for document in documents for row in lab.metric_points(document)]
    assert not any(row["name"] == "ssgrep_estimated_net_work_seconds" for row in points)
    known = [row for row in points if row["name"] == "ssgrep_known_net_work_seconds"]
    assert sorted(int(row["point"]["asInt"]) for row in known) == [10, 20, 30]
    assert all(row["attributes"]["method"] == "capped_gap_v1" for row in known)
    assert all(
        row["resource"]["tenant"] == "tenant-a" and row["resource"]["device"] == "device-a"
        for row in points
    )
    lower = [
        row
        for row in points
        if row["name"] == "ssgrep_estimate_bound_work_seconds"
        and row["attributes"]["quantity"] == "net"
        and row["attributes"]["bound"] == "lower"
    ]
    assert lower == []
    latency = [
        row for row in points if row["name"] == "ssgrep_measured_service_latency_milliseconds"
    ]
    assert (
        len(latency) == 1 and latency[0]["unit"] == "ms" and latency[0]["point"]["asInt"] == "2000"
    )
    assert all("subject" not in row["resource"] for row in points)
    assert fx.state().accepted_means == "collector_acceptance"
    before = collector.count("metrics")
    snapshots.publish(fx.PROFILE, envelope(2, missing=False, milliseconds=25000))
    assert fx.cycle(seconds=1, deps=fx.deps(environ={fx.ENV_NAME: lab.TOKEN_A})).acked == 1
    documents = collector.wait_records("metrics", 1, since=before)
    nets = [
        int(row["point"]["asInt"])
        for document in documents
        for row in lab.metric_points(document)
        if row["name"] == "ssgrep_estimated_net_work_seconds"
    ]
    assert sorted(nets) == [-15, -5, 5]


def test_real_collector_reordered_revisions_converge_and_credentials_isolate_tenants(collector):
    original = work_candidate(True)
    first = export_registry.plan_revisions(export_registry.Registry(), (original,))[0]
    registry = queue(export_registry.Registry(), (first,), "queued-first")
    revised = work_candidate(False)
    second = export_registry.plan_revisions(registry, (revised,), current_namespace=METHOD.hash)[0]
    assert second.revision == 2 and second.supersedes == 1
    resource = Resource("synthetic-test", "a" * 32, 1)
    batches = {}
    for planned in (first, second):
        logs = export_records.log_records((planned,), time_unix_nano=1_791_806_400_000_000_000)
        ((body, count),) = otlp.split_logs(resource, logs)
        assert count == 1
        batches[planned.revision] = body
    before = collector.count("logs")
    for revision in (2, 1, 2, 1):
        result = export_http.post(target(collector, "logs"), batches[revision])
        assert result.status == 200
        assert acks.parse_response(fx.L, result.body).outcome == "full"
    documents = collector.wait_records("logs", 4, since=before)
    state = ReceiverState()
    for document in documents:
        for record in lab.log_records(document):
            body = json.loads(record["body"])
            identity = record["resource"]
            assert (identity["tenant"], identity["device"]) == ("tenant-a", "device-a")
            key = Key(identity["tenant"], identity["device"], body["record_id"])
            state, outcome = apply(state, key, body)
            assert outcome is not Outcome.REJECTED_INVALID
    assert len(state.current) == 1
    current = next(iter(state.current.values()))
    assert current.revision == 2
    estimate = current.body["scenarios"]["conservative"]
    assert estimate["net"]["value"] == estimate["known_net"] == 8 and estimate["complete"]
    assert estimate["bounds"]["net_lower"]["value"] == estimate["bounds"]["net_upper"]["value"] == 8
    before = collector.count("logs")
    assert export_http.post(target(collector, "logs", lab.TOKEN_B), batches[2]).status == 200
    other = collector.wait_records("logs", 1, since=before)
    record = lab.log_records(other[0])[0]
    assert (record["resource"]["tenant"], record["resource"]["device"]) == ("tenant-b", "device-b")
    body = json.loads(record["body"])
    state, outcome = apply(state, Key("tenant-b", "device-b", body["record_id"]), body)
    assert outcome is Outcome.APPLIED_NEW and len(state.current) == 2
    unit = ("work_seconds", "capped_gap_v1", None, None)
    with pytest.raises(ValueError, match="mix tenants"):
        current_totals(state, methodology=METHOD.hash, unit_key=unit)
    selected = current_totals(state, methodology=METHOD.hash, unit_key=unit, tenant="tenant-a")
    assert selected["rows"] == 1 and selected["scenarios"]["conservative"]["net"]["total"] == 8
    before = collector.count("logs")
    denied = export_http.post(target(collector, "logs", "not-enrolled"), batches[2])
    assert denied.status in (401, 403) and collector.count("logs") == before
