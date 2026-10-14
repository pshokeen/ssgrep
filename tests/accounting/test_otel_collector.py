"""Real OpenTelemetry Collector interop (tasks 12.2, 12.6, 13.7, 14.7).

The system under test talks over REAL loopback sockets to a REAL
``otel/opentelemetry-collector-contrib:0.137.0`` container (``otel_collector_lab.py``,
config ``docs/examples/otel-collector-reference.yaml``). Every assertion compares what the
collector's OWN file exporter wrote against the values this test put in: nothing here is a mock
or a call-count spy.

The module skips ONLY when docker or the pinned image is absent (named reason, evaluated once);
a collector that fails to start or a failing assertion is a failure, never a skip.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest

from ssgrep.accounting import (
    acks,
    collector_state,
    collector_store,
    export,
    export_build,
    export_http,
    export_ids,
    export_records,
    export_retry,
)
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.receiver_contract import Key, Outcome, ReceiverState, apply, current_totals
from ssgrep.accounting.revisions import seal_day
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting import otel_collector_lab as lab
from tests.accounting.http_builders import make_snapshot
from tests.accounting.valuation_builders import DAY, METHOD, candidate, debit

_REASON = lab.unavailable_reason()
pytestmark = [pytest.mark.skipif(_REASON is not None, reason=str(_REASON))]

PROFILE = "personal-local"
DEST_ID = "lab-collector"
KEY = bytes(range(32))
UNIT_KEY = ("context_tokens", "chars_div4_v1", None, None)
CUMULATIVE = 2  # AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE


def destination(endpoint: str = "http://127.0.0.1:4318") -> DestinationPolicy:
    return DestinationPolicy(
        DEST_ID, endpoint, (Signal.METRICS, Signal.LOGS), True, "env:LAB_INGEST_TOKEN"
    )


def profile_policy() -> ProfilePolicy:
    return ProfilePolicy(PROFILE, destinations=(destination(),))


def target(
    collector: lab.Collector, signal: str, token: str | None = lab.TOKEN_A
) -> export_http.Target:
    headers = (("Authorization", f"Bearer {token}"),) if token is not None else ()
    return export_http.Target(f"{collector.base_url}/v1/{signal}", headers=headers)


# ---- lifecycle fixtures (module scope; teardown verified by label even when tests fail) ------


@pytest.fixture(scope="module")
def run_id() -> Iterator[str]:
    run = lab.new_run_id()
    yield run
    assert lab.labelled(run) == [], "lab containers survived teardown"


@pytest.fixture(scope="module")
def collector(run_id: str) -> Iterator[lab.Collector]:
    instance = lab.Collector.start(run_id, "main")
    try:
        yield instance
    finally:
        instance.stop()
        assert all(a["owned"] for a in instance.actions)
        assert {a["container_id"] for a in instance.actions} <= {instance.container_id}


# ---- helpers ---------------------------------------------------------------------------------


def assert_fully_accepted(signal: Signal, outcome: export_http.Outcome) -> None:
    """The real collector's 200 body ({"partialSuccess":{}}) is a plain full acknowledgment."""
    result = acks.parse_response(signal, outcome.body)
    assert result == acks.AckResult("full", 0, None, False)


def point_key(name: str, attributes: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]]:
    return name, tuple(sorted(attributes.items()))


def assert_metrics_match(
    documents: list[dict[str, Any]],
    envelope: Any,
    *,
    identity: tuple[str, str],
    key: bytes = KEY,
) -> dict[str, int]:
    """Every projected point is present in the collector's file with identical content."""
    expected = export.project_metrics(envelope).points
    rows = [row for document in documents for row in lab.metric_points(document)]
    assert len(rows) == len(expected) > 20
    by_key = {point_key(r["name"], r["attributes"]): r for r in rows}
    assert len(by_key) == len(rows)  # no point arrived twice
    installation = export_ids.installation_id(key, DEST_ID, PROFILE)
    for row in rows:
        assert row["resource"] == {
            "service.name": "ssgrep",
            "service.version": export.installed_version(),
            "ssgrep.installation.id": installation,
            "ssgrep.policy.version": envelope.policy_version,
            "ssgrep.counter.epoch": envelope.counter_epoch,
            "tenant": identity[0],  # from the credential, not from the payload
            "device": identity[1],
        }
    kinds = {"sum": 0, "gauge": 0, "histogram": 0}
    for point in expected:
        row = by_key[point_key(point.name, dict(point.attributes))]
        wire = row["point"]
        assert (row["kind"], row["unit"]) == (point.kind, point.unit)
        assert wire["timeUnixNano"] == str(point.time_unix_nano)
        kinds[point.kind] += 1
        if point.kind == "histogram":
            hist = point.hist
            assert hist is not None
            assert row["temporality"] == CUMULATIVE
            assert wire["startTimeUnixNano"] == str(point.start_unix_nano)
            assert wire["count"] == str(hist.count)
            assert wire["bucketCounts"] == [str(c) for c in hist.counts]
            assert Decimal(str(wire["sum"])) == Decimal(hist.sum)
            assert [Decimal(str(b)) for b in wire["explicitBounds"]] == [
                Decimal(b) for b in hist.bounds
            ]
            assert sum(int(c) for c in wire["bucketCounts"]) == hist.count
        else:
            assert wire["asInt"] == str(point.value)  # int64 as a decimal string
        if point.kind == "sum":
            assert row["temporality"] == CUMULATIVE and row["monotonic"] is True
            assert wire["startTimeUnixNano"] == str(point.start_unix_nano)
        if point.kind == "gauge":
            assert "startTimeUnixNano" not in wire
    assert all(kinds.values())
    return kinds


# ---- 1. metrics interop (12.2) ---------------------------------------------------------------


def test_real_metrics_payload_is_accepted_and_every_point_survives_in_the_file(
    collector: lab.Collector,
) -> None:
    envelope = make_snapshot(5)
    candidate_ = export_build.build_metrics(PROFILE, destination(), profile_policy(), envelope, KEY)
    before = collector.count("metrics")
    for batch in candidate_.batches:
        outcome = export_http.post(target(collector, "metrics"), batch.body)
        assert (outcome.kind, outcome.status) == ("response", 200)
        assert json.loads(outcome.body or b"") == {"partialSuccess": {}}  # valid OTLP/JSON reply
        assert export_retry.classify(outcome) == export_retry.ACK_CANDIDATE
        assert_fully_accepted(Signal.METRICS, outcome)
    documents = collector.wait_records("metrics", len(candidate_.batches), since=before)

    kinds = assert_metrics_match(documents, envelope, identity=("tenant-a", "device-a"))

    assert kinds["sum"] == 3 and kinds["histogram"] == 2 and kinds["gauge"] > 20
    rows = {
        point_key(r["name"], r["attributes"]): r for d in documents for r in lab.metric_points(d)
    }
    negatives = {
        key[1]: row["point"]["asInt"]
        for key, row in rows.items()
        if row["name"] == "ssgrep_estimated_net_tokens"
    }
    assert (
        negatives[(("method", "chars_div4_v1"), ("scenario", "conservative"), ("window", "7d"))]
        == "-12000"
    )
    assert (
        negatives[(("method", "chars_div4_v1"), ("scenario", "central"), ("window", "7d"))]
        == "-7000"
    )
    first = next(r for k, r in rows.items() if k[0] == "ssgrep_operation_duration_seconds")
    assert first["point"]["bucketCounts"] == ["0", "3", "0", "0", "0", "0", "0", "0", "0", "0"]
    assert Decimal(str(first["point"]["sum"])) == Decimal("0.006")


@pytest.mark.parametrize("variant", ["gauge_asint_object", "resource_metrics_string", "not_json"])
def test_malformed_payload_is_rejected_4xx_by_the_real_collector_and_classified_permanent(
    collector: lab.Collector, variant: str
) -> None:
    body = json.loads(
        export_build.build_metrics(PROFILE, destination(), profile_policy(), make_snapshot(6), KEY)
        .batches[0]
        .body
    )
    if variant == "gauge_asint_object":
        metrics = body["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        gauge = next(m for m in metrics if "gauge" in m)
        gauge["gauge"]["dataPoints"][0]["asInt"] = {"wrong": "type"}
        raw = json.dumps(body).encode()
    elif variant == "resource_metrics_string":
        body["resourceMetrics"] = "nope"
        raw = json.dumps(body).encode()
    else:
        raw = b'{"resourceMetrics": [ {'
    before = collector.count("metrics")

    outcome = export_http.post(target(collector, "metrics"), raw)

    assert outcome.kind == "response" and outcome.status == 400
    assert export_retry.classify(outcome) == export_retry.PERMANENT  # never retried
    assert not export_retry.is_retryable(outcome.kind, outcome.status)
    assert collector.count("metrics") == before  # nothing reached the exporter


# ---- 2. logs (14.7) --------------------------------------------------------------------------


def commit_ledger(ledger: VersionedLedger) -> None:
    """Persist ledger state + checkpoint through the collector's own writers."""
    from ssgrep.accounting.policy import collection_scope

    scope = collection_scope(profile_policy())
    blobs: dict[str, str] = {}
    state = collector_state.State(
        ledger=ledger.state(), superseded=dict(ledger.superseded), collection_scope=scope
    )
    collector_state.save_state(PROFILE, state, METHOD.hash, blobs)
    checkpoint = collector_store.Checkpoint(PROFILE, generation=1, methodology=METHOD.hash)
    checkpoint.blobs = blobs
    checkpoint.collection_scope = scope
    collector_store.commit_checkpoint(PROFILE, checkpoint)


def send_logs(collector: lab.Collector, batches: Any, token: str = lab.TOKEN_A) -> None:
    for batch in batches:
        outcome = export_http.post(target(collector, "logs", token), batch.body)
        assert (outcome.kind, outcome.status) == ("response", 200)
        assert_fully_accepted(Signal.LOGS, outcome)


def test_logs_revision_two_and_a_tombstone_arrive_as_separate_records(
    collector: lab.Collector,
) -> None:
    envelope, dest, policy = make_snapshot(7), destination(), profile_policy()
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": EvidenceBundle((candidate("a"),), (debit("x", None),))}, today="2026-11-02"
    )
    commit_ledger(ledger)
    first = export_build.build_logs(PROFILE, dest, policy, envelope, KEY)
    (planned_1,) = (p for group in first.planned for p in group)
    assert planned_1.revision == 1 and planned_1.record_type == "contribution"
    before = collector.count("logs")
    send_logs(collector, first.batches)
    export_records.mark_queued(PROFILE, DEST_ID, [planned_1], first.batches[0].batch_id)
    export_records.mark_acked(PROFILE, DEST_ID, first.batches[0].batch_id)

    ledger.reconcile(  # more evidence after the day sealed: revision 2 of the SAME record id
        {"root-1": EvidenceBundle((candidate("a"),), (debit("x", 640),))}, today=seal_day(DAY)
    )
    commit_ledger(ledger)
    second = export_build.build_logs(PROFILE, dest, policy, envelope, KEY)
    (planned_2,) = (p for group in second.planned for p in group)
    assert (planned_2.record_id, planned_2.revision) == (planned_1.record_id, 2)
    assert planned_2.record_type == "adjustment" and planned_2.body != planned_1.body
    send_logs(collector, second.batches)
    export_records.mark_queued(PROFILE, DEST_ID, [planned_2], second.batches[0].batch_id)
    export_records.mark_acked(PROFILE, DEST_ID, second.batches[0].batch_id)

    registry = export_records.load_registry(PROFILE, DEST_ID)
    (deletion,) = export_records.build_deletions(
        registry, [planned_1.record_id], key=KEY, destination=DEST_ID
    )
    tombstone = export_build.build_tombstones(
        PROFILE, dest, policy, [deletion], KEY, stamp=1_791_806_700
    )
    send_logs(collector, tombstone.batches)

    documents = collector.wait_records("logs", 3, since=before)
    rows = [row for document in documents for row in lab.log_records(document)]
    assert len(rows) == 3  # vanilla OTLP keeps ALL revisions and the tombstone: it never dedupes
    expected = [planned_1, planned_2, deletion]
    for row, plan in zip(rows, expected, strict=True):
        assert row["attributes"] == {
            "ssgrep.record.id": plan.record_id,
            "ssgrep.record.revision": plan.revision,
            "ssgrep.record.type": plan.record_type,
            "ssgrep.record.digest": plan.body_digest,
        }
        body = json.loads(row["body"])
        assert body == plan.body and digest_of(body) == row["attributes"]["ssgrep.record.digest"]
        assert row["resource"]["tenant"] == "tenant-a" and row["resource"]["device"] == "device-a"
        assert row["resource"]["ssgrep.policy.version"] == 1
        assert row["resource"]["service.name"] == "ssgrep"
    assert [r["attributes"]["ssgrep.record.type"] for r in rows] == [
        "contribution", "adjustment", "deletion",
    ]  # fmt: skip
    assert json.loads(rows[2]["body"])["scenarios"] == {}

    # Only revision-aware receiver logic converges: apply the FILE's records, in file order,
    # under the identity the collector attached (never a payload claim).
    state = ReceiverState()
    outcomes = []
    totals = []
    for row in rows:
        key = Key(
            row["resource"]["tenant"],
            row["resource"]["device"],
            row["attributes"]["ssgrep.record.id"],
        )
        state, outcome = apply(state, key, json.loads(row["body"]))
        outcomes.append(outcome)
        totals.append(current_totals(state, methodology=METHOD.hash, unit_key=UNIT_KEY))
    assert outcomes == [Outcome.APPLIED_NEW, Outcome.APPLIED_UPSERT, Outcome.APPLIED_TOMBSTONE]
    assert [t["rows"] for t in totals] == [1, 1, 0]  # upsert replaced rev 1; tombstone removed it
    assert (
        totals[0]["scenarios"]["central"]["net"]["total"]
        != totals[1]["scenarios"]["central"]["net"]["total"]
    )
    stale = Key("tenant-a", "device-a", planned_1.record_id)
    _, replay = apply(state, stale, json.loads(rows[1]["body"]))
    assert replay in {Outcome.REJECTED_OLD_REVISION, Outcome.REJECTED_TOMBSTONED}


def test_logs_and_metrics_go_to_separate_files_and_a_logs_payload_on_metrics_is_not_mixed(
    collector: lab.Collector,
) -> None:
    metrics_before, logs_before = collector.count("metrics"), collector.count("logs")
    envelope = make_snapshot(8)
    metrics = export_build.build_metrics(PROFILE, destination(), profile_policy(), envelope, KEY)
    outcome = export_http.post(target(collector, "metrics"), metrics.batches[0].body)
    assert outcome.status == 200
    collector.wait_records("metrics", 1, since=metrics_before)
    assert collector.count("logs") == logs_before  # a metrics request never lands in the logs file
    for document in collector.records("logs"):
        assert "resourceMetrics" not in document
    for document in collector.records("metrics"):
        assert "resourceLogs" not in document


# ---- 3. authentication and identity ----------------------------------------------------------


@pytest.mark.parametrize("token", [None, "wrong-token", lab.TOKEN_A + "x", lab.TOKEN_A[:-1]])
def test_missing_or_wrong_token_is_401_blocked_and_nothing_is_stored(
    collector: lab.Collector, token: str | None
) -> None:
    body = (
        export_build.build_metrics(PROFILE, destination(), profile_policy(), make_snapshot(9), KEY)
        .batches[0]
        .body
    )
    before = collector.count("metrics")

    outcome = export_http.post(target(collector, "metrics", token), body)

    assert (outcome.kind, outcome.status, outcome.category) == ("response", 401, "unauthorized")
    assert outcome.body is None  # the collector's error text (it echoes the token) is dropped
    assert export_retry.classify(outcome) == export_retry.BLOCKED
    assert collector.count("metrics") == before


def test_logs_endpoint_requires_the_same_credential(collector: lab.Collector) -> None:
    request = json.dumps({"resourceLogs": []}).encode()
    denied = export_http.post(target(collector, "logs", None), request)
    assert denied.status == 401 and export_retry.classify(denied) == export_retry.BLOCKED
    accepted = export_http.post(target(collector, "logs", lab.TOKEN_B), request)
    assert accepted.status == 200 and export_retry.classify(accepted) == export_retry.ACK_CANDIDATE


def forged_metrics_request(claimed_tenant: str, claimed_device: str) -> bytes:
    def string(key: str, value: str) -> dict[str, Any]:
        return {"key": key, "value": {"stringValue": value}}

    return json.dumps(
        {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            string("service.name", "ssgrep"),
                            string("tenant", claimed_tenant),
                            string("device", claimed_device),
                            # an attacker also tries to plant the private identity-carrier name
                            string("ssgrep.auth.credential", f"Bearer {lab.TOKEN_B}"),
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "metrics": [
                                {
                                    "name": "forged_probe",
                                    "gauge": {
                                        "dataPoints": [{"asInt": "-1", "timeUnixNano": "1000"}]
                                    },
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    ).encode()


@pytest.mark.parametrize(
    ("token", "identity"),
    [(lab.TOKEN_A, ("tenant-a", "device-a")), (lab.TOKEN_B, ("tenant-b", "device-b"))],
)
def test_payload_asserted_tenant_and_device_are_overwritten_from_the_credential(
    collector: lab.Collector, token: str, identity: tuple[str, str]
) -> None:
    before = collector.count("metrics")

    outcome = export_http.post(
        target(collector, "metrics", token), forged_metrics_request("EVIL-TENANT", "EVIL-DEVICE")
    )

    assert outcome.status == 200
    (document,) = collector.wait_records("metrics", 1, since=before)[:1]
    (row,) = lab.metric_points(document)
    assert row["resource"] == {
        "service.name": "ssgrep",
        "tenant": identity[0],
        "device": identity[1],
    }  # claims overwritten; the planted carrier attribute is gone
    raw = (collector.out_dir / "metrics.jsonl").read_text()
    assert "EVIL" not in raw and lab.TOKEN_A not in raw and lab.TOKEN_B not in raw
    assert row["point"]["asInt"] == "-1"


def test_forged_identity_on_the_logs_pipeline_is_overwritten_too(collector: lab.Collector) -> None:
    before = collector.count("logs")
    request = json.dumps(
        {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "tenant", "value": {"stringValue": "EVIL-TENANT"}},
                            {"key": "device", "value": {"stringValue": "EVIL-DEVICE"}},
                        ]
                    },
                    "scopeLogs": [
                        {"logRecords": [{"timeUnixNano": "1000", "body": {"stringValue": "{}"}}]}
                    ],
                }
            ]
        }
    ).encode()

    assert export_http.post(target(collector, "logs", lab.TOKEN_B), request).status == 200

    (document,) = collector.wait_records("logs", 1, since=before)[:1]
    (row,) = lab.log_records(document)
    assert row["resource"] == {"tenant": "tenant-b", "device": "device-b"}
    assert "EVIL" not in (collector.out_dir / "logs.jsonl").read_text()


def test_collector_logs_never_contain_the_enrolled_tokens(collector: lab.Collector) -> None:
    export_http.post(target(collector, "metrics", lab.TOKEN_A), b'{"resourceMetrics": []}')
    text = collector.logs()
    assert "Everything is ready" in text
    assert lab.TOKEN_A not in text and lab.TOKEN_B not in text


# ---- 5. TLS ----------------------------------------------------------------------------------


def test_tls_receiver_with_the_lab_certificate(run_id: str) -> None:
    instance = lab.Collector.start(run_id, "tls", tls=True)
    try:
        assert instance.base_url.startswith("https://127.0.0.1:")
        assert instance.cert is not None
        body = (
            export_build.build_metrics(
                PROFILE, destination(), profile_policy(), make_snapshot(10), KEY
            )
            .batches[0]
            .body
        )
        trusted = export_http.post(
            target(instance, "metrics"), body, context=instance.cert.client_context()
        )
        assert (trusted.kind, trusted.status) == ("response", 200)
        untrusted = export_http.post(target(instance, "metrics"), body)  # system trust store
        assert untrusted.kind == "tls_error"
        assert export_retry.classify(untrusted) == export_retry.REFUSED
        (document,) = instance.wait_records("metrics", 1)[:1]
        assert lab.metric_points(document)[0]["resource"]["tenant"] == "tenant-a"
    finally:
        instance.stop()
        assert all(a["owned"] for a in instance.actions)
