"""Delivery journey against a REAL OpenTelemetry Collector (task 12.6/13.7/14.7, test 4).

``delivery.run_destination`` is driven exactly as production drives it: policy saved with
``policy.save_profile`` + ``export_consent.apply_enable``, snapshots published with
``snapshots.publish``, the ledger persisted through the collector's own writers, credentials
resolved from an ``env:`` reference at send time, and requests sent by the real
``export_http.post`` over loopback to the container's published port (through a stable
``lab.Forwarder`` so the destination endpoint survives ``docker restart``). Assertions read the
delivery document, the spool files, the bytes that crossed the socket and the collector's own
file-exporter output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import ssgrep.accounting.views.snapshots as snapshots
from ssgrep.accounting.collect import store as collector_store
from ssgrep.accounting.collect.state import VersionedLedger
from ssgrep.accounting.export import (
    acks,
    consent as export_consent,
    delivery,
    ids as export_ids,
    registry as export_registry,
    spool as export_spool,
)
from ssgrep.accounting.export.receiver import Key, Outcome, ReceiverState, apply
from ssgrep.accounting.export.registry import DELETION
from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle
from ssgrep.accounting.privacy import policy, published as tombstones
from tests.accounting.export import otel_collector_lab as lab
from tests.accounting.export.http_builders import make_snapshot, synthetic_current_checkpoint
from tests.accounting.export.test_otel_collector import assert_metrics_match, commit_ledger
from tests.accounting.ledger.valuation_builders import METHOD, candidate, debit

_REASON = lab.unavailable_reason()
pytestmark = [pytest.mark.skipif(_REASON is not None, reason=str(_REASON))]

PROFILE = "personal-local"
DEST = "lab-collector"
GENERATION = 3


@pytest.fixture(scope="module")
def run_id() -> Iterator[str]:
    run = lab.new_run_id()
    yield run
    assert lab.labelled(run) == [], "lab containers survived teardown"


@pytest.fixture(scope="module")
def collector(run_id: str) -> Iterator[lab.Collector]:
    instance = lab.Collector.start(run_id, "delivery")
    try:
        yield instance
    finally:
        instance.stop()
        assert all(a["owned"] for a in instance.actions)
        assert {a["container_id"] for a in instance.actions} <= {instance.container_id}


@pytest.fixture
def forwarder(collector: lab.Collector) -> Iterator[lab.Forwarder]:
    front = lab.Forwarder(lambda: collector.port)
    try:
        yield front
    finally:
        front.close()


def deps(**overrides: Any) -> delivery.Deps:
    environ = {"LAB_INGEST_TOKEN": lab.TOKEN_A}
    return delivery.Deps(environ=environ, connect_timeout=3.0, read_timeout=8.0, **overrides)


def enable(endpoint: str, *signals: Signal) -> ProfilePolicy:
    """Save a profile that permits and enables ``signals`` toward the loopback collector."""
    dest = DestinationPolicy(
        DEST, endpoint, (Signal.METRICS, Signal.LOGS), True, "env:LAB_INGEST_TOKEN"
    )
    current = policy.save_profile(ProfilePolicy(PROFILE, destinations=(dest,)))
    checkpoint = synthetic_current_checkpoint(PROFILE, current)
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    for signal in signals:
        export_consent.apply_enable(PROFILE, DEST, signal)
    return policy.load_profile_policy(PROFILE)


def ledger_with_two_revisions() -> VersionedLedger:
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": EvidenceBundle((candidate("a"),), (debit("x", None),))}, today="2026-11-02"
    )
    return ledger


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def spool_bytes(signal: Signal) -> dict[str, bytes]:
    return {
        meta.batch_id: export_spool.read_body(PROFILE, DEST, signal, meta.batch_id)
        for meta in export_spool.list_batches(PROFILE, DEST, signal)
    }


# ---- metrics + separately authorized logs end to end -----------------------------------------


def test_metrics_and_logs_cycles_are_fully_acknowledged_and_visible_in_the_collector_file(
    collector: lab.Collector, forwarder: lab.Forwarder
) -> None:
    enable(forwarder.url, Signal.METRICS, Signal.LOGS)
    envelope = make_snapshot(GENERATION)
    snapshots.publish(PROFILE, envelope)
    ledger = ledger_with_two_revisions()
    commit_ledger(ledger)
    metrics_before, logs_before = collector.count("metrics"), collector.count("logs")

    metrics = delivery.run_destination(PROFILE, DEST, Signal.METRICS, deps=deps())
    logs = delivery.run_destination(PROFILE, DEST, Signal.LOGS, deps=deps())

    assert (metrics.state, metrics.reason) == ("sent", None)
    assert (logs.state, logs.reason) == ("sent", None)
    assert metrics.enqueued_batches == metrics.acked == metrics.sent >= 1
    assert logs.enqueued_batches == logs.acked == logs.sent >= 1
    assert metrics.quarantined == metrics.partial == logs.quarantined == logs.partial == 0
    for signal in (Signal.METRICS, Signal.LOGS):
        state = acks.read_delivery(PROFILE, DEST, signal)
        assert state is not None
        assert state.last_acked_generation == GENERATION  # fully acked: every batch accepted
        assert state.last_acked_snapshot_id == envelope.snapshot_id
        assert state.accepted_means == "collector_acceptance"
        assert (state.pending_batches, state.rejected_points, state.blocked_reason) == (0, 0, None)
        assert all(m.state == "acked" for m in export_spool.list_batches(PROFILE, DEST, signal))

    key = export_ids.load_key(PROFILE, create=False)
    assert key is not None
    metric_docs = collector.wait_records("metrics", metrics.sent, since=metrics_before)
    kinds = assert_metrics_match(metric_docs, envelope, identity=("tenant-a", "device-a"), key=key)
    assert kinds["sum"] and kinds["histogram"] and kinds["gauge"]

    log_docs = collector.wait_records("logs", logs.sent, since=logs_before)
    rows = [row for document in log_docs for row in lab.log_records(document)]
    (row,) = rows
    assert row["attributes"]["ssgrep.record.revision"] == 1
    assert row["attributes"]["ssgrep.record.type"] == "contribution"
    assert row["resource"]["tenant"] == "tenant-a"
    assert row["resource"]["ssgrep.installation.id"] == export_ids.installation_id(
        key, DEST, PROFILE
    )
    (record_id,) = export_registry.load_registry(PROFILE, DEST).entries
    assert row["attributes"]["ssgrep.record.id"] == record_id

    # the same snapshot again: nothing new is enqueued or sent (idempotent), file unchanged
    again = delivery.run_destination(PROFILE, DEST, Signal.METRICS, deps=deps())
    assert again.sent == 0 and again.enqueued_batches == 0
    assert collector.count("metrics") == metrics_before + metrics.sent


# ---- restart between enqueue and send --------------------------------------------------------


def test_collector_restart_between_enqueue_and_send_retries_the_same_spool_bytes(
    collector: lab.Collector, forwarder: lab.Forwarder
) -> None:
    enable(forwarder.url, Signal.METRICS)
    envelope = make_snapshot(GENERATION + 1)
    snapshots.publish(PROFILE, envelope)
    before = collector.count("metrics")
    collector.halt()  # `docker stop` of OUR container: the batch is enqueued while it is down

    first = delivery.run_destination(PROFILE, DEST, Signal.METRICS, deps=deps())

    assert first.state == "waiting" or first.retry_scheduled_utc is not None, first
    assert first.enqueued_batches >= 1 and first.acked == 0 and first.sent >= 1
    (queued,) = export_spool.list_batches(PROFILE, DEST, Signal.METRICS)
    assert queued.state == "queued" and queued.attempts == 1  # a failed attempt, same batch
    assert queued.next_attempt_utc is not None
    spooled = export_spool.read_body(PROFILE, DEST, Signal.METRICS, queued.batch_id)
    assert sha(spooled) == queued.sha256 and len(spooled) == queued.bytes
    state = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert state is not None and state.last_acked_generation in (None, GENERATION)
    assert state.last_acked_generation != GENERATION + 1  # NOT acknowledged yet
    assert forwarder.request_bodies() == []  # nothing reached any collector
    assert collector.count("metrics") == before

    collector.restart()  # `docker restart` of OUR container (new published port)
    later = datetime.now(UTC) + timedelta(hours=1)  # past the scheduled retry time
    second = delivery.run_destination(
        PROFILE, DEST, Signal.METRICS, now=later, enqueue=False, deps=deps()
    )

    assert (second.state, second.acked) == ("sent", 1)
    (settled,) = export_spool.list_batches(PROFILE, DEST, Signal.METRICS)
    assert (settled.batch_id, settled.state, settled.attempts) == (queued.batch_id, "acked", 2)
    assert settled.sha256 == queued.sha256  # the retry reused the immutable spool bytes
    crossed = forwarder.request_bodies()
    assert [(path, sha(body)) for path, body in crossed] == [("/v1/metrics", queued.sha256)]
    assert crossed[0][1] == spooled
    final = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert final is not None and final.last_acked_generation == GENERATION + 1
    assert final.last_acked_snapshot_id == envelope.snapshot_id

    (document,) = collector.wait_records("metrics", 1, since=before)
    key = export_ids.load_key(PROFILE, create=False)
    assert key is not None
    assert_metrics_match([document], envelope, identity=("tenant-a", "device-a"), key=key)


# ---- tombstone delivery ----------------------------------------------------------------------


def test_tombstone_batch_is_delivered_as_a_separate_deletion_record(
    collector: lab.Collector, forwarder: lab.Forwarder
) -> None:
    current = enable(forwarder.url, Signal.METRICS, Signal.LOGS)
    snapshots.publish(PROFILE, make_snapshot(GENERATION + 2))
    commit_ledger(ledger_with_two_revisions())
    before = collector.count("logs")
    sent = delivery.run_destination(PROFILE, DEST, Signal.LOGS, deps=deps())
    assert (sent.state, sent.acked) == ("sent", 1)
    (contribution_doc,) = collector.wait_records("logs", 1, since=before)
    (contribution,) = lab.log_records(contribution_doc)

    plans = tombstones.plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (plan,) = plans
    assert plan.permitted and plan.record_ids == (contribution["attributes"]["ssgrep.record.id"],)
    now = datetime.now(UTC)
    (queued,) = tombstones.enqueue_tombstones(PROFILE, current, plans, now=now)
    assert queued.queued_records == 1 and queued.batches == 1 and queued.skipped_reason is None
    assert tombstones.pending_tombstones(PROFILE, DEST) == 1
    (tomb_meta,) = (
        m for m in export_spool.list_batches(PROFILE, DEST, Signal.LOGS) if m.kind == "tombstone"
    )
    tomb_bytes = export_spool.read_body(PROFILE, DEST, Signal.LOGS, tomb_meta.batch_id)

    result = delivery.run_destination(PROFILE, DEST, Signal.LOGS, now=now, deps=deps())

    assert result.state == "sent" and result.acked >= 1 and result.quarantined == 0
    documents = collector.wait_records("logs", 2, since=before)
    assert len(documents) == 2  # a separate OTLP request, never merged into the first
    (deletion,) = lab.log_records(documents[1])
    assert deletion["attributes"]["ssgrep.record.type"] == DELETION
    assert deletion["attributes"]["ssgrep.record.id"] == plan.record_ids[0]
    assert deletion["attributes"]["ssgrep.record.revision"] == 2
    body = json.loads(deletion["body"])
    assert body["scenarios"] == {} and body["known_costs"] == {} and body["coverage"] == {}
    assert body["supersedes_revision"] == 1
    assert tomb_bytes in [
        b for _, b in forwarder.request_bodies()
    ]  # exactly the spooled tombstone bytes crossed the wire
    settled = export_spool.read_meta(PROFILE, DEST, Signal.LOGS, tomb_meta.batch_id)
    assert settled is not None and settled.state == "acked"
    # acceptance of the deletion lets delivery forget the linkage it kept for this record
    assert plan.record_ids[0] not in export_registry.load_registry(PROFILE, DEST).entries
    assert tombstones.pending_tombstones(PROFILE, DEST) == 0

    state = ReceiverState()
    for row in (contribution, deletion):  # file order, identity attached by the collector
        key = Key(
            row["resource"]["tenant"],
            row["resource"]["device"],
            row["attributes"]["ssgrep.record.id"],
        )
        state, outcome = apply(state, key, json.loads(row["body"]))
    assert outcome == Outcome.APPLIED_TOMBSTONE
