"""Durable delivery cycle against the deterministic loopback lab receiver (real sockets/files).

Every assertion is on state and bytes: the receiver's recorded requests (bodies, sha256, headers),
the spool metas, the delivery document and the registry. Policy, snapshots and the ledger are the
real writers' output (see ``delivery_fixtures``); the only injected seams are ``Deps.environ`` for
the credential, a deterministic jitter and the cycle clock.
"""

from __future__ import annotations

import errno
import errno as spool_errno
import hashlib
import json
import json as ack_json
import json as recovery_json
import json as spool_json
import json as tomb_json
import multiprocessing as ack_multiprocessing
import multiprocessing as spool_multiprocessing
import os as ack_os
import os as spool_os
import re
import stat as ack_stat
import stat as spool_stat
from collections.abc import Callable
from dataclasses import (
    replace,
    replace as ack_replace,
    replace as spool_replace,
    replace as tomb_replace,
)
from datetime import UTC, datetime, datetime as tomb_datetime
from http.client import HTTPResponse
from pathlib import Path, Path as ack_Path, Path as spool_Path, Path as tomb_Path
from typing import Any, Any as ack_Any, Any as spool_Any, Any as tomb_Any, Literal, cast

import pytest
import pytest as ack_pytest
import pytest as recovery_pytest
import pytest as retry_pytest
import pytest as spool_pytest
import pytest as tomb_pytest

import ssgrep.accounting.export.otlp as retry
from ssgrep.accounting.collect import store as collector_store
from ssgrep.accounting.collect.replay import Ledger
from ssgrep.accounting.export import (
    acks,
    acks as ack_acks,
    acks as spool_acks,
    acks as tomb_acks,
    consent as export_consent,
    consent as export_policy,
    consent as tomb_export_consent,
    delivery,
    ids as export_ids,
    ids as tomb_export_ids,
    otlp as export_http,
    projection as delivery_candidate,
    registry as export_registry,
    spool,
    spool as ack_export_spool,
    spool as codec,
    spool as export_spool,
    spool as recovery_export_spool,
    spool as tomb_export_spool,
)
from ssgrep.accounting.export.delivery import CycleResult, ExportPaused, check
from ssgrep.accounting.export.otlp import MAX_BATCH_BYTES, Batch, Outcome as retry_Outcome
from ssgrep.accounting.export.receiver import (
    Key,
    Outcome as tomb_Outcome,
    ReceiverState,
    apply,
    current_totals,
)
from ssgrep.accounting.export.records import project_candidates, root_key_of
from ssgrep.accounting.export.registry import (
    Planned,
    Registry,
    load_registry,
    mark_acked,
    mark_queued,
    plan_revisions as spool_plan_revisions,
    plan_revisions as tomb_plan_revisions,
)
from ssgrep.accounting.export.spool import BatchConflict, Meta, SpoolCorrupt, Usage
from ssgrep.accounting.ledger.contracts import TaskId, digest_of
from ssgrep.accounting.ledger.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    Signal as ack_Signal,
    Signal as recovery_Signal,
    Signal as spool_Signal,
    Signal as tomb_Signal,
)
from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle
from ssgrep.accounting.privacy import (
    policy,
    policy as tomb_policy,
    published as tombstones,
    purge,
    purge_spool,
    storage,
    storage as ack_storage,
    storage as plainfile,
    storage as recovery_storage,
    storage as spool_storage,
    storage as tomb_storage,
)
from ssgrep.accounting.privacy.published import (
    MAX_OBLIGATIONS,
    TombstonePlan,
    enqueue_tombstones,
    forget_acknowledged_deletions,
    pending_tombstones,
    plan_tombstones,
)
from ssgrep.accounting.views import snapshots, snapshots as ack_snapshots
from tests.accounting.export import (
    delivery_fixtures as ack_fx,
    delivery_fixtures as fx,
    delivery_fixtures as recovery_fx,
    delivery_fixtures as spool_fx,
)
from tests.accounting.export.delivery_fixtures import (
    DEST,
    DEST as recovery_DEST,
    PROFILE,
    PROFILE as recovery_PROFILE,
    SECRET,
    L,
    L as recovery_L,
    M,
    M as recovery_M,
    at,
    cycle,
    deps,
)
from tests.accounting.export.http_builders import synthetic_current_checkpoint
from tests.accounting.export.lab_receiver import (
    DropAfterRead,
    Hang,
    Hold,
    LabReceiver,
    LabReceiver as recovery_LabReceiver,
    LabReceiver as spool_LabReceiver,
    Oversize,
    Partial,
    Redirect,
    Respond,
    Respond as recovery_Respond,
)
from tests.accounting.export.snapshots_driver import make_envelope
from tests.accounting.export.test_consent import CANARY
from tests.accounting.export.test_export_records import (
    CANARY as tomb_CANARY,
    QUANTITY,
    canary_rows,
    project,
    rows_a,
)
from tests.accounting.ledger.valuation_builders import METHOD, atom, candidate
from tests.accounting.privacy.test_purge import (
    build_scene,
    configure_policy,
    export_all,
    old_record_ids,
    select,
    tree as disk_tree,
)
from tests.accounting.privacy.test_retention import (
    NOW_Z as ack_NOW_Z,
    delivery as ack_delivery,
    put as ack_put,
    seed_delivery,
)

PATH_M, PATH_L = "/v1/metrics", "/v1/logs"


@pytest.fixture
def lab():
    with LabReceiver.start() as receiver:
        yield receiver


def setup_metrics(lab: LabReceiver, generation: int = 1, **config) -> str:
    fx.configure(lab.url, **config)
    return fx.publish(generation)


def setup_logs(lab: LabReceiver, records: int = 3, generation: int = 1) -> str:
    fx.configure(
        lab.url, signals=(M, L), logs_permitted=True, enabled=(L,), roots=(), policy_version=1
    )
    fx.commit_ledger(records)
    return fx.publish(generation)


# --- happy path: bytes, auth, acknowledgment -------------------------------------------------


def test_cycle_sends_the_spooled_bytes_with_auth_and_acknowledges_the_exact_snapshot(
    lab: LabReceiver,
) -> None:
    snapshot_id = setup_metrics(lab)

    result = cycle()

    assert (result.state, result.sent, result.acked, result.enqueued_batches) == ("sent", 1, 1, 1)
    (request,) = lab.requests
    (meta,) = fx.metas()
    assert request.path == PATH_M and request.method == "POST"
    assert request.sha256 == meta.sha256 == fx.sha(fx.body_of(meta)) and request.body == fx.body_of(
        meta
    )
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert request.headers["content-type"] == "application/json"
    state = fx.delivery()
    assert state is not None
    assert (state.last_collected_generation, state.last_attempted_generation) == (1, 1)
    assert (state.last_acked_generation, state.last_acked_snapshot_id) == (1, snapshot_id)
    assert (state.pending_batches, state.unknown_batches, state.in_flight) == (0, 0, None)
    assert state.accepted_means == "collector_acceptance"
    assert (meta.state, meta.attempts, meta.last_status) == ("acked", 1, 200)
    resolved = acks.resolve_acknowledged(PROFILE, DEST, M)
    assert resolved.state == "acknowledged" and resolved.envelope is not None
    assert resolved.envelope.snapshot_id == snapshot_id


def test_a_second_cycle_on_the_same_snapshot_sends_nothing(lab: LabReceiver) -> None:
    setup_metrics(lab)
    cycle()
    again = cycle(seconds=400)
    assert (again.state, again.sent, again.enqueued_batches) == ("idle", 0, 0)
    assert len(lab.requests) == 1


def test_secret_and_response_body_never_reach_disk_or_results(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Respond(200, b'{"partialSuccess":{"errorMessage":"leak-me-RESPONSE-CANARY"}}'))
    result = cycle()
    assert result.state == "sent" and result.warnings == ("receiver_warning",)
    hay = fx.everything_on_disk() + repr(result)
    assert SECRET not in hay and "RESPONSE-CANARY" not in hay
    assert lab.requests[0].headers["authorization"].endswith(SECRET)  # it DID travel


def test_destination_without_credential_ref_sends_without_auth_header(lab: LabReceiver) -> None:
    setup_metrics(lab, credential_ref=None)
    assert cycle().state == "sent"
    assert "authorization" not in lab.requests[0].headers


def test_missing_credential_blocks_with_the_stable_condition_and_makes_no_request(
    lab: LabReceiver,
) -> None:
    setup_metrics(lab)
    result = cycle(deps=deps(environ={}))
    assert (result.state, result.reason) == ("blocked", "blocked_credential_invalid")
    assert result.blocked_reason == "credential_invalid" and not lab.requests
    state = fx.delivery()
    assert state is not None and state.blocked_reason == "credential_invalid"
    assert "credential_missing" in (state.last_error_category or "")
    assert "SSGREP_TEST_INGEST" not in fx.everything_on_disk() + repr(result)
    # still blocked (no request) until the credential value changes ...
    assert cycle(seconds=60, deps=deps(environ={})).state == "blocked" and not lab.requests
    # ... and sending resumes once it is provided
    assert cycle(seconds=120).state == "sent" and len(lab.requests) == 1


def test_unsafe_credential_file_blocks_and_never_sends_it(lab: LabReceiver, tmp_path: Path) -> None:
    secret_file = tmp_path / "ingest.token"
    secret_file.write_text("sk-FILE-CANARY-1")
    secret_file.chmod(0o644)  # group/other readable: refused
    setup_metrics(lab, credential_ref=f"file:{secret_file}")
    result = cycle()
    assert (result.state, result.blocked_reason) == ("blocked", "credential_invalid")
    assert not lab.requests and "sk-FILE-CANARY-1" not in fx.everything_on_disk() + repr(result)
    secret_file.chmod(0o600)
    assert cycle(seconds=60).state == "sent"
    assert lab.requests[0].headers["authorization"] == "Bearer sk-FILE-CANARY-1"


# --- retries: same bytes, floors, honest state -----------------------------------------------


@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_is_a_floor_and_the_same_bytes_are_resent(
    lab: LabReceiver, status: int
) -> None:
    setup_metrics(lab)
    lab.queue(Respond(status, b"overloaded-CANARY", {"Retry-After": "120"}))
    first = cycle(seconds=0)
    assert (first.state, first.reason, first.sent) == ("waiting", "retry_pending", 1)
    (meta,) = fx.metas()
    assert meta.state == "queued" and meta.attempts == 1 and meta.last_status == status
    assert meta.next_attempt_utc is not None and meta.next_attempt_utc >= fx.stamp(at(120))
    state = fx.delivery()
    assert state is not None and state.next_retry_utc == meta.next_attempt_utc
    assert state.last_acked_generation is None and state.pending_batches == 1
    assert first.retry_scheduled_utc == meta.next_attempt_utc
    # not due: no request
    assert cycle(seconds=60).state == "waiting" and len(lab.requests) == 1
    # due: identical bytes, now acked
    done = cycle(seconds=130)
    assert done.state == "sent" and done.acked == 1
    assert [r.sha256 for r in lab.requests] == [meta.sha256, meta.sha256]
    assert fx.metas()[0].attempts == 2 and fx.metas()[0].state == "acked"
    assert "CANARY" not in fx.everything_on_disk()


def test_unreachable_destination_grows_a_backlog_that_is_delivered_oldest_first_unchanged() -> None:
    port = fx.free_port()
    fx.configure(f"http://127.0.0.1:{port}")  # connection refused: the destination is down
    for generation in (1, 2, 3):
        fx.publish(generation)
        result = cycle(seconds=generation * 1000)  # each cycle is past the previous backoff
        assert result.state == "waiting" and result.reason == "retry_pending"
        assert result.error_category == "network"
    queued = fx.metas()
    assert [m.generation for m in queued] == [1, 2, 3]  # distinct generations, none lost
    assert len({m.batch_id for m in queued}) == 3
    # strictly oldest first: only the head of the queue was ever attempted (and kept retrying)
    assert queued[0].attempts == 3 and [m.attempts for m in queued[1:]] == [0, 0]
    created = {m.batch_id: m.created_utc for m in queued}
    originals = {m.batch_id: fx.body_of(m) for m in queued}
    state = fx.delivery()
    assert state is not None and state.last_acked_generation is None
    assert (state.last_collected_generation, state.pending_batches) == (3, 3)
    assert state.last_error_category == "network"

    revived = fx.receiver_on(port)  # the SAME endpoint comes back (late)
    try:
        results = fx.drain(start=20_000, step=400, limit=8)
        assert sum(r.acked for r in results) == 3
        assert [r.body for r in revived.requests] == [originals[m.batch_id] for m in queued]
        after = fx.metas()
        assert all(m.state == "acked" for m in after)
        assert {m.batch_id: m.created_utc for m in after} == created  # identity unchanged
        assert {m.batch_id: fx.body_of(m) for m in after} == originals
        done = fx.delivery()
        assert done is not None and done.last_acked_generation == 3
        assert (done.pending_batches, done.unknown_batches) == (0, 0)
    finally:
        revived.stop()


# --- 401/403 blocks until something changes --------------------------------------------------


@pytest.mark.parametrize(("status", "reason"), [(401, "unauthorized"), (403, "forbidden")])
def test_auth_failure_blocks_until_the_credential_or_endpoint_changes(
    lab: LabReceiver, status: int, reason: str
) -> None:
    setup_metrics(lab)
    lab.queue(Respond(status, b"nope-RESPONSE-CANARY"))
    first = cycle()
    assert (first.state, first.blocked_reason, first.reason) == (
        "blocked",
        reason,
        f"blocked_{reason}",
    )
    state = fx.delivery()
    assert state is not None and state.blocked_reason == reason
    assert (fx.metas()[0].state, fx.metas()[0].attempts) == ("queued", 1)
    for step in range(1, 4):  # further cycles make NO request
        assert cycle(seconds=step * 1000).state == "blocked"
    assert len(lab.requests) == 1
    # a different credential value clears the block and delivery resumes with the SAME bytes
    again = cycle(seconds=5000, deps=deps(environ={fx.ENV_NAME: "sk-ROTATED-0001"}))
    assert again.state == "sent" and again.acked == 1
    assert lab.requests[1].headers["authorization"] == "Bearer sk-ROTATED-0001"
    assert lab.requests[1].sha256 == lab.requests[0].sha256
    blob = fx.everything_on_disk() + repr(first) + repr(again)
    assert "RESPONSE-CANARY" not in blob and SECRET not in blob and "sk-ROTATED" not in blob


def test_endpoint_change_also_unblocks(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Respond(401))
    assert cycle().state == "blocked"
    with LabReceiver.start() as other:
        fx.amend_destination(endpoint=other.url)
        assert cycle(seconds=1000).state == "sent"
        assert len(other.requests) == 1 and len(lab.requests) == 1


def test_permanent_4xx_quarantines_the_batch_and_never_resends_it(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Respond(400, b"bad-request-RESPONSE-CANARY"))
    result = cycle()
    assert result.quarantined == 1 and result.acked == 0
    assert [m.state for m in fx.metas()] == ["quarantined"]
    state = fx.delivery()
    assert state is not None
    assert (state.quarantined_batches, state.last_acked_generation) == (1, None)
    for step in range(1, 4):
        cycle(seconds=step * 1000)
    assert len(lab.requests) == 1  # never blindly retried
    assert "RESPONSE-CANARY" not in fx.everything_on_disk()


# --- timeouts, drops, redirects, oversize: never acked, same bytes -------------------------


def test_timeout_after_accept_is_unknown_and_resent_with_identical_bytes(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Hang(8.0))
    first = cycle(deps=deps(connect_timeout=0.5, read_timeout=0.5))
    assert first.state == "waiting" and first.error_category == "timeout"
    (meta,) = fx.metas()
    assert (meta.state, meta.attempts, meta.last_category) == ("queued", 1, "timeout")
    state = fx.delivery()
    assert state is not None and state.last_acked_generation is None
    assert (state.pending_batches, state.unknown_batches) == (1, 1)  # sent, fate unknown
    assert acks.describe(state)["state"] == "unknown"
    done = cycle(seconds=400)
    assert done.state == "sent" and done.acked == 1
    assert [r.sha256 for r in lab.requests] == [meta.sha256, meta.sha256]  # receiver saw it twice
    final = fx.delivery()
    assert final is not None and final.last_acked_generation == 1 and final.unknown_batches == 0
    assert fx.metas()[0].state == "acked"


def test_dropped_connection_is_retried_later_with_the_same_bytes(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(DropAfterRead())
    assert cycle().state == "waiting"
    assert cycle(seconds=400).acked == 1
    assert len({r.sha256 for r in lab.requests}) == 1 and len(lab.requests) == 2


def test_redirect_is_never_followed_and_the_credential_is_never_sent_elsewhere(
    lab: LabReceiver,
) -> None:
    with LabReceiver.start() as elsewhere:
        setup_metrics(lab)
        lab.queue(Redirect(elsewhere.url_for("/v1/metrics")))
        result = cycle()
        assert result.state == "waiting" and result.error_category == "redirect"
        assert elsewhere.requests == [] and elsewhere.connections == 0
        (meta,) = fx.metas()
        assert (meta.state, meta.attempts, meta.last_category) == ("queued", 1, "redirect")
        state = fx.delivery()
        assert state is not None and state.last_acked_generation is None
        assert state.last_error_category == "redirect"
        assert fx.metas()[0].state != "acked"


@pytest.mark.parametrize("mode", ["content-length", "chunked", "stream"])
def test_oversize_response_is_bounded_and_not_acknowledged(
    lab: LabReceiver, monkeypatch, mode: Literal["content-length", "chunked", "stream"]
) -> None:
    read_bytes = 0
    original_read = HTTPResponse.read

    def measured_read(response, *args, **kwargs):
        nonlocal read_bytes
        body = original_read(response, *args, **kwargs)
        read_bytes += len(body)
        return body

    monkeypatch.setattr(HTTPResponse, "read", measured_read)
    setup_metrics(lab)
    assert cycle().acked == 1
    assert read_bytes > 0, "the client read observer must see a real accepted response"
    read_bytes = 0
    fx.publish(2)
    lab.queue(Oversize(512 * 1024, mode=mode))
    result = cycle(seconds=30)
    assert result.state == "waiting" and result.error_category == "oversize"
    assert read_bytes == (0 if mode == "content-length" else 64 * 1024 + 1)
    pending = [meta for meta in fx.metas() if meta.generation == 2]
    assert pending and all(meta.state == "queued" for meta in pending)
    state = fx.delivery()
    assert state is not None and state.last_acked_generation == 1 and state.unknown_batches == 1


def test_a_200_with_an_unparseable_body_is_quarantined_not_acknowledged(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Respond(200, b"<<not json>>"))
    result = cycle()
    assert result.quarantined == 1 and result.acked == 0
    assert fx.metas()[0].state == "quarantined"
    state = fx.delivery()
    assert state is not None and state.last_acked_generation is None
    assert state.quarantined_batches == 1


# --- transport policy ------------------------------------------------------------------------


def test_remote_plaintext_http_never_leaves_the_machine() -> None:
    fx.configure("https://collector.example.test", credential_ref=None)
    fx.plant_policy(lambda p: p["destinations"][0].update(endpoint="http://203.0.113.9:4318"))
    fx.publish(1)
    dispatched: list[object] = []

    def recording_post(*args: object, **kwargs: object) -> export_http.Outcome:
        dispatched.append(args)
        return fx.offline_post()

    result = cycle(deps=deps(post=recording_post))
    assert (result.state, result.reason) == ("blocked", "endpoint_invalid")
    assert dispatched == [] and fx.metas() == [] and fx.delivery() is None
    # the transport refuses plaintext remote too (defence in depth)
    outcome = export_http.post(export_http.Target("http://203.0.113.9:4318/v1/metrics"), b"{}")
    assert (outcome.kind, outcome.category) == ("transport_refused", "refused_transport")


def test_tls_succeeds_only_when_the_lab_certificate_is_trusted() -> None:
    with LabReceiver.start(tls=True) as secure:
        fx.configure(secure.url)
        fx.publish(1)
        default_context = cycle()  # system trust store: the self-signed cert is NOT trusted
        assert default_context.state == "waiting" and default_context.error_category == "tls"
        assert secure.requests == [] and fx.metas()[0].state == "queued"
        state = fx.delivery()
        assert state is not None and state.last_acked_generation is None
        assert secure.cert is not None
        trusted = cycle(seconds=400, deps=deps(context=secure.cert.client_context()))
        assert trusted.state == "sent" and trusted.acked == 1
        assert len(secure.requests) == 1 and secure.requests[0].tls


def test_explicit_managed_proxy_is_used_and_ambient_proxy_env_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.accounting.export.lab_receiver import ProxyLab

    with LabReceiver.start() as direct, ProxyLab.start() as proxy:
        assert isinstance(proxy, ProxyLab)
        monkeypatch.setenv("HTTPS_PROXY", direct.url)
        monkeypatch.setenv("HTTP_PROXY", direct.url)
        monkeypatch.setenv("https_proxy", direct.url)
        fx.configure("https://collector.example.test", proxy=proxy.url, credential_ref=None)
        fx.publish(1)
        result = cycle()
        assert result.state == "waiting"  # the lab proxy refuses to tunnel (502)
        assert len(proxy.connects) == 1  # the explicit managed proxy was dialled
        assert proxy.connects[0].path == "collector.example.test:443"
        assert direct.requests == [] and direct.connections == 0  # ambient env never consulted


# --- partial success -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("signal", "kind", "field"),
    [(M, "metrics", "rejectedDataPoints"), (L, "logs", "rejectedLogRecords")],
)
def test_partial_success_is_never_acknowledged_and_never_resent(
    lab: LabReceiver, signal: Signal, kind: Literal["metrics", "logs"], field: str
) -> None:
    if signal is M:
        setup_metrics(lab)
    else:
        setup_logs(lab)
    lab.queue(Partial(5, kind=kind, message="rejected-RESPONSE-CANARY"))
    first = cycle(signal)
    assert first.partial == 1 and first.acked == 0 and "partial_success" in first.warnings
    state = fx.delivery(signal)
    assert state is not None
    assert (state.partial_batches, state.rejected_points) == (1, 5)
    assert state.identity_unavailable is True and state.last_acked_generation is None
    assert [m.state for m in fx.metas(signal)] == ["partial"]
    assert fx.metas(signal)[0].rejected == 5
    assert acks.describe(state)["state"] == "partial"
    for step in range(1, 4):  # three more cycles: the batch is never resent
        again = cycle(signal, seconds=step * 1000)
        assert again.state == "idle" and again.sent == 0
    assert len(lab.requests) == 1
    assert fx.state(signal).last_acked_generation is None
    assert "RESPONSE-CANARY" not in fx.everything_on_disk()
    assert field in {"rejectedDataPoints", "rejectedLogRecords"}


def test_partial_with_zero_rejections_is_acked_with_a_sanitized_warning(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Partial(0, kind="metrics", message="slow-down-RESPONSE-CANARY"))
    result = cycle()
    assert result.acked == 1 and result.warnings == ("receiver_warning",)
    state = fx.delivery()
    assert state is not None and state.last_acked_generation == 1
    assert state.warning_category == "receiver_warning" and state.partial_batches == 0
    assert fx.metas()[0].warning == "receiver_warning"
    assert "RESPONSE-CANARY" not in fx.everything_on_disk()


def test_the_other_signals_rejection_field_does_not_make_a_partial(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Respond(200, b'{"partialSuccess":{"rejectedLogRecords":9}}'))  # wrong field
    result = cycle()
    assert result.acked == 1 and result.partial == 0


# --- multi-batch snapshots: acknowledged only when EVERY batch was accepted ------------------


def test_snapshot_is_acknowledged_only_after_every_batch_is_accepted(
    lab: LabReceiver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.split_small(monkeypatch)
    setup_metrics(lab)
    lab.queue(Respond(200), Respond(503, headers={"Retry-After": "30"}))
    first = cycle()
    total = fx.metas()[0].total
    assert total >= 3
    assert (first.acked, first.state) == (1, "waiting")
    state = fx.delivery()
    assert state is not None
    assert state.last_acked_generation is None  # 1 of N accepted: NOT acknowledged
    assert (state.last_attempted_generation, state.last_collected_generation) == (1, 1)
    assert [m.state for m in fx.metas()] == ["acked", "queued"] + ["queued"] * (total - 2)
    assert state.pending_batches == total - 1
    final = fx.drain(start=400)
    assert sum(r.acked for r in final) == total - 1
    done = fx.delivery()
    assert done is not None and done.last_acked_generation == 1
    assert [m.state for m in fx.metas()] == ["acked"] * total
    sent = [r.body for r in lab.requests]
    assert len(sent) == total + 1  # exactly one resend (the 503'd batch), nothing else


def test_metrics_and_logs_advance_independently(lab: LabReceiver) -> None:
    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(M, L))
    fx.commit_ledger(2)
    fx.publish(1)
    lab.queue(Respond(503), path=PATH_L)
    assert cycle(M).acked == 1
    assert cycle(L).state == "waiting"
    metrics, logs = fx.delivery(M), fx.delivery(L)
    assert metrics is not None and logs is not None
    assert metrics.last_acked_generation == 1 and logs.last_acked_generation is None
    assert cycle(L, seconds=400).acked == 1
    again = fx.delivery(L)
    assert again is not None and again.last_acked_generation == 1


def test_an_older_late_ack_cannot_regress_the_acknowledged_generation(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Respond(503))
    cycle()  # generation 1 stays queued
    fx.publish(2)
    done = fx.drain(start=400)  # generation 1 (late) then generation 2, oldest first
    assert sum(r.acked for r in done) == 2
    state = fx.delivery()
    assert state is not None and state.last_acked_generation == 2
    assert [m.generation for m in fx.metas()] == [1, 2]
    assert [r.sha256 for r in lab.requests][-2:] == [m.sha256 for m in fx.metas()]


# --- restart after an uncertain send (a REAL killed process) ---------------------------------


def killed_after_post(signal) -> None:
    """Run a cycle in a subprocess that dies (os._exit) right after the POST returned."""
    done = fx.run_driver("crash", signal.value)
    assert done.returncode == fx.CRASH_CODE and done.stdout == ""


@pytest.mark.parametrize("signal", [M, L])
def test_killed_between_post_and_ack_resends_identical_bytes_and_reports_unknown(
    lab: LabReceiver, signal
) -> None:
    setup_metrics(lab) if signal is M else setup_logs(lab, records=2)
    killed_after_post(signal)
    assert len(lab.requests) == 1  # the receiver really processed the batch
    (meta,) = fx.metas(signal)
    assert meta.state == "in_flight" and meta.attempts == 1  # the process died before settling
    died = fx.delivery(signal)
    assert died is not None and died.in_flight is not None
    assert died.last_acked_generation is None
    assert (died.pending_batches, died.unknown_batches) == (1, 1)  # unknown BEFORE any recovery
    assert acks.describe(died)["state"] == "unknown"
    if signal is L:
        assert all(e.queued and e.unknown is None for e in fx.registry().entries.values())

    lab.queue(Respond(503))  # a restart cannot resolve it while the receiver still errors
    retry = cycle(signal, seconds=60)
    assert retry.state == "waiting"
    still = fx.delivery(signal)
    assert still is not None and still.in_flight is None
    assert (still.unknown_batches, still.pending_batches) == (1, 1)  # unknown until resolved
    assert still.last_acked_generation is None
    resent = fx.metas(signal)[0]
    assert (resent.state, resent.attempts) == ("queued", 2)
    if signal is L:
        assert all(e.unknown is not None and not e.queued for e in fx.registry().entries.values())

    final = cycle(signal, seconds=600)
    assert final.state == "sent" and final.acked == 1
    shas = [r.sha256 for r in lab.requests]
    assert shas == [meta.sha256] * 3  # identical bytes on all three requests
    assert all(r.body == fx.body_of(meta) for r in lab.requests)
    state = fx.delivery(signal)
    assert state is not None and state.last_acked_generation == 1
    assert (state.unknown_batches, state.pending_batches, state.in_flight) == (0, 0, None)
    if signal is L:
        entries = fx.registry().entries.values()
        assert entries and all(e.acked is not None and e.unknown is None for e in entries)


def test_a_cycle_that_is_not_authorized_reports_the_dead_in_flight_as_nonrecallable(
    lab: LabReceiver,
) -> None:
    setup_metrics(lab)
    killed_after_post(M)
    fx.set_enabled(M, False)  # consent revoked while that request was (possibly) delivered
    result = cycle(seconds=60)
    assert (result.state, result.reason) == ("not_enabled", "not_enabled")
    assert result.nonrecallable_batch == fx.metas()[0].batch_id
    state = fx.delivery()
    assert state is not None and state.in_flight is None
    assert state.in_flight_nonrecallable == fx.metas()[0].batch_id
    assert acks.describe(state)["state"] == "unknown"
    assert len(lab.requests) == 1  # nothing further was dispatched
    assert fx.metas()[0].state == "queued"  # bounded queue preserved, not discarded


# --- one in flight per destination/signal ----------------------------------------------------


def test_a_second_runner_while_a_request_is_in_flight_is_busy_and_sends_nothing(
    lab: LabReceiver,
) -> None:
    import threading

    snapshot_id = setup_metrics(lab)
    held = Hold()
    lab.queue(held)
    results: list[CycleResult] = []
    # Complete child startup before the POST; only its actual cycle runs while held.
    with fx.ready_driver("metrics", "2") as runner:
        first = threading.Thread(target=lambda: results.append(cycle(deps=deps(read_timeout=30.0))))
        first.start()
        try:
            assert held.entered.wait(timeout=10)
            assert first.is_alive() and lab.requests[0].status is None
            other_thread = cycle(seconds=1)
            assert (other_thread.state, other_thread.reason, other_thread.sent) == (
                "busy",
                "send_in_progress",
                0,
            )
            stdout, stderr = runner.communicate(input="run\n", timeout=10)
            assert runner.returncode == 0, stderr
            assert json.loads(stdout) == {
                "state": "busy",
                "reason": "send_in_progress",
                "sent": 0,
            }
            assert first.is_alive() and lab.requests[0].status is None
            assert len(lab.requests) == lab.connections == 1
        finally:
            held.release.set()
            first.join(timeout=10)
            assert not first.is_alive()
    assert len(results) == 1
    assert (results[0].state, results[0].sent, results[0].acked) == ("sent", 1, 1)
    assert len(lab.requests) == lab.connections == 1
    (meta,) = fx.metas()
    assert (meta.state, meta.attempts, meta.last_status) == ("acked", 1, 200)
    state = fx.delivery()
    assert state is not None and state.in_flight is None
    assert (state.last_acked_generation, state.last_acked_snapshot_id) == (1, snapshot_id)


# --- --check: empty request, no side effects -------------------------------------------------


def test_check_sends_only_an_empty_request_and_writes_nothing(lab: LabReceiver) -> None:
    setup_metrics(lab)
    fx.publish(1)
    before = fx.tree()
    result = check(PROFILE, DEST, M, deps=deps())
    assert (result.state, result.ok, result.status, result.category) == ("ok", True, 200, "ok")
    (request,) = lab.requests
    assert request.body == b'{"resourceMetrics":[]}' and request.path == PATH_M
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert fx.tree() == before  # not one byte of delivery/spool/registry/ack/lock state written
    assert fx.delivery() is None and fx.metas() == []
    assert SECRET not in repr(result)


def test_check_with_logs_sends_the_empty_logs_request(lab: LabReceiver) -> None:
    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=())
    assert check(PROFILE, DEST, L, deps=deps()).ok
    assert lab.requests[0].body == b'{"resourceLogs":[]}' and lab.requests[0].path == PATH_L


def test_check_with_partial_success_on_an_empty_request_is_still_ok(lab: LabReceiver) -> None:
    setup_metrics(lab)
    lab.queue(Partial(0, kind="metrics", message="hello-RESPONSE-CANARY"))
    result = check(PROFILE, DEST, M, deps=deps())
    assert (result.state, result.ok) == ("ok", True)
    assert "RESPONSE-CANARY" not in repr(result)


def test_a_check_401_neither_blocks_nor_marks_the_destination(lab: LabReceiver) -> None:
    setup_metrics(lab)
    before = fx.tree()
    lab.queue(Respond(401, b"denied-RESPONSE-CANARY"))
    result = check(PROFILE, DEST, M, deps=deps())
    assert (result.state, result.ok, result.status) == ("unauthorized", False, 401)
    assert fx.tree() == before and fx.delivery() is None
    sent = cycle()  # a real send is not blocked by a failed check
    assert sent.state == "sent" and sent.acked == 1


@pytest.mark.parametrize(
    ("behavior", "state", "category"),
    [
        (Respond(403), "forbidden", "forbidden"),
        (Respond(429), "rate_limited", "rate_limited"),
        (Respond(503), "http_error", "http_5xx"),
        (Respond(404), "http_error", "http_4xx"),
        (DropAfterRead(), "unreachable", "network"),
        (Oversize(300_000), "oversize_response", "oversize"),
    ],
)
def test_check_maps_outcomes_to_stable_states(
    lab: LabReceiver, behavior, state: str, category: str
) -> None:
    setup_metrics(lab)
    lab.queue(behavior)
    result = check(PROFILE, DEST, M, deps=deps())
    assert (result.state, result.ok, result.category) == (state, False, category)


def test_check_reports_redirect_timeout_and_tls_failures(lab: LabReceiver) -> None:
    setup_metrics(lab)
    with LabReceiver.start() as elsewhere:
        lab.queue(Redirect(elsewhere.url_for("/v1/metrics")))
        assert check(PROFILE, DEST, M, deps=deps()).state == "redirect_refused"
        assert elsewhere.connections == 0
    lab.queue(Hang(5.0))
    slow = check(PROFILE, DEST, M, deps=deps(connect_timeout=0.4, read_timeout=0.4))
    assert (slow.state, slow.ok) == ("timeout", False)
    with LabReceiver.start(tls=True) as secure:
        fx.amend_destination(endpoint=secure.url)
        assert check(PROFILE, DEST, M, deps=deps()).state == "tls_error"
        assert secure.cert is not None
        trusted = check(PROFILE, DEST, M, deps=deps(context=secure.cert.client_context()))
        assert trusted.ok and len(secure.requests) == 1


def test_check_on_a_dead_port_is_unreachable() -> None:
    fx.configure(f"http://127.0.0.1:{fx.free_port()}")
    result = check(PROFILE, DEST, M, deps=deps())
    assert (result.state, result.ok, result.status) == ("unreachable", False, None)


def test_a_paused_profile_refuses_check_before_any_credential_read_or_network(
    lab: LabReceiver,
) -> None:
    from ssgrep.accounting.privacy import policy

    setup_metrics(lab)
    policy.pause(PROFILE)
    spy = fx.SpyEnviron(fx.ENV)
    before = fx.tree()
    with pytest.raises(ExportPaused) as raised:
        check(PROFILE, DEST, M, deps=deps(environ=spy))
    assert raised.value.condition == "export_paused"
    assert spy.looked_up == [] and lab.connections == 0 and fx.tree() == before


def test_check_refuses_an_unknown_destination_and_a_missing_credential_without_network(
    lab: LabReceiver,
) -> None:
    setup_metrics(lab)
    unknown = check(PROFILE, "nowhere", M, deps=deps())
    assert (unknown.state, unknown.category) == ("refused", "unknown_destination")
    missing = check(PROFILE, DEST, M, deps=deps(environ={}))
    assert (missing.state, missing.category) == ("refused", "credential_missing")
    assert lab.connections == 0


# --- logs: the registry follows the batch ----------------------------------------------------


def test_logs_batches_register_queued_then_acked_and_resend_nothing_when_unchanged(
    lab: LabReceiver,
) -> None:
    setup_logs(lab, records=3)
    before_send = cycle(L, enqueue=True)
    assert before_send.state == "sent" and before_send.acked == 1
    entries = fx.registry().entries
    assert len(entries) == 3 and all(e.acked and e.acked.revision == 1 for e in entries.values())
    assert all(not e.queued and e.unknown is None for e in entries.values())
    body = json.loads(lab.requests[0].body)
    records = body["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
    assert len(records) == 3
    assert cycle(L, seconds=400).sent == 0 and len(lab.requests) == 1  # nothing new, nothing sent


def test_a_changed_ledger_sends_a_higher_revision_not_a_duplicate_contribution(
    lab: LabReceiver,
) -> None:
    setup_logs(lab, records=2)
    cycle(L)
    fx.commit_ledger(2, extra_for=(1,), generation=2)
    fx.publish(2)
    result = cycle(L, seconds=400)
    assert result.acked == 1
    revisions = {
        json.loads(r["body"]["stringValue"])["revision"]
        for r in json.loads(lab.requests[1].body)["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
    }
    assert revisions == {2}  # only the changed record, as the next revision
    assert sorted(e.acked.revision for e in fx.registry().entries.values()) == [1, 2]


# --- tombstones ------------------------------------------------------------------------------


def record_types(body: bytes) -> set[str]:
    records = json.loads(body)["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
    return {json.loads(r["body"]["stringValue"])["record_type"] for r in records}


def enqueue_deletions() -> list[str]:
    from datetime import UTC, datetime

    from ssgrep.accounting.privacy import policy, published as tombstones

    current = policy.load_profile_policy(PROFILE)
    plans = tombstones.plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    out = tombstones.enqueue_tombstones(
        PROFILE, current, plans, now=datetime(2026, 10, 14, tzinfo=UTC)
    )
    return [bid for result in out for bid in result.batch_ids]


def test_tombstones_follow_data_batches_and_never_touch_generation_progress(
    lab: LabReceiver,
) -> None:
    from ssgrep.accounting.privacy import published as tombstones

    setup_logs(lab, records=2)
    cycle(L)
    assert fx.state(L).last_acked_generation == 1
    fx.commit_ledger(3, generation=2)
    fx.publish(2)
    lab.queue(Respond(503))
    cycle(L, seconds=400)  # generation 2's data batch is queued (503)
    (batch_id,) = enqueue_deletions()
    assert batch_id.startswith("t") and tombstones.pending_tombstones(PROFILE, DEST) == 1
    done = fx.drain(L, start=1000)
    assert sum(r.acked for r in done) == 2
    wire = [record_types(r.body) for r in lab.requests]
    # generation 1 data, generation 2 data (503'd), generation 2 data again, THEN the deletion
    assert wire == [{"contribution"}, {"contribution"}, {"contribution"}, {"deletion"}][: len(wire)]
    assert wire[-1] == {"deletion"} and all(w != {"deletion"} for w in wire[:-1])
    tomb = next(m for m in fx.metas(L) if m.kind == "tombstone")
    assert (tomb.generation, tomb.state) == (0, "acked")
    state = fx.delivery(L)
    assert state is not None and state.last_acked_generation == 2  # data progress only
    assert tombstones.pending_tombstones(PROFILE, DEST) == 0
    # the two accepted deletions are forgotten (minimal linkage); the third record (new in
    # generation 2, never deleted) keeps its acked contribution
    (survivor,) = fx.registry().entries.values()
    assert survivor.acked is not None and survivor.acked.type == "contribution"


def test_a_partial_tombstone_is_unknown_not_forgotten_and_not_a_generation(
    lab: LabReceiver,
) -> None:
    setup_logs(lab, records=1)
    cycle(L)
    enqueue_deletions()
    lab.queue(Partial(1, kind="logs"))
    result = cycle(L, seconds=400)
    assert result.partial == 1 and result.acked == 0
    (entry,) = fx.registry().entries.values()
    assert entry.acked is not None and entry.acked.type == "contribution"
    assert entry.unknown is not None and entry.unknown.type == "deletion"
    state = fx.delivery(L)
    assert state is not None and state.last_acked_generation == 1 and state.partial_batches == 0
    assert cycle(L, seconds=2000).sent == 0  # a partial deletion is never blindly resent


def test_a_revoked_logs_permission_discards_queued_tombstones_with_an_explicit_loss(
    lab: LabReceiver,
) -> None:
    setup_logs(lab, records=1)
    cycle(L)
    enqueue_deletions()
    current = fx.policy.load_profile_policy(PROFILE)
    narrowed = fx.replace(current.destinations[0], logs_permitted=False, signals=(M,))
    fx.amend(destinations=(narrowed,), enabled_exports=())  # logs revoked in ONE valid save
    result = cycle(L, seconds=400)
    assert (result.state, result.reason) == ("revoked", "signal_not_permitted")
    assert len(lab.requests) == 1
    assert [m for m in fx.metas(L) if m.kind == "tombstone"] == []
    state = fx.delivery(L)
    assert state is not None
    assert [loss.reason for loss in state.loss_summary] == ["policy_contraction"]
    assert state.lost_records == 1


# --- the worker callback ---------------------------------------------------------------------


def test_export_callback_runs_every_enabled_destination_and_never_raises(lab: LabReceiver) -> None:
    from ssgrep.accounting.export.delivery import export_callback

    setup_metrics(lab)
    assert export_callback(PROFILE, deps=deps())() is None
    assert len(lab.requests) == 1 and fx.state().last_acked_generation == 1
    assert not (fx.profile_root() / "export" / "last_error.json").exists()


def test_export_callback_records_only_a_class_name_when_everything_explodes(
    lab: LabReceiver, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting.export import delivery
    from ssgrep.accounting.export.delivery import export_callback

    setup_metrics(lab)

    def explode(*args: object, **kwargs: object) -> list:
        raise RuntimeError("secret-EXCEPTION-TEXT /home/user/private " + SECRET)

    monkeypatch.setattr(delivery, "run_export", explode)
    export_callback(PROFILE, deps=deps())()  # must not raise
    record = json.loads((fx.profile_root() / "export" / "last_error.json").read_text())
    assert record["error"] == "RuntimeError" and set(record) == {"schema", "at", "error"}
    hay = fx.everything_on_disk()
    assert "EXCEPTION-TEXT" not in hay and SECRET not in hay and "/home/user" not in hay


def test_a_destination_that_fails_does_not_stop_the_others(lab: LabReceiver) -> None:
    from ssgrep.accounting.export.delivery import run_export

    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(M, L))
    fx.publish(1)  # logs have no ledger: a vacuous generation; metrics send normally
    results = run_export(PROFILE, deps=deps())
    assert {(r.destination, r.signal.value) for r in results} == {(DEST, "metrics"), (DEST, "logs")}
    assert next(r for r in results if r.signal is M).acked == 1
    assert fx.state(M).last_acked_generation == 1


# --- a hostile snapshot cannot leak text or earn an acknowledgment ----------------------------

HOSTILE = "not-a-time-SNAPSHOT-CANARY"


@pytest.mark.parametrize("signal", [M, L])
def test_an_unparseable_snapshot_is_a_stable_failure_never_an_acknowledgment(
    lab: LabReceiver, signal
) -> None:
    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(signal,))
    fx.commit_ledger(2)
    fx.publish(1, collected_at=HOSTILE)
    result = cycle(signal)
    assert result.state == "failed" and result.sent == 0 and not lab.requests
    assert result.reason in {"snapshot_invalid", "snapshot_unexportable"}
    assert HOSTILE not in repr(result) + fx.everything_on_disk()
    state = fx.delivery(signal)
    assert state is None or state.last_acked_generation is None  # nothing was claimed
    assert fx.metas(signal) == []
    # the failure is recoverable: a good snapshot afterwards is delivered normally
    fx.publish(2)
    assert cycle(signal, seconds=400).acked == 1


# --- retention inside a cycle, enqueue=False, run_export over revoked destinations -----------


def test_an_expired_spool_is_an_explicit_loss_never_a_send_and_never_a_false_ack(
    lab: LabReceiver,
) -> None:
    setup_logs(lab, records=2)
    lab.queue(Respond(503))
    cycle(L)  # queued + attempted once
    (old,) = fx.metas(L)
    assert all(e.queued for e in fx.registry().entries.values())
    week_later = 8 * 24 * 3600
    result = cycle(L, seconds=week_later)
    assert len(lab.requests) == 1 and result.sent == 0  # nothing went out on expiry
    assert fx.metas(L) == []  # the aged-out bytes are gone ...
    assert fx.registry().entries == {} or all(not e.queued for e in fx.registry().entries.values())
    state = fx.state(L)
    assert state.last_acked_generation is None  # ... and were NEVER marked accepted
    assert state.lost_records == old.items and state.loss_summary[0].records == old.items
    assert [p.lost for p in state.progress] == [(0,)]  # the generation can never be acknowledged
    assert result.state == "idle"


def test_enqueue_false_sends_what_is_queued_but_never_builds_a_new_snapshot(
    lab: LabReceiver,
) -> None:
    setup_metrics(lab)
    lab.queue(Respond(503))
    cycle()  # generation 1 queued
    fx.publish(2)
    result = cycle(seconds=400, enqueue=False)
    assert result.enqueued_batches == 0 and result.acked == 1
    assert [m.generation for m in fx.metas()] == [1]  # generation 2 was NOT spooled
    assert fx.state().last_collected_generation == 1


def test_run_export_discards_the_queue_of_a_destination_dropped_from_policy(
    lab: LabReceiver,
) -> None:
    from ssgrep.accounting.export.delivery import run_export
    from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy

    setup_metrics(lab)
    lab.queue(Respond(503))
    cycle()
    fx.amend(
        destinations=(DestinationPolicy("other-org", "http://127.0.0.1:1"),), enabled_exports=()
    )
    results = run_export(PROFILE, now=at(1000), deps=deps())
    assert [(r.destination, r.state, r.reason) for r in results] == [
        (DEST, "revoked", "unknown_destination")
    ]
    assert len(lab.requests) == 1 and fx.metas() == []
    assert [loss.reason for loss in fx.state().loss_summary] == ["policy_contraction"]


def test_a_corrupt_state_file_is_a_stable_failure_that_never_sends_or_repairs(
    lab: LabReceiver,
) -> None:
    setup_metrics(lab)
    cycle()  # a healthy first cycle writes the delivery document
    path = fx.profile_root() / "delivery" / DEST / "metrics.json"
    path.write_text("{ not json CORRUPT-CANARY")
    fx.publish(2)
    result = cycle(seconds=400)
    assert (result.state, result.reason) == ("failed", "state_corrupt")
    assert len(lab.requests) == 1
    assert path.read_text() == "{ not json CORRUPT-CANARY"  # never rewritten
    assert "CORRUPT-CANARY" not in repr(result)


# --- orchestration and dispatch fault boundaries (real queued generations) ------------------


def test_no_snapshot_cycle_and_unreadable_preview_recover_after_collection(
    tmp_path: Path,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url, credential_ref=None)
        assert export_consent.plan_enable(PROFILE, DEST, M).credential_ref is None
        empty = fx.cycle()
        assert (empty.state, empty.sent) == ("no_snapshot", 0)
        verdict = export_consent.classify_cycle(empty, PROFILE)
        assert not verdict.ok and verdict.condition == "no_snapshot"
        assert fx.delivery() is None and lab.requests == []
        directory = fx.profile_root() / "snapshots"
        saved = tmp_path / "snapshots"
        if directory.exists():
            directory.rename(saved)
        directory.symlink_to(tmp_path, target_is_directory=True)
        assert delivery_candidate.current_envelope(PROFILE) is None
        assert fx.cycle().state == "no_snapshot"
        directory.unlink()
        if saved.exists():
            saved.rename(directory)
        sid = fx.publish(1)
        assert fx.cycle(seconds=1).acked == 1
        assert fx.state().last_acked_snapshot_id == sid
        assert lab.requests[0].body == fx.body_of(fx.metas()[0])


def test_disabled_target_under_send_lock_is_busy_without_losing_queued_bytes() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        (meta,) = fx.metas()
        body = fx.body_of(meta)
        fx.set_enabled(M, False)
        with export_spool.locked(PROFILE, ("export", DEST), "send-metrics"):
            busy = fx.cycle(seconds=1)
        assert (busy.state, busy.reason, busy.sent) == ("busy", "send_in_progress", 0)
        assert export_consent.classify_cycle(busy, PROFILE).condition == "export_busy"
        assert fx.body_of(meta) == body and lab.requests == []
        fx.set_enabled(M, True)
        assert fx.cycle(seconds=2).acked == 1
        assert [r.body for r in lab.requests] == [body]


@pytest.mark.parametrize("argument", ["profile", "destination", "clock"])
def test_programmer_argument_errors_create_no_export_state(argument: str) -> None:
    options: dict[str, Any] = {"profile": PROFILE, "destination": DEST, "signal": M, "now": fx.at()}
    if argument == "clock":
        options["now"] = datetime(2026, 10, 13)
    else:
        options[argument] = "../escape"
    before = fx.tree()
    with pytest.raises(ValueError):
        delivery.run_destination(**options)
    assert fx.tree() == before
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        assert fx.cycle().acked == 1 and len(lab.requests) == 1


@pytest.mark.parametrize("damage", ["short-key", "unsafe-key", "too-large", "unexpected-write"])
def test_failed_enqueue_exposes_only_stable_code_then_delivers_after_repair(
    damage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "short-key": "storage_unsafe",
        "unsafe-key": "storage_unsafe",
        "too-large": "export_too_large",
        "unexpected-write": "internal_error",
    }[damage]
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        sid = fx.publish(1)
        export_ids.load_key(PROFILE, create=True)
        key_path = fx.profile_root() / "export" / "pseudonym.key"
        key = key_path.read_bytes()
        with monkeypatch.context() as fault:
            if damage == "short-key":
                key_path.write_bytes(b"DAMAGED-KEY")
            elif damage == "unsafe-key":
                target = tmp_path / "secret"
                key_path.rename(target)
                key_path.symlink_to(target)
            elif damage == "too-large":
                real_split = export_http.split_metrics
                fault.setattr(
                    export_http, "split_metrics", lambda r, p: real_split(r, p, max_bytes=1)
                )
            else:
                real_write = storage.atomic_write

                def failed_write(fd, name, data):
                    if name.endswith(".meta.json"):
                        raise RuntimeError("PRIVATE-WRITE-CANARY")
                    return real_write(fd, name, data)

                fault.setattr(storage, "atomic_write", failed_write)
            failed = fx.cycle()
        assert (failed.state, failed.reason, failed.sent, failed.acked) == (
            "failed",
            expected,
            0,
            0,
        )
        verdict = export_consent.classify_cycle(failed, PROFILE)
        assert not verdict.ok and verdict.condition == "send_failed"
        assert lab.requests == [] and fx.delivery() is None
        assert "PRIVATE-WRITE-CANARY" not in repr(failed) + fx.everything_on_disk()
        if key_path.is_symlink():
            key_path.unlink()
        key_path.write_bytes(key)
        key_path.chmod(0o600)
        assert fx.cycle(seconds=1).acked == 1
        assert fx.state().last_acked_snapshot_id == sid
        assert len(lab.requests) == 1


def test_transport_size_refusal_blocks_without_post_and_changed_credential_unblocks() -> None:
    def bounded_post(target, body, **kwargs):
        return export_http.post(target, body, max_request_bytes=1, **kwargs)

    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        blocked = fx.cycle(deps=fx.deps(post=bounded_post))
        assert (blocked.state, blocked.blocked_reason, blocked.acked) == (
            "blocked",
            "policy_blocked",
            0,
        )
        assert lab.requests == [] and fx.state().last_acked_generation is None
        original = fx.body_of(fx.metas()[0])
        assert fx.cycle(seconds=1).state == "blocked"
        changed = {fx.ENV_NAME: "ROTATED-INGEST-CONTROL"}
        assert fx.cycle(seconds=2, deps=fx.deps(environ=changed)).acked == 1
        assert [r.body for r in lab.requests] == [original]


@pytest.mark.parametrize("window", ["before-begin", "before-read"])
def test_privacy_marker_racing_dispatch_prevents_request_and_later_generation_succeeds(
    window: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        (meta,) = fx.metas()
        body = fx.body_of(meta)
        real_read = export_spool.read_body
        marked = []

        def privacy_race(profile, destination, signal, batch_id):
            if not marked:
                marked.append(batch_id)
                export_spool.update_meta(
                    profile,
                    destination,
                    signal,
                    batch_id,
                    lambda m: replace(m, purge_requested=True),
                )
                if window == "before-read":
                    path = fx.profile_root() / "spool" / DEST / M.value / f"{batch_id}.json"
                    path.write_bytes(b"privacy-erased")
            return real_read(profile, destination, signal, batch_id)

        with monkeypatch.context() as race:
            race.setattr(export_spool, "read_body", privacy_race)
            refused = fx.cycle(seconds=1)
        assert marked == [meta.batch_id]
        assert (refused.sent, refused.acked, refused.quarantined) == (0, 0, 0)
        assert lab.requests == [] and fx.state().last_acked_generation is None
        current = fx.metas()[0]
        assert current.purge_requested and current.attempts == 0
        if window == "before-begin":
            assert fx.body_of(current) == body
        sid = fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1
        assert fx.state().last_acked_snapshot_id == sid
        assert len(lab.requests) == 1 and lab.requests[0].body != body


def test_log_ack_survives_body_becoming_unreadable_after_receiver_accepts() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(L,), logs_permitted=True, enabled=(L,))
        fx.commit_ledger(1)
        sid = fx.publish(1)
        captured = []

        def post_then_damage(target, body, **kwargs):
            result = export_http.post(target, body, **kwargs)
            captured.append(body)
            (meta,) = fx.metas(L)
            path = fx.profile_root() / "spool" / DEST / L.value / f"{meta.batch_id}.json"
            path.write_bytes(b"corrupt-after-accept")
            return result

        result = fx.cycle(L, deps=fx.deps(post=post_then_damage))
        assert result.acked == 1 and fx.state(L).last_acked_snapshot_id == sid
        assert [r.body for r in lab.requests] == captured
        (entry,) = fx.registry().entries.values()
        assert entry.acked and entry.acked.revision == 1 and not entry.queued
        assert fx.cycle(L, seconds=1).sent == 0
        preview = delivery_candidate.preview_candidate(PROFILE, DEST, L, now=fx.at(2))
        assert preview.state == "no_changes" and preview.candidate is not None
        assert preview.candidate.batches == ()


def test_policy_becomes_unreadable_after_scope_refresh_holds_queue_until_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        body = fx.body_of(fx.metas()[0])
        path = policy.policy_path()
        original = path.read_bytes()
        real_observe = export_policy.observe_scope

        def revoke_after_scope(*args, **kwargs):
            result = real_observe(*args, **kwargs)
            path.write_bytes(b"{bad policy")
            return result

        with monkeypatch.context() as race:
            race.setattr(export_policy, "observe_scope", revoke_after_scope)
            held = fx.cycle(seconds=1, enqueue=False)
        assert held.state == "paused" and held.sent == 0
        assert fx.body_of(fx.metas()[0]) == body and lab.requests == []
        path.write_bytes(original)
        assert fx.cycle(seconds=2).acked == 1
        assert [r.body for r in lab.requests] == [body]


def test_missing_generation_during_revalidation_discards_without_false_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        real_read = snapshots.read_generation

        def unavailable(*args, **kwargs):
            raise OSError("snapshot device temporarily unavailable")

        with monkeypatch.context() as fault:
            fault.setattr(snapshots, "read_generation", unavailable)
            result = fx.cycle(seconds=1, enqueue=False)
        assert result.invalidated_batches == 1 and result.sent == 0
        assert fx.state().last_acked_generation is None and fx.metas() == []
        assert fx.state().loss_summary[0].reason == "policy_contraction"
        assert lab.requests == []
        assert real_read(PROFILE, 1) is not None
        sid = fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1
        assert fx.state().last_acked_snapshot_id == sid


def test_disabled_settled_spool_cannot_recreate_a_deleted_ack_document() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        assert fx.cycle().acked == 1
        (settled,) = fx.metas()
        body = fx.body_of(settled)
        path = fx.profile_root() / "delivery" / DEST / M.value
        path.with_suffix(".json").unlink()
        fx.set_enabled(M, False)
        held = fx.cycle(seconds=1)
        assert held.state == "not_enabled" and held.sent == 0
        assert fx.delivery() is None
        assert fx.metas() == [settled] and fx.body_of(settled) == body
        assert len(lab.requests) == 1
        fx.set_enabled(M, True)
        sid = fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1
        assert fx.state().last_acked_snapshot_id == sid
        assert len(lab.requests) == 2 and lab.requests[1].body != body


# --- export callback and consent control ----------------------------------------------------


def test_callback_absent_export_directory_does_not_create_state() -> None:
    before = fx.tree()
    delivery.export_callback(PROFILE)()
    assert fx.tree() == before
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        delivery.export_callback(PROFILE, deps=fx.deps())()
        assert len(lab.requests) == 1 and fx.state().last_acked_generation == 1


@pytest.mark.parametrize("failure", ["report-write", "report-delete"])
def test_callback_reporting_failures_never_stop_recovery(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        export_ids.load_key(PROFILE, create=True)
        key_path = fx.profile_root() / "export" / "pseudonym.key"
        key = key_path.read_bytes()
        key_path.write_bytes(b"broken")
        callback = delivery.export_callback(PROFILE, deps=fx.deps())
        error_path = fx.profile_root() / "export" / "last_error.json"
        callback()
        report = error_path.read_bytes()
        assert json.loads(report)["error"] == "storage_unsafe"
        assert lab.requests == [] and fx.delivery() is None
        if failure == "report-delete":
            key_path.write_bytes(key)
        real_write, real_delete = storage.atomic_write, storage.delete_file

        def fail_write(fd, name, data):
            if name == "last_error.json":
                raise OSError(errno.ENOSPC, "REPORT-WRITE-CANARY")
            return real_write(fd, name, data)

        def fail_delete(fd, name):
            if name == "last_error.json":
                raise OSError(errno.EACCES, "REPORT-DELETE-CANARY")
            return real_delete(fd, name)

        with monkeypatch.context() as fault:
            fault.setattr(
                storage, "atomic_write", fail_write if failure == "report-write" else real_write
            )
            fault.setattr(
                storage, "delete_file", fail_delete if failure == "report-delete" else real_delete
            )
            callback()
        assert error_path.read_bytes() == report
        assert "REPORT-WRITE-CANARY" not in fx.everything_on_disk()
        assert "REPORT-DELETE-CANARY" not in fx.everything_on_disk()
        key_path.write_bytes(key)
        callback()
        assert not error_path.exists()
        assert len(lab.requests) == 1 and fx.state().last_acked_generation == 1


def test_target_discovery_failure_does_not_erase_queued_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        before = fx.tree()
        body = fx.body_of(fx.metas()[0])

        def unavailable(profile):
            raise OSError(errno.EIO, "spool mount disappeared")

        with monkeypatch.context() as fault:
            fault.setattr(delivery.export_spool, "iter_spools", unavailable)
            assert delivery.run_export(PROFILE, now=fx.at(), deps=fx.deps()) == []
        assert fx.tree() == before and lab.requests == []
        results = delivery.run_export(PROFILE, now=fx.at(1), deps=fx.deps())
        assert len(results) == 1 and results[0].acked == 1
        assert [r.body for r in lab.requests] == [body]


def test_malformed_enabled_signal_is_ignored_without_suppressing_valid_destination() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.plant_policy(lambda p: p["enabled_exports"].append([DEST, "bogus-signal"]))
        results = delivery.run_export(PROFILE, now=fx.at(), deps=fx.deps())
        assert [(r.destination, r.signal, r.acked) for r in results] == [(DEST, M, 1)]
        assert len(lab.requests) == 1 and fx.state().last_acked_generation == 1


# ---- Consolidated from test_acknowledgments.py (source-specific globals are prefixed) ----


ack_PROFILE = "personal-local"
ack_DEST = "org-primary"
ack_NOW = "2026-10-13T10:00:00Z"
LATER = "2026-10-13T10:05:00Z"
SNAP7 = "a7" * 32
SNAP8 = "a8" * 32
SNAP9 = "a9" * 32
ack_CANARY = "sk-CANARY-SECRET-0123456789"


def fresh(signal: ack_Signal = ack_Signal.METRICS) -> ack_acks.Delivery:
    return ack_acks.Delivery(ack_DEST, signal)


def ack(
    d: ack_acks.Delivery, gen: int, snap: str, index: int, total: int, **kw: ack_Any
) -> ack_acks.Delivery:
    kw.setdefault("now", ack_NOW)
    return ack_acks.outcome(
        d, generation=gen, snapshot_id=snap, index=index, total=total, kind="acked", **kw
    )


def doc_path(tmp_path: ack_Path, signal: str = "metrics", dest: str = ack_DEST) -> ack_Path:
    return tmp_path / "data" / "accounting" / ack_PROFILE / "delivery" / dest / f"{signal}.json"


def ack_tree(root: ack_Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


# --- response parsing -------------------------------------------------------------------------


@ack_pytest.mark.parametrize(
    "body", [None, b"", b"{}", b'{"other":1}', b'{"x":{"rejectedDataPoints":9}}']
)
@ack_pytest.mark.parametrize("signal", list(ack_Signal))
def test_absent_partial_success_is_full(signal: ack_Signal, body: bytes | None) -> None:
    assert ack_acks.parse_response(signal, body) == ack_acks.AckResult("full", 0, None, False)


@ack_pytest.mark.parametrize(
    ("signal", "field"),
    [(ack_Signal.METRICS, "rejectedDataPoints"), (ack_Signal.LOGS, "rejectedLogRecords")],
)
@ack_pytest.mark.parametrize("zero", [0, "0", None])
def test_zero_rejections_are_full_with_sanitized_warning(
    signal: ack_Signal, field: str, zero: object
) -> None:
    partial: dict[str, object] = {"errorMessage": f"quota soon {CANARY}"}
    if zero is not None:
        partial[field] = zero
    body = ack_json.dumps({"partialSuccess": partial}).encode()
    result = ack_acks.parse_response(signal, body)
    assert result == ack_acks.AckResult("full_with_warning", 0, "receiver_warning", False)
    assert ack_CANARY not in repr(result)


@ack_pytest.mark.parametrize("signal", list(ack_Signal))
def test_empty_partial_success_object_is_plain_full_acceptance(signal: ack_Signal) -> None:
    """A real OTel Collector answers every fully accepted request with ``{"partialSuccess":{}}``.

    The object carries no information (proto3 JSON default values), so it must not turn every
    healthy delivery into a permanent warning; any content in it still does.
    """
    assert ack_acks.parse_response(signal, b'{"partialSuccess":{}}') == ack_acks.AckResult("full")
    assert (
        ack_acks.parse_response(signal, b'{"partialSuccess":{"errorMessage":""}}').outcome == "full"
    )
    zero_only = b'{"partialSuccess":{"rejectedDataPoints":0,"rejectedLogRecords":"0"}}'
    assert ack_acks.parse_response(signal, zero_only).outcome == "full_with_warning"


@ack_pytest.mark.parametrize(
    ("signal", "field", "value", "expected"),
    [
        (ack_Signal.METRICS, "rejectedDataPoints", 5, 5),
        (ack_Signal.METRICS, "rejectedDataPoints", "7", 7),
        (ack_Signal.LOGS, "rejectedLogRecords", 3, 3),
        (ack_Signal.LOGS, "rejectedLogRecords", "12", 12),
        (ack_Signal.LOGS, "rejectedLogRecords", str((1 << 63) - 1), (1 << 63) - 1),
    ],
)
def test_nonzero_rejections_are_partial_with_unavailable_identities(
    signal: ack_Signal, field: str, value: object, expected: int
) -> None:
    body = ack_json.dumps({"partialSuccess": {field: value, "errorMessage": ack_CANARY}}).encode()
    result = ack_acks.parse_response(signal, body)
    assert result == ack_acks.AckResult("partial", expected, None, True)
    assert ack_CANARY not in repr(result)


def test_other_signals_rejection_field_is_not_mistaken() -> None:
    logs_field_on_metrics = b'{"partialSuccess":{"rejectedLogRecords":9}}'
    metrics_field_on_logs = b'{"partialSuccess":{"rejectedDataPoints":9}}'
    assert (
        ack_acks.parse_response(ack_Signal.METRICS, logs_field_on_metrics).outcome
        == "full_with_warning"
    )
    assert (
        ack_acks.parse_response(ack_Signal.LOGS, metrics_field_on_logs).outcome
        == "full_with_warning"
    )
    both = b'{"partialSuccess":{"rejectedLogRecords":4,"rejectedDataPoints":0}}'
    assert ack_acks.parse_response(ack_Signal.LOGS, both) == ack_acks.AckResult(
        "partial", 4, None, True
    )
    assert ack_acks.parse_response(ack_Signal.METRICS, both).outcome == "full_with_warning"


@ack_pytest.mark.parametrize(
    "body",
    [
        b'{"partialSuccess":{"rejectedDataPoints":-1}}',
        b'{"partialSuccess":{"rejectedDataPoints":true}}',
        b'{"partialSuccess":{"rejectedDataPoints":false}}',
        b'{"partialSuccess":{"rejectedDataPoints":1.0}}',
        b'{"partialSuccess":{"rejectedDataPoints":0.0}}',
        b'{"partialSuccess":{"rejectedDataPoints":"1.5"}}',
        b'{"partialSuccess":{"rejectedDataPoints":"1e3"}}',
        b'{"partialSuccess":{"rejectedDataPoints":"abc"}}',
        b'{"partialSuccess":{"rejectedDataPoints":"-3"}}',
        b'{"partialSuccess":{"rejectedDataPoints":" 3"}}',
        b'{"partialSuccess":{"rejectedDataPoints":""}}',
        b'{"partialSuccess":{"rejectedDataPoints":"007"}}',
        b'{"partialSuccess":{"rejectedDataPoints":null}}',
        b'{"partialSuccess":{"rejectedDataPoints":[1]}}',
        b'{"partialSuccess":{"rejectedDataPoints":9223372036854775808}}',
        b'{"partialSuccess":{"rejectedDataPoints":"9223372036854775808"}}',
        b'{"partialSuccess":{"rejectedDataPoints":NaN}}',
        b'{"partialSuccess":"oops"}',
        b'{"partialSuccess":null}',
        b'{"partialSuccess":[]}',
        b'{"partialSuccess":7}',
        b"not json",
        b'{"partialSuccess":',
        b"\xff\xfe\x00",
        b"[]",
        b"7",
        b'"text"',
        b"null",
    ],
)
def test_malformed_responses_are_unparseable_never_full(body: bytes) -> None:
    result = ack_acks.parse_response(ack_Signal.METRICS, body)
    assert result == ack_acks.AckResult("unparseable", 0, None, False)


def test_deeply_nested_body_is_unparseable() -> None:
    body = b"[" * 100_000 + b"]" * 100_000
    assert ack_acks.parse_response(ack_Signal.LOGS, body).outcome == "unparseable"


# --- pure transitions -------------------------------------------------------------------------


def test_collected_and_attempted_are_max_monotonic() -> None:
    d = ack_acks.collected(ack_acks.collected(fresh(), 8), 5)
    assert d.last_collected_generation == 8
    d = ack_acks.attempted(ack_acks.attempted(d, 8, ack_NOW), 6, LATER)
    assert (d.last_attempted_generation, d.last_attempt_at) == (8, LATER)


def test_in_flight_lifecycle_and_nonrecallable() -> None:
    d = ack_acks.start_in_flight(fresh(), "g0000000008-a8a8a8a8a8a8a8a8-0001", 8, ack_NOW)
    assert d.in_flight == ack_acks.InFlight("g0000000008-a8a8a8a8a8a8a8a8-0001", 8, ack_NOW)
    d = ack_acks.mark_nonrecallable(d, "g0000000008-a8a8a8a8a8a8a8a8-0001")
    assert d.in_flight_nonrecallable == "g0000000008-a8a8a8a8a8a8a8a8-0001"
    done = ack_acks.finish_in_flight(d)
    assert done.in_flight is None
    assert done.in_flight_nonrecallable == "g0000000008-a8a8a8a8a8a8a8a8-0001"


def test_full_generation_acknowledges_only_when_every_index_acked() -> None:
    d = fresh()
    d = ack(d, 8, SNAP8, 0, 3)
    d = ack(d, 8, SNAP8, 2, 3)
    assert (d.last_acked_generation, d.last_acked_snapshot_id, d.last_acked_at) == (None,) * 3
    assert d.progress == (ack_acks.Progress(8, SNAP8, 3, acked=(0, 2)),)
    d = ack(d, 8, SNAP8, 1, 3, now=LATER)
    assert (d.last_acked_generation, d.last_acked_snapshot_id, d.last_acked_at) == (8, SNAP8, LATER)
    assert d.progress == ()  # entries up to the acknowledged generation are pruned
    assert d.late_acks == 0


def test_duplicate_ack_of_one_index_does_not_complete_a_generation() -> None:
    d = ack(ack(fresh(), 8, SNAP8, 0, 2), 8, SNAP8, 0, 2)
    assert d.last_acked_generation is None
    assert d.progress == (ack_acks.Progress(8, SNAP8, 2, acked=(0,)),)


def test_late_older_generation_never_regresses_the_acknowledgment() -> None:
    d = ack(fresh(), 7, SNAP7, 1, 2)  # gen 7 half done
    d = ack(d, 8, SNAP8, 0, 1)  # gen 8 completes first
    assert (d.last_acked_generation, d.last_acked_snapshot_id) == (8, SNAP8)
    assert d.progress == ()  # gen 7 is older than the acknowledged generation: pruned
    late = ack(d, 7, SNAP7, 0, 2)
    assert (late.last_acked_generation, late.last_acked_snapshot_id) == (8, SNAP8)
    assert late.late_acks == 0  # pruned progress cannot complete: nothing to credit


def test_progress_of_a_superseded_generation_is_pruned_when_a_newer_one_acks() -> None:
    d = ack(ack(fresh(), 7, SNAP7, 0, 2), 9, SNAP9, 0, 2)
    d = ack(ack(d, 9, SNAP9, 1, 2, now=LATER), 7, SNAP7, 1, 2)
    assert (d.last_acked_generation, d.last_acked_snapshot_id, d.last_acked_at) == (9, SNAP9, LATER)
    assert d.late_acks == 0  # gen 7 progress was pruned: the remainder cannot complete it


def test_late_completion_of_older_generation_keeps_acknowledged_generation() -> None:
    # gen 8 acknowledged first; gen 7's progress arrives only afterwards (restart replays it)
    d = ack(fresh(), 8, SNAP8, 0, 1, now=LATER)
    d = ack(d, 7, SNAP7, 0, 2)
    d = ack(d, 7, SNAP7, 1, 2)
    assert (d.last_acked_generation, d.last_acked_snapshot_id, d.last_acked_at) == (8, SNAP8, LATER)
    assert d.late_acks == 1
    assert d.progress == (ack_acks.Progress(7, SNAP7, 2, acked=(0, 1)),)


def test_partial_batch_poisons_its_generation_but_not_a_later_one() -> None:
    d = fresh()
    d = ack_acks.outcome(
        d, generation=8, snapshot_id=SNAP8, index=1, total=3, kind="partial",
        rejected=4, identity_unavailable=True, now=ack_NOW,
    )  # fmt: skip
    assert (d.partial_batches, d.rejected_points, d.identity_unavailable) == (1, 4, True)
    d = ack(ack(d, 8, SNAP8, 0, 3), 8, SNAP8, 2, 3)
    assert d.last_acked_generation is None and d.last_acked_snapshot_id is None
    d = ack(d, 8, SNAP8, 1, 3)  # re-acking the partial index must not rescue the generation
    assert d.last_acked_generation is None
    assert d.progress == (ack_acks.Progress(8, SNAP8, 3, acked=(0, 2), partial=(1,)),)
    later = ack(d, 9, SNAP9, 0, 1, now=LATER)
    assert (later.last_acked_generation, later.last_acked_snapshot_id) == (9, SNAP9)
    assert later.progress == ()
    assert (later.partial_batches, later.rejected_points) == (1, 4)  # the history stays reported


def test_partial_counters_are_idempotent_per_index() -> None:
    kw: dict[str, ack_Any] = {"generation": 8, "snapshot_id": SNAP8, "index": 0, "total": 2}
    d = ack_acks.outcome(fresh(), kind="partial", rejected=3, now=ack_NOW, **kw)
    again = ack_acks.outcome(d, kind="partial", rejected=3, now=ack_NOW, **kw)
    assert again == d
    assert (again.partial_batches, again.rejected_points) == (1, 3)


@ack_pytest.mark.parametrize("kind", ["quarantined", "lost", "discarded"])
def test_other_terminal_outcomes_block_full_ack_of_that_generation(kind: str) -> None:
    d = ack_acks.outcome(
        fresh(), generation=8, snapshot_id=SNAP8, index=0, total=2, kind=kind, now=ack_NOW
    )
    d = ack(d, 8, SNAP8, 1, 2)
    assert d.last_acked_generation is None
    assert getattr(d.progress[0], kind) == (0,) and d.progress[0].acked == (1,)
    assert d.quarantined_batches == (1 if kind == "quarantined" else 0)


def test_index_moves_between_sets_staying_disjoint() -> None:
    d = ack_acks.outcome(
        fresh(), generation=8, snapshot_id=SNAP8, index=0, total=2, kind="quarantined", now=ack_NOW
    )
    d = ack_acks.outcome(
        d, generation=8, snapshot_id=SNAP8, index=0, total=2, kind="lost", now=ack_NOW
    )
    assert d.progress == (ack_acks.Progress(8, SNAP8, 2, lost=(0,)),)


def test_snapshot_id_mismatch_for_existing_generation_raises() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 2)
    with ack_pytest.raises(ValueError, match="mismatch"):
        ack(d, 8, SNAP9, 1, 2)
    with ack_pytest.raises(ValueError, match="mismatch"):
        ack(d, 8, SNAP8, 1, 3)


@ack_pytest.mark.parametrize(
    "bad",
    [
        {"kind": "weird"},
        {"index": 2},
        {"index": -1},
        {"total": 0},
        {"generation": 0},
        {"snapshot_id": "NOT-HEX"},
        {"rejected": -1},
        {"warning": "Has Spaces And Secrets"},
    ],
)
def test_outcome_rejects_invalid_arguments(bad: dict[str, ack_Any]) -> None:
    args: dict[str, ack_Any] = {
        "generation": 8,
        "snapshot_id": SNAP8,
        "index": 0,
        "total": 2,
        "kind": "acked",
        "now": ack_NOW,
    }
    args.update(bad)
    with ack_pytest.raises(ValueError):
        ack_acks.outcome(fresh(), **args)


def test_acked_warning_sets_sanitized_category_and_progress_is_bounded() -> None:
    d = ack(fresh(), 3, SNAP7, 0, 2, warning="receiver_warning")
    assert d.warning_category == "receiver_warning"
    d = fresh()
    for generation in range(1, 21):  # twenty never-completing generations
        d = ack(d, generation, SNAP7, 0, 2)
    assert [p.generation for p in d.progress] == list(range(5, 21))  # newest 16


def test_success_clears_block_retry_and_error_category() -> None:
    d = ack_acks.retry_scheduled(fresh(), "2026-10-13T10:01:00Z", "http_503")
    d = ack_acks.blocked(d, "unauthorized", "fp-one")
    assert (d.next_retry_utc, d.last_error_category) == ("2026-10-13T10:01:00Z", "http_503")
    assert (d.blocked_reason, d.blocked_fingerprint) == ("unauthorized", "fp-one")
    d = ack(d, 8, SNAP8, 0, 1)
    assert (d.blocked_reason, d.blocked_fingerprint, d.next_retry_utc, d.last_error_category) == (
        None, None, None, None,
    )  # fmt: skip


def test_blocked_clears_only_when_the_fingerprint_changes() -> None:
    d = ack_acks.blocked(fresh(), "credential_invalid", "fp-one")
    assert ack_acks.unblock_if_changed(d, "fp-one") == d
    cleared = ack_acks.unblock_if_changed(d, "fp-two")
    assert (cleared.blocked_reason, cleared.blocked_fingerprint) == (None, None)
    assert ack_acks.unblock_if_changed(fresh(), "anything") == fresh()


def test_blocked_reason_is_an_enumerated_code() -> None:
    for reason in ack_acks.BLOCK_REASONS:
        assert ack_acks.blocked(fresh(), reason, "fp").blocked_reason == reason
    with ack_pytest.raises(ValueError):
        ack_acks.blocked(fresh(), f"HTTP 401: {CANARY}", "fp")
    with ack_pytest.raises(ValueError):
        ack_acks.record_loss(fresh(), f"bad reason {CANARY}", 1, 1, ack_NOW)


def test_loss_is_bounded_newest_first_and_never_moves_the_ack() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 1)
    for n in range(40):
        d = ack_acks.record_loss(d, f"spool_expired_{n}", 1, 10, f"2026-10-13T10:{n:02d}:00Z")
    assert len(d.loss_summary) == 32
    assert d.loss_summary[0] == ack_acks.Loss("spool_expired_39", 1, 10, "2026-10-13T10:39:00Z")
    assert d.loss_summary[-1].reason == "spool_expired_8"
    assert d.lost_records == 400
    assert (d.last_acked_generation, d.last_acked_snapshot_id) == (8, SNAP8)


def test_release_moves_the_pin_to_expired_and_keeps_the_generation() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 1)
    released = ack_acks.release_acked_snapshot(d)
    assert released.last_acked_snapshot_id is None
    assert released.expired_snapshot_id == SNAP8
    assert released.last_acked_generation == 8
    assert ack_acks.release_acked_snapshot(fresh()) == fresh()
    renewed = ack(released, 9, SNAP9, 0, 1)
    assert (renewed.last_acked_snapshot_id, renewed.expired_snapshot_id) == (SNAP9, None)


def test_reset_generation_drops_only_that_progress_and_touches_nothing_else() -> None:
    d = ack(fresh(), 5, SNAP7, 0, 1, now=LATER)  # acknowledged generation 5
    d = ack_acks.record_loss(d, "spool_age_expired", 1, 7, ack_NOW)
    d = ack_acks.outcome(
        d, generation=8, snapshot_id=SNAP8, index=0, total=3, kind="partial",
        rejected=2, identity_unavailable=True, now=ack_NOW,
    )  # fmt: skip
    d = ack(ack(d, 9, SNAP9, 0, 2), 7, SNAP7, 0, 2)
    d = ack_replace(d, late_acks=4)
    reset = ack_acks.reset_generation(d, 8)
    assert reset == ack_replace(d, progress=tuple(p for p in d.progress if p.generation != 8))
    assert [p.generation for p in d.progress] == [7, 8, 9]
    assert [p.generation for p in reset.progress] == [7, 9]
    assert (reset.last_acked_generation, reset.last_acked_snapshot_id, reset.last_acked_at) == (
        5, SNAP7, LATER,
    )  # fmt: skip
    assert reset.late_acks == 4 and reset.loss_summary == d.loss_summary
    assert (reset.partial_batches, reset.rejected_points) == (1, 2)
    # the generation can be re-driven from scratch afterwards
    again = ack(ack(reset, 8, SNAP9, 0, 1), 8, SNAP9, 0, 1)
    assert again.last_acked_generation == 8 and again.last_acked_snapshot_id == SNAP9


def test_reset_generation_is_a_no_op_without_progress_or_for_acknowledged_generations() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 1)
    d = ack(d, 9, SNAP9, 0, 2)
    assert ack_acks.reset_generation(d, 12) == d  # no progress for that generation
    assert ack_acks.reset_generation(d, 8) == d  # the acknowledged generation is never reset
    assert ack_acks.reset_generation(d, 3) == d  # older than the acknowledged generation
    stale = ack_replace(d, progress=(*d.progress, ack_acks.Progress(7, SNAP7, 2, acked=(0,))))
    assert ack_acks.reset_generation(stale, 7) == stale  # <= last acked: untouched
    assert ack_acks.reset_generation(fresh(), 4) == fresh()
    with ack_pytest.raises(ValueError):
        ack_acks.reset_generation(d, 0)


@ack_pytest.mark.parametrize(
    "reason",
    [
        "spool_age_expired",
        "spool_size_exceeded",
        "policy_contraction",
        "reprojected_scope_changed",
        "tombstone_expired",
    ],
)
def test_loss_reason_codes_used_by_retention_are_accepted_and_persisted(reason: str) -> None:
    d = ack_acks.record_loss(fresh(), reason, 1, 2, ack_NOW)
    assert d.loss_summary == (ack_acks.Loss(reason, 1, 2, ack_NOW),)
    stored = ack_acks.decode(ack_acks.encode(d), ack_DEST, ack_Signal.METRICS)
    assert stored.loss_summary == d.loss_summary


def test_ack_vacuous_advances_exactly_like_a_full_ack() -> None:
    d = ack(fresh(), 5, SNAP7, 0, 1)
    d = ack(ack(d, 6, SNAP8, 0, 2), 9, SNAP9, 0, 2)  # progress for 6 (older) and 9 (newer)
    d = ack_acks.retry_scheduled(ack_acks.blocked(d, "forbidden", "fp"), ack_NOW, "http_503")
    d = ack_replace(d, late_acks=2, partial_batches=3, rejected_points=11)
    d = ack_acks.record_loss(d, "spool_age_expired", 1, 5, ack_NOW)
    done = ack_acks.ack_vacuous(d, 8, SNAP8, LATER)
    expected = ack_replace(
        d,
        last_acked_generation=8,
        last_acked_snapshot_id=SNAP8,
        last_acked_at=LATER,
        blocked_reason=None,
        blocked_fingerprint=None,
        next_retry_utc=None,
        last_error_category=None,
        progress=(ack_acks.Progress(9, SNAP9, 2, acked=(0,)),),  # older entry pruned, newer kept
    )
    assert done == expected
    assert (done.late_acks, done.partial_batches, done.rejected_points) == (2, 3, 11)
    assert done.loss_summary == d.loss_summary and done.lost_records == d.lost_records
    assert ack_acks.ack_vacuous(fresh(), 1, SNAP7, ack_NOW).last_acked_generation == 1


def test_ack_vacuous_accepts_a_generation_whose_progress_is_all_acked() -> None:
    d = ack_replace(fresh(), progress=(ack_acks.Progress(8, SNAP8, 2, acked=(0, 1)),))
    done = ack_acks.ack_vacuous(d, 8, SNAP8, LATER)
    assert (done.last_acked_generation, done.last_acked_at, done.progress) == (8, LATER, ())


def test_ack_vacuous_never_regresses_or_repeats() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 1, now=LATER)
    assert ack_acks.ack_vacuous(d, 8, SNAP9, ack_NOW) == d  # equal generation
    assert ack_acks.ack_vacuous(d, 7, SNAP7, ack_NOW) == d  # older generation
    with ack_pytest.raises(ValueError):
        ack_acks.ack_vacuous(d, 0, SNAP7, ack_NOW)


@ack_pytest.mark.parametrize("kind", ["partial", "quarantined", "lost", "discarded"])
def test_ack_vacuous_is_refused_for_a_generation_with_a_bad_batch(kind: str) -> None:
    d = ack_acks.outcome(
        fresh(), generation=8, snapshot_id=SNAP8, index=0, total=2, kind=kind, now=ack_NOW
    )
    d = ack(d, 8, SNAP8, 1, 2)
    assert ack_acks.ack_vacuous(d, 8, SNAP8, LATER) == d
    assert d.last_acked_generation is None


def test_ack_vacuous_is_refused_for_a_generation_with_unacked_batches() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 3)
    assert ack_acks.ack_vacuous(d, 8, SNAP8, LATER) == d
    other = ack_acks.ack_vacuous(d, 9, SNAP9, LATER)  # a different generation is unaffected
    assert (other.last_acked_generation, other.progress) == (9, ())  # gen 8 pruned as older


def test_ack_vacuous_keeps_metrics_and_logs_documents_independent() -> None:
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.LOGS, lambda d: ack(d, 7, SNAP7, 0, 1)
    )
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack(d, 8, SNAP8, 0, 2)
    )
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.LOGS, lambda d: ack_acks.ack_vacuous(d, 8, SNAP8, LATER)
    )
    metrics = ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    logs = ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.LOGS)
    assert metrics is not None and logs is not None
    assert (logs.last_acked_generation, logs.last_acked_snapshot_id) == (8, SNAP8)
    assert metrics.last_acked_generation is None  # metrics gen 8 still has index 1 outstanding
    assert metrics.progress == (ack_acks.Progress(8, SNAP8, 2, acked=(0,)),)


def test_set_counts_are_not_derived_from_outcomes() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 2)
    assert (d.pending_batches, d.unknown_batches) == (0, 0)
    d = ack_acks.set_counts(d, 5, 2)
    assert (d.pending_batches, d.unknown_batches) == (5, 2)
    with ack_pytest.raises(ValueError):
        ack_acks.set_counts(d, -1, 0)


# --- durable state ----------------------------------------------------------------------------


def test_update_creates_private_files_and_roundtrips_exactly(tmp_path: ack_Path) -> None:
    def change(d: ack_acks.Delivery) -> ack_acks.Delivery:
        d = ack_acks.collected(d, 8)
        d = ack(d, 8, SNAP8, 0, 1, now=LATER, warning="receiver_warning")
        d = ack_acks.record_loss(d, "spool_expired", 2, 9, ack_NOW)
        d = ack_acks.start_in_flight(d, "g0000000009-a9a9a9a9a9a9a9a9-0000", 9, ack_NOW)
        return ack_acks.set_counts(d, 3, 1)

    written = ack_acks.update_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS, change)
    path = doc_path(tmp_path)
    assert ack_stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ack_stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS) == written
    assert ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.LOGS) is None
    document = ack_json.loads(path.read_text())
    assert document["schema"] == 1 and document["accepted_means"] == "collector_acceptance"
    assert document["last_acked_snapshot_id"] == SNAP8 and document["signal"] == "metrics"
    assert document["loss_summary"] == [
        {"reason": "spool_expired", "batches": 2, "records": 9, "at": ack_NOW}
    ]
    assert document["in_flight"]["generation"] == 9


def test_update_roundtrips_progress_sets(tmp_path: ack_Path) -> None:
    def change(d: ack_acks.Delivery) -> ack_acks.Delivery:
        d = ack_acks.outcome(
            d, generation=8, snapshot_id=SNAP8, index=1, total=4, kind="partial",
            rejected=2, identity_unavailable=True, now=ack_NOW,
        )  # fmt: skip
        return ack(d, 8, SNAP8, 0, 4)

    written = ack_acks.update_delivery(ack_PROFILE, ack_DEST, ack_Signal.LOGS, change)
    stored = ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.LOGS)
    assert stored == written
    assert stored is not None
    assert stored.progress == (ack_acks.Progress(8, SNAP8, 4, acked=(0,), partial=(1,)),)
    assert ack_json.loads(doc_path(tmp_path, "logs").read_text())["progress"][0]["partial"] == [1]


def test_update_that_produces_an_invalid_state_writes_nothing(tmp_path: ack_Path) -> None:
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 3)
    )
    before = doc_path(tmp_path).read_bytes()
    with ack_pytest.raises(ValueError):
        ack_acks.update_delivery(
            ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_replace(d, pending_batches=-1)
        )
    with ack_pytest.raises(ValueError):
        ack_acks.update_delivery(
            ack_PROFILE,
            ack_DEST,
            ack_Signal.METRICS,
            lambda d: ack_replace(d, warning_category="Bad Text"),
        )
    assert doc_path(tmp_path).read_bytes() == before
    assert not [p for p in doc_path(tmp_path).parent.iterdir() if p.name.endswith(".tmp")]


@ack_pytest.mark.parametrize(
    ("profile", "destination"),
    [
        ("../evil", ack_DEST),
        (ack_PROFILE, "../evil"),
        (ack_PROFILE, "a/b"),
        (ack_PROFILE, ""),
        ("Bad", ack_DEST),
    ],
)
def test_ids_are_identifiers_never_paths(
    tmp_path: ack_Path, profile: str, destination: str
) -> None:
    with ack_pytest.raises(ValueError):
        ack_acks.update_delivery(profile, destination, ack_Signal.METRICS, lambda d: d)
    with ack_pytest.raises(ValueError):
        ack_acks.read_delivery(profile, destination, ack_Signal.METRICS)
    assert ack_tree(tmp_path / "data") == []


def raw_document(**overrides: object) -> bytes:
    document: dict[str, object] = {"schema": 1, "destination_id": ack_DEST, "signal": "metrics"}
    document.update(overrides)
    return ack_json.dumps(document).encode()


@ack_pytest.mark.parametrize(
    "data",
    [
        b"{not json",
        b"",
        b"[]",
        b"\xff\xfe",
        raw_document(schema=2),
        raw_document(schema=True),
        raw_document(extra_field=1),
        raw_document(late_acks=True),
        raw_document(late_acks=-1),
        raw_document(pending_batches=1.5),
        b'{"schema":1,"destination_id":"org-primary","signal":"metrics","late_acks":1.0}',
        raw_document(last_acked_generation=0),
        raw_document(last_acked_generation=10**10),
        raw_document(last_acked_snapshot_id="NOT-HEX"),
        raw_document(blocked_reason="free text from receiver"),
        raw_document(warning_category="Free Text"),
        raw_document(next_retry_utc="tomorrow"),
        raw_document(accepted_means="durable_in_backend"),
        raw_document(identity_unavailable=1),
        raw_document(destination_id="someone-else"),
        raw_document(signal="logs"),
        raw_document(loss_summary=[{"reason": "x", "batches": 1, "records": 1}]),
        raw_document(
            loss_summary=[{"reason": "ok", "batches": 1, "records": 1, "at": ack_NOW}] * 33
        ),
        raw_document(
            progress=[
                {"generation": 8, "snapshot_id": SNAP8, "total": 2, "acked": [0], "partial": [0]}
            ]
        ),
        raw_document(progress=[{"generation": 8, "snapshot_id": SNAP8, "total": 2, "acked": [5]}]),
    ],
)
def test_corrupt_documents_raise_and_are_skipped_by_listing(
    tmp_path: ack_Path, data: bytes
) -> None:
    path = doc_path(tmp_path)
    ack_os.close(ack_storage.open_profile_dir(ack_PROFILE, "delivery", ack_DEST))
    path.write_bytes(data)
    path.chmod(0o600)
    with ack_pytest.raises(ack_acks.DeliveryCorrupt):
        ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    with ack_pytest.raises(ack_acks.DeliveryCorrupt):
        ack_acks.update_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: d)
    assert path.read_bytes() == data  # a corrupt document is never silently overwritten
    assert ack_acks.list_deliveries(ack_PROFILE) == []


def test_listing_skips_corrupt_entries_but_returns_the_good_ones(tmp_path: ack_Path) -> None:
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 4)
    )
    ack_acks.update_delivery(
        ack_PROFILE, "second", ack_Signal.LOGS, lambda d: ack_acks.collected(d, 6)
    )
    doc_path(tmp_path, "logs").parent.mkdir(exist_ok=True)
    doc_path(tmp_path, "logs").write_text("{garbage")
    doc_path(tmp_path, "logs").chmod(0o600)
    (doc_path(tmp_path).parent.parent / "stray.json").write_text("{}")
    found = ack_acks.list_deliveries(ack_PROFILE)
    assert sorted((d.destination_id, d.signal.value) for d in found) == [
        (ack_DEST, "metrics"),
        ("second", "logs"),
    ]


def test_symlinked_destination_directory_is_refused(tmp_path: ack_Path) -> None:
    ack_delivery = tmp_path / "data" / "accounting" / ack_PROFILE / "delivery"
    ack_os.close(ack_storage.open_profile_dir(ack_PROFILE, "delivery"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (elsewhere / "metrics.json").write_bytes(raw_document(late_acks=3))
    (elsewhere / "metrics.json").chmod(0o600)
    (ack_delivery / ack_DEST).symlink_to(elsewhere)
    with ack_pytest.raises(ack_acks.DeliveryCorrupt):
        ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    with ack_pytest.raises(ack_acks.DeliveryCorrupt):
        ack_acks.update_delivery(
            ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 1)
        )
    assert sorted(p.name for p in elsewhere.iterdir()) == ["metrics.json"]
    assert ack_acks.list_deliveries(ack_PROFILE) == []


def test_symlinked_document_is_refused_and_never_followed(tmp_path: ack_Path) -> None:
    path = doc_path(tmp_path)
    ack_os.close(ack_storage.open_profile_dir(ack_PROFILE, "delivery", ack_DEST))
    target = tmp_path / "target.json"
    target.write_bytes(raw_document(late_acks=3))
    target.chmod(0o600)
    path.symlink_to(target)
    with ack_pytest.raises(ack_acks.DeliveryCorrupt):
        ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    assert ack_acks.list_deliveries(ack_PROFILE) == []


def test_reads_never_create_anything(tmp_path: ack_Path) -> None:
    assert ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS) is None
    assert ack_acks.list_deliveries(ack_PROFILE) == []
    assert (
        ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS).state
        == "not_acknowledged"
    )
    assert ack_acks.export_health(ack_PROFILE) == []
    assert ack_tree(tmp_path / "data") == []
    ack_os.close(ack_storage.open_profile_dir(ack_PROFILE))
    before = ack_tree(tmp_path / "data")
    ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.LOGS)
    ack_acks.list_deliveries(ack_PROFILE)
    ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.LOGS)
    assert ack_tree(tmp_path / "data") == before


def _increment(profile: str, destination: str, count: int, ready: ack_Any) -> None:
    ready.wait(30)
    for _ in range(count):
        ack_acks.update_delivery(
            profile,
            destination,
            ack_Signal.METRICS,
            lambda d: ack_replace(d, late_acks=d.late_acks + 1),
        )


def ack_test_two_processes_updating_concurrently_lose_no_updates(tmp_path: ack_Path) -> None:
    context = ack_multiprocessing.get_context("spawn")
    ready = context.Event()
    workers = [
        context.Process(target=_increment, args=(ack_PROFILE, ack_DEST, 25, ready))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    ready.set()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    state = ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    assert state is not None and state.late_acks == 50
    assert ack_json.loads(doc_path(tmp_path).read_text())["late_acks"] == 50
    assert ack_os.path.exists(doc_path(tmp_path).parent / ".metrics.lock")


# --- independence of signals (spec scenario) --------------------------------------------------


def test_metrics_ack_does_not_move_logs_ack() -> None:
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.LOGS, lambda d: ack(d, 7, SNAP7, 0, 1)
    )
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.LOGS, lambda d: ack(d, 8, SNAP8, 0, 2)
    )  # index 1 pending
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack(d, 8, SNAP8, 0, 1)
    )
    metrics = ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    logs = ack_acks.read_delivery(ack_PROFILE, ack_DEST, ack_Signal.LOGS)
    assert metrics is not None and logs is not None
    assert (metrics.last_acked_generation, metrics.last_acked_snapshot_id) == (8, SNAP8)
    assert (logs.last_acked_generation, logs.last_acked_snapshot_id) == (7, SNAP7)
    assert logs.progress == (ack_acks.Progress(8, SNAP8, 2, acked=(0,)),)


# --- acknowledged resolution against real snapshot files ---------------------------------------


def acknowledge_real(generation: int, signal: ack_Signal = ack_Signal.METRICS) -> str:
    snapshot_id = make_envelope(generation).snapshot_id
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, signal, lambda d: ack(d, generation, snapshot_id, 0, 1)
    )
    return snapshot_id


def test_resolve_returns_the_exact_older_acknowledged_snapshot_not_current(
    tmp_path: ack_Path,
) -> None:
    older, newer = make_envelope(7), make_envelope(8)
    ack_snapshots.publish(ack_PROFILE, older)
    ack_snapshots.publish(ack_PROFILE, newer)
    acknowledge_real(7)
    result = ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    assert result.state == "acknowledged"
    assert result.envelope == older and result.envelope is not None
    assert result.envelope.snapshot_id == older.snapshot_id != newer.snapshot_id
    assert result.envelope.generation == 7
    assert result.delivery is not None and result.delivery.last_acked_generation == 7
    assert (
        ack_snapshots.current_generation(ack_PROFILE) == 8
    )  # current is newer, and was NOT returned
    assert (
        ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.LOGS).state
        == "not_acknowledged"
    )


def test_resolve_without_an_acknowledgment_never_falls_back_to_current() -> None:
    ack_snapshots.publish(ack_PROFILE, make_envelope(3))
    absent = ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    assert (absent.state, absent.delivery, absent.envelope) == ("not_acknowledged", None, None)
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack(d, 3, SNAP7, 0, 2)
    )
    pending = ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    assert pending.state == "not_acknowledged" and pending.envelope is None
    assert pending.delivery is not None and pending.delivery.last_acked_snapshot_id is None


def test_resolve_reports_expired_when_the_snapshot_file_is_gone(tmp_path: ack_Path) -> None:
    ack_snapshots.publish(ack_PROFILE, make_envelope(7))
    ack_snapshots.publish(ack_PROFILE, make_envelope(8))
    snapshot_id = acknowledge_real(7)
    gen_dir = tmp_path / "data" / "accounting" / ack_PROFILE / "snapshots"
    (gen_dir / f"gen-0000000007-{snapshot_id}.json").unlink()
    result = ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    assert (result.state, result.envelope) == ("acknowledged_snapshot_expired", None)
    assert result.delivery is not None and result.delivery.last_acked_generation == 7


def test_resolve_reports_expired_for_a_corrupt_snapshot_and_for_a_released_pin(
    tmp_path: ack_Path,
) -> None:
    ack_snapshots.publish(ack_PROFILE, make_envelope(7))
    ack_snapshots.publish(ack_PROFILE, make_envelope(8))
    snapshot_id = acknowledge_real(7)
    path = tmp_path / "data" / "accounting" / ack_PROFILE / "snapshots"
    (path / f"gen-0000000007-{snapshot_id}.json").write_bytes(b"{torn")
    corrupt = ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    assert (corrupt.state, corrupt.envelope) == ("acknowledged_snapshot_expired", None)
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, ack_acks.release_acked_snapshot
    )
    released = ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS)
    assert (released.state, released.envelope) == ("acknowledged_snapshot_expired", None)
    assert released.delivery is not None and released.delivery.expired_snapshot_id == snapshot_id


def test_pin_is_visible_and_protects_the_acknowledged_snapshot_from_prune(
    tmp_path: ack_Path,
) -> None:
    for generation in range(1, 6):
        ack_snapshots.publish(ack_PROFILE, make_envelope(generation))
    pinned_id = acknowledge_real(2)
    assert ack_snapshots.pinned_snapshot_ids(ack_PROFILE) == frozenset({pinned_id})
    assert ack_snapshots.prune(ack_PROFILE, keep=1) == [1, 3, 4]  # 2 survives: pinned; 5 is current
    assert ack_snapshots.generations(ack_PROFILE) == [2, 5]
    assert (
        ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS).state
        == "acknowledged"
    )
    # negative twin: after the pin is released the same prune deletes it
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, ack_acks.release_acked_snapshot
    )
    assert ack_snapshots.pinned_snapshot_ids(ack_PROFILE) == frozenset()
    assert ack_snapshots.prune(ack_PROFILE, keep=1) == [2]
    assert (
        ack_acks.resolve_acknowledged(ack_PROFILE, ack_DEST, ack_Signal.METRICS).state
        == "acknowledged_snapshot_expired"
    )


def test_lock_files_do_not_confuse_the_pin_scan(tmp_path: ack_Path) -> None:
    for generation in (1, 2, 3):
        ack_snapshots.publish(ack_PROFILE, make_envelope(generation))
    acknowledge_real(1)
    assert (doc_path(tmp_path).parent / ".metrics.lock").exists()
    assert ack_snapshots.pinned_snapshot_ids(ack_PROFILE) == {make_envelope(1).snapshot_id}


# --- health and status views -------------------------------------------------------------------


def test_export_health_reports_epoch_counts_and_aggregated_losses() -> None:
    def change(d: ack_acks.Delivery) -> ack_acks.Delivery:
        d = ack(d, 8, SNAP8, 0, 1, now="2026-10-13T10:00:00Z")
        d = ack_acks.outcome(
            d, generation=9, snapshot_id=SNAP9, index=0, total=2, kind="partial",
            rejected=6, identity_unavailable=True, now=ack_NOW,
        )  # fmt: skip
        d = ack_acks.record_loss(d, "spool_expired", 1, 4, ack_NOW)
        d = ack_acks.record_loss(d, "spool_expired", 2, 5, ack_NOW)
        d = ack_acks.record_loss(d, "tombstone_unsent", 1, 1, ack_NOW)
        return ack_acks.set_counts(d, 2, 0)

    ack_acks.update_delivery(ack_PROFILE, ack_DEST, ack_Signal.LOGS, change)
    ack_acks.update_delivery(
        ack_PROFILE, "other", ack_Signal.METRICS, lambda d: ack_acks.collected(d, 1)
    )
    health = {(h.destination, h.signal): h for h in ack_acks.export_health(ack_PROFILE)}
    logs = health[(ack_DEST, "logs")]
    assert logs.last_full_ack_epoch == 1_791_885_600  # 2026-10-13T10:00:00Z
    assert (logs.pending_batches, logs.rejected_points) == (2, 6)
    assert logs.lost == (("spool_expired", 9), ("tombstone_unsent", 1))
    other = health[("other", "metrics")]
    assert (other.last_full_ack_epoch, other.pending_batches, other.lost) == (None, 0, ())
    assert isinstance(logs.signal, str) and not isinstance(logs.signal, ack_Signal)


def test_describe_states_and_sanitization() -> None:
    assert ack_acks.describe(fresh())["state"] == "idle"
    d = ack(ack_acks.collected(fresh(), 8), 8, SNAP8, 0, 1)
    assert ack_acks.describe(d)["state"] == "acknowledged"
    assert ack_acks.describe(ack_acks.collected(d, 9))["state"] == "pending"
    assert ack_acks.describe(ack_acks.set_counts(d, 1, 0))["state"] == "pending"
    assert ack_acks.describe(ack_acks.set_counts(d, 0, 2))["state"] == "unknown"
    partial = ack_acks.outcome(
        d, generation=9, snapshot_id=SNAP9, index=0, total=2, kind="partial",
        rejected=3, identity_unavailable=True, now=ack_NOW,
    )  # fmt: skip
    assert ack_acks.describe(partial)["state"] == "partial"
    blocked = ack_acks.blocked(partial, "forbidden", "fp")
    view = ack_acks.describe(blocked)
    assert view["state"] == "blocked" and view["blocked_reason"] == "forbidden"
    assert view["accepted_means"] == "collector_acceptance"
    assert view["last_acked_generation"] == 8 and view["identity_unavailable"] is True
    assert "blocked_fingerprint" not in view
    ack_json.dumps(view)  # JSON-safe
    listing = ack_acks.describe(ack_acks.record_loss(blocked, "spool_expired", 1, 3, ack_NOW))
    assert listing["loss_summary"] == [
        {"reason": "spool_expired", "batches": 1, "records": 3, "at": ack_NOW}
    ]


def test_canary_error_message_never_reaches_any_state_or_view(tmp_path: ack_Path) -> None:
    body = ack_json.dumps(
        {"partialSuccess": {"rejectedLogRecords": 2, "errorMessage": f"bad token {CANARY}"}}
    ).encode()
    result = ack_acks.parse_response(ack_Signal.LOGS, body)

    def change(d: ack_acks.Delivery) -> ack_acks.Delivery:
        return ack_acks.outcome(
            d, generation=8, snapshot_id=SNAP8, index=0, total=1, kind="partial",
            rejected=result.rejected, identity_unavailable=result.identity_unavailable,
            warning=result.warning_category, now=ack_NOW,
        )  # fmt: skip

    state = ack_acks.update_delivery(ack_PROFILE, ack_DEST, ack_Signal.LOGS, change)
    assert ack_CANARY not in doc_path(tmp_path, "logs").read_text()
    assert ack_CANARY not in ack_json.dumps(ack_acks.describe(state))
    assert state.rejected_points == 2 and state.identity_unavailable is True


# --- damaged documents, unsafe paths and ack/meta settlement races --------------------------


@ack_pytest.mark.parametrize("damage", ["nested", "duplicate-generation", "descending-generation"])
def test_invalid_ack_document_is_not_rewritten_and_restoration_retains_progress(
    damage: str,
) -> None:
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 3)
    )
    path = ack_fx.profile_root() / "delivery" / ack_DEST / "metrics.json"
    original = path.read_bytes()
    if damage == "nested":
        damaged = b"[" * 2000 + b"0" + b"]" * 2000
    else:
        document = ack_json.loads(original)
        generations = [2, 2] if damage == "duplicate-generation" else [3, 2]
        document["progress"] = [
            {"generation": g, "snapshot_id": f"{g:032x}", "total": 1, "acked": []}
            for g in generations
        ]
        damaged = ack_json.dumps(document).encode()
    path.write_bytes(damaged)
    with ack_pytest.raises(ack_acks.DeliveryCorrupt):
        ack_acks.update_delivery(
            ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 4)
        )
    assert path.read_bytes() == damaged
    assert ack_acks.list_deliveries(ack_PROFILE) == []
    path.write_bytes(original)
    recovered = ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 4)
    )
    assert recovered.last_collected_generation == 4
    assert ack_acks.list_deliveries(ack_PROFILE) == [recovered]


def test_ack_update_refuses_symlink_without_touching_target_then_recovers(
    tmp_path: ack_Path,
) -> None:
    ack_acks.update_delivery(
        ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 1)
    )
    path = ack_fx.profile_root() / "delivery" / ack_DEST / "metrics.json"
    target = tmp_path / "saved-delivery"
    path.rename(target)
    original = target.read_bytes()
    path.symlink_to(target)
    with ack_pytest.raises(ack_acks.DeliveryCorrupt):
        ack_acks.update_delivery(
            ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 2)
        )
    assert target.read_bytes() == original and path.is_symlink()
    path.unlink()
    target.rename(path)
    assert (
        ack_acks.update_delivery(
            ack_PROFILE, ack_DEST, ack_Signal.METRICS, lambda d: ack_acks.collected(d, 2)
        ).last_collected_generation
        == 2
    )


def test_ack_committed_before_meta_settlement_is_not_counted_as_delivery_loss() -> None:
    stale = ack_put(3)
    original = seed_delivery(
        lambda d: ack_acks.outcome(
            d,
            generation=stale.generation,
            snapshot_id=stale.snapshot_id,
            index=0,
            total=1,
            kind="acked",
            now=ack_NOW_Z,
        )
    )
    queued = ack_export_spool.read_meta(ack_PROFILE, ack_DEST, ack_Signal.METRICS, stale.batch_id)
    assert queued is not None and queued.state == "queued"
    assert purge_spool.apply_spool(ack_PROFILE, purge_spool.SpoolPlan((stale,)), ack_NOW_Z) == (
        1,
        0,
    )
    assert ack_export_spool.list_batches(ack_PROFILE, ack_DEST, ack_Signal.METRICS) == []
    after = ack_delivery()
    assert (after.last_acked_generation, after.last_acked_snapshot_id, after.last_acked_at) == (
        original.last_acked_generation,
        original.last_acked_snapshot_id,
        original.last_acked_at,
    )
    assert after.lost_records == 0 and after.loss_summary == ()


# ---- Consolidated from test_export_spool.py (source-specific globals are prefixed) ----


spool_PROFILE = "personal-local"
spool_DEST = "org-primary"
spool_NOW_Z = "2026-10-13T12:00:00Z"
SCOPE: dict[str, object] = {
    "roots": ["a", "b"],
    "endpoint": "https://collector.example/otlp",
    "fields_version": "logs-v1",
    "policy_version": 3,
    "logs_permitted": True,
    "identity": None,
}
spool_M, spool_L = spool_Signal.METRICS, spool_Signal.LOGS


def snap(gen: int) -> str:
    return f"{gen:04x}" * 8


def make(
    gen: int = 1,
    index: int = 0,
    *,
    total: int = 1,
    signal: spool_Signal = spool_M,
    kind: str = "data",
    body: bytes | None = None,
    items: int = 2,
    batch_id: str | None = None,
) -> Batch:
    if batch_id is None:
        batch_id = (
            f"t1760000000-ab12cd34-{index:04d}"
            if kind == "tombstone"
            else f"g{gen:010d}-{snap(gen)[:16]}-{index:04d}"
        )
    payload = (
        body
        if body is not None
        else spool_json.dumps({"n": gen * 100 + index, "pad": "x" * 40}).encode()
    )
    ids = tuple(f"{gen:08x}{index:04x}{i:020x}" for i in range(items)) if signal is spool_L else ()
    return Batch(batch_id, signal, gen, snap(gen), index, total, kind, payload, items, ids)


def spool_put(
    batch: Batch, *, dest: str = spool_DEST, scope: dict[str, object] | None = None
) -> Meta:
    return spool.write_batch(
        spool_PROFILE,
        dest,
        batch.signal,
        batch,
        created_utc=spool_NOW_Z,
        scope=SCOPE if scope is None else scope,
    )


def folder(
    tmp_path: spool_Path, signal: spool_Signal = spool_M, dest: str = spool_DEST
) -> spool_Path:
    return tmp_path / "data" / "accounting" / spool_PROFILE / "spool" / dest / signal.value


def spool_tree(root: spool_Path) -> dict[str, tuple[int, int, int]]:
    """Every entry under ``root`` as (mode, size, mtime_ns); empty when ``root`` is absent."""
    if not root.exists():
        return {}
    found = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        found[str(path.relative_to(root))] = (info.st_mode, info.st_size, info.st_mtime_ns)
    return found


def names(directory: spool_Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name != ".spool.lock")


def ids(signal: spool_Signal = spool_M) -> list[str]:
    return [m.batch_id for m in spool.list_batches(spool_PROFILE, spool_DEST, signal)]


# --- write: exact bytes, idempotency, conflict, validation -----------------------------------


def test_write_batch_persists_exact_body_and_meta_derived_from_the_bytes(
    tmp_path: spool_Path,
) -> None:
    batch = make(7, 2, total=3, signal=spool_L, items=2)
    meta = spool_put(batch)
    body = folder(tmp_path, spool_L) / f"{batch.batch_id}.json"
    assert body.read_bytes() == batch.body
    expected = Meta(
        batch_id=batch.batch_id,
        destination_id=spool_DEST,
        signal=spool_L,
        snapshot_id=snap(7),
        generation=7,
        index=2,
        total=3,
        kind="data",
        created_utc=spool_NOW_Z,
        bytes=len(batch.body),
        sha256=hashlib.sha256(batch.body).hexdigest(),
        items=2,
        record_ids=batch.record_ids,
        scope=SCOPE,
    )
    assert meta == expected
    assert spool.read_meta(spool_PROFILE, spool_DEST, spool_L, batch.batch_id) == expected
    on_disk = spool_json.loads(
        (folder(tmp_path, spool_L) / f"{batch.batch_id}.meta.json").read_text()
    )
    assert on_disk["schema"] == 1 and on_disk["state"] == "queued" and on_disk["attempts"] == 0
    assert on_disk["sha256"] == hashlib.sha256(batch.body).hexdigest()
    assert on_disk["record_ids"] == list(batch.record_ids)
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_L, batch.batch_id) == batch.body


def test_identical_rewrite_returns_the_existing_meta_without_resetting_it(
    tmp_path: spool_Path,
) -> None:
    batch = make()
    spool_put(batch)
    advanced = spool.update_meta(
        spool_PROFILE,
        spool_DEST,
        spool_M,
        batch.batch_id,
        lambda m: spool_replace(m, state="acked", attempts=3),
    )
    meta_file = folder(tmp_path) / f"{batch.batch_id}.meta.json"
    before = meta_file.read_bytes(), spool_tree(folder(tmp_path))
    again = spool.write_batch(
        spool_PROFILE,
        spool_DEST,
        spool_M,
        batch,
        created_utc="2027-01-01T00:00:00Z",
        scope={"other": 1},
    )
    assert again == advanced
    assert (again.created_utc, again.state, again.attempts) == (spool_NOW_Z, "acked", 3)
    assert (meta_file.read_bytes(), spool_tree(folder(tmp_path))) == before


def test_different_bytes_for_the_same_id_conflict_and_change_nothing(tmp_path: spool_Path) -> None:
    batch = make()
    spool_put(batch)
    before = spool_tree(folder(tmp_path))
    changed = spool_replace(batch, body=b'{"n":999}')
    with spool_pytest.raises(BatchConflict):
        spool_put(changed)
    assert spool_tree(folder(tmp_path)) == before
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body


def test_orphan_body_with_different_bytes_conflicts_and_is_left_for_the_sweep(
    tmp_path: spool_Path,
) -> None:
    batch = make()
    directory = folder(tmp_path)
    directory.parent.mkdir(parents=True, mode=0o700)
    directory.mkdir(mode=0o700)
    orphan = directory / f"{batch.batch_id}.json"
    orphan.write_bytes(b'{"older":true}')
    orphan.chmod(0o600)
    with spool_pytest.raises(BatchConflict):
        spool_put(batch)
    assert orphan.read_bytes() == b'{"older":true}' and ids() == []
    assert spool.sweep_orphans(spool_PROFILE, spool_DEST, spool_M) == 1
    spool_put(batch)
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body


@spool_pytest.mark.parametrize(
    "bad",
    [
        spool_pytest.param(make(body=b"x" * (MAX_BATCH_BYTES + 1)), id="too-large"),
        spool_pytest.param(make(body=b""), id="empty"),
        spool_pytest.param(make(body=b"not json"), id="not-json"),
        spool_pytest.param(make(body=b"[1,2]"), id="json-array"),
        spool_pytest.param(make(body=b'"text"'), id="json-string"),
        spool_pytest.param(make(index=1, total=1), id="index-not-below-total"),
        spool_pytest.param(make(kind="weird"), id="bad-kind"),
        spool_pytest.param(make(kind="tombstone"), id="tombstone-for-metrics"),
        spool_pytest.param(make(batch_id="../evil-0000"), id="path-id"),
        spool_pytest.param(make(batch_id="g1-abc-0"), id="short-id"),
        spool_pytest.param(make(batch_id="G0000000001-0001000100010001-0000"), id="uppercase-id"),
        spool_pytest.param(make(signal=spool_L), id="signal-mismatch"),
    ],
)
def test_invalid_batches_are_rejected_and_nothing_is_written(
    tmp_path: spool_Path, bad: Batch
) -> None:
    with spool_pytest.raises(ValueError):
        spool.write_batch(
            spool_PROFILE, spool_DEST, spool_M, bad, created_utc=spool_NOW_Z, scope=SCOPE
        )
    assert spool_tree(tmp_path / "data") == {}


@spool_pytest.mark.parametrize(
    "scope",
    [
        {"ratio": 0.5},
        {"nested": [{"deep": [1.0]}]},
        {"blob": "x" * (16 * 1024)},
        {"obj": object()},
        {1: "int key"},
    ],
    ids=["float", "nested-float", "over-16KiB", "non-json", "non-str-key"],
)
def test_invalid_scope_is_rejected_before_anything_is_written(
    tmp_path: spool_Path, scope: dict[spool_Any, spool_Any]
) -> None:
    with spool_pytest.raises(ValueError):
        spool_put(make(), scope=scope)
    assert spool_tree(tmp_path / "data") == {}


def test_scope_is_capped_at_exactly_16_kib_canonical_bytes_and_is_copied(
    tmp_path: spool_Path,
) -> None:
    def canonical(blob: str) -> int:
        return len(
            spool_json.dumps(
                {"blob": blob, "n": [1, None, True]}, separators=(",", ":"), sort_keys=True
            )
        )

    fill = 16 * 1024 - canonical("")
    scope: dict[str, object] = {"blob": "y" * fill, "n": [1, None, True]}
    assert canonical("y" * fill) == 16 * 1024
    meta = spool_put(make(), scope=scope)
    scope["blob"] = "mutated"
    assert meta.scope["blob"] == "y" * fill
    assert spool.read_meta(spool_PROFILE, spool_DEST, spool_M, make().batch_id) == meta
    with spool_pytest.raises(ValueError):
        spool_put(make(2), scope={"blob": "y" * (fill + 1), "n": [1, None, True]})
    assert ids() == [make().batch_id]


@spool_pytest.mark.parametrize(
    ("profile", "destination"),
    [
        ("../x", spool_DEST),
        (spool_PROFILE, "../x"),
        (spool_PROFILE, "UPPER"),
        ("a/b", spool_DEST),
        (spool_PROFILE, ""),
    ],
)
def test_profile_and_destination_are_identifiers_never_paths(
    tmp_path: spool_Path, profile: str, destination: str
) -> None:
    with spool_pytest.raises(ValueError):
        spool.write_batch(
            profile, destination, spool_M, make(), created_utc=spool_NOW_Z, scope=SCOPE
        )
    with spool_pytest.raises(ValueError):
        spool.list_batches(profile, destination, spool_M)
    with spool_pytest.raises(ValueError):
        spool.usage(profile, destination)
    assert spool_tree(tmp_path / "data") == {}


@spool_pytest.mark.parametrize(
    "batch_id", ["../../victim", "g1", "x" * 64, "g0000000001-zz-0000", ""]
)
def test_only_well_formed_batch_ids_ever_become_filenames(
    tmp_path: spool_Path, batch_id: str
) -> None:
    sentinel = tmp_path / "data" / "accounting" / spool_PROFILE / "spool" / "victim.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"keep")
    spool_put(make())
    for call in (spool.read_meta, spool.read_body, spool.delete_batch):
        with spool_pytest.raises(ValueError):
            call(spool_PROFILE, spool_DEST, spool_M, batch_id)
    with spool_pytest.raises(ValueError):
        spool.update_meta(spool_PROFILE, spool_DEST, spool_M, batch_id, lambda m: m)
    assert sentinel.read_bytes() == b"keep"


# --- crash safety and durability ordering -----------------------------------------------------


def test_crash_between_body_and_meta_leaves_an_orphan_that_the_sweep_removes(
    tmp_path: spool_Path, monkeypatch: spool_pytest.MonkeyPatch
) -> None:
    batch = make()
    real = spool_storage.atomic_write

    def crashing(dir_fd: int, name: str, data: bytes) -> None:
        if name.endswith(".meta.json"):
            raise OSError("power cut")
        real(dir_fd, name, data)

    monkeypatch.setattr(spool_storage, "atomic_write", crashing)
    with spool_pytest.raises(OSError, match="power cut"):
        spool_put(batch)
    monkeypatch.setattr(spool_storage, "atomic_write", real)
    assert names(folder(tmp_path)) == [f"{batch.batch_id}.json"]
    assert (
        ids() == [] and spool.read_meta(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) is None
    )
    assert spool.sweep_orphans(spool_PROFILE, spool_DEST, spool_M) == 1
    assert names(folder(tmp_path)) == []
    meta = spool_put(batch)
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body
    assert ids() == [meta.batch_id]
    assert (
        spool.sweep_orphans(spool_PROFILE, spool_DEST, spool_M) == 0
    )  # positive twin: a healthy batch survives


def test_orphan_body_with_identical_bytes_is_adopted_by_a_retry_without_a_sweep(
    tmp_path: spool_Path, monkeypatch: spool_pytest.MonkeyPatch
) -> None:
    batch = make()
    real = spool_storage.atomic_write
    monkeypatch.setattr(
        spool_storage,
        "atomic_write",
        lambda *a: (_ for _ in ()).throw(OSError("power cut")),
    )
    with spool_pytest.raises(OSError):
        spool_put(batch)
    monkeypatch.setattr(spool_storage, "atomic_write", real)
    spool_put(batch)
    assert ids() == [batch.batch_id]
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body


def test_failure_while_writing_the_body_leaves_no_final_named_file(
    tmp_path: spool_Path, monkeypatch: spool_pytest.MonkeyPatch
) -> None:
    batch = make()
    real = spool_storage.write_all

    def failing(descriptor: int, data: bytes) -> None:
        real(descriptor, data[:5])
        raise OSError(28, "disk full")

    monkeypatch.setattr(spool_storage, "write_all", failing)
    with spool_pytest.raises(OSError, match="disk full"):
        spool_put(batch)
    monkeypatch.setattr(spool_storage, "write_all", real)
    assert names(folder(tmp_path)) == []  # no body, no meta, no leaked temporary
    assert ids() == [] and spool.corrupt_batches(spool_PROFILE, spool_DEST, spool_M) == []
    spool_put(batch)
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body


def test_body_is_fsynced_and_linked_and_its_directory_fsynced_before_the_meta_is_written(
    tmp_path: spool_Path, monkeypatch: spool_pytest.MonkeyPatch
) -> None:
    batch = make()
    events: list[tuple[spool_Any, ...]] = []
    real_fsync, real_link, real_rename = spool_os.fsync, spool_os.link, spool_os.rename

    def fsync(fd: int) -> None:
        info = spool_os.fstat(fd)
        events.append(
            ("fsync-dir",) if spool_stat.S_ISDIR(info.st_mode) else ("fsync-file", info.st_size)
        )
        real_fsync(fd)

    def link(src: spool_Any, dst: spool_Any, **kwargs: spool_Any) -> None:
        events.append(("link", dst))
        real_link(src, dst, **kwargs)

    def rename(src: spool_Any, dst: spool_Any, **kwargs: spool_Any) -> None:
        events.append(("rename", dst))
        real_rename(src, dst, **kwargs)

    monkeypatch.setattr(spool_os, "fsync", fsync)
    monkeypatch.setattr(spool_os, "link", link)
    monkeypatch.setattr(spool_os, "rename", rename)
    spool_put(batch)
    meta_name = f"{batch.batch_id}.meta.json"
    meta_size = (folder(tmp_path) / meta_name).stat().st_size
    assert events == [
        ("fsync-file", len(batch.body)),  # temporary body durable ...
        ("link", f"{batch.batch_id}.json"),  # ... then published under its final name ...
        ("fsync-dir",),  # ... and the directory entry durable ...
        ("fsync-file", meta_size),  # ... BEFORE the meta even starts
        ("rename", meta_name),
        ("fsync-dir",),
    ]


# --- integrity --------------------------------------------------------------------------------


def test_read_body_returns_the_exact_bytes_and_refuses_a_flipped_byte(tmp_path: spool_Path) -> None:
    batch = make(body=b'{"resourceMetrics":[{"v":"abcdef"}]}')
    spool_put(batch)
    path = folder(tmp_path) / f"{batch.batch_id}.json"
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body
    flipped = bytearray(batch.body)
    flipped[-5] ^= 0x01  # same size, different content
    path.write_bytes(bytes(flipped))
    with spool_pytest.raises(SpoolCorrupt):
        spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    path.write_bytes(batch.body)  # restored: readable again (the meta was never the problem)
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body


@spool_pytest.mark.parametrize("damage", ["truncate", "append", "empty", "delete"])
def test_a_damaged_body_is_never_returned_and_is_reported_corrupt(
    tmp_path: spool_Path, damage: str
) -> None:
    batch = make()
    spool_put(batch)
    healthy = make(2)
    spool_put(healthy)
    path = folder(tmp_path) / f"{batch.batch_id}.json"
    if damage == "truncate":
        path.write_bytes(batch.body[:-3])
    elif damage == "append":
        path.write_bytes(batch.body + b" ")
    elif damage == "empty":
        path.write_bytes(b"")
    else:
        path.unlink()
    with spool_pytest.raises(SpoolCorrupt):
        spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    assert spool.corrupt_batches(spool_PROFILE, spool_DEST, spool_M) == [batch.batch_id]
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, healthy.batch_id) == healthy.body


def test_read_body_of_an_unknown_batch_is_file_not_found_not_corruption() -> None:
    with spool_pytest.raises(FileNotFoundError):
        spool.read_body(spool_PROFILE, spool_DEST, spool_M, make().batch_id)
    spool_put(make(2))
    with spool_pytest.raises(FileNotFoundError):
        spool.read_body(spool_PROFILE, spool_DEST, spool_M, make().batch_id)


def test_symlinked_body_is_refused_and_its_target_is_never_read_or_removed(
    tmp_path: spool_Path,
) -> None:
    batch = make()
    spool_put(batch)
    target = tmp_path / "outside.json"
    target.write_bytes(batch.body)  # identical bytes: only the symlink itself can betray it
    path = folder(tmp_path) / f"{batch.batch_id}.json"
    path.unlink()
    path.symlink_to(target)
    with spool_pytest.raises(SpoolCorrupt):
        spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    assert spool.corrupt_batches(spool_PROFILE, spool_DEST, spool_M) == [batch.batch_id]
    with spool_pytest.raises(SpoolCorrupt):
        spool.delete_batch(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    with spool_pytest.raises(SpoolCorrupt):
        spool_put(batch)
    assert target.read_bytes() == batch.body and path.is_symlink()


def test_symlinked_spool_directory_is_refused_and_nothing_outside_is_touched(
    tmp_path: spool_Path,
) -> None:
    batch = make()
    spool_put(batch)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / f"{batch.batch_id}.json"
    victim_meta = outside / f"{batch.batch_id}.meta.json"
    stray = outside / ".g.tmp"
    victim.write_bytes(b'{"victim":true}')
    victim_meta.write_bytes(b"{}")
    stray.write_bytes(b"x")
    real = folder(tmp_path)
    moved = tmp_path / "moved-away"
    real.rename(moved)
    real.symlink_to(outside, target_is_directory=True)
    before = spool_tree(outside)
    calls: list[Callable[[], object]] = [
        lambda: spool_put(make(2)),
        lambda: spool.list_batches(spool_PROFILE, spool_DEST, spool_M),
        lambda: spool.read_meta(spool_PROFILE, spool_DEST, spool_M, batch.batch_id),
        lambda: spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id),
        lambda: spool.delete_batch(spool_PROFILE, spool_DEST, spool_M, batch.batch_id),
        lambda: spool.sweep_orphans(spool_PROFILE, spool_DEST, spool_M),
        lambda: spool.drop_destination(spool_PROFILE, spool_DEST, spool_M),
        lambda: spool.update_meta(spool_PROFILE, spool_DEST, spool_M, batch.batch_id, lambda m: m),
        lambda: spool.corrupt_batches(spool_PROFILE, spool_DEST, spool_M),
    ]
    for call in calls:
        with spool_pytest.raises(SpoolCorrupt):
            call()
    assert spool_tree(outside) == before
    assert victim.read_bytes() == b'{"victim":true}' and victim_meta.read_bytes() == b"{}"
    assert (moved / f"{batch.batch_id}.json").read_bytes() == batch.body  # not swept through it
    assert spool.iter_spools(spool_PROFILE) == []  # a symlinked spool is not a spool


# --- update_meta ------------------------------------------------------------------------------


def _bump(profile: str, destination: str, batch_id: str, count: int, ready: spool_Any) -> None:
    ready.wait(30)
    for _ in range(count):
        spool.update_meta(
            profile,
            destination,
            spool_M,
            batch_id,
            lambda m: spool_replace(m, attempts=m.attempts + 1),
        )


def spool_test_two_processes_updating_concurrently_lose_no_updates(tmp_path: spool_Path) -> None:
    batch = make()
    spool_put(batch)
    context = spool_multiprocessing.get_context("spawn")
    ready = context.Event()
    workers = [
        context.Process(target=_bump, args=(spool_PROFILE, spool_DEST, batch.batch_id, 25, ready))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    ready.set()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    meta = spool.read_meta(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    assert meta is not None and meta.attempts == 50
    on_disk = spool_json.loads((folder(tmp_path) / f"{batch.batch_id}.meta.json").read_text())
    assert on_disk["attempts"] == 50 and on_disk["state"] == "queued"
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body


def test_mutable_progress_fields_update_and_persist() -> None:
    batch = make()
    spool_put(batch)
    updated = spool.update_meta(
        spool_PROFILE,
        spool_DEST,
        spool_M,
        batch.batch_id,
        lambda m: spool_replace(
            m,
            state="partial",
            attempts=2,
            last_attempt_utc="2026-10-13T12:01:00Z",
            next_attempt_utc="2026-10-13T12:06:00Z",
            last_status=429,
            last_category="rate_limited",
            rejected=3,
            warning="collector rejected 3 points",
        ),
    )
    assert spool.read_meta(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == updated
    assert (updated.state, updated.attempts, updated.last_status, updated.rejected) == (
        "partial",
        2,
        429,
        3,
    )
    assert updated.warning == "collector rejected 3 points"


IDENTITY_CHANGES: dict[str, spool_Any] = {
    "batch_id": "g0000000002-0002000200020002-0000",
    "destination_id": "other-dest",
    "signal": spool_M,
    "snapshot_id": "ff" * 32,
    "generation": 99,
    "index": 1,
    "total": 4,
    "kind": "tombstone",
    "created_utc": "2026-01-01T00:00:00Z",
    "bytes": 5,
    "sha256": "0" * 64,
    "items": 9,
    "record_ids": ("b" * 32,),
    "scope": {"changed": 1},
}


@spool_pytest.mark.parametrize("field", sorted(IDENTITY_CHANGES))
def test_identity_fields_are_immutable(tmp_path: spool_Path, field: str) -> None:
    batch = make(3, 0, total=3, signal=spool_L, items=1)
    spool_put(batch)
    path = folder(tmp_path, spool_L) / f"{batch.batch_id}.meta.json"
    before = path.read_bytes()
    with spool_pytest.raises(ValueError):
        spool.update_meta(
            spool_PROFILE,
            spool_DEST,
            spool_L,
            batch.batch_id,
            lambda m: spool_replace(m, **{field: IDENTITY_CHANGES[field]}),
        )
    assert path.read_bytes() == before


def test_every_identity_field_is_covered_by_the_immutability_test() -> None:
    from ssgrep.accounting.export.spool import IDENTITY

    assert sorted(IDENTITY) == sorted(IDENTITY_CHANGES)


@spool_pytest.mark.parametrize("state", ["sent", "", "QUEUED", "done"])
def test_invalid_state_is_rejected_and_every_valid_state_is_accepted(
    tmp_path: spool_Path, state: str
) -> None:
    batch = make()
    spool_put(batch)
    path = folder(tmp_path) / f"{batch.batch_id}.meta.json"
    before = path.read_bytes()
    with spool_pytest.raises(ValueError):
        spool.update_meta(
            spool_PROFILE,
            spool_DEST,
            spool_M,
            batch.batch_id,
            lambda m: spool_replace(m, state=state),
        )
    assert path.read_bytes() == before
    for good in spool.STATES:
        moved = spool.update_meta(
            spool_PROFILE,
            spool_DEST,
            spool_M,
            batch.batch_id,
            lambda m, s=good: spool_replace(m, state=s),
        )
        assert moved.state == good


def test_update_of_an_absent_batch_or_spool_is_file_not_found_and_creates_nothing(
    tmp_path: spool_Path,
) -> None:
    with spool_pytest.raises(FileNotFoundError):
        spool.update_meta(spool_PROFILE, spool_DEST, spool_M, make().batch_id, lambda m: m)
    assert spool_tree(tmp_path / "data") == {}
    spool_put(make(2))
    with spool_pytest.raises(FileNotFoundError):
        spool.update_meta(spool_PROFILE, spool_DEST, spool_M, make().batch_id, lambda m: m)


# --- strict stored-meta codec -----------------------------------------------------------------

OTHER_ID = "g0000000002-0002000200020002-0000"


def _doc(mutate: Callable[[dict[str, spool_Any]], None]) -> Callable[[dict[str, spool_Any]], bytes]:
    def build(doc: dict[str, spool_Any]) -> bytes:
        mutate(doc)
        return spool_json.dumps(doc).encode()

    return build


def _set(key: str, value: spool_Any) -> Callable[[dict[str, spool_Any]], bytes]:
    return _doc(lambda doc: doc.__setitem__(key, value))


def _drop(key: str) -> Callable[[dict[str, spool_Any]], bytes]:
    return _doc(lambda doc: doc.pop(key))


CODEC_CASES: dict[str, Callable[[dict[str, spool_Any]], bytes]] = {
    "unknown-key": _set("extra", 1),
    "float": _set("attempts", 1.5),
    "integral-float": _set("attempts", 1.0),
    "nan": _set("attempts", float("nan")),
    "float-in-scope": _set("scope", {"ratio": 0.5}),
    "bool-attempts": _set("attempts", True),
    "bool-generation": _set("generation", True),
    "bool-rejected": _set("rejected", False),
    "bool-schema": _set("schema", True),
    "newer-schema": _set("schema", 2),
    "older-schema": _set("schema", 0),
    "missing-identity": _drop("sha256"),
    "wrong-batch-id": _set("batch_id", OTHER_ID),
    "wrong-destination": _set("destination_id", "elsewhere"),
    "wrong-signal": _set("signal", "logs"),
    "bad-state": _set("state", "sent"),
    "bad-time": _set("created_utc", "2026-02-30T00:00:00Z"),
    "bad-sha": _set("sha256", "XYZ"),
    "negative-attempts": _set("attempts", -1),
    "string-attempts": _set("attempts", "1"),
    "record-ids-not-list": _set("record_ids", "abc"),
    "tombstone-for-metrics": _set("kind", "tombstone"),
    "index-past-total": _set("index", 5),
    "not-object": lambda doc: b"[1]",
    "not-json": lambda doc: b"{nope",
    "empty": lambda doc: b"",
    "escape-destination": _set("destination_id", "../escape"),
    "scope-not-object": _set("scope", []),
    "scope-int-overflow": _set("scope", {"count": 1 << 63}),
    "scope-depth": _set("scope", {"nested": [[[[[[[[[0]]]]]]]]]}),
    "purge-not-bool": _set("purge_requested", 1),
    "warning-type": _set("warning", 4),
    "warning-length": _set("warning", "x" * 513),
    "duplicate-key": lambda doc: spool_json.dumps(doc)[:-1].encode() + b',"attempts":7}',
}


@spool_pytest.mark.parametrize("case", sorted(CODEC_CASES))
def test_a_stored_meta_outside_schema_1_is_corrupt_and_never_rewritten(
    tmp_path: spool_Path, case: str
) -> None:
    batch = make()
    meta = spool_put(batch)
    healthy = spool_put(make(2))
    path = folder(tmp_path) / f"{batch.batch_id}.meta.json"
    original = path.read_bytes()
    good = spool_json.loads(original)
    assert good["schema"] == 1
    path.write_bytes(CODEC_CASES[case](good))
    broken = path.read_bytes()
    with spool_pytest.raises(SpoolCorrupt):
        spool.read_meta(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    with spool_pytest.raises(SpoolCorrupt):
        spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    with spool_pytest.raises(SpoolCorrupt):
        spool.update_meta(
            spool_PROFILE,
            spool_DEST,
            spool_M,
            batch.batch_id,
            lambda m: spool_replace(m, attempts=1),
        )
    assert path.read_bytes() == broken  # left untouched for a newer/older reader or an operator
    assert spool.list_batches(spool_PROFILE, spool_DEST, spool_M) == [healthy]  # skipped, not fatal
    assert spool.corrupt_batches(spool_PROFILE, spool_DEST, spool_M) == [batch.batch_id]
    assert meta != healthy
    path.write_bytes(original)  # operator restores the bytes: readable again, nothing was rewritten
    assert spool.read_meta(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == meta
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body
    assert meta in spool.list_batches(spool_PROFILE, spool_DEST, spool_M)
    assert spool.corrupt_batches(spool_PROFILE, spool_DEST, spool_M) == []


def test_a_meta_round_trips_byte_for_byte_through_the_codec(tmp_path: spool_Path) -> None:
    batch = make(4, 1, total=2, signal=spool_L, items=3)
    spool_put(batch)
    path = folder(tmp_path, spool_L) / f"{batch.batch_id}.meta.json"
    before = path.read_bytes()
    spool.update_meta(
        spool_PROFILE, spool_DEST, spool_L, batch.batch_id, lambda m: spool_replace(m)
    )
    assert path.read_bytes() == before
    spool.update_meta(
        spool_PROFILE, spool_DEST, spool_L, batch.batch_id, lambda m: spool_replace(m, attempts=1)
    )
    after = spool_json.loads(path.read_text())
    assert after == {**spool_json.loads(before), "attempts": 1}
    assert spool.corrupt_batches(spool_PROFILE, spool_DEST, spool_L) == []


# --- listing, usage, deletion -----------------------------------------------------------------


def test_list_batches_is_ordered_by_generation_then_data_before_tombstone_then_index(
    tmp_path: spool_Path,
) -> None:
    tombstone = make(10, kind="tombstone", signal=spool_L)
    for batch in (
        make(11, 0, signal=spool_L),
        make(10, 1, total=2, signal=spool_L),
        tombstone,
        make(9, 0, signal=spool_L),
        make(10, 0, total=2, signal=spool_L),
    ):
        spool_put(batch)
    assert ids(spool_L) == [
        f"g{9:010d}-{snap(9)[:16]}-0000",
        f"g{10:010d}-{snap(10)[:16]}-0000",
        f"g{10:010d}-{snap(10)[:16]}-0001",
        tombstone.batch_id,
        f"g{11:010d}-{snap(11)[:16]}-0000",
    ]
    assert ids(spool_M) == []  # the other signal's spool is separate (and was never created)
    assert not folder(tmp_path, spool_M).exists()


def test_usage_equals_the_bytes_and_batches_actually_on_disk(tmp_path: spool_Path) -> None:
    assert spool.usage(spool_PROFILE, spool_DEST) == Usage(0, 0)
    for gen in (1, 2, 3):
        spool_put(make(gen, body=spool_json.dumps({"gen": gen, "pad": "z" * (gen * 300)}).encode()))
    spool_put(make(4, signal=spool_L))
    spool_put(make(1, signal=spool_L), dest="other-dest")

    def disk(signal: spool_Signal, dest: str = spool_DEST) -> int:
        return sum(
            spool_os.stat(p).st_size
            for p in folder(tmp_path, signal, dest).iterdir()
            if p.name != ".spool.lock"
        )

    assert spool.usage(spool_PROFILE, spool_DEST, spool_M) == Usage(3, disk(spool_M))
    assert spool.usage(spool_PROFILE, spool_DEST, spool_L) == Usage(1, disk(spool_L))
    assert spool.usage(spool_PROFILE, spool_DEST) == Usage(4, disk(spool_M) + disk(spool_L))
    assert spool.usage(spool_PROFILE, "other-dest") == Usage(1, disk(spool_L, "other-dest"))
    assert spool.usage(spool_PROFILE, "unknown-dest") == Usage(0, 0)
    bodies = sum(len(spool.read_body(spool_PROFILE, spool_DEST, spool_M, i)) for i in ids(spool_M))
    metas = sum((folder(tmp_path) / f"{i}.meta.json").stat().st_size for i in ids(spool_M))
    assert metas > 0 and spool.usage(spool_PROFILE, spool_DEST, spool_M).bytes == bodies + metas
    spool.delete_batch(spool_PROFILE, spool_DEST, spool_M, ids(spool_M)[0])
    assert spool.usage(spool_PROFILE, spool_DEST, spool_M) == Usage(2, disk(spool_M))


def test_delete_batch_removes_meta_then_body_and_fsyncs_the_directory(
    tmp_path: spool_Path, monkeypatch: spool_pytest.MonkeyPatch
) -> None:
    keep, gone = make(1), make(2)
    spool_put(keep)
    spool_put(gone)
    events: list[str] = []
    real_unlink, real_fsync = spool_os.unlink, spool_os.fsync

    def unlink(path: str, **kwargs: spool_Any) -> None:
        events.append(str(path))
        real_unlink(path, **kwargs)

    def fsync(fd: int) -> None:
        events.append(
            "fsync-dir" if spool_stat.S_ISDIR(spool_os.fstat(fd).st_mode) else "fsync-file"
        )
        real_fsync(fd)

    monkeypatch.setattr(spool_os, "unlink", unlink)
    monkeypatch.setattr(spool_os, "fsync", fsync)
    assert spool.delete_batch(spool_PROFILE, spool_DEST, spool_M, gone.batch_id) is True
    monkeypatch.setattr(spool_os, "unlink", real_unlink)
    monkeypatch.setattr(spool_os, "fsync", real_fsync)
    assert events == [f"{gone.batch_id}.meta.json", f"{gone.batch_id}.json", "fsync-dir"]
    assert names(folder(tmp_path)) == [f"{keep.batch_id}.json", f"{keep.batch_id}.meta.json"]
    assert spool.delete_batch(spool_PROFILE, spool_DEST, spool_M, gone.batch_id) is False
    assert spool.delete_batch(spool_PROFILE, "unknown-dest", spool_M, gone.batch_id) is False
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, keep.batch_id) == keep.body


# --- sweep, purge, discovery, permissions -----------------------------------------------------


def test_sweep_removes_orphan_bodies_leaked_temporaries_and_bodyless_metas_only(
    tmp_path: spool_Path,
) -> None:
    healthy, orphan_body, orphan_meta = make(1), make(2), make(3)
    spool_put(healthy)
    spool_put(orphan_body)
    spool_put(orphan_meta)
    directory = folder(tmp_path)
    (directory / f"{orphan_body.batch_id}.meta.json").unlink()
    (directory / f"{orphan_meta.batch_id}.json").unlink()
    leaked = directory / f".{healthy.batch_id}.json.123.abcd1234.tmp"
    leaked.write_bytes(b"half written")
    unrelated = [directory / "README.txt", directory / "notes.json", directory / "g1.json"]
    for path in unrelated:
        path.write_bytes(b"not ours")
    assert spool.sweep_orphans(spool_PROFILE, spool_DEST, spool_M) == 3
    assert names(directory) == sorted(
        [f"{healthy.batch_id}.json", f"{healthy.batch_id}.meta.json", *(p.name for p in unrelated)]
    )
    assert all(p.read_bytes() == b"not ours" for p in unrelated)
    assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, healthy.batch_id) == healthy.body
    assert spool.sweep_orphans(spool_PROFILE, spool_DEST, spool_M) == 0
    assert spool.sweep_orphans(spool_PROFILE, "unknown-dest", spool_M) == 0
    assert not folder(tmp_path, spool_M, "unknown-dest").exists()


def test_drop_destination_purges_every_batch_of_that_destination_only(tmp_path: spool_Path) -> None:
    for gen in (1, 2):
        spool_put(make(gen))
    spool_put(make(1, signal=spool_L))
    spool_put(make(5), dest="other-dest")
    (folder(tmp_path) / f".{make(1).batch_id}.json.9.deadbeef.tmp").write_bytes(b"leak")
    orphan = make(9)
    spool_put(orphan)
    (folder(tmp_path) / f"{orphan.batch_id}.meta.json").unlink()  # an orphan body counts as a batch
    assert spool.drop_destination(spool_PROFILE, spool_DEST, spool_L) == 1
    assert ids(spool_M) == [make(1).batch_id, make(2).batch_id] and ids(spool_L) == []
    assert spool.drop_destination(spool_PROFILE, spool_DEST) == 3
    assert names(folder(tmp_path)) == [] and names(folder(tmp_path, spool_L)) == []
    assert [m.batch_id for m in spool.list_batches(spool_PROFILE, "other-dest", spool_M)] == [
        make(5).batch_id
    ]
    assert spool.drop_destination(spool_PROFILE, spool_DEST) == 0
    assert spool.drop_destination(spool_PROFILE, "unknown-dest") == 0


def test_iter_spools_lists_existing_spools_and_creates_nothing(tmp_path: spool_Path) -> None:
    root = tmp_path / "data"
    assert spool.iter_spools(spool_PROFILE) == [] and spool_tree(root) == {}
    spool_put(make(1))
    spool_put(make(1, signal=spool_L), dest="dest-b")
    spool_put(make(2), dest="dest-b")
    spools = tmp_path / "data" / "accounting" / spool_PROFILE / "spool"
    (spools / "stray-file").write_bytes(b"x")
    (spools / "BAD NAME").mkdir(mode=0o700)
    (spools / "not-a-signal-dest").mkdir(mode=0o700)
    (spools / "not-a-signal-dest" / "traces").mkdir(mode=0o700)
    (spools / "linked").symlink_to(tmp_path, target_is_directory=True)
    before = spool_tree(root)
    assert sorted(spool.iter_spools(spool_PROFILE), key=lambda t: (t[0], t[1].value)) == [
        ("dest-b", spool_L),
        ("dest-b", spool_M),
        (spool_DEST, spool_M),
    ]
    assert spool.iter_spools("unknown-profile") == [] and spool.iter_spools("../x") == []
    for call in (
        lambda: spool.list_batches(spool_PROFILE, "ghost", spool_M),
        lambda: spool.read_meta(spool_PROFILE, "ghost", spool_M, make().batch_id),
        lambda: spool.corrupt_batches(spool_PROFILE, "ghost", spool_L),
        lambda: spool.usage(spool_PROFILE, "ghost"),
        lambda: spool.sweep_orphans(spool_PROFILE, "ghost", spool_M),
        lambda: spool.drop_destination(spool_PROFILE, "ghost"),
        lambda: spool.delete_batch(spool_PROFILE, "ghost", spool_M, make().batch_id),
    ):
        call()
    assert spool_tree(root) == before


def test_files_are_0600_and_directories_0700(tmp_path: spool_Path) -> None:
    batch = make(1, signal=spool_L)
    spool_put(batch)
    spool.update_meta(
        spool_PROFILE, spool_DEST, spool_L, batch.batch_id, lambda m: spool_replace(m, attempts=1)
    )
    base = tmp_path / "data" / "accounting"
    entries = [base, *base.rglob("*")]
    assert len(entries) >= 8
    for path in entries:
        mode = spool_stat.S_IMODE(path.lstat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), path
    assert {p.name for p in folder(tmp_path, spool_L).iterdir()} == {
        f"{batch.batch_id}.json",
        f"{batch.batch_id}.meta.json",
        ".spool.lock",
    }


# --- persistence failures preserve bytes and recover through public operations ---------------


@spool_pytest.mark.parametrize("owner", ["acks", "state", "registry", "spool"])
@spool_pytest.mark.parametrize("exhausted", [False, True])
def test_lock_creation_race_is_bounded_and_recovers_without_partial_state(
    owner: str, exhausted: bool, monkeypatch: spool_pytest.MonkeyPatch
) -> None:
    planned = spool_plan_revisions(export_registry.Registry(), project(rows_a()))
    batch = make()
    lock_name = {
        "acks": ".metrics.lock",
        "state": ".control.lock",
        "registry": ".records.lock",
        "spool": ".spool.lock",
    }[owner]
    module = {
        "acks": spool_acks,
        "state": spool,
        "registry": export_registry,
        "spool": spool,
    }[owner]

    def persist():
        if owner == "acks":
            return spool_acks.update_delivery(
                spool_PROFILE, spool_DEST, spool_M, lambda d: spool_acks.collected(d, 7)
            )
        if owner == "registry":
            from ssgrep.accounting.export.registry import mark_queued

            return mark_queued(spool_PROFILE, spool_DEST, planned, batch.batch_id)
        if owner == "spool":
            return spool_put(batch)
        with spool.locked(spool_PROFILE, ("export", spool_DEST), "control") as fd:
            assert fd is not None
            spool.write_json(fd, "control.json", {"generation": 7})
        return None

    real_open = spool_os.open
    attempts = []
    fail_count = module.LOCK_OPEN_ATTEMPTS - (0 if exhausted else 1)

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == lock_name:
            attempts.append(path)
            if len(attempts) <= fail_count:
                raise FileNotFoundError(spool_errno.ENOENT, "concurrent lock creation")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as fault:
        fault.setattr(spool_os, "open", racing_open)
        if exhausted:
            with spool_pytest.raises(
                (OSError, export_registry.RegistryCorrupt, spool.SpoolCorrupt)
            ):
                persist()
        else:
            persist()
    assert len(attempts) == module.LOCK_OPEN_ATTEMPTS
    if exhausted:
        assert not [p for p in spool_fx.profile_root().rglob("*.json")]
    persist()
    if owner == "acks":
        assert spool_fx.state().last_collected_generation == 7
    elif owner == "registry":
        entry = spool_fx.registry().entries[planned[0].record_id]
        assert [(s.revision, s.batch) for s in entry.queued] == [(1, batch.batch_id)]
    elif owner == "spool":
        assert spool.read_body(spool_PROFILE, spool_DEST, spool_M, batch.batch_id) == batch.body
        assert spool_fx.metas()[0].attempts == 0
    elif owner == "state":
        path = spool_fx.profile_root() / "export" / spool_DEST / "control.json"
        assert spool_json.loads(path.read_bytes()) == {"generation": 7}


@spool_pytest.mark.parametrize("owner", ["acks", "spool"])
def test_directory_listing_failure_is_read_only_and_next_listing_recovers(
    owner: str, monkeypatch: spool_pytest.MonkeyPatch
) -> None:
    meta = spool_put(make())
    saved = spool_acks.update_delivery(
        spool_PROFILE, spool_DEST, spool_M, lambda d: spool_acks.collected(d, 1)
    )
    before = spool_tree(spool_fx.profile_root())
    real_list = spool_storage.list_names
    wanted = spool_fx.profile_root() / ("delivery" if owner == "acks" else "spool")
    identity = wanted.stat()

    def unreadable(fd):
        info = spool_os.fstat(fd)
        if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            raise OSError(spool_errno.EIO, "directory unavailable")
        return real_list(fd)

    with monkeypatch.context() as fault:
        fault.setattr(spool_storage, "list_names", unreadable)
        found = (
            spool_acks.list_deliveries(spool_PROFILE)
            if owner == "acks"
            else spool.iter_spools(spool_PROFILE)
        )
        assert found == []
    assert spool_tree(spool_fx.profile_root()) == before
    assert spool_acks.list_deliveries(spool_PROFILE) == [saved]
    assert spool.iter_spools(spool_PROFILE) == [(spool_DEST, spool_M)]
    assert spool_fx.body_of(meta) == make().body


def test_metadata_encoding_cap_refuses_write_without_replacing_original(
    monkeypatch: spool_pytest.MonkeyPatch,
) -> None:
    meta = spool_put(make())
    path = (
        spool_fx.profile_root()
        / "spool"
        / spool_DEST
        / spool_M.value
        / f"{meta.batch_id}.meta.json"
    )
    original = path.read_bytes()
    with monkeypatch.context() as fault:
        fault.setattr(codec, "MAX_META_BYTES", len(original) - 1)
        with spool_pytest.raises(spool.SpoolCorrupt):
            spool.update_meta(
                spool_PROFILE,
                spool_DEST,
                spool_M,
                meta.batch_id,
                lambda m: spool_replace(m, attempts=1),
            )
    assert path.read_bytes() == original and spool_fx.body_of(meta) == make().body
    assert (
        spool.update_meta(
            spool_PROFILE,
            spool_DEST,
            spool_M,
            meta.batch_id,
            lambda m: spool_replace(m, attempts=1),
        ).attempts
        == 1
    )


def test_invalid_update_result_cannot_replace_a_valid_spool_meta() -> None:
    meta = spool_put(make())
    before = spool_tree(spool_fx.profile_root())
    with spool_pytest.raises(ValueError):
        invalid = cast(Callable[[codec.Meta], codec.Meta], lambda _meta: None)
        spool.update_meta(spool_PROFILE, spool_DEST, spool_M, meta.batch_id, invalid)
    assert spool_tree(spool_fx.profile_root()) == before
    assert (
        spool.update_meta(
            spool_PROFILE,
            spool_DEST,
            spool_M,
            meta.batch_id,
            lambda m: spool_replace(m, state="acked"),
        ).state
        == "acked"
    )


def test_privacy_erase_requires_marker_and_cannot_erase_deletion_requests() -> None:
    batch = make()
    assert not spool.erase_marked_batch(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    meta = spool_put(batch)
    before = spool_tree(spool_fx.profile_root())
    with spool_pytest.raises(ValueError):
        spool.erase_marked_batch(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    assert spool_tree(spool_fx.profile_root()) == before and spool_fx.body_of(meta) == batch.body
    tomb = spool_put(make(signal=spool_L, kind="tombstone"))
    with spool_pytest.raises(ValueError):
        spool.update_meta(
            spool_PROFILE,
            spool_DEST,
            spool_L,
            tomb.batch_id,
            lambda m: spool_replace(m, purge_requested=True),
        )
    assert spool_fx.body_of(tomb) == make(signal=spool_L, kind="tombstone").body
    spool.update_meta(
        spool_PROFILE,
        spool_DEST,
        spool_M,
        batch.batch_id,
        lambda m: spool_replace(m, purge_requested=True),
    )
    assert spool.erase_marked_batch(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    assert not spool.erase_marked_batch(spool_PROFILE, spool_DEST, spool_M, batch.batch_id)
    assert spool_fx.metas() == [] and spool_fx.metas(spool_L) == [tomb]


@spool_pytest.mark.parametrize("operation", ["corrupt", "usage"])
def test_spool_file_removed_after_listing_is_not_reported_as_existing(
    operation: str, monkeypatch: spool_pytest.MonkeyPatch
) -> None:
    meta = spool_put(make())
    real_list = spool_storage.list_names
    folder = spool_fx.profile_root() / "spool" / spool_DEST / spool_M.value
    removed = []

    def concurrent_delete(fd):
        names = real_list(fd)
        if f"{meta.batch_id}.meta.json" in names and not removed:
            removed.append(meta.batch_id)
            for path in folder.glob(f"{meta.batch_id}.*"):
                path.unlink()
        return names

    with monkeypatch.context() as race:
        race.setattr(spool_storage, "list_names", concurrent_delete)
        if operation == "corrupt":
            assert spool.corrupt_batches(spool_PROFILE, spool_DEST, spool_M) == []
        else:
            assert spool.usage(spool_PROFILE, spool_DEST, spool_M) == spool.Usage(0, 0)
    assert removed == [meta.batch_id] and spool_fx.metas() == []
    spool_put(make())
    assert spool_fx.body_of(meta) == make().body
    assert spool.usage(spool_PROFILE, spool_DEST, spool_M).batches == 1


def test_invalid_state_path_is_refused_before_creating_anything() -> None:
    before = spool_tree(spool_fx.profile_root())
    with spool_pytest.raises(ValueError):
        with spool.locked(spool_PROFILE, ("../escape",), "control"):
            spool_pytest.fail("invalid path acquired a lock")
    assert spool_tree(spool_fx.profile_root()) == before
    with spool.locked(spool_PROFILE, ("export", spool_DEST), "control") as fd:
        assert fd is not None
        spool.write_json(fd, "control.json", {"ok": True})
    assert spool_json.loads(
        (spool_fx.profile_root() / "export" / spool_DEST / "control.json").read_bytes()
    ) == {"ok": True}


def test_spool_codec_rejects_invalid_namespace_even_when_expected_binding_matches() -> None:
    meta = spool_put(make())
    original = codec.encode_meta(meta)
    document = spool_json.loads(original)
    document["destination_id"] = "../escape"
    with spool_pytest.raises(codec.SpoolCorrupt, match="destination id"):
        codec.decode_meta(spool_json.dumps(document).encode(), meta.batch_id, "../escape", spool_M)
    restored = codec.decode_meta(original, meta.batch_id, spool_DEST, spool_M)
    assert restored == meta and spool_fx.body_of(meta) == make().body


def test_spool_codec_bounds_record_count_independently_of_identifier_shape() -> None:
    meta = spool_put(make())
    original = codec.encode_meta(meta)
    document = spool_json.loads(original)
    document["record_ids"] = ["r"] * (codec.MAX_RECORD_IDS + 1)
    with spool_pytest.raises(codec.SpoolCorrupt, match="record_ids must be a list of at most"):
        codec.decode_meta(spool_json.dumps(document).encode(), meta.batch_id, spool_DEST, spool_M)
    restored = codec.decode_meta(original, meta.batch_id, spool_DEST, spool_M)
    assert restored == meta and spool_fx.body_of(meta) == make().body


def test_conflicting_orphan_created_during_enqueue_preserves_bytes_until_recovery(
    monkeypatch: spool_pytest.MonkeyPatch,
) -> None:
    with spool_LabReceiver.start() as lab:
        spool_fx.configure(lab.url)
        spool_fx.publish(1)
        real_write = spool.write_batch
        orphan = []

        def conflicting_writer(profile, destination, signal, batch, **kwargs):
            fd = spool_storage.open_profile_dir(profile, "spool", destination, signal.value)
            try:
                assert spool_storage.write_once(fd, f"{batch.batch_id}.json", b"old orphan bytes")
            finally:
                spool_os.close(fd)
            orphan.append(batch.batch_id)
            return real_write(profile, destination, signal, batch, **kwargs)

        with monkeypatch.context() as race:
            race.setattr(spool, "write_batch", conflicting_writer)
            failed = spool_fx.cycle()
        assert (failed.state, failed.reason, failed.sent) == ("failed", "batch_conflict", 0)
        assert len(orphan) == 1 and spool_fx.metas() == [] and lab.requests == []
        path = spool_fx.profile_root() / "spool" / spool_DEST / spool_M.value / f"{orphan[0]}.json"
        assert path.read_bytes() == b"old orphan bytes"
        assert spool.sweep_orphans(spool_PROFILE, spool_DEST, spool_M) == 1
        assert spool_fx.cycle(seconds=1).acked == 1
        assert lab.requests[0].body == spool_fx.body_of(spool_fx.metas()[0])
        assert b"old orphan bytes" not in lab.requests[0].body


# ---- Consolidated from test_export_retry.py (source-specific globals are prefixed) ----


def _always(value: float):
    return lambda: value


def test_growth_is_monotone_exponential_with_full_jitter_top():
    delays = [retry.backoff_seconds(n, None, rand=_always(1.0)) for n in range(1, 12)]
    assert delays[:9] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert delays == sorted(delays)


def test_cap_is_five_minutes_even_for_absurd_attempts():
    for attempt in (10, 11, 50, 10_000):
        assert retry.backoff_seconds(attempt, None, rand=_always(1.0)) == 300


def test_jitter_stays_within_half_and_full_delay():
    for attempt, delay in ((3, 4), (6, 32), (9, 256), (12, 300)):
        low = retry.backoff_seconds(attempt, None, rand=_always(0.0))
        high = retry.backoff_seconds(attempt, None, rand=_always(1.0))
        mid = retry.backoff_seconds(attempt, None, rand=_always(0.5))
        assert (low, high) == (delay // 2, delay)
        assert low <= mid <= high


def test_first_attempt_is_at_least_one_second():
    assert retry.backoff_seconds(1, None, rand=_always(0.0)) == 1
    assert retry.backoff_seconds(2, None, rand=_always(0.0)) == 1


def test_out_of_range_random_values_cannot_escape_the_bounds():
    assert retry.backoff_seconds(4, None, rand=_always(7.5)) == 8
    assert retry.backoff_seconds(4, None, rand=_always(-3.0)) == 4


def test_retry_after_is_a_floor_not_a_replacement():
    assert retry.backoff_seconds(1, 30, rand=_always(1.0)) == 30
    assert retry.backoff_seconds(8, 5, rand=_always(1.0)) == 128  # computed delay is larger
    assert retry.backoff_seconds(2, 0, rand=_always(1.0)) == 2


def test_final_delay_is_clamped_to_one_hour():
    assert retry.backoff_seconds(1, 86_400, rand=_always(0.0)) == 3600
    assert retry.backoff_seconds(1, 3600, rand=_always(0.0)) == 3600


@retry_pytest.mark.parametrize("attempt", [0, -1, -100])
def test_non_positive_attempt_is_rejected(attempt):
    with retry_pytest.raises(ValueError):
        retry.backoff_seconds(attempt, None)


def test_default_randomness_stays_in_bounds():
    for _ in range(200):
        assert 4 <= retry.backoff_seconds(4, None) <= 8


def _response(status: int) -> retry_Outcome:
    return retry_Outcome("response", status, None, None, "x")


@retry_pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_response(200), "ack_candidate"),
        (retry_Outcome("network_error"), "retry"),
        (retry_Outcome("timeout"), "retry"),
        (_response(429), "retry"),
        (_response(502), "retry"),
        (_response(503), "retry"),
        (_response(504), "retry"),
        (_response(401), "blocked"),
        (_response(403), "blocked"),
        (_response(400), "permanent"),
        (_response(404), "permanent"),
        (_response(413), "permanent"),
        (_response(500), "permanent"),
        (_response(501), "permanent"),
        (retry_Outcome("tls_error"), "refused"),
        (retry_Outcome("redirect_refused", 307), "refused"),
        (retry_Outcome("oversize_response"), "refused"),
        (retry_Outcome("malformed_response"), "refused"),
        (retry_Outcome("transport_refused"), "refused"),
    ],
)
def test_classification_table(outcome, expected):
    assert retry.classify(outcome) == expected


def test_is_retryable_agrees_with_classification_for_every_status():
    for status in range(100, 600):
        outcome = _response(status)
        assert retry.is_retryable("response", status) == (retry.classify(outcome) == "retry")
    assert not retry.is_retryable("tls_error", None)
    assert retry.is_retryable("timeout", None)


# ---- Consolidated from test_tombstones.py (source-specific globals are prefixed) ----


tomb_PROFILE = "personal-local"
tomb_DEST = "acme-org"
OTHER = "other-org"
ENDPOINT = "https://collector.example.test"
tomb_NOW = tomb_datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)
STAMP = int(tomb_NOW.timestamp())
SCENE_NOW = tomb_datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
T_ID = re.compile(r"^t[0-9]{10}-[0-9a-f]{8}-[0-9]{4}$")
UNIT_KEY = ("context_tokens", "chars_div4_v1", None, None)
ROOT1, ROOT2, ROOT3 = (TaskId(f"root-{n}", "turn-1", "main") for n in (1, 2, 3))
DAY1, DAY2, DAY3 = "2026-11-01", "2026-11-02", "2026-11-03"


# ---- builders --------------------------------------------------------------------------------


def data_root(tmp_path: tomb_Path) -> tomb_Path:
    return tmp_path / "data"


def tomb_tree(root: tomb_Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def make_policy(
    *, enabled: tuple[str, ...] = (tomb_DEST, OTHER), logs: tuple[str, ...] = (tomb_DEST, OTHER)
) -> ProfilePolicy:
    """Explicitly approve current logs fields only for the declared enabled destinations."""
    destinations = tuple(
        DestinationPolicy(
            name,
            ENDPOINT,
            (tomb_Signal.METRICS, tomb_Signal.LOGS) if name in logs else (tomb_Signal.METRICS,),
            logs_permitted=name in logs,
        )
        for name in (tomb_DEST, OTHER)
    )
    exports = tuple((name, "logs") for name in enabled if name in logs)
    current = tomb_policy.save_profile(
        ProfilePolicy(destinations=destinations, enabled_exports=exports, collection_enabled=True)
    )
    checkpoint = synthetic_current_checkpoint(tomb_PROFILE, current)
    collector_store.commit_checkpoint(tomb_PROFILE, checkpoint)
    for destination, signal in exports:
        tomb_export_consent.apply_enable(tomb_PROFILE, destination, tomb_Signal(signal))
    return current


def key() -> bytes:
    minted = tomb_export_ids.load_key(tomb_PROFILE, create=True)
    assert minted is not None
    return minted


def ledger_rows() -> tomb_Any:
    ledger = Ledger(METHOD)
    ledger.ingest(
        EvidenceBundle(
            (
                candidate("a", task=ROOT1, day=DAY1),
                candidate("b", task=ROOT2, day=DAY3, immediate=(atom("s2", 80, 0, "e2"),)),
                candidate("c", task=ROOT3, day=DAY2, immediate=(atom("s3", 120, 0, "e3"),)),
            ),
            (),
        )
    )
    return ledger.rows()


def plan_for(destination: str, rows: tomb_Any = None) -> list[Planned]:
    candidates, omitted = project_candidates(
        rows if rows is not None else ledger_rows(), key=key(), destination=destination
    )
    assert omitted == 0
    return tomb_plan_revisions(Registry(), candidates)


def seed(destination: str) -> dict[str, Planned]:
    """Root-1 and root-2 records are SENT (acked); root-3's only ever sat queued."""
    planned = plan_for(destination)
    by_root = {p.root_key: p for p in planned}
    assert set(by_root) == {root_key_of(t.root) for t in (ROOT1, ROOT2, ROOT3)}
    sent = [by_root[root_key_of(ROOT1.root)], by_root[root_key_of(ROOT2.root)]]
    mark_queued(tomb_PROFILE, destination, sent, "b-sent-0001")
    mark_acked(tomb_PROFILE, destination, "b-sent-0001")
    mark_queued(tomb_PROFILE, destination, [by_root[root_key_of(ROOT3.root)]], "b-unsent-0001")
    return by_root


def ids_of(by_root: dict[str, Planned], *roots: TaskId) -> tuple[str, ...]:
    return tuple(sorted(by_root[root_key_of(r.root)].record_id for r in roots))


def plan_of(plans: list[TombstonePlan], destination: str) -> TombstonePlan:
    (found,) = [p for p in plans if p.destination_id == destination]
    return found


def logs_bodies(request: bytes) -> list[dict[str, tomb_Any]]:
    """Record bodies of one serialized OTLP logs request, in request order."""
    document = tomb_json.loads(request)
    found = []
    for resource in document["resourceLogs"]:
        for scope in resource["scopeLogs"]:
            found += [tomb_json.loads(r["body"]["stringValue"]) for r in scope["logRecords"]]
    return found


def spooled(destination: str) -> list[tomb_export_spool.Meta]:
    return tomb_export_spool.list_batches(tomb_PROFILE, destination, tomb_Signal.LOGS)


def tombstone_bodies(destination: str) -> list[dict[str, tomb_Any]]:
    bodies: list[dict[str, tomb_Any]] = []
    for meta in spooled(destination):
        if meta.kind == "tombstone":
            bodies += logs_bodies(
                tomb_export_spool.read_body(
                    tomb_PROFILE, destination, tomb_Signal.LOGS, meta.batch_id
                )
            )
    return bodies


def receive(
    state: ReceiverState, bodies: list[dict[str, tomb_Any]]
) -> tuple[ReceiverState, list[tomb_Outcome]]:
    outcomes = []
    for body in bodies:
        state, outcome = apply(state, Key("t", "dev", body["record_id"]), body)
        outcomes.append(outcome)
    return state, outcomes


def rows_total(state: ReceiverState) -> int:
    return current_totals(state, methodology=METHOD.hash, unit_key=UNIT_KEY)["rows"]


def synthetic(index: int, destination: str = tomb_DEST) -> Planned:
    """A receiver-valid revision-1 body that needs no ledger."""
    record_id = tomb_export_ids.pseudonym(key(), "record", destination, f"i{index}")
    body: dict[str, tomb_Any] = {
        "schema": 1, "record_id": record_id, "revision": 1, "supersedes_revision": None,
        "record_type": "contribution", "task_id": record_id, "period": DAY1,
        "unit": {"unit": "context_tokens", "method": "chars_div4_v1", "tokenizer": None,
                 "currency": None},
        "scenarios": {"central": {k: dict(QUANTITY) for k in ("credit", "debit", "net")}},
        "known_costs": {}, "coverage": {}, "methodology": METHOD.hash,
    }  # fmt: skip
    return Planned(
        record_id, 1, None, "contribution", body, digest_of(body), DAY1, record_id,
        root_key_of(f"w{index}"), digest_of(["content", index]),
    )  # fmt: skip


# ---- plan selection --------------------------------------------------------------------------


def test_plan_selects_sent_records_by_day_roots_and_their_intersection() -> None:
    current = make_policy()
    by_root = {d: seed(d) for d in (tomb_DEST, OTHER)}
    one, two = ids_of(by_root[tomb_DEST], ROOT1), ids_of(by_root[tomb_DEST], ROOT2)
    both = ids_of(by_root[tomb_DEST], ROOT1, ROOT2)
    r1, r2, r3 = (frozenset({root_key_of(t.root)}) for t in (ROOT1, ROOT2, ROOT3))

    everything = plan_of(
        plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None), tomb_DEST
    )
    assert everything.record_ids == both and everything.record_count == 2  # root-3 never sent

    def by_day(day: str) -> tuple[str, ...]:
        plans = plan_tombstones(tomb_PROFILE, current, before_day=day, root_keys=None)
        return plan_of(plans, tomb_DEST).record_ids

    assert by_day("2026-11-02") == one  # strictly before the day
    assert by_day("2026-11-04") == both
    assert by_day(DAY1) == ()

    def by_roots(roots: frozenset[str], day: str | None = None) -> tuple[str, ...]:
        plans = plan_tombstones(tomb_PROFILE, current, before_day=day, root_keys=roots)
        return plan_of(plans, tomb_DEST).record_ids

    assert by_roots(r2) == two
    assert by_roots(r1 | r2) == both
    assert by_roots(r1, "2026-11-04") == one  # intersection: root-1 AND before day
    assert by_roots(r2, "2026-11-02") == ()  # root-2 is dated after the cutoff
    assert by_roots(r3) == ()  # queued-never-sent: nothing to retract
    assert by_roots(frozenset({"f" * 16})) == ()


def test_each_destination_is_planned_from_its_own_registry() -> None:
    current = make_policy()
    by_root = {d: seed(d) for d in (tomb_DEST, OTHER)}
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    assert [p.destination_id for p in plans] == [tomb_DEST, OTHER]
    for destination in (tomb_DEST, OTHER):
        found = plan_of(plans, destination)
        assert found.record_ids == ids_of(by_root[destination], ROOT1, ROOT2)
    assert not set(plans[0].record_ids) & set(plans[1].record_ids)  # unrelated pseudonyms


def test_queued_only_registry_has_nothing_sent_to_retract() -> None:
    current = make_policy()
    planned = plan_for(tomb_DEST)
    mark_queued(tomb_PROFILE, tomb_DEST, planned, "b-unsent-0001")
    found = plan_of(
        plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None), tomb_DEST
    )
    assert (found.permitted, found.reason, found.record_ids, found.record_count) == (
        True,
        "no_records_to_delete",
        (),
        0,
    )


def test_destination_without_any_registry_or_metrics_ack_gets_no_plan() -> None:
    current = make_policy()
    seed(tomb_DEST)  # OTHER is in the policy but never exported anything
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    assert [p.destination_id for p in plans] == [tomb_DEST]


# ---- permission ------------------------------------------------------------------------------


@tomb_pytest.mark.parametrize(
    ("saved", "permitted", "reason"),
    [
        (lambda: make_policy(), True, None),
        (lambda: make_policy(enabled=()), False, "not_enabled"),
        (lambda: make_policy(enabled=(), logs=()), False, "signal_not_permitted"),
    ],
    ids=["enabled", "not_enabled", "logs_not_allowed"],
)
def test_permission_follows_current_policy_with_stable_codes(
    saved: tomb_Any, permitted: bool, reason: str | None
) -> None:
    seed(tomb_DEST)
    current = saved()
    found = plan_of(
        plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None), tomb_DEST
    )
    assert (found.permitted, found.reason) == (permitted, reason)
    assert found.record_count == 2 and len(found.record_ids) == 2
    assert (found.obligations == ()) is permitted
    assert len(found.obligations) == (0 if permitted else 2)


def test_unreadable_policy_file_denies_and_unknown_destination_cannot_be_planned(
    tmp_path: tomb_Path,
) -> None:
    current = make_policy()
    seed(tomb_DEST)
    (data_root(tmp_path) / "accounting" / "policy.json").write_text("{not json")
    found = plan_of(
        plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None), tomb_DEST
    )
    assert (found.permitted, found.reason) == (False, "policy_unreadable")
    assert len(found.obligations) == 2


def test_paused_profile_still_permits_spooling_but_not_unpaused_sending() -> None:
    make_policy()
    seed(tomb_DEST)
    tomb_policy.pause(tomb_PROFILE)
    current = tomb_policy.load_profile_policy(tomb_PROFILE)
    (found,) = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    assert found.permitted and found.reason is None
    (result,) = enqueue_tombstones(tomb_PROFILE, current, [found], now=tomb_NOW)
    assert result.batches == 1 and pending_tombstones(tomb_PROFILE, tomb_DEST) == 1


def test_non_reversible_metrics_only_when_a_metrics_ack_exists() -> None:
    current = make_policy()
    seed(tomb_DEST)
    seed(OTHER)
    assert [
        p.non_reversible_metrics
        for p in plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    ] == [False, False]
    tomb_acks.update_delivery(
        tomb_PROFILE,
        tomb_DEST,
        tomb_Signal.METRICS,
        lambda d: tomb_acks.ack_vacuous(d, 1, "a" * 16, "2026-11-01T00:00:00Z"),
    )
    tomb_acks.update_delivery(
        tomb_PROFILE, OTHER, tomb_Signal.LOGS, lambda d: tomb_acks.set_counts(d, 1, 0)
    )
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    assert plan_of(plans, tomb_DEST).non_reversible_metrics is True
    assert plan_of(plans, OTHER).non_reversible_metrics is False  # a logs document is not metrics


def test_metrics_ack_alone_reports_the_limit_even_without_per_task_records() -> None:
    current = make_policy()
    tomb_acks.update_delivery(
        tomb_PROFILE,
        tomb_DEST,
        tomb_Signal.METRICS,
        lambda d: tomb_acks.ack_vacuous(d, 1, "a" * 16, "2026-11-01T00:00:00Z"),
    )
    (found,) = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    assert (found.destination_id, found.record_count, found.reason) == (
        tomb_DEST,
        0,
        "no_records_to_delete",
    )
    assert found.non_reversible_metrics is True


# ---- plan is read-only -----------------------------------------------------------------------


def test_plan_creates_and_changes_nothing(tmp_path: tomb_Path) -> None:
    current = make_policy()
    seed(tomb_DEST)
    before = tomb_tree(data_root(tmp_path))
    r1 = frozenset({root_key_of(ROOT1.root)})
    for day, roots in ((None, None), ("2026-11-09", r1)):
        plans = plan_tombstones(tomb_PROFILE, current, before_day=day, root_keys=roots)
        assert plan_of(plans, tomb_DEST).record_count >= 1  # positive twin: it really planned
    assert tomb_tree(data_root(tmp_path)) == before


# ---- enqueue ---------------------------------------------------------------------------------


def test_enqueue_spools_receiver_valid_tombstones_and_queues_deletion_revisions() -> None:
    current = make_policy()
    by_root = seed(tomb_DEST)
    sent = [by_root[root_key_of(t.root)] for t in (ROOT1, ROOT2)]
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    before = load_registry(tomb_PROFILE, tomb_DEST)

    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert (result.destination_id, result.queued_records, result.batches) == (tomb_DEST, 2, 1)
    assert result.skipped_reason is None and result.obligations == ()
    (batch_id,) = result.batch_ids
    assert T_ID.fullmatch(batch_id) and batch_id.startswith(f"t{STAMP:010d}-")

    (meta,) = spooled(tomb_DEST)
    assert (meta.batch_id, meta.kind, meta.state, meta.generation, meta.snapshot_id) == (
        batch_id,
        "tombstone",
        "queued",
        0,
        "",
    )
    assert meta.created_utc == "2026-11-05T12:00:00Z" and meta.items == 2
    assert meta.record_ids == tuple(sorted(p.record_id for p in sent))
    assert meta.scope["endpoint"] == ENDPOINT and meta.scope["logs_permitted"] is True

    after = load_registry(tomb_PROFILE, tomb_DEST)
    for item in sent:
        entry = after.entries[item.record_id]
        assert entry.acked == before.entries[item.record_id].acked  # the contribution stays acked
        (queued,) = entry.queued
        assert (queued.revision, queued.type, queued.batch) == (2, "deletion", batch_id)
        assert queued.revision == item.revision + 1
    unsent = by_root[root_key_of(ROOT3.root)]
    assert after.entries[unsent.record_id].queued[0].type == "contribution"  # untouched

    delivery = tomb_acks.read_delivery(tomb_PROFILE, tomb_DEST, tomb_Signal.LOGS)
    assert delivery is not None and delivery.pending_batches == 1  # the one tombstone batch
    assert delivery.unknown_batches == 0
    assert pending_tombstones(tomb_PROFILE, tomb_DEST) == 1
    assert pending_tombstones(tomb_PROFILE, OTHER) == 0


def test_tombstone_bodies_delete_at_the_receiver_and_replays_are_refused() -> None:
    current = make_policy()
    by_root = seed(tomb_DEST)
    sent = [by_root[root_key_of(t.root)] for t in (ROOT1, ROOT2)]
    state, outcomes = receive(ReceiverState(), [p.body for p in sent])
    assert outcomes == [tomb_Outcome.APPLIED_NEW] * 2 and rows_total(state) == 2

    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    bodies = tombstone_bodies(tomb_DEST)
    assert len(bodies) == 2
    state, outcomes = receive(state, bodies)
    assert outcomes == [tomb_Outcome.APPLIED_TOMBSTONE] * 2
    assert rows_total(state) == 0  # current_totals dropped them

    state, replays = receive(state, [p.body for p in sent])
    assert replays == [tomb_Outcome.REJECTED_TOMBSTONED] * 2
    assert rows_total(state) == 0
    state, again = receive(state, bodies)
    assert again == [tomb_Outcome.DUPLICATE_NOOP] * 2  # re-delivery of the tombstone is harmless


def test_tombstone_batches_contain_no_contribution_data() -> None:
    current = make_policy()
    by_root = seed(tomb_DEST)
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    bodies = tombstone_bodies(tomb_DEST)
    assert len(bodies) == 2
    originals = {p.record_id: p for p in by_root.values()}
    for body in bodies:
        assert (body["scenarios"], body["known_costs"], body["coverage"]) == ({}, {}, {})
        assert body["record_type"] == "deletion" and body["revision"] == 2
        assert body["supersedes_revision"] == 1
        original = originals[body["record_id"]]
        assert body["task_id"] == original.task and body["period"] == original.period


def test_second_enqueue_is_a_no_op() -> None:
    current = make_policy()
    seed(tomb_DEST)
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (first,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    registry = load_registry(tomb_PROFILE, tomb_DEST)
    files = sorted(m.batch_id for m in spooled(tomb_DEST))

    later = tomb_datetime(2026, 11, 6, 9, 0, 0, tzinfo=UTC)
    (second,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=later)
    assert (second.queued_records, second.batches, second.batch_ids) == (0, 0, ())
    assert second.skipped_reason is None
    assert sorted(m.batch_id for m in spooled(tomb_DEST)) == files == list(first.batch_ids)
    assert load_registry(tomb_PROFILE, tomb_DEST) == registry
    fresh = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (third,) = enqueue_tombstones(tomb_PROFILE, current, fresh, now=later)  # a fresh plan too
    assert third.batches == 0 and len(spooled(tomb_DEST)) == 1


def test_enqueue_signals_only_the_planned_selection() -> None:
    current = make_policy()
    by_root = seed(tomb_DEST)
    r1 = frozenset({root_key_of(ROOT1.root)})
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=r1)
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert result.queued_records == 1
    (body,) = tombstone_bodies(tomb_DEST)
    assert body["record_id"] == by_root[root_key_of(ROOT1.root)].record_id
    other = load_registry(tomb_PROFILE, tomb_DEST).entries[
        by_root[root_key_of(ROOT2.root)].record_id
    ]
    assert other.queued == ()


def test_destinations_are_enqueued_independently() -> None:
    current = make_policy(enabled=(tomb_DEST,))  # OTHER logs permitted but not switched on
    for destination in (tomb_DEST, OTHER):
        seed(destination)
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    results = {
        r.destination_id: r for r in enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    }
    assert results[tomb_DEST].batches == 1 and results[tomb_DEST].skipped_reason is None
    assert (results[OTHER].batches, results[OTHER].skipped_reason) == (0, "not_enabled")
    assert len(results[OTHER].obligations) == 2
    assert spooled(OTHER) == [] and pending_tombstones(tomb_PROFILE, OTHER) == 0
    assert all(
        not e.queued or e.queued[0].type == "contribution"
        for e in load_registry(tomb_PROFILE, OTHER).entries.values()
    )


# ---- revocation is never bypassed ------------------------------------------------------------


def test_not_permitted_enqueue_writes_nothing_and_reports_obligations(tmp_path: tomb_Path) -> None:
    current = make_policy(enabled=())
    by_root = seed(tomb_DEST)
    before = tomb_tree(data_root(tmp_path))
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert result.skipped_reason == "not_enabled"
    assert (result.queued_records, result.batches, result.batch_ids) == (0, 0, ())
    assert result.obligations == ids_of(by_root, ROOT1, ROOT2) == plans[0].obligations
    assert tomb_tree(data_root(tmp_path)) == before  # no spool, no registry, no delivery document


def test_obligations_are_bounded_to_a_thousand_ids() -> None:
    current = make_policy(enabled=())
    planned = [synthetic(i) for i in range(1500)]
    mark_queued(tomb_PROFILE, tomb_DEST, planned, "b-big-0001")
    mark_acked(tomb_PROFILE, tomb_DEST, "b-big-0001")
    (found,) = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    ordered = tuple(sorted(p.record_id for p in planned))
    assert found.record_count == 1500 and len(found.record_ids) == 1500
    assert MAX_OBLIGATIONS == 1000 and found.obligations == ordered[:1000]
    (result,) = enqueue_tombstones(tomb_PROFILE, current, [found], now=tomb_NOW)
    assert result.obligations == ordered[:1000] and result.batches == 0


def test_permitted_plan_batches_all_records_beyond_the_obligation_bound() -> None:
    current = make_policy()
    planned = [synthetic(i) for i in range(1500)]
    mark_queued(tomb_PROFILE, tomb_DEST, planned, "b-big-0001")
    mark_acked(tomb_PROFILE, tomb_DEST, "b-big-0001")
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    assert plans[0].obligations == ()
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert result.queued_records == 1500 and result.batches >= 1
    assert len({b["record_id"] for b in tombstone_bodies(tomb_DEST)}) == 1500
    assert all(T_ID.fullmatch(b) for b in result.batch_ids)


@tomb_pytest.mark.parametrize(
    ("revoke", "reason"),
    [
        ("disable", "not_enabled"),
        ("remove_destination", "unknown_destination"),
        ("drop_logs", "signal_not_permitted"),
    ],
)
def test_revoked_after_plan_writes_nothing(tmp_path: tomb_Path, revoke: str, reason: str) -> None:
    current = make_policy()
    seed(tomb_DEST)
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    assert plans[0].permitted  # the plan itself was fine
    if revoke == "disable":
        tomb_policy.save_profile(tomb_replace(current, enabled_exports=()))
    elif revoke == "remove_destination":
        tomb_policy.save_profile(
            tomb_replace(
                current,
                destinations=tuple(
                    d for d in current.destinations if d.destination_id != tomb_DEST
                ),
                enabled_exports=(),
            )
        )
    else:
        narrowed = tuple(
            tomb_replace(d, signals=(tomb_Signal.METRICS,), logs_permitted=False)
            for d in current.destinations
        )
        tomb_policy.save_profile(tomb_replace(current, destinations=narrowed, enabled_exports=()))
    before = tomb_tree(data_root(tmp_path))
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert (result.batches, result.queued_records, result.skipped_reason) == (0, 0, reason)
    assert len(result.obligations) == 2
    assert tomb_tree(data_root(tmp_path)) == before


def test_unreadable_policy_at_enqueue_time_blocks_spooling(tmp_path: tomb_Path) -> None:
    current = make_policy()
    seed(tomb_DEST)
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (data_root(tmp_path) / "accounting" / "policy.json").write_text("{broken")
    before = tomb_tree(data_root(tmp_path))
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert (result.skipped_reason, result.batches) == ("policy_unreadable", 0)
    assert tomb_tree(data_root(tmp_path)) == before


def test_missing_pseudonym_key_is_never_minted_by_enqueue(tmp_path: tomb_Path) -> None:
    current = make_policy()
    seed(tomb_DEST)
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    key_file = data_root(tmp_path) / "accounting" / tomb_PROFILE / "export" / "pseudonym.key"
    key_file.unlink()
    before = tomb_tree(data_root(tmp_path))
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert (result.skipped_reason, result.batches) == ("no_export_key", 0)
    assert tomb_tree(data_root(tmp_path)) == before and not key_file.exists()


# ---- acknowledgement bookkeeping -------------------------------------------------------------


def test_forget_removes_exactly_the_acknowledged_deletions() -> None:
    current = make_policy()
    by_root = seed(tomb_DEST)
    unsent = by_root[root_key_of(ROOT3.root)]
    assert forget_acknowledged_deletions(tomb_PROFILE, tomb_DEST) == 0  # nothing deleted yet
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert forget_acknowledged_deletions(tomb_PROFILE, tomb_DEST) == 0  # queued is not accepted
    assert set(load_registry(tomb_PROFILE, tomb_DEST).entries) == {
        p.record_id for p in by_root.values()
    }

    mark_acked(tomb_PROFILE, tomb_DEST, result.batch_ids[0])
    acked = load_registry(tomb_PROFILE, tomb_DEST)
    assert all(
        (state := acked.entries[i].acked) is not None and state.type == "deletion"
        for i in ids_of(by_root, ROOT1, ROOT2)
    )
    assert forget_acknowledged_deletions(tomb_PROFILE, tomb_DEST) == 2
    left = load_registry(tomb_PROFILE, tomb_DEST)
    assert set(left.entries) == {unsent.record_id}  # unrelated records keep their linkage
    assert left.entries[unsent.record_id].queued[0].type == "contribution"
    assert forget_acknowledged_deletions(tomb_PROFILE, tomb_DEST) == 0  # idempotent


def test_forget_keeps_a_record_recreated_after_its_deletion() -> None:
    current = make_policy()
    by_root = seed(tomb_DEST)
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    mark_acked(tomb_PROFILE, tomb_DEST, result.batch_ids[0])
    # changed content re-creates root-1's record above the deletion (adjustment at revision 3)
    more = Ledger(METHOD)
    more.ingest(
        EvidenceBundle(
            (
                candidate("a", task=ROOT1, day=DAY1),
                candidate("z", task=ROOT1, day=DAY1, immediate=(atom("s9", 200, 0, "e9"),)),
            ),
            (),
        )
    )
    candidates, _ = project_candidates(more.rows(), key=key(), destination=tomb_DEST)
    (recreated,) = tomb_plan_revisions(load_registry(tomb_PROFILE, tomb_DEST), candidates)
    assert (recreated.revision, recreated.record_type) == (3, "adjustment")
    mark_queued(tomb_PROFILE, tomb_DEST, [recreated], "b-new-0001")
    assert forget_acknowledged_deletions(tomb_PROFILE, tomb_DEST) == 1  # only root-2's
    kept = load_registry(tomb_PROFILE, tomb_DEST).entries
    assert by_root[root_key_of(ROOT1.root)].record_id in kept
    assert by_root[root_key_of(ROOT2.root)].record_id not in kept


def test_forget_never_creates_a_registry(tmp_path: tomb_Path) -> None:
    assert forget_acknowledged_deletions(tomb_PROFILE, tomb_DEST) == 0
    assert tomb_tree(data_root(tmp_path)) == {}


def test_pending_counts_only_live_tombstone_batches() -> None:
    from ssgrep.accounting.export.spool import update_meta

    current = make_policy()
    seed(tomb_DEST)
    assert pending_tombstones(tomb_PROFILE, tomb_DEST) == 0
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    (batch_id,) = result.batch_ids
    assert pending_tombstones(tomb_PROFILE, tomb_DEST) == 1
    update_meta(
        tomb_PROFILE,
        tomb_DEST,
        tomb_Signal.LOGS,
        batch_id,
        lambda m: tomb_replace(m, state="in_flight"),
    )
    assert pending_tombstones(tomb_PROFILE, tomb_DEST) == 1
    update_meta(
        tomb_PROFILE,
        tomb_DEST,
        tomb_Signal.LOGS,
        batch_id,
        lambda m: tomb_replace(m, state="acked"),
    )
    assert pending_tombstones(tomb_PROFILE, tomb_DEST) == 0


def test_delivery_counts_include_unresolved_attempts() -> None:
    from ssgrep.accounting.export.spool import update_meta

    current = make_policy()
    seed(tomb_DEST)
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (first,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    update_meta(
        tomb_PROFILE,
        tomb_DEST,
        tomb_Signal.LOGS,
        first.batch_ids[0],
        lambda m: tomb_replace(m, attempts=2),
    )
    # a second selection later spools a second tombstone batch and refreshes the counts
    more = Ledger(METHOD)
    more.ingest(
        EvidenceBundle(
            (candidate("y", task=ROOT2, day=DAY3, immediate=(atom("s7", 40, 0, "e7"),)),), ()
        )
    )
    candidates, _ = project_candidates(more.rows(), key=key(), destination=tomb_DEST)
    (follow,) = tomb_plan_revisions(load_registry(tomb_PROFILE, tomb_DEST), candidates)
    mark_queued(tomb_PROFILE, tomb_DEST, [follow], "b-new-0001")
    mark_acked(tomb_PROFILE, tomb_DEST, "b-new-0001")
    again = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (second,) = enqueue_tombstones(
        tomb_PROFILE, current, again, now=tomb_datetime(2026, 11, 7, tzinfo=UTC)
    )
    assert second.batches == 1
    delivery = tomb_acks.read_delivery(tomb_PROFILE, tomb_DEST, tomb_Signal.LOGS)
    assert delivery is not None
    assert (delivery.pending_batches, delivery.unknown_batches) == (2, 1)


# ---- canary ----------------------------------------------------------------------------------


def test_no_root_task_or_path_string_reaches_a_tombstone_batch(tmp_path: tomb_Path) -> None:
    current = make_policy()
    rows = canary_rows()
    candidates, omitted = project_candidates(rows, key=key(), destination=tomb_DEST)
    assert omitted == 0 and tomb_CANARY.encode() in rows.canonical_bytes()
    planned = tomb_plan_revisions(Registry(), candidates)
    mark_queued(tomb_PROFILE, tomb_DEST, planned, "b-sent-0001")
    mark_acked(tomb_PROFILE, tomb_DEST, "b-sent-0001")
    plans = plan_tombstones(tomb_PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(tomb_PROFILE, current, plans, now=tomb_NOW)
    assert result.queued_records == 1
    spool_dir = data_root(tmp_path) / "accounting" / tomb_PROFILE / "spool" / tomb_DEST / "logs"
    files = [p for p in spool_dir.iterdir() if p.name.startswith("t")]
    assert len(files) == 2  # body + meta
    hay = " ".join(p.read_text().lower() for p in files)
    for fragment in (tomb_CANARY, "secret-project", "/home/", "witness", "action:", "exposure"):
        assert fragment not in hay, fragment
    for token in ("origin-", "atom-", "cand-", "debit-", "turn-", "branch-"):
        assert token not in hay, token
    assert tombstone_bodies(tomb_DEST)[0]["scenarios"] == {}


@tomb_pytest.mark.parametrize("damage", [b"{torn", b"[]", b'{"schema_version":999}'])
def test_corrupt_metrics_delivery_cannot_hide_nonreversible_remote_data(
    tmp_path: tomb_Path, damage: bytes
) -> None:
    saved = make_policy()
    assert tombstones.plan_tombstones(tomb_PROFILE, saved, before_day=None, root_keys=None) == []
    plainfile.write_private(
        tomb_storage.data_root(),
        "accounting",
        tomb_PROFILE,
        "delivery",
        tomb_DEST,
        "metrics.json",
        data=damage,
    )
    before = disk_tree(tmp_path / "data")
    plans = tombstones.plan_tombstones(tomb_PROFILE, saved, before_day=None, root_keys=None)
    assert len(plans) == 1
    assert plans[0].destination_id == tomb_DEST
    assert plans[0].non_reversible_metrics and plans[0].record_count == 0
    assert disk_tree(tmp_path / "data") == before


def test_policy_revocation_after_plan_preserves_explicit_remote_deletion_obligations(
    tmp_path: tomb_Path,
) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    export_all(tomb_DEST, ack=True)
    expected = tuple(old_record_ids(tomb_DEST))
    planned = purge.plan(select())
    assert planned.tomb_plans and all(p.permitted for p in planned.tomb_plans)
    tomb_policy.save_profile(tomb_replace(planned.policy, enabled_exports=()))
    original = disk_tree(tmp_path / "data")
    results = tombstones.enqueue_tombstones(
        tomb_PROFILE, planned.policy, planned.tomb_plans, now=SCENE_NOW
    )
    report = purge.report(planned, performed=True, spool_removed=0, results=results)
    assert len(report.tombstones) == 1 and report.tombstones[0].status == "skipped"
    assert len(report.obligations) == 1
    owed = report.obligations[0]
    assert (owed.destination, owed.records, owed.record_ids) == (
        tomb_DEST,
        len(expected),
        expected,
    )
    assert owed.reason == results[0].skipped_reason
    assert disk_tree(tmp_path / "data") == original


# ---- Consolidated from test_delivery_recovery_boundaries.py ----
# Source-specific globals use a prefix.


def configure(lab: recovery_LabReceiver, signal: recovery_Signal, *, records: int = 2) -> None:
    recovery_fx.configure(
        lab.url, signals=(recovery_M, recovery_L), logs_permitted=True, enabled=(signal,)
    )
    if signal is recovery_L:
        recovery_fx.commit_ledger(records)
    recovery_fx.publish(1)


@recovery_pytest.mark.parametrize("signal", [recovery_M, recovery_L])
def test_partial_build_before_registry_commit_is_recreated_before_any_dispatch(
    signal: recovery_Signal, monkeypatch: recovery_pytest.MonkeyPatch
) -> None:
    recovery_fx.split_small(monkeypatch)
    with recovery_LabReceiver.start() as lab:
        configure(lab, signal, records=8)
        real_write = recovery_storage.atomic_write

        def fail_second_meta(dir_fd, name, data):
            if name.endswith(".meta.json") and recovery_json.loads(data)["index"] == 1:
                raise OSError("second batch metadata write interrupted")
            return real_write(dir_fd, name, data)

        with monkeypatch.context() as fault:
            fault.setattr(recovery_storage, "atomic_write", fail_second_meta)
            interrupted = recovery_fx.cycle(signal)
        assert (interrupted.state, interrupted.reason, interrupted.sent) == (
            "failed",
            "storage_error",
            0,
        )
        (first,) = recovery_fx.metas(signal)
        assert first.total > 1 and first.attempts == 0 and first.state == "queued"
        first_body = recovery_fx.body_of(first)
        assert lab.requests == [] and recovery_fx.delivery(signal) is None
        if signal is recovery_L:
            assert recovery_fx.registry().entries == {}

        recovered = recovery_fx.cycle(signal, seconds=1)
        assert recovered.enqueued_batches == recovered.acked == first.total
        assert len(lab.requests) == first.total
        assert lab.requests[0].body == first_body
        assert [request.body for request in lab.requests] == [
            recovery_fx.body_of(meta) for meta in recovery_fx.metas(signal)
        ]
        assert all(
            meta.state == "acked" and meta.attempts == 1 for meta in recovery_fx.metas(signal)
        )
        assert recovery_fx.state(signal).last_acked_generation == 1
        assert recovery_fx.state(signal).loss_summary == ()
        if signal is recovery_L:
            entries = list(recovery_fx.registry().entries.values())
            assert len(entries) == 8
            assert all(entry.acked and entry.acked.revision == 1 for entry in entries)
            assert all(not entry.queued and entry.unknown is None for entry in entries)
        assert recovery_fx.cycle(signal, seconds=2).sent == 0
        assert len(lab.requests) == first.total


@recovery_pytest.mark.parametrize("gap", ["registry", "registry-entry"])
def test_spooled_logs_with_incomplete_registration_are_rebuilt_identically(
    gap: str, monkeypatch: recovery_pytest.MonkeyPatch
) -> None:
    recovery_fx.split_small(monkeypatch)
    with recovery_LabReceiver.start() as lab:
        configure(lab, recovery_L, records=8)
        seeded = recovery_fx.cycle(recovery_L, deps=recovery_fx.deps(max_batches=0))
        original = recovery_fx.metas(recovery_L)
        assert len(original) > 1 and seeded.enqueued_batches == len(original)
        expected = [recovery_fx.body_of(meta) for meta in original]
        path = recovery_fx.profile_root() / "export" / recovery_DEST / "records.json"
        if gap == "registry":
            path.unlink()
        else:
            document = recovery_json.loads(path.read_bytes())
            document["records"].pop(next(iter(document["records"])))
            path.write_text(recovery_json.dumps(document))

        recovered = recovery_fx.cycle(recovery_L, seconds=1)
        assert recovered.enqueued_batches == recovered.acked == len(original)
        assert [request.body for request in lab.requests] == expected
        assert [meta.batch_id for meta in recovery_fx.metas(recovery_L)] == [
            m.batch_id for m in original
        ]
        assert (
            recovery_fx.state(recovery_L).last_acked_generation == 1
            and recovery_fx.state(recovery_L).loss_summary == ()
        )
        entries = list(recovery_fx.registry().entries.values())
        assert len(entries) == 8
        assert all(entry.acked and entry.acked.revision == 1 for entry in entries)
        assert all(not entry.queued and entry.unknown is None for entry in entries)


def test_attempted_incomplete_generation_is_not_rebuilt_as_a_new_request(
    monkeypatch: recovery_pytest.MonkeyPatch,
) -> None:
    recovery_fx.split_small(monkeypatch)
    with recovery_LabReceiver.start() as lab:
        configure(lab, recovery_M)
        lab.queue(recovery_Respond(503))
        assert recovery_fx.cycle().sent == 1
        original = recovery_fx.metas()
        assert len(original) > 1 and original[0].attempts == 1
        missing = original[-1]
        assert recovery_export_spool.delete_batch(
            recovery_PROFILE, recovery_DEST, recovery_M, missing.batch_id
        )
        surviving = [recovery_fx.body_of(meta) for meta in original[:-1]]

        recovered = recovery_fx.cycle(seconds=400)
        assert recovered.enqueued_batches == 0
        assert recovered.acked == len(surviving)
        assert [request.body for request in lab.requests[1:]] == surviving
        assert recovery_fx.state().last_acked_generation is None
        (progress,) = recovery_fx.state().progress
        assert progress.total == len(original) and missing.index not in progress.acked
        assert recovery_fx.cycle(seconds=800).sent == 0
        assert len(lab.requests) == 1 + len(surviving)


@recovery_pytest.mark.parametrize("signal", [recovery_M, recovery_L])
@recovery_pytest.mark.parametrize("boundary", ["before-attempt", "after-ack"])
def test_durable_write_failure_recovers_without_losing_or_changing_request_bytes(
    signal: recovery_Signal, boundary: str, monkeypatch: recovery_pytest.MonkeyPatch
) -> None:
    with recovery_LabReceiver.start() as lab:
        configure(lab, signal)
        real_write = recovery_storage.atomic_write
        failures = []

        def interrupted_write(dir_fd, name, data):
            if name == f"{signal.value}.json":
                fail_here = boundary == "before-attempt" and recovery_json.loads(data).get(
                    "in_flight"
                )
            elif name.endswith(".meta.json"):
                fail_here = (
                    boundary == "after-ack" and recovery_json.loads(data).get("state") == "acked"
                )
            else:
                fail_here = False
            if fail_here and not failures:
                failures.append(name)
                raise OSError("WRITE-FAILURE-CANARY")
            return real_write(dir_fd, name, data)

        with monkeypatch.context() as fault:
            fault.setattr(recovery_storage, "atomic_write", interrupted_write)
            interrupted = recovery_fx.cycle(signal)
        assert (interrupted.state, interrupted.reason) == ("failed", "storage_error")
        assert len(failures) == 1
        (pending,) = recovery_fx.metas(signal)
        assert pending.state == "in_flight" and pending.attempts == 1
        expected_body = recovery_fx.body_of(pending)
        state = recovery_fx.state(signal)
        if boundary == "before-attempt":
            assert lab.requests == []
            assert state.last_attempted_generation is None and state.last_acked_generation is None
        else:
            assert [request.body for request in lab.requests] == [expected_body]
            assert state.last_acked_generation == 1
        assert "WRITE-FAILURE-CANARY" not in repr(interrupted) + recovery_fx.everything_on_disk()

        restarted = recovery_fx.cycle(signal, seconds=1)
        assert (restarted.sent, restarted.acked) == (1, 1)
        count = 1 if boundary == "before-attempt" else 2
        assert [request.body for request in lab.requests] == [expected_body] * count
        (settled,) = recovery_fx.metas(signal)
        assert settled.state == "acked" and settled.attempts == 2
        final = recovery_fx.state(signal)
        assert final.last_acked_snapshot_id == pending.snapshot_id
        assert (final.pending_batches, final.unknown_batches, final.in_flight) == (0, 0, None)
        if signal is recovery_L:
            assert all(
                entry.acked and entry.unknown is None
                for entry in recovery_fx.registry().entries.values()
            )
        assert recovery_fx.cycle(signal, seconds=2).sent == 0 and len(lab.requests) == count


def test_corrupt_queued_body_is_quarantined_without_poisoning_a_later_snapshot() -> None:
    with recovery_LabReceiver.start() as lab:
        configure(lab, recovery_M)
        assert recovery_fx.cycle(deps=recovery_fx.deps(max_batches=0)).enqueued_batches == 1
        (original,) = recovery_fx.metas()
        body_path = (
            recovery_fx.profile_root()
            / "spool"
            / recovery_DEST
            / recovery_M.value
            / f"{original.batch_id}.json"
        )
        body_path.write_bytes(b"TAMPERED-BODY-CANARY")
        rejected = recovery_fx.cycle(seconds=1)
        assert rejected.sent == 0 and rejected.quarantined == 1
        assert lab.requests == []
        (quarantined,) = recovery_fx.metas()
        assert quarantined.state == "quarantined" and quarantined.attempts == 0
        assert quarantined.last_category == "spool_corrupt"
        assert recovery_fx.state().last_acked_generation is None
        assert recovery_fx.state().progress[0].quarantined == (0,)

        fresh_id = recovery_fx.publish(2)
        assert recovery_fx.cycle(seconds=2).acked == 1
        fresh = next(meta for meta in recovery_fx.metas() if meta.generation == 2)
        assert [request.body for request in lab.requests] == [recovery_fx.body_of(fresh)]
        assert b"TAMPERED-BODY-CANARY" not in lab.requests[0].body
        assert recovery_fx.state().last_acked_snapshot_id == fresh_id
        assert recovery_fx.state().quarantined_batches == 1
        assert recovery_fx.cycle(seconds=3).sent == 0 and len(lab.requests) == 1


def test_batch_removed_between_listing_and_body_read_is_not_dispatched_or_acknowledged(
    monkeypatch: recovery_pytest.MonkeyPatch,
) -> None:
    with recovery_LabReceiver.start() as lab:
        configure(lab, recovery_M)
        assert recovery_fx.cycle(deps=recovery_fx.deps(max_batches=0)).enqueued_batches == 1
        (original,) = recovery_fx.metas()
        expected_body = recovery_fx.body_of(original)
        real_read = recovery_export_spool.read_body
        removed = []

        def concurrent_removal(profile, destination, signal, batch_id):
            if not removed:
                removed.append(batch_id)
                recovery_export_spool.delete_batch(profile, destination, signal, batch_id)
            return real_read(profile, destination, signal, batch_id)

        with monkeypatch.context() as race:
            race.setattr(recovery_export_spool, "read_body", concurrent_removal)
            vanished = recovery_fx.cycle(seconds=1)
        assert removed == [original.batch_id]
        assert vanished.sent == 0 and vanished.acked == 0 and lab.requests == []
        assert recovery_fx.metas() == [] and recovery_fx.state().last_acked_generation is None
        assert recovery_fx.cycle(seconds=2).acked == 1
        assert [request.body for request in lab.requests] == [expected_body]


@recovery_pytest.mark.parametrize("missing", ["body", "meta", "both"])
@recovery_pytest.mark.parametrize("new_snapshot", [False, True])
def test_missing_registered_batch_blocks_rebuild_and_false_ack_until_original_files_restored(
    missing: str, new_snapshot: bool, monkeypatch: recovery_pytest.MonkeyPatch
) -> None:
    recovery_fx.split_small(monkeypatch)
    with recovery_LabReceiver.start() as lab:
        configure(lab, recovery_L, records=8)
        assert (
            recovery_fx.cycle(recovery_L, deps=recovery_fx.deps(max_batches=0)).enqueued_batches > 1
        )
        original = recovery_fx.metas(recovery_L)
        expected_bodies = [recovery_fx.body_of(meta) for meta in original]
        lost = original[-1]
        folder = recovery_fx.profile_root() / "spool" / recovery_DEST / recovery_L.value
        body_path = folder / f"{lost.batch_id}.json"
        meta_path = folder / f"{lost.batch_id}.meta.json"
        removed = {}
        for path in (
            (body_path,)
            if missing == "body"
            else (meta_path,)
            if missing == "meta"
            else (body_path, meta_path)
        ):
            removed[path] = path.read_bytes()
            path.unlink()
        registry_path = recovery_fx.profile_root() / "export" / recovery_DEST / "records.json"
        registry_before = registry_path.read_bytes()
        if new_snapshot:
            recovery_fx.publish(2)

        for second in (1, 2):
            refused = recovery_fx.cycle(recovery_L, seconds=second, enqueue=second == 1)
            assert (refused.state, refused.reason, refused.sent, refused.acked) == (
                "failed",
                "spool_corrupt",
                0,
                0,
            )
            assert refused.enqueued_batches == 0
            assert registry_path.read_bytes() == registry_before
            assert recovery_fx.state(recovery_L).last_acked_generation is None
            assert recovery_fx.state(recovery_L).last_collected_generation == 1
            assert all(meta.attempts == 0 for meta in recovery_fx.metas(recovery_L))
            assert lab.requests == []
        # Do not fabricate the lost record's attempt history or reuse its revision. Restoring
        # the original immutable pair is a positive recovery control for the entire generation.
        for path, content in removed.items():
            path.write_bytes(content)
            path.chmod(0o600)
        recovered = recovery_fx.cycle(recovery_L, seconds=3)
        assert recovered.acked == len(original)
        assert [request.body for request in lab.requests] == expected_bodies
        assert recovery_fx.state(recovery_L).last_acked_generation == (2 if new_snapshot else 1)
        assert len(recovery_fx.registry().entries) == 8
        assert all(
            entry.acked and not entry.queued and entry.unknown is None
            for entry in recovery_fx.registry().entries.values()
        )
        assert recovery_fx.cycle(recovery_L, seconds=4).sent == 0
        assert len(lab.requests) == len(original)
