"""Durable delivery cycle against the deterministic loopback lab receiver (real sockets/files).

Every assertion is on state and bytes: the receiver's recorded requests (bodies, sha256, headers),
the spool metas, the delivery document and the registry. Policy, snapshots and the ledger are the
real writers' output (see ``delivery_fixtures``); the only injected seams are ``Deps.environ`` for
the credential, a deterministic jitter and the cycle clock.
"""

from __future__ import annotations

import errno
import json
from dataclasses import replace
from datetime import datetime
from http.client import HTTPResponse
from pathlib import Path
from typing import Any, Literal

import pytest

from ssgrep.accounting import (
    acks,
    delivery,
    delivery_candidate,
    export_consent,
    export_files,
    export_http,
    export_ids,
    export_policy,
    export_spool,
    otlp,
    policy,
    snapshots,
    storage,
)
from ssgrep.accounting.delivery import CycleResult, ExportPaused, check
from ssgrep.accounting.ledger_contracts import Signal
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import DEST, PROFILE, SECRET, L, M, at, cycle, deps
from tests.accounting.lab_receiver import (
    DropAfterRead,
    Hang,
    Hold,
    LabReceiver,
    Oversize,
    Partial,
    Redirect,
    Respond,
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
    from tests.accounting.lab_receiver import ProxyLab

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
    from ssgrep.accounting import policy

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

    from ssgrep.accounting import policy, tombstones

    current = policy.load_profile_policy(PROFILE)
    plans = tombstones.plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    out = tombstones.enqueue_tombstones(
        PROFILE, current, plans, now=datetime(2026, 10, 14, tzinfo=UTC)
    )
    return [bid for result in out for bid in result.batch_ids]


def test_tombstones_follow_data_batches_and_never_touch_generation_progress(
    lab: LabReceiver,
) -> None:
    from ssgrep.accounting import tombstones

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
    from ssgrep.accounting.delivery import export_callback

    setup_metrics(lab)
    assert export_callback(PROFILE, deps=deps())() is None
    assert len(lab.requests) == 1 and fx.state().last_acked_generation == 1
    assert not (fx.profile_root() / "export" / "last_error.json").exists()


def test_export_callback_records_only_a_class_name_when_everything_explodes(
    lab: LabReceiver, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting import delivery
    from ssgrep.accounting.delivery import export_callback

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
    from ssgrep.accounting.delivery import run_export

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
    from ssgrep.accounting.delivery import run_export
    from ssgrep.accounting.ledger_contracts import DestinationPolicy

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
        with export_files.locked(PROFILE, ("export", DEST), "send-metrics"):
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
                real_split = otlp.split_metrics
                fault.setattr(otlp, "split_metrics", lambda r, p: real_split(r, p, max_bytes=1))
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
