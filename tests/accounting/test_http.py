"""Loopback server over REAL sockets: auth, Host, CORS, protocol bounds, snapshot/ack views."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from prometheus_client.openmetrics.parser import text_string_to_metric_families

from ssgrep.accounting import http, http_auth, snapshots
from ssgrep.accounting.metrics_render import CONTENT_TYPE
from tests.accounting.http_builders import (
    COLLECTED_EPOCH,
    PROFILE,
    get,
    make_snapshot,
    publish,
    raw_exchange,
    start,
    token_for,
    tree_state,
    window_with_unknown_optimistic,
    write_delivery,
)
from tests.accounting.snapshot_builders import enable_policy

FRESH_NOW = datetime(2026, 10, 12, 12, 0, 30, tzinfo=UTC)
STALE_NOW = datetime(2026, 10, 12, 12, 10, 0, tzinfo=UTC)


@pytest.fixture
def server():
    handle = start(clock=lambda: FRESH_NOW)
    yield handle
    handle.stop()


def data_root() -> Path:
    return Path(os.environ["SSGREP_DATA_DIR"])


# ---- binding / profile isolation ------------------------------------------------------------


@pytest.mark.parametrize("bind", ["0.0.0.0", "::", "192.168.1.5", "example.com", "", "localhost"])
def test_non_loopback_binds_are_refused_before_anything_is_created(bind: str) -> None:
    with pytest.raises(http.BindRefused):
        http.serve(PROFILE, port=0, bind=bind)
    assert not (data_root() / "accounting" / PROFILE / http_auth.TOKEN_FILE).exists()


def test_server_listens_only_on_loopback_and_reports_its_real_port(
    server: http.ServerHandle,
) -> None:
    assert server.port != 0
    with socket.create_connection(("127.0.0.1", server.port), timeout=2):
        pass
    assert server.binds == ("127.0.0.1",)
    assert {s.getsockname()[0] for s in server._listeners} == {"127.0.0.1"}


def test_ipv6_loopback_is_optional_and_bound_to_the_same_port() -> None:
    try:
        handle = http.serve(PROFILE, port=0, bind=("127.0.0.1", "::1"))
    except http.PortInUse:
        pytest.skip("IPv6 loopback is unavailable on this host")
    with handle:
        with socket.create_connection(("::1", handle.port), timeout=2):
            pass
        assert {s.getsockname()[0] for s in handle._listeners} == {"127.0.0.1", "::1"}


def test_port_conflict_is_an_error_not_a_reassignment(server: http.ServerHandle) -> None:
    with pytest.raises(http.PortInUse):
        http.serve(PROFILE, port=server.port)
    assert get(server, "/healthz", token=False).status == 200  # the original is undisturbed


def test_default_port_is_9469() -> None:
    import inspect

    assert inspect.signature(http.serve).parameters["port"].default == 9469


def test_invalid_profile_ids_never_reach_the_filesystem() -> None:
    for bad in ("../x", "a/b", "A", "", "x" * 40):
        with pytest.raises(ValueError):
            http.serve(bad, port=0)
    assert not (data_root() / "accounting").exists() or not any(
        (data_root() / "accounting").iterdir()
    )


def test_token_is_created_owner_only_and_is_not_the_fingerprint_key() -> None:
    with start() as handle:
        token_path = data_root() / "accounting" / PROFILE / http_auth.TOKEN_FILE
        assert oct(token_path.stat().st_mode & 0o777) == "0o600"
        assert token_path.name != "fingerprint.key"
        assert len(token_for()) == 64 and handle.port


def test_unsafe_token_file_prevents_start_and_authenticates_nobody() -> None:
    with start():
        pass
    token_path = data_root() / "accounting" / PROFILE / http_auth.TOKEN_FILE
    token_path.chmod(0o644)
    with pytest.raises(http_auth.TokenFileError):
        http.serve(PROFILE, port=0)


def test_cross_profile_token_is_rejected_and_each_profile_has_its_own_state() -> None:
    with start("personal-local") as one, start("acme-org") as two:
        publish("personal-local", make_snapshot(5))
        publish("acme-org", make_snapshot(9, methodology_id="m-acme-v1"))
        assert token_for("personal-local") != token_for("acme-org")
        mine = get(one, "/v1/snapshot")
        assert mine.status == 200 and mine.json()["generation"] == 5
        theirs = get(two, "/v1/snapshot")
        assert theirs.json()["generation"] == 9
        cross = get(one, "/v1/snapshot", token=token_for("acme-org"))
        assert cross.status == 401 and b"generation" not in cross.body
        assert get(two, "/metrics", token=token_for("personal-local")).status == 401


# ---- authentication / Host / CORS ------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/metrics",
        "/v1/snapshot",
        "/v1/export-status",
        "/v1/snapshot?view=acknowledged&destination=acme",
    ],
)
def test_data_endpoints_require_a_valid_bearer_and_reveal_nothing(
    server: http.ServerHandle, path: str
) -> None:
    publish(PROFILE, make_snapshot(3))
    for token in (False, "0" * 64, "x", token_for() + "0"):
        reply = get(server, path, token=token)
        assert reply.status == 401
        assert reply.json() == {"error": "unauthorized"}
        assert reply.headers["www-authenticate"] == "Bearer"
    sent = get(server, path, headers={"Authorization": f"Basic {token_for()}"}, token=False)
    assert sent.status == 401
    assert get(server, path).status in (200, 404, 410)  # authenticated, whatever the view holds


def test_unauthenticated_existence_of_a_specific_receipt_or_ack_is_not_revealed(
    server: http.ServerHandle,
) -> None:
    write_delivery(
        PROFILE, "acme", "metrics", last_acked_snapshot_id="a" * 16, last_acked_generation=1
    )
    exists = get(server, "/v1/snapshot?view=acknowledged&destination=acme", token=False)
    missing = get(server, "/v1/snapshot?view=acknowledged&destination=nope", token=False)
    assert (exists.status, exists.body) == (missing.status, missing.body) == (401, exists.body)


def test_token_in_url_is_rejected_even_when_correct_and_never_echoed(
    server: http.ServerHandle,
) -> None:
    token = token_for()
    for path in (
        f"/metrics?token={token}",
        f"/v1/snapshot?access_token={token}",
        f"/v1/snapshot?api_key={token}",
    ):
        reply = get(server, path, token=True)
        assert reply.status == 400 and reply.json() == {"error": "credential_in_url"}
        assert token.encode() not in reply.body
        assert get(server, path, token=False).status == 400


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "127.0.0.1.evil.example",
        "127.0.0.1:1",
        "localhost:1",
        "[::1]:1",
        "0.0.0.0",
        "",
    ],
)
def test_unapproved_host_is_rejected_before_authentication(
    server: http.ServerHandle, host: str
) -> None:
    payload = f"GET /healthz HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode()
    answer = raw_exchange(server.port, payload)
    assert answer.startswith(b"HTTP/1.1 421 ")
    assert b"host_not_allowed" in answer


def test_missing_host_header_is_rejected(server: http.ServerHandle) -> None:
    answer = raw_exchange(server.port, b"GET /healthz HTTP/1.1\r\n\r\n")
    assert answer.startswith(b"HTTP/1.1 421 ")


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost"])
def test_approved_loopback_host_names_work_with_the_bound_port(
    server: http.ServerHandle, name: str
) -> None:
    reply = get(server, "/healthz", token=False, headers={"Host": f"{name}:{server.port}"})
    assert reply.status == 200


def test_no_cors_headers_ever_and_preflight_gets_no_grant(server: http.ServerHandle) -> None:
    publish(PROFILE, make_snapshot(3))
    preflight = get(
        server,
        "/v1/snapshot",
        method="OPTIONS",
        token=False,
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert preflight.status == 405
    cross_origin = [
        get(server, "/v1/snapshot", headers={"Origin": "https://evil.example"}),
        get(server, "/metrics", headers={"Origin": "https://evil.example"}),
        get(server, "/healthz", token=False, headers={"Origin": "null"}),
        get(server, "/v1/snapshot", token=False, headers={"Origin": "https://evil.example"}),
        preflight,
    ]
    for reply in cross_origin:
        assert not [h for h in reply.headers if h.startswith("access-control-")]
        assert "vary" not in reply.headers or "origin" not in reply.headers["vary"].lower()


# ---- liveness --------------------------------------------------------------------------------


def test_healthz_is_content_free_and_independent_of_snapshot_age(server: http.ServerHandle) -> None:
    before = get(server, "/healthz", token=False)
    publish(PROFILE, make_snapshot(3, collected_at="2020-01-01T00:00:00Z"))
    after = get(server, "/healthz", token=False)
    assert before.status == after.status == 200
    assert before.body == after.body == b"ok\n"
    for reply in (before, after):
        assert "server" not in reply.headers and not any(
            h.startswith("x-ssgrep") for h in reply.headers
        )
        for leaked in (b"version", str(data_root()).encode(), b"personal", b"generation"):
            assert leaked not in reply.body


# ---- protocol bounds -------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_mutation_methods_are_rejected_without_side_effects(
    server: http.ServerHandle, method: str
) -> None:
    publish(PROFILE, make_snapshot(3))
    write_delivery(PROFILE, "acme", "metrics", pending_batches=1)
    before = tree_state(data_root())
    for path in ("/v1/snapshot", "/metrics", "/v1/export-status", "/healthz"):
        reply = get(server, path, method=method)
        assert reply.status == 405 and reply.headers["allow"] == "GET, HEAD"
    assert tree_state(data_root()) == before


def test_request_bodies_are_refused_and_never_parsed_as_a_second_request(
    server: http.ServerHandle,
) -> None:
    smuggled = (
        f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\nContent-Length: 40\r\n\r\n"
        "GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n"
    ).encode()
    answer = raw_exchange(server.port, smuggled)
    assert answer.startswith(b"HTTP/1.1 400 ") and answer.count(b"HTTP/1.1 ") == 1
    chunked = (
        f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
        "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
    ).encode()
    assert raw_exchange(server.port, chunked).startswith(b"HTTP/1.1 400 ")
    both = (
        f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
        "Content-Length: 4\r\nContent-Length: 5\r\n\r\nabcd"
    ).encode()
    assert raw_exchange(server.port, both).startswith(b"HTTP/1.1 400 ")


def test_pipelined_second_request_is_not_served(server: http.ServerHandle) -> None:
    one = f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n\r\n"
    answer = raw_exchange(server.port, (one + one).encode())
    assert answer.count(b"HTTP/1.1 200") == 1
    assert b"Connection: close" in answer


def test_oversized_headers_get_431_without_reading_forever(server: http.ServerHandle) -> None:
    padding = "a" * 9000
    payload = (
        f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\nX-Pad: {padding}\r\n\r\n"
    ).encode()
    assert raw_exchange(server.port, payload).startswith(b"HTTP/1.1 431 ")
    no_terminator = b"GET /healthz HTTP/1.1\r\nX-Pad: " + b"b" * 20000
    started = time.monotonic()
    assert raw_exchange(server.port, no_terminator).startswith(b"HTTP/1.1 431 ")
    assert time.monotonic() - started < 2


def test_header_block_just_under_8_kib_is_accepted(server: http.ServerHandle) -> None:
    head = f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\nX-Pad: "
    pad = "a" * (8 * 1024 - len(head) - 8)
    answer = raw_exchange(server.port, f"{head}{pad}\r\n\r\n".encode())
    assert answer.startswith(b"HTTP/1.1 200 ")


def test_slow_client_is_cut_off_after_five_seconds_and_good_reads_continue(
    server: http.ServerHandle,
) -> None:
    publish(PROFILE, make_snapshot(3))
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as slow:
        slow.sendall(b"GET /healthz HT")
        started = time.monotonic()
        # a drip that would re-arm a naive per-recv timeout
        for _ in range(3):
            time.sleep(1.2)
            slow.sendall(b"T")
        answer = slow.recv(4096)
        elapsed = time.monotonic() - started
        assert answer.startswith(b"HTTP/1.1 408 ") or answer == b""
        assert 4.0 <= elapsed <= 7.5
        assert get(server, "/v1/snapshot").json()["generation"] == 3  # last snapshot still served


def test_seventeenth_concurrent_connection_is_rejected_and_the_first_sixteen_still_work(
    server: http.ServerHandle,
) -> None:
    held = [socket.create_connection(("127.0.0.1", server.port), timeout=10) for _ in range(16)]
    try:
        time.sleep(0.4)
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as extra:
            answer = extra.recv(4096)
        assert answer.startswith(b"HTTP/1.1 503 ") and b"too_many_connections" in answer
        held[0].sendall(f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n\r\n".encode())
        assert held[0].recv(4096).startswith(b"HTTP/1.1 200 ")
    finally:
        for sock in held:
            sock.close()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and get(server, "/healthz", token=False).status != 200:
        time.sleep(0.1)  # slots free up after the held clients leave
    assert get(server, "/healthz", token=False).status == 200


@pytest.mark.parametrize(
    "target",
    [
        "/../../etc/passwd",
        "/v1/snapshot?path=/etc/passwd",
        "/v1/snapshot?file=../../x",
        "/v1/snapshot?view=acknowledged&destination=../../../etc&signal=metrics",
        "/v1/snapshot?view=acknowledged&destination=acme/../x",
        "/%2e%2e/%2e%2e/etc/passwd",
        "/v1/snapshot?view=../current",
        "/metrics?path=/tmp",
        "/v1/export-status?dir=/",
        "/v1/snapshot?signal=../logs",
        "//etc/passwd",
        "/v1/snapshot%00",
        "/v1/snapshot?destination=a&destination=b",
    ],
)
def test_traversal_shaped_requests_never_touch_files_or_expose_content(
    server: http.ServerHandle, target: str
) -> None:
    publish(PROFILE, make_snapshot(3))
    before = tree_state(data_root())
    reply = get(server, target)
    assert reply.status in (400, 404)
    assert b"root:" not in reply.body and b"snapshot_id" not in reply.body
    assert target.encode() not in reply.body  # untrusted input is never echoed
    assert tree_state(data_root()) == before


def test_http_1_0_and_garbage_request_lines_are_rejected(server: http.ServerHandle) -> None:
    for payload in (
        b"\x00\x01\x02\r\n\r\n",
        b"GET\r\n\r\n",
        b"GET / HTTP/9.9\r\n\r\n",
        b"GET /healthz HTTP/1.1\r\nbad header\r\n\r\n",
        b"get /healthz HTTP/1.1\r\n\r\n",
    ):
        assert raw_exchange(server.port, payload).startswith(b"HTTP/1.1 400 ")


def test_reads_never_create_collect_or_modify_anything(server: http.ServerHandle) -> None:
    publish(PROFILE, make_snapshot(3))
    write_delivery(PROFILE, "acme", "metrics", pending_batches=1)
    before = tree_state(data_root())
    for path in (
        "/healthz",
        "/metrics",
        "/v1/snapshot",
        "/v1/export-status",
        "/v1/snapshot?view=acknowledged&destination=acme",
    ):
        get(server, path, token=path != "/healthz")
        get(server, path, method="HEAD", token=path != "/healthz")
    assert tree_state(data_root()) == before


# ---- snapshot reads --------------------------------------------------------------------------


def test_snapshot_before_first_publish_is_503_and_health_stays_live(
    server: http.ServerHandle,
) -> None:
    reply = get(server, "/v1/snapshot")
    assert reply.status == 503 and reply.json() == {"error": "snapshot_unavailable"}
    assert get(server, "/healthz", token=False).status == 200


def test_snapshot_body_etag_and_conditional_304(server: http.ServerHandle) -> None:
    envelope = make_snapshot(3)
    publish(PROFILE, envelope)
    reply = get(server, "/v1/snapshot")
    assert reply.status == 200 and reply.headers["content-type"].startswith("application/json")
    document = reply.json()
    assert document["snapshot_id"] == envelope.snapshot_id and document["generation"] == 3
    assert reply.headers["etag"] == f'"{envelope.snapshot_id}"'
    assert reply.headers["x-ssgrep-state"] == "fresh"
    before = tree_state(data_root())
    cached = get(server, "/v1/snapshot", headers={"If-None-Match": reply.headers["etag"]})
    assert cached.status == 304 and cached.body == b"" and "content-length" not in cached.headers
    assert cached.headers["etag"] == reply.headers["etag"]
    assert (
        get(server, "/v1/snapshot", headers={"If-None-Match": f"W/{reply.headers['etag']}"}).status
        == 304
    )
    assert (
        get(server, "/v1/snapshot", headers={"If-None-Match": '"deadbeefdeadbeef"'}).status == 200
    )
    assert tree_state(data_root()) == before
    publish(PROFILE, make_snapshot(4, extra_searches=1))
    assert (
        get(server, "/v1/snapshot", headers={"If-None-Match": reply.headers["etag"]}).status == 200
    )


def test_head_has_get_auth_and_headers_but_no_body(server: http.ServerHandle) -> None:
    publish(PROFILE, make_snapshot(3))
    full = get(server, "/v1/snapshot")
    head = get(server, "/v1/snapshot", method="HEAD")
    assert head.status == 200 and head.body == b""
    assert head.headers["etag"] == full.headers["etag"]
    assert head.headers["content-length"] == str(len(full.body))
    assert head.headers["x-ssgrep-state"] == full.headers["x-ssgrep-state"]
    denied = get(server, "/v1/snapshot", method="HEAD", token=False)
    assert denied.status == 401 and denied.body == b""
    metrics_head = get(server, "/metrics", method="HEAD")
    assert metrics_head.status == 200 and metrics_head.body == b""
    assert metrics_head.headers["content-type"] == CONTENT_TYPE


def test_stale_snapshot_is_still_served_with_freshness_state_and_growing_age() -> None:
    clock = [FRESH_NOW]
    with start(clock=lambda: clock[0]) as handle:
        publish(PROFILE, make_snapshot(3))
        fresh = get(handle, "/v1/snapshot")
        assert (
            fresh.headers["x-ssgrep-state"] == "fresh"
            and fresh.headers["x-ssgrep-age-seconds"] == "30"
        )
        clock[0] = STALE_NOW
        stale = get(handle, "/v1/snapshot")
        assert stale.status == 200 and stale.json()["generation"] == 3
        assert stale.headers["x-ssgrep-state"] == "stale"
        assert stale.headers["x-ssgrep-age-seconds"] == "600"
        assert stale.headers["x-ssgrep-stale-after-seconds"] == "180"
        get(handle, "/v1/snapshot")
        clock[0] = datetime(2026, 10, 12, 12, 20, 0, tzinfo=UTC)
        assert (
            get(handle, "/v1/snapshot").headers["x-ssgrep-age-seconds"] == "1200"
        )  # reads do not reset


def test_paused_state_is_separate_from_stale_and_age_keeps_growing() -> None:
    enable_policy(PROFILE, paused=True)
    clock = [STALE_NOW]
    with start(clock=lambda: clock[0]) as handle:
        publish(PROFILE, make_snapshot(3))
        reply = get(handle, "/v1/snapshot")
        assert reply.status == 200
        assert (
            reply.headers["x-ssgrep-paused"] == "true"
            and reply.headers["x-ssgrep-state"] == "stale"
        )
        clock[0] = datetime(2026, 10, 12, 13, 0, 0, tzinfo=UTC)
        assert int(get(handle, "/v1/snapshot").headers["x-ssgrep-age-seconds"]) == 3600
        assert get(handle, "/healthz", token=False).status == 200
    enable_policy(PROFILE)


def test_unpaused_fresh_snapshot_is_not_marked_paused(server: http.ServerHandle) -> None:
    publish(PROFILE, make_snapshot(3))
    assert get(server, "/v1/snapshot").headers["x-ssgrep-paused"] == "false"


def test_reads_during_an_incomplete_generation_return_the_previous_complete_one(
    server: http.ServerHandle,
) -> None:
    publish(PROFILE, make_snapshot(3))
    first = get(server, "/v1/snapshot").json()["snapshot_id"]
    snapshots.stage_generation(PROFILE, make_snapshot(4, extra_searches=5))  # staged, not current
    during = get(server, "/v1/snapshot").json()
    assert (during["generation"], during["snapshot_id"]) == (3, first)
    snapshots.make_current(PROFILE, 4)
    assert get(server, "/v1/snapshot").json()["generation"] == 4


def test_corrupt_current_snapshot_is_unavailable_not_a_crash(server: http.ServerHandle) -> None:
    publish(PROFILE, make_snapshot(3))
    current = data_root() / "accounting" / PROFILE / "snapshots" / "current.json"
    current.write_bytes(b'{"schema_version": 1, "oops"')
    reply = get(server, "/v1/snapshot")
    assert reply.status == 503 and reply.json() == {"error": "snapshot_unreadable"}
    assert get(server, "/healthz", token=False).status == 200


# ---- /metrics --------------------------------------------------------------------------------


def parsed(reply) -> dict:
    return {f.name: f for f in text_string_to_metric_families(reply.body.decode())}


def test_metrics_is_valid_openmetrics_with_content_type_and_eof(server: http.ServerHandle) -> None:
    publish(PROFILE, make_snapshot(3))
    reply = get(server, "/metrics")
    assert reply.status == 200 and reply.headers["content-type"] == CONTENT_TYPE
    assert reply.body.endswith(b"# EOF\n")
    families = parsed(reply)
    executions = {
        tuple(sorted(s.labels.items())): s.value
        for s in families["ssgrep_executions"].samples
        if s.name.endswith("_total")
    }
    assert sum(executions.values()) == 4
    assert reply.headers["x-ssgrep-state"] == "fresh"


def test_metrics_before_first_snapshot_exposes_only_availability_and_exporter_health(
    server: http.ServerHandle,
) -> None:
    write_delivery(PROFILE, "acme", "metrics", pending_batches=2, rejected_points=3)
    families = parsed(get(server, "/metrics"))
    assert set(families) == {
        "ssgrep_snapshot_available",
        "ssgrep_export_pending_batches",
        "ssgrep_export_rejected_points",
    }
    assert families["ssgrep_snapshot_available"].samples[0].value == 0


def test_stale_metrics_with_unknown_estimate_are_valid_and_untimestamped() -> None:
    with start(clock=lambda: STALE_NOW) as handle:
        publish(PROFILE, make_snapshot(3, window=window_with_unknown_optimistic()))
        reply = get(handle, "/metrics")
        families = parsed(reply)
        gauge = families["ssgrep_snapshot_collected_timestamp_seconds"].samples[0]
        assert gauge.value == COLLECTED_EPOCH
        assert all(s.timestamp is None for f in families.values() for s in f.samples)
        nets = {s.labels["scenario"] for s in families["ssgrep_estimated_net_tokens"].samples}
        assert nets == {"conservative", "central"}
        assert reply.headers["x-ssgrep-state"] == "stale"


def test_metrics_include_exporter_health_from_delivery_state(server: http.ServerHandle) -> None:
    publish(PROFILE, make_snapshot(3))
    write_delivery(
        PROFILE,
        "acme",
        "metrics",
        pending_batches=2,
        rejected_points=5,
        last_acked_at="2026-10-12T11:59:00Z",
        loss_summary=[
            {"reason": "spool_expired", "batches": 1, "records": 7, "at": "2026-10-12T11:00:00Z"}
        ],
    )
    families = parsed(get(server, "/metrics"))
    assert families["ssgrep_export_pending_batches"].samples[0].value == 2
    assert (
        families["ssgrep_export_last_full_ack_timestamp_seconds"].samples[0].value
        == COLLECTED_EPOCH - 60
    )
    lost = [s for s in families["ssgrep_export_lost_records"].samples if s.name.endswith("_total")]
    assert [(s.labels["destination"], s.labels["reason"], s.value) for s in lost] == [
        ("acme", "spool_expired", 7)
    ]


# ---- export status ---------------------------------------------------------------------------


def test_export_status_distinguishes_signals_and_never_leaks_secrets_or_remote_text(
    server: http.ServerHandle,
) -> None:
    write_delivery(
        PROFILE,
        "acme",
        "metrics",
        last_attempted_generation=12,
        last_acked_generation=10,
        last_acked_snapshot_id="c" * 16,
        pending_batches=1,
        rejected_points=4,
        unknown_batches=0,
        partial_batches=1,
        last_error_category="partial_rejection",
        last_attempt_at="2026-10-12T11:58:00Z",
        last_acked_at="2026-10-12T11:50:00Z",
        raw_remote_error="Bearer sk-live-REMOTE-SECRET /home/u/.ssh upstream said: nope",
    )
    write_delivery(
        PROFILE,
        "acme",
        "logs",
        blocked_reason="credentials_expired",
        pending_batches=3,
        last_error_category="Bearer sk-live-REMOTE-SECRET spaces and text",
        in_flight_nonrecallable=False,
        enabled=True,
    )
    reply = get(server, "/v1/export-status")
    assert reply.status == 200
    assert (
        b"REMOTE-SECRET" not in reply.body
        and b".ssh" not in reply.body
        and b"upstream" not in reply.body
    )
    rows = {(d["destination_id"], d["signal"]): d for d in reply.json()["destinations"]}
    metrics_row, logs_row = rows[("acme", "metrics")], rows[("acme", "logs")]
    assert metrics_row["last_full_ack"]["generation"] == 10
    assert metrics_row["last_attempt"]["generation"] == 12
    assert (metrics_row["pending_batches"], metrics_row["rejected_points"]) == (1, 4)
    assert metrics_row["last_error_category"] == "partial_rejection"
    assert logs_row["blocked_reason"] == "credentials_expired"
    assert logs_row["last_full_ack"] == {"at": None, "generation": None, "snapshot_id": None}
    assert logs_row["last_error_category"] is None  # free text dropped, not echoed
    assert "durability_not_claimed" in reply.json()["acknowledgment"]


def test_export_status_with_no_delivery_files_is_an_empty_list(server: http.ServerHandle) -> None:
    reply = get(server, "/v1/export-status")
    assert reply.status == 200 and reply.json()["destinations"] == []


def test_unreadable_delivery_state_is_reported_unknown_not_zero(server: http.ServerHandle) -> None:
    path = write_delivery(PROFILE, "acme", "metrics", pending_batches=1)
    path.write_text("{not json")
    (row,) = get(server, "/v1/export-status").json()["destinations"]
    assert row["state"] == "unreadable" and row["pending_batches"] is None


# ---- acknowledged views ----------------------------------------------------------------------


def seed_generations() -> tuple[str, str, str]:
    ids = []
    for generation in (10, 11, 12):
        envelope = make_snapshot(generation, extra_searches=generation)
        publish(PROFILE, envelope)
        ids.append(envelope.snapshot_id)
    return ids[0], ids[1], ids[2]


def test_default_acknowledged_view_returns_generation_10_while_current_is_12(
    server: http.ServerHandle,
) -> None:
    ten, _, twelve = seed_generations()
    write_delivery(PROFILE, "acme", "metrics", last_acked_generation=10, last_acked_snapshot_id=ten)
    current = get(server, "/v1/snapshot").json()
    acked = get(server, "/v1/snapshot?view=acknowledged&destination=acme")
    assert (current["generation"], current["snapshot_id"]) == (12, twelve)
    document = acked.json()
    assert acked.status == 200 and (document["generation"], document["snapshot_id"]) == (10, ten)
    assert acked.headers["etag"] == f'"{ten}"' and acked.headers["x-ssgrep-view"] == "acknowledged"
    assert acked.headers["x-ssgrep-signal"] == "metrics"
    again = get(
        server,
        "/v1/snapshot?view=acknowledged&destination=acme",
        headers={"If-None-Match": acked.headers["etag"]},
    )
    assert again.status == 304


def test_per_signal_acknowledged_views_are_independent(server: http.ServerHandle) -> None:
    ten, eleven, _ = seed_generations()
    write_delivery(PROFILE, "acme", "metrics", last_acked_generation=10, last_acked_snapshot_id=ten)
    write_delivery(PROFILE, "acme", "logs", last_acked_generation=11, last_acked_snapshot_id=eleven)
    metrics = get(server, "/v1/snapshot?view=acknowledged&destination=acme&signal=metrics").json()
    logs = get(server, "/v1/snapshot?view=acknowledged&destination=acme&signal=logs").json()
    default = get(server, "/v1/snapshot?view=acknowledged&destination=acme").json()
    assert (metrics["generation"], logs["generation"], default["generation"]) == (10, 11, 10)
    assert "records" not in logs and "snapshot_id" in logs  # sanitized aggregate, not log records


def test_never_acknowledged_is_404_not_acknowledged_and_never_current(
    server: http.ServerHandle,
) -> None:
    seed_generations()
    for path in (
        "/v1/snapshot?view=acknowledged&destination=acme",
        "/v1/snapshot?view=acknowledged&destination=acme&signal=logs",
    ):
        reply = get(server, path)
        assert reply.status == 404 and reply.json() == {"error": "not_acknowledged"}
    write_delivery(PROFILE, "acme", "logs", pending_batches=1)  # state without any ack
    assert get(server, "/v1/snapshot?view=acknowledged&destination=acme&signal=logs").status == 404
    write_delivery(
        PROFILE, "acme", "metrics", last_acked_generation=12, last_acked_snapshot_id="d" * 16
    )  # id that matches no retained snapshot
    expired = get(server, "/v1/snapshot?view=acknowledged&destination=acme")
    assert expired.status == 410 and expired.json() == {"error": "acknowledged_snapshot_expired"}


def test_expired_acknowledged_snapshot_is_reported_after_prune_and_not_replaced(
    server: http.ServerHandle,
) -> None:
    ten, _, twelve = seed_generations()
    write_delivery(PROFILE, "acme", "metrics", last_acked_generation=10, last_acked_snapshot_id=ten)
    assert snapshots.prune(PROFILE, keep=1) == [11]  # generation 10 is pinned by the delivery state
    assert get(server, "/v1/snapshot?view=acknowledged&destination=acme").json()["generation"] == 10
    write_delivery(
        PROFILE, "acme", "metrics", last_acked_generation=11, last_acked_snapshot_id="f" * 16
    )
    reply = get(server, "/v1/snapshot?view=acknowledged&destination=acme")
    assert reply.status == 410 and reply.json() == {"error": "acknowledged_snapshot_expired"}
    assert twelve.encode() not in reply.body


@pytest.mark.parametrize(
    "query",
    [
        "view=acknowledged",
        "view=acknowledged&destination=ACME",
        "view=acknowledged&destination=a&signal=traces",
        "view=other",
        "destination=acme",
        "view=acknowledged&destination=a&extra=1",
        "view=",
    ],
)
def test_acknowledged_view_parameter_validation(server: http.ServerHandle, query: str) -> None:
    seed_generations()
    reply = get(server, f"/v1/snapshot?{query}")
    assert reply.status == 400 and reply.json() == {"error": "unsupported_parameter"}


# ---- composition with a worker ---------------------------------------------------------------


class FakeWorker:
    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.ran = threading.Event()

    def run(self) -> str:
        self.ran.set()
        self.stopped.wait(10)
        return "done"

    def stop(self) -> None:
        self.stopped.set()


def test_worker_composition_runs_alongside_the_server_and_stop_stops_both() -> None:
    worker = FakeWorker()
    handle = start(worker=worker)
    result: list[str] = []
    thread = threading.Thread(target=lambda: result.append(handle.run_worker()))
    thread.start()
    assert worker.ran.wait(3)
    assert get(handle, "/healthz", token=False).status == 200
    handle.stop()
    thread.join(5)
    assert result == ["done"] and worker.stopped.is_set()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", handle.port), timeout=1)


def test_stop_is_idempotent_and_releases_the_port() -> None:
    handle = start()
    port = handle.port
    handle.stop()
    handle.stop()
    again = http.serve(PROFILE, port=port)
    again.stop()


def test_token_matches_is_constant_time_and_rejects_prefixes() -> None:
    expected = b"a" * 64
    assert http_auth.token_matches(expected, expected)
    assert not http_auth.token_matches(b"a" * 63, expected)
    assert not http_auth.token_matches(b"a" * 65, expected)
    assert not http_auth.token_matches(None, expected) and not http_auth.token_matches(
        expected, None
    )
    assert http_auth.presented_token("Bearer abc") == b"abc"
    assert http_auth.presented_token("bearer  abc") == b"abc"
    for bad in ("Bearer", "Bearer a b", "Basic abc", "abc", "Bearer " + "x" * 600, ""):
        assert http_auth.presented_token(bad) is None
    assert json.dumps(None) == "null"
