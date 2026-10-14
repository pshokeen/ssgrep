"""Permission failures, real timestamps and transport recovery at read-only endpoints."""

from __future__ import annotations

import errno
import json
import os
import socket
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting import http, http_auth, http_views, http_wire, snapshots, storage
from tests.accounting.http_builders import (
    PROFILE,
    get,
    make_snapshot,
    publish,
    start,
    tree_state,
    write_delivery,
)

NOW = datetime(2026, 10, 12, 12, 0, 30, tzinfo=UTC)


@pytest.mark.parametrize("damage", ["directory_permissions", "file_permissions", "missing"])
def test_current_snapshot_read_failure_never_repairs_and_recovers(tmp_path: Path, damage: str):
    envelope = make_snapshot(3)
    publish(PROFILE, envelope)
    directory = tmp_path / "data" / "accounting" / PROFILE / snapshots.SNAPSHOT_DIR
    target = directory / snapshots.CURRENT_NAME
    saved = target.read_bytes()
    if damage == "directory_permissions":
        directory.chmod(0o777)
    elif damage == "file_permissions":
        target.chmod(0o644)
    else:
        target.unlink()
    before = tree_state(tmp_path / "data")
    current, error = http_views.load_current(PROFILE)
    assert current is None
    assert error == (None if damage == "missing" else "snapshot_unreadable")
    assert tree_state(tmp_path / "data") == before
    directory.chmod(0o700)
    if damage == "missing":
        target.write_bytes(saved)
    target.chmod(0o600)
    recovered, error = http_views.load_current(PROFILE)
    assert error is None and recovered is not None
    assert recovered.envelope.generation == 3 and recovered.raw == saved


def test_unsafe_profile_revokes_read_but_explicit_token_setup_may_tighten_permissions(
    tmp_path: Path,
):
    token = http_auth.ensure_token(PROFILE)
    profile = tmp_path / "data" / "accounting" / PROFILE
    profile.chmod(0o777)
    before = tree_state(tmp_path / "data")
    with pytest.raises(http_auth.TokenFileError) as caught:
        http_auth.load_token(PROFILE)
    assert caught.value.errno is not None
    assert http_auth.load_token_or_none(PROFILE) is None
    assert profile.stat().st_mode & 0o777 == 0o777
    assert tree_state(tmp_path / "data") == before
    assert http_auth.ensure_token(PROFILE) == token
    assert profile.stat().st_mode & 0o777 == 0o700
    assert http_auth.load_token(PROFILE) == token


def test_token_setup_rejects_symlinked_profile_without_changing_target(tmp_path: Path):
    token = http_auth.ensure_token(PROFILE)
    profile = tmp_path / "data" / "accounting" / PROFILE
    outside = tmp_path / "outside-profile"
    profile.rename(outside)
    profile.symlink_to(outside, target_is_directory=True)
    before = tree_state(outside)
    for operation in (http_auth.load_token, http_auth.ensure_token):
        with pytest.raises(http_auth.TokenFileError):
            operation(PROFILE)
    assert tree_state(outside) == before
    assert profile.is_symlink()
    profile.unlink()
    outside.rename(profile)
    assert http_auth.load_token(PROFILE) == token


def test_token_deleted_after_atomic_creation_fails_closed(monkeypatch: pytest.MonkeyPatch):
    original = storage.write_once

    def deleted(directory, name, data):
        result = original(directory, name, data)
        os.unlink(name, dir_fd=directory)
        return result

    with monkeypatch.context() as race:
        race.setattr(storage, "write_once", deleted)
        with pytest.raises(http_auth.TokenFileError) as caught:
            http_auth.ensure_token(PROFILE)
        assert caught.value.errno == errno.EIO
    token = http_auth.ensure_token(PROFILE)
    assert http_auth.token_matches(token, http_auth.load_token(PROFILE))


@pytest.mark.parametrize("damage", ["delivery", "retained"])
def test_acknowledged_snapshot_failure_never_substitutes_current(tmp_path: Path, damage: str):
    old, new = make_snapshot(1), make_snapshot(2, extra_searches=1)
    publish(PROFILE, old)
    publish(PROFILE, new)
    delivery = write_delivery(PROFILE, "otel", "metrics", last_acked_snapshot_id=old.snapshot_id)
    current, error = http_views.acknowledged_current(PROFILE, "otel", "metrics")
    assert error is None and current is not None and current.envelope.generation == 1
    target = (
        delivery
        if damage == "delivery"
        else tmp_path / "data" / "accounting" / PROFILE / snapshots.SNAPSHOT_DIR
    )
    original_mode = target.stat().st_mode & 0o777
    target.chmod(0o777)
    before = tree_state(tmp_path / "data")
    current, error = http_views.acknowledged_current(PROFILE, "otel", "metrics")
    assert current is None and error is not None and error.status == 500
    assert json.loads(error.body) == {
        "error": "delivery_state_unreadable"
        if damage == "delivery"
        else "acknowledged_snapshot_unreadable"
    }
    assert tree_state(tmp_path / "data") == before
    target.chmod(original_mode)
    current, error = http_views.acknowledged_current(PROFILE, "otel", "metrics")
    assert error is None and current is not None and current.envelope.snapshot_id == old.snapshot_id


def test_delivery_directory_failure_is_unknown_not_absent(tmp_path: Path):
    path = write_delivery(PROFILE, "otel", "metrics", pending_batches=4)
    path.parent.chmod(0o777)
    before = tree_state(tmp_path / "data")
    assert http_views.read_delivery_file(PROFILE, "otel", "metrics") == ("unreadable", None)
    rows = http_views.delivery_rows(PROFILE, (("otel", ("metrics",)),))
    assert rows[0]["state"] == "unreadable" and rows[0]["pending_batches"] is None
    assert tree_state(tmp_path / "data") == before
    path.parent.chmod(0o700)
    assert http_views.delivery_rows(PROFILE, ())[0]["pending_batches"] == 4


def test_directory_listing_failure_keeps_configured_delivery_visible(
    monkeypatch: pytest.MonkeyPatch,
):
    write_delivery(PROFILE, "otel", "metrics", pending_batches=4)
    with monkeypatch.context() as failed:
        failed.setattr(
            storage, "list_names", lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "disk"))
        )
        rows = http_views.delivery_rows(PROFILE, (("otel", ("metrics",)),))
        assert len(rows) == 1 and rows[0]["pending_batches"] == 4
    assert http_views.destination_ids(PROFILE, ()) == ["otel"]


def test_invalid_collection_date_is_stale_unknown_not_zero_age():
    headers = dict(
        http_views.freshness(replace(make_snapshot(2), collected_at="invalid"), NOW, 180, True)
    )
    assert headers["X-Ssgrep-State"] == "stale"
    assert headers["X-Ssgrep-Age-Seconds"] == "unknown"
    assert headers["X-Ssgrep-Paused"] == "true"
    sanitized = http_views.sanitize_delivery(
        {
            "loss_summary": [
                None,
                "secret",
                {"reason": "expired", "records": 5, "batches": 1, "at": "2026-10-12T12:00:00Z"},
            ]
        }
    )
    assert sanitized["loss_summary"] == [
        {"reason": "expired", "records": 5, "batches": 1, "at": "2026-10-12T12:00:00Z"}
    ]


@pytest.mark.parametrize(
    "target", ["/v1/snapshot#fragment", "/v1/snapshot?view", "/healthz?view=current"]
)
def test_malformed_query_rejected_and_next_request_recovers(target: str):
    with start(clock=lambda: NOW) as handle:
        reply = get(handle, target)
        assert reply.status == 400
        assert reply.json()["error"] in {"bad_request", "unsupported_parameter"}
        assert get(handle, "/healthz", token=False).body == b"ok\n"


def test_handler_failure_returns_fixed_error_without_secret_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
):
    with start(clock=lambda: NOW) as handle:
        publish(PROFILE, make_snapshot(3))
        with monkeypatch.context() as failed:
            failed.setattr(
                http_views,
                "load_current",
                lambda profile: (_ for _ in ()).throw(OSError("private-path-secret")),
            )
            response = get(handle, "/v1/snapshot")
            assert response.status == 500 and response.json() == {"error": "internal_error"}
        assert get(handle, "/v1/snapshot").json()["generation"] == 3


def test_read_transport_boundaries_have_precise_errors():
    for payload, limit, timeout, expected in (
        (b"GET / HTTP/1.1\r\n\r\n", 17, 1, (431, "headers_too_large")),
        (b"GET / HTTP/1.1\r\n", 8192, 1, (400, "bad_request")),
        (b"", 8192, 0, (408, "request_timeout")),
    ):
        server, client = socket.socketpair()
        with server, client:
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            with pytest.raises(http_wire.WireError) as caught:
                http_wire.read_request(server, timeout=timeout, max_bytes=limit)
            assert (caught.value.status, caught.value.code) == expected


def test_disconnected_writer_cannot_break_next_response():
    server, client = socket.socketpair()
    with server:
        client.close()
        http_wire.write_response(server, 200, [], b"discarded")
    server, client = socket.socketpair()
    with server, client:
        http_wire.write_response(server, 200, [], b"alive")
        assert client.recv(4096).endswith(b"\r\n\r\nalive")


def test_workerless_handle_rejects_run_without_starting_worker():
    handle = start(start=False)
    try:
        with pytest.raises(RuntimeError):
            handle.run_worker()
        handle.start()
        assert get(handle, "/healthz", token=False).status == 200
    finally:
        handle.stop()


def test_url_userinfo_is_credentials_even_without_query():
    assert http_auth.credential_in_url("https://secret@localhost/v1/snapshot")
    assert not http_auth.credential_in_url("/v1/snapshot?view=current")


def test_partial_multi_address_bind_failure_releases_first_socket(monkeypatch: pytest.MonkeyPatch):
    opened = []
    real_listener = http._listener

    def listener(host, port):
        if opened:
            raise OSError(errno.EADDRINUSE, "second address occupied")
        result = real_listener(host, port)
        opened.append(result)
        return result

    with monkeypatch.context() as failing:
        failing.setattr(http, "check_binds", lambda bind: ("127.0.0.1", "::1"))
        failing.setattr(http, "_listener", listener)
        with pytest.raises(OSError) as caught:
            start(start=False)
        assert caught.value.errno == errno.EADDRINUSE
    assert len(opened) == 1 and opened[0].fileno() == -1
    with start(clock=lambda: NOW) as handle:
        assert get(handle, "/healthz", token=False).status == 200


def test_socket_read_failure_is_client_gone_not_an_internal_response():
    class FailingReceive(socket.socket):
        def recv(self, size, flags=0):
            self.close()
            return super().recv(size, flags)

    with FailingReceive() as server:
        with pytest.raises(http_wire.ClientGone):
            http_wire.read_request(server)
    server, client = socket.socketpair()
    with server, client:
        client.sendall(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
        request = http_wire.read_request(server)
        assert request.target == "/healthz" and not request.has_body


def test_stop_closes_idle_clients_without_waiting_for_header_deadline(
    monkeypatch: pytest.MonkeyPatch,
):
    reading = threading.Event()
    release_reader = threading.Event()
    original = http.read_request

    def observed(sock):
        reading.set()
        try:
            return original(sock)
        finally:
            # Callback completion is not the TCP shutdown contract. Hold it pending
            # so the test cannot rely on favorable reader-thread scheduling.
            release_reader.wait(10)

    monkeypatch.setattr(http, "read_request", observed)
    handle = start()
    client = socket.create_connection(("127.0.0.1", handle.port), timeout=2)
    stopper = threading.Thread(target=handle.stop, daemon=True)
    try:
        assert reading.wait(2)
        stopper.start()
        stopper.join(timeout=2)
        assert not stopper.is_alive(), "stop waited for reader finalization"
        assert client.recv(1) == b""
    finally:
        release_reader.set()
        client.close()
        handle.stop()
        if stopper.ident is not None:
            stopper.join(timeout=2)


def test_failed_listener_exits_accept_loop_and_new_server_can_bind():
    handle = start()
    try:
        port = handle.port
        for listener in handle._listeners:
            listener.close()
        for thread in handle._threads:
            thread.join(timeout=2)
            assert not thread.is_alive()
        with http.serve(PROFILE, port=port) as replacement:
            assert get(replacement, "/healthz", token=False).status == 200
    finally:
        handle.stop()
