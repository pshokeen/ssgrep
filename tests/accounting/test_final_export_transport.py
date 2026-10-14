"""Transport and credential failures use real files/sockets with successful neighboring controls."""

from __future__ import annotations

import errno
import http.client
import importlib.metadata
import json
import os
import socket
import ssl
from pathlib import Path

import pytest

from ssgrep.accounting import export_creds, export_http
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import PROFILE
from tests.accounting.lab_receiver import LabReceiver, Respond


@pytest.mark.parametrize("failure", ["foreign-owner", "grew-after-stat"])
def test_file_credential_owner_and_growth_races_refuse_secret_without_changing_it(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "ingest-token"
    path.write_bytes(b"INGEST-POSITIVE-CONTROL")
    path.chmod(0o600)
    ref = f"file:{path}"
    real_read = os.read
    identity = path.stat()
    reads = []

    def growing_read(fd, count):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            reads.append(count)
            path.write_bytes(b"X" * (export_creds.MAX_CREDENTIAL_BYTES + 1))
        return real_read(fd, count)

    with monkeypatch.context() as fault:
        if failure == "foreign-owner":
            fault.setattr(os, "geteuid", lambda: identity.st_uid + 1)
        else:
            fault.setattr(os, "read", growing_read)
        with pytest.raises(export_creds.CredentialError) as error:
            export_creds.resolve(PROFILE, ref)
    assert error.value.condition == (
        "credential_unsafe_file" if failure == "foreign-owner" else "credential_malformed"
    )
    assert "INGEST-POSITIVE-CONTROL" not in str(error.value)
    if failure == "foreign-owner":
        assert path.read_bytes() == b"INGEST-POSITIVE-CONTROL"
    else:
        assert reads == [export_creds.MAX_CREDENTIAL_BYTES + 1]
        assert path.stat().st_size == export_creds.MAX_CREDENTIAL_BYTES + 1
    path.write_bytes(b"INGEST-POSITIVE-CONTROL")
    resolved = export_creds.resolve(PROFILE, ref)
    assert resolved is not None and resolved.value == "Bearer INGEST-POSITIVE-CONTROL"


def test_unreadable_local_secret_does_not_poison_distinct_ingest_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx.profile_root().mkdir(parents=True)
    local = fx.profile_root() / "endpoint.token"
    local.write_bytes(b"LOCAL-ONLY-CANARY")
    local.chmod(0o600)
    identity = local.stat()
    real_read = os.read
    failures = []

    def unreadable(fd, count):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            failures.append(fd)
            raise OSError(errno.EIO, "local secret device failed")
        return real_read(fd, count)

    with monkeypatch.context() as fault:
        fault.setattr(os, "read", unreadable)
        credential = export_creds.resolve(
            PROFILE, "env:INGEST", environ={"INGEST": "DISTINCT-INGEST"}
        )
    assert credential is not None and credential.value == "Bearer DISTINCT-INGEST"
    assert len(failures) == 1 and local.read_bytes() == b"LOCAL-ONLY-CANARY"
    with pytest.raises(export_creds.CredentialError):
        export_creds.resolve(PROFILE, "env:INGEST", environ={"INGEST": "LOCAL-ONLY-CANARY"})
    assert (
        export_creds.resolve(PROFILE, "env:INGEST", environ={"INGEST": "DISTINCT-INGEST"})
        == credential
    )


def test_invalid_url_repr_redacts_credentials_and_length_refusal_never_opens_socket() -> None:
    target = export_http.Target("https://user:SECRET-CANARY@example.test:bad/path?private=1")
    assert "<invalid-url>" in repr(target)
    assert "SECRET-CANARY" not in repr(target) and "private" not in repr(target)
    with LabReceiver.start() as lab:
        oversized = export_http.Target(lab.url + "/" + "x" * export_http.MAX_URL_CHARS)
        rejected = export_http.post(oversized, b"{}")
        assert rejected.kind == "transport_refused" and lab.requests == []
        control = export_http.post(export_http.Target(lab.url + "/v1/metrics"), b"{}")
        assert control.kind == "response" and control.status == 200
        assert [r.body for r in lab.requests] == [b"{}"]


@pytest.mark.parametrize("error_type", [ssl.SSLEOFError, ssl.SSLZeroReturnError, ValueError])
def test_response_decoder_failure_discards_exception_text_and_next_request_recovers(
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_response = http.client.HTTPConnection.getresponse

    def fail_after_receiving(connection):
        response = real_response(connection)
        response.read()
        response.close()
        raise error_type("PRIVATE-RESPONSE-CANARY")

    with LabReceiver.start() as lab:
        target = export_http.Target(lab.url + "/v1/metrics")
        with monkeypatch.context() as fault:
            fault.setattr(http.client.HTTPConnection, "getresponse", fail_after_receiving)
            failed = export_http.post(target, b'{"resourceMetrics":[]}')
        assert failed.kind == (
            "malformed_response" if error_type is ValueError else "network_error"
        )
        assert failed.body is None and "PRIVATE-RESPONSE-CANARY" not in repr(failed)
        control = export_http.post(target, b'{"resourceMetrics":[]}')
        assert control.kind == "response" and control.status == 200 and control.body == b"{}"
        assert [r.body for r in lab.requests] == [b'{"resourceMetrics":[]}'] * 2


def test_deadline_fired_during_connection_setup_prevents_request_even_if_socket_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_start = export_http._Watchdog.start
    closed_socket = socket.socket()
    closed_socket.close()

    def expire_before_connect(dog, connection):
        real_start(dog, connection)
        dog.sock = closed_socket
        dog._fire()

    with LabReceiver.start() as lab:
        target = export_http.Target(lab.url + "/v1/metrics")
        with monkeypatch.context() as fault:
            fault.setattr(export_http._Watchdog, "start", expire_before_connect)
            expired = export_http.post(target, b"{}", connect_timeout=2, read_timeout=2)
        assert expired.kind == expired.category == "timeout"
        assert expired.body is None and lab.requests == []
        control = export_http.post(target, b"{}")
        assert control.status == 200 and [r.body for r in lab.requests] == [b"{}"]


def test_retry_after_without_timezone_is_not_interpreted_in_machine_local_time() -> None:
    with LabReceiver.start() as lab:
        target = export_http.Target(lab.url + "/v1/metrics")
        lab.queue(Respond(503, headers={"Retry-After": "Tue, 13 Oct 2026 12:00:00"}))
        ambiguous = export_http.post(target, b"{}")
        assert ambiguous.status == 503 and ambiguous.retry_after_seconds is None
        lab.queue(Respond(503, headers={"Retry-After": "17"}))
        bounded = export_http.post(target, b"{}")
        assert bounded.status == 503 and bounded.retry_after_seconds == 17
        assert export_http.post(target, b"{}").status == 200
        assert [r.body for r in lab.requests] == [b"{}"] * 3


def test_missing_distribution_metadata_exports_explicit_unknown_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_version = importlib.metadata.version

    def uninstalled(name):
        if name == "ssgrep":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        with monkeypatch.context() as fault:
            fault.setattr(importlib.metadata, "version", uninstalled)
            assert fx.cycle().acked == 1
        first = json.loads(lab.requests[0].body)["resourceMetrics"][0]["resource"]["attributes"]
        assert (
            next(a["value"]["stringValue"] for a in first if a["key"] == "service.version")
            == "unknown"
        )
        fx.publish(2)
        assert fx.cycle(seconds=1).acked == 1
        second = json.loads(lab.requests[1].body)["resourceMetrics"][0]["resource"]["attributes"]
        assert next(
            a["value"]["stringValue"] for a in second if a["key"] == "service.version"
        ) == real_version("ssgrep")
