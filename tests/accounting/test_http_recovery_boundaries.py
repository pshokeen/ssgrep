"""Real socket recovery, authorization ordering, and damaged delivery health."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from prometheus_client.openmetrics.parser import text_string_to_metric_families

from ssgrep.accounting import http_auth, http_views
from tests.accounting.http_builders import (
    PROFILE,
    get,
    make_snapshot,
    publish,
    raw_exchange,
    start,
    token_for,
    tree_state,
    write_delivery,
)

NOW = datetime(2026, 10, 12, 12, 0, 30, tzinfo=UTC)


@pytest.mark.parametrize("damage", ["symlink", "directory", "malformed", "permissions"])
def test_damaged_token_revokes_access_without_reading_snapshot_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    with start(clock=lambda: NOW) as handle:
        publish(PROFILE, make_snapshot(3))
        original = token_for()
        assert get(handle, "/v1/snapshot", token=original).json()["generation"] == 3
        token_path = Path(os.environ["SSGREP_DATA_DIR"]) / "accounting" / PROFILE / "endpoint.token"
        original_bytes = token_path.read_bytes()
        token_path.unlink()
        if damage == "symlink":
            outside = tmp_path / "outside-token"
            outside.write_bytes(original_bytes)
            outside.chmod(0o600)
            token_path.symlink_to(outside)
        elif damage == "directory":
            token_path.mkdir(mode=0o700)
        else:
            token_path.write_bytes(b"not-a-token" if damage == "malformed" else original_bytes)
            token_path.chmod(0o644 if damage == "permissions" else 0o600)
        before = tree_state(tmp_path / "data")
        with monkeypatch.context() as denied:

            def forbidden_read(*args, **kwargs):
                pytest.fail("unauthorized request reached snapshot storage")

            denied.setattr(http_views, "load_current", forbidden_read)
            response = get(handle, "/v1/snapshot", token=original)
            assert response.status == 401 and response.json() == {"error": "unauthorized"}
            assert get(handle, "/healthz", token=False).body == b"ok\n"
        assert tree_state(tmp_path / "data") == before
        if damage == "directory":
            token_path.rmdir()
        else:
            token_path.unlink()
        token_path.write_bytes(original_bytes)
        token_path.chmod(0o600)
        assert get(handle, "/v1/snapshot", token=original).json()["generation"] == 3


@pytest.mark.parametrize(
    ("length", "status", "error"),
    [
        ("0" * 5000, 200, None),
        ("9" * 5000, 400, "request_body_not_allowed"),
        ("-1", 400, "bad_request"),
        ("1x", 400, "bad_request"),
    ],
)
def test_content_length_is_bounded_without_big_integer_conversion(
    length: str, status: int, error: str | None
) -> None:
    with start(clock=lambda: NOW) as handle:
        request = (
            f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{handle.port}\r\n"
            f"Content-Length: {length}\r\n\r\n"
        ).encode()
        response = raw_exchange(handle.port, request)
        assert response.startswith(f"HTTP/1.1 {status} ".encode())
        if error:
            assert f'"error":"{error}"'.encode() in response
        else:
            assert response.endswith(b"ok\n")
        assert get(handle, "/healthz", token=False).status == 200


def test_header_count_boundary_rejects_even_when_total_bytes_fit() -> None:
    with start(clock=lambda: NOW) as handle:
        base = f"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{handle.port}\r\n"
        for count, expected in ((63, 200), (64, 431)):
            headers = "".join(f"X-{i}: x\r\n" for i in range(count))
            response = raw_exchange(handle.port, (base + headers + "\r\n").encode())
            assert response.startswith(f"HTTP/1.1 {expected} ".encode())


@pytest.mark.parametrize(
    "stamp",
    ["2026-02-30T12:00:00Z", "2026-13-01T00:00:00Z", "2026-10-12T24:00:00Z"],
)
def test_impossible_delivery_dates_leave_other_metrics_available(stamp: str) -> None:
    with start(clock=lambda: NOW) as handle:
        publish(PROFILE, make_snapshot(3))
        write_delivery(
            PROFILE,
            "acme",
            "metrics",
            pending_batches=2,
            last_acked_at=stamp,
            last_attempt_at=stamp,
            next_retry_utc=stamp,
            loss_summary=[{"reason": "expired", "records": 7, "batches": 1, "at": stamp}],
        )
        response = get(handle, "/metrics")
        assert response.status == 200
        families = {
            family.name: family for family in text_string_to_metric_families(response.body.decode())
        }
        assert families["ssgrep_export_pending_batches"].samples[0].value == 2
        assert "ssgrep_export_last_full_ack_timestamp_seconds" not in families
        assert families["ssgrep_snapshot_available"].samples[0].value == 1
        (row,) = get(handle, "/v1/export-status").json()["destinations"]
        assert row["last_full_ack"]["at"] is None
        assert row["last_attempt"]["at"] is None and row["next_retry_utc"] is None
        assert row["loss_summary"] == [
            {"reason": "expired", "records": 7, "batches": 1, "at": None}
        ]
        write_delivery(PROFILE, "acme", "metrics", last_acked_at="2026-10-12T12:00:00Z")
        recovered = get(handle, "/metrics")
        assert recovered.status == 200
        assert b"ssgrep_export_last_full_ack_timestamp_seconds{" in recovered.body


def test_delivery_cardinality_failure_is_explicit_not_partial_and_recovers() -> None:
    with start(clock=lambda: NOW) as handle:
        for index in range(8):
            write_delivery(PROFILE, f"d{index}", "metrics", pending_batches=index)
        accepted = get(handle, "/metrics")
        assert accepted.status == 200
        families = list(text_string_to_metric_families(accepted.body.decode()))
        pending = next(f for f in families if f.name == "ssgrep_export_pending_batches")
        assert {s.labels["destination"]: s.value for s in pending.samples} == {
            f"d{i}": i for i in range(8)
        }
        ninth = write_delivery(PROFILE, "d8", "metrics", pending_batches=8)
        rejected = get(handle, "/metrics")
        assert rejected.status == 500 and rejected.json() == {"error": "destination_limit"}
        assert b"ssgrep_export_pending_batches" not in rejected.body
        ninth.unlink()
        assert get(handle, "/metrics").status == 200


def test_absent_profile_token_read_creates_nothing(tmp_path: Path) -> None:
    assert http_auth.load_token("missing") is None
    assert not (tmp_path / "data" / "accounting").exists()
