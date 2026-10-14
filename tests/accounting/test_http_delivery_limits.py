"""The HTTP view must retain delivery limitations without exposing batch identifiers."""

from __future__ import annotations

import pytest

from tests.accounting.http_builders import PROFILE, get, start, write_delivery


def test_status_exposes_nonrecallable_presence_and_partial_identity_limit() -> None:
    batch = "g0000000001-abcd1234-0000"
    write_delivery(
        PROFILE,
        "acme",
        "metrics",
        in_flight_nonrecallable=batch,
        identity_unavailable=True,
        rejected_points=4,
        partial_batches=1,
        last_acked_generation=2,
        last_attempted_generation=3,
    )
    with start() as server:
        response = get(server, "/v1/export-status")

    assert response.status == 200
    (row,) = response.json()["destinations"]
    assert row["in_flight_nonrecallable"] is True
    assert row["identity_unavailable"] is True
    assert row["rejected_points"] == 4
    assert row["last_full_ack"]["generation"] == 2
    assert row["last_attempt"]["generation"] == 3
    assert batch.encode() not in response.body


@pytest.mark.parametrize("batch,expected", [(None, False), (False, False), (True, True)])
def test_status_distinguishes_no_inflight_send_from_unknown(batch: object, expected: bool) -> None:
    write_delivery(PROFILE, "acme", "logs", in_flight_nonrecallable=batch)
    with start() as server:
        response = get(server, "/v1/export-status")
    (row,) = response.json()["destinations"]
    assert row["in_flight_nonrecallable"] is expected
    assert row["identity_unavailable"] is None


def test_status_rejects_malformed_limitation_values_without_leaking_text() -> None:
    secret = "private-destination-token"
    write_delivery(
        PROFILE,
        "acme",
        "logs",
        in_flight_nonrecallable=secret,
        identity_unavailable=secret,
    )
    with start() as server:
        response = get(server, "/v1/export-status")
    (row,) = response.json()["destinations"]
    assert row["in_flight_nonrecallable"] is None
    assert row["identity_unavailable"] is None
    assert secret.encode() not in response.body
