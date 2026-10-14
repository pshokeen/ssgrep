"""OTLP response parsing, monotonic durable delivery state, and acknowledged-snapshot lookup."""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import acks, snapshots, storage
from ssgrep.accounting.ledger_contracts import Signal
from tests.accounting.snapshots_driver import make_envelope

PROFILE = "personal-local"
DEST = "org-primary"
NOW = "2026-10-13T10:00:00Z"
LATER = "2026-10-13T10:05:00Z"
SNAP7 = "a7" * 32
SNAP8 = "a8" * 32
SNAP9 = "a9" * 32
CANARY = "sk-CANARY-SECRET-0123456789"


def fresh(signal: Signal = Signal.METRICS) -> acks.Delivery:
    return acks.Delivery(DEST, signal)


def ack(d: acks.Delivery, gen: int, snap: str, index: int, total: int, **kw: Any) -> acks.Delivery:
    kw.setdefault("now", NOW)
    return acks.outcome(
        d, generation=gen, snapshot_id=snap, index=index, total=total, kind="acked", **kw
    )


def doc_path(tmp_path: Path, signal: str = "metrics", dest: str = DEST) -> Path:
    return tmp_path / "data" / "accounting" / PROFILE / "delivery" / dest / f"{signal}.json"


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


# --- response parsing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body", [None, b"", b"{}", b'{"other":1}', b'{"x":{"rejectedDataPoints":9}}']
)
@pytest.mark.parametrize("signal", list(Signal))
def test_absent_partial_success_is_full(signal: Signal, body: bytes | None) -> None:
    assert acks.parse_response(signal, body) == acks.AckResult("full", 0, None, False)


@pytest.mark.parametrize(
    ("signal", "field"),
    [(Signal.METRICS, "rejectedDataPoints"), (Signal.LOGS, "rejectedLogRecords")],
)
@pytest.mark.parametrize("zero", [0, "0", None])
def test_zero_rejections_are_full_with_sanitized_warning(
    signal: Signal, field: str, zero: object
) -> None:
    partial: dict[str, object] = {"errorMessage": f"quota soon {CANARY}"}
    if zero is not None:
        partial[field] = zero
    body = json.dumps({"partialSuccess": partial}).encode()
    result = acks.parse_response(signal, body)
    assert result == acks.AckResult("full_with_warning", 0, "receiver_warning", False)
    assert CANARY not in repr(result)


@pytest.mark.parametrize("signal", list(Signal))
def test_empty_partial_success_object_is_plain_full_acceptance(signal: Signal) -> None:
    """A real OTel Collector answers every fully accepted request with ``{"partialSuccess":{}}``.

    The object carries no information (proto3 JSON default values), so it must not turn every
    healthy delivery into a permanent warning; any content in it still does.
    """
    assert acks.parse_response(signal, b'{"partialSuccess":{}}') == acks.AckResult("full")
    assert acks.parse_response(signal, b'{"partialSuccess":{"errorMessage":""}}').outcome == "full"
    zero_only = b'{"partialSuccess":{"rejectedDataPoints":0,"rejectedLogRecords":"0"}}'
    assert acks.parse_response(signal, zero_only).outcome == "full_with_warning"


@pytest.mark.parametrize(
    ("signal", "field", "value", "expected"),
    [
        (Signal.METRICS, "rejectedDataPoints", 5, 5),
        (Signal.METRICS, "rejectedDataPoints", "7", 7),
        (Signal.LOGS, "rejectedLogRecords", 3, 3),
        (Signal.LOGS, "rejectedLogRecords", "12", 12),
        (Signal.LOGS, "rejectedLogRecords", str((1 << 63) - 1), (1 << 63) - 1),
    ],
)
def test_nonzero_rejections_are_partial_with_unavailable_identities(
    signal: Signal, field: str, value: object, expected: int
) -> None:
    body = json.dumps({"partialSuccess": {field: value, "errorMessage": CANARY}}).encode()
    result = acks.parse_response(signal, body)
    assert result == acks.AckResult("partial", expected, None, True)
    assert CANARY not in repr(result)


def test_other_signals_rejection_field_is_not_mistaken() -> None:
    logs_field_on_metrics = b'{"partialSuccess":{"rejectedLogRecords":9}}'
    metrics_field_on_logs = b'{"partialSuccess":{"rejectedDataPoints":9}}'
    assert acks.parse_response(Signal.METRICS, logs_field_on_metrics).outcome == "full_with_warning"
    assert acks.parse_response(Signal.LOGS, metrics_field_on_logs).outcome == "full_with_warning"
    both = b'{"partialSuccess":{"rejectedLogRecords":4,"rejectedDataPoints":0}}'
    assert acks.parse_response(Signal.LOGS, both) == acks.AckResult("partial", 4, None, True)
    assert acks.parse_response(Signal.METRICS, both).outcome == "full_with_warning"


@pytest.mark.parametrize(
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
    result = acks.parse_response(Signal.METRICS, body)
    assert result == acks.AckResult("unparseable", 0, None, False)


def test_deeply_nested_body_is_unparseable() -> None:
    body = b"[" * 100_000 + b"]" * 100_000
    assert acks.parse_response(Signal.LOGS, body).outcome == "unparseable"


# --- pure transitions -------------------------------------------------------------------------


def test_collected_and_attempted_are_max_monotonic() -> None:
    d = acks.collected(acks.collected(fresh(), 8), 5)
    assert d.last_collected_generation == 8
    d = acks.attempted(acks.attempted(d, 8, NOW), 6, LATER)
    assert (d.last_attempted_generation, d.last_attempt_at) == (8, LATER)


def test_in_flight_lifecycle_and_nonrecallable() -> None:
    d = acks.start_in_flight(fresh(), "g0000000008-a8a8a8a8a8a8a8a8-0001", 8, NOW)
    assert d.in_flight == acks.InFlight("g0000000008-a8a8a8a8a8a8a8a8-0001", 8, NOW)
    d = acks.mark_nonrecallable(d, "g0000000008-a8a8a8a8a8a8a8a8-0001")
    assert d.in_flight_nonrecallable == "g0000000008-a8a8a8a8a8a8a8a8-0001"
    done = acks.finish_in_flight(d)
    assert done.in_flight is None
    assert done.in_flight_nonrecallable == "g0000000008-a8a8a8a8a8a8a8a8-0001"


def test_full_generation_acknowledges_only_when_every_index_acked() -> None:
    d = fresh()
    d = ack(d, 8, SNAP8, 0, 3)
    d = ack(d, 8, SNAP8, 2, 3)
    assert (d.last_acked_generation, d.last_acked_snapshot_id, d.last_acked_at) == (None,) * 3
    assert d.progress == (acks.Progress(8, SNAP8, 3, acked=(0, 2)),)
    d = ack(d, 8, SNAP8, 1, 3, now=LATER)
    assert (d.last_acked_generation, d.last_acked_snapshot_id, d.last_acked_at) == (8, SNAP8, LATER)
    assert d.progress == ()  # entries up to the acknowledged generation are pruned
    assert d.late_acks == 0


def test_duplicate_ack_of_one_index_does_not_complete_a_generation() -> None:
    d = ack(ack(fresh(), 8, SNAP8, 0, 2), 8, SNAP8, 0, 2)
    assert d.last_acked_generation is None
    assert d.progress == (acks.Progress(8, SNAP8, 2, acked=(0,)),)


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
    assert d.progress == (acks.Progress(7, SNAP7, 2, acked=(0, 1)),)


def test_partial_batch_poisons_its_generation_but_not_a_later_one() -> None:
    d = fresh()
    d = acks.outcome(
        d, generation=8, snapshot_id=SNAP8, index=1, total=3, kind="partial",
        rejected=4, identity_unavailable=True, now=NOW,
    )  # fmt: skip
    assert (d.partial_batches, d.rejected_points, d.identity_unavailable) == (1, 4, True)
    d = ack(ack(d, 8, SNAP8, 0, 3), 8, SNAP8, 2, 3)
    assert d.last_acked_generation is None and d.last_acked_snapshot_id is None
    d = ack(d, 8, SNAP8, 1, 3)  # re-acking the partial index must not rescue the generation
    assert d.last_acked_generation is None
    assert d.progress == (acks.Progress(8, SNAP8, 3, acked=(0, 2), partial=(1,)),)
    later = ack(d, 9, SNAP9, 0, 1, now=LATER)
    assert (later.last_acked_generation, later.last_acked_snapshot_id) == (9, SNAP9)
    assert later.progress == ()
    assert (later.partial_batches, later.rejected_points) == (1, 4)  # the history stays reported


def test_partial_counters_are_idempotent_per_index() -> None:
    kw: dict[str, Any] = {"generation": 8, "snapshot_id": SNAP8, "index": 0, "total": 2}
    d = acks.outcome(fresh(), kind="partial", rejected=3, now=NOW, **kw)
    again = acks.outcome(d, kind="partial", rejected=3, now=NOW, **kw)
    assert again == d
    assert (again.partial_batches, again.rejected_points) == (1, 3)


@pytest.mark.parametrize("kind", ["quarantined", "lost", "discarded"])
def test_other_terminal_outcomes_block_full_ack_of_that_generation(kind: str) -> None:
    d = acks.outcome(fresh(), generation=8, snapshot_id=SNAP8, index=0, total=2, kind=kind, now=NOW)
    d = ack(d, 8, SNAP8, 1, 2)
    assert d.last_acked_generation is None
    assert getattr(d.progress[0], kind) == (0,) and d.progress[0].acked == (1,)
    assert d.quarantined_batches == (1 if kind == "quarantined" else 0)


def test_index_moves_between_sets_staying_disjoint() -> None:
    d = acks.outcome(
        fresh(), generation=8, snapshot_id=SNAP8, index=0, total=2, kind="quarantined", now=NOW
    )
    d = acks.outcome(d, generation=8, snapshot_id=SNAP8, index=0, total=2, kind="lost", now=NOW)
    assert d.progress == (acks.Progress(8, SNAP8, 2, lost=(0,)),)


def test_snapshot_id_mismatch_for_existing_generation_raises() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 2)
    with pytest.raises(ValueError, match="mismatch"):
        ack(d, 8, SNAP9, 1, 2)
    with pytest.raises(ValueError, match="mismatch"):
        ack(d, 8, SNAP8, 1, 3)


@pytest.mark.parametrize(
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
def test_outcome_rejects_invalid_arguments(bad: dict[str, Any]) -> None:
    args: dict[str, Any] = {
        "generation": 8, "snapshot_id": SNAP8, "index": 0, "total": 2, "kind": "acked", "now": NOW,
    }  # fmt: skip
    args.update(bad)
    with pytest.raises(ValueError):
        acks.outcome(fresh(), **args)


def test_acked_warning_sets_sanitized_category_and_progress_is_bounded() -> None:
    d = ack(fresh(), 3, SNAP7, 0, 2, warning="receiver_warning")
    assert d.warning_category == "receiver_warning"
    d = fresh()
    for generation in range(1, 21):  # twenty never-completing generations
        d = ack(d, generation, SNAP7, 0, 2)
    assert [p.generation for p in d.progress] == list(range(5, 21))  # newest 16


def test_success_clears_block_retry_and_error_category() -> None:
    d = acks.retry_scheduled(fresh(), "2026-10-13T10:01:00Z", "http_503")
    d = acks.blocked(d, "unauthorized", "fp-one")
    assert (d.next_retry_utc, d.last_error_category) == ("2026-10-13T10:01:00Z", "http_503")
    assert (d.blocked_reason, d.blocked_fingerprint) == ("unauthorized", "fp-one")
    d = ack(d, 8, SNAP8, 0, 1)
    assert (d.blocked_reason, d.blocked_fingerprint, d.next_retry_utc, d.last_error_category) == (
        None, None, None, None,
    )  # fmt: skip


def test_blocked_clears_only_when_the_fingerprint_changes() -> None:
    d = acks.blocked(fresh(), "credential_invalid", "fp-one")
    assert acks.unblock_if_changed(d, "fp-one") == d
    cleared = acks.unblock_if_changed(d, "fp-two")
    assert (cleared.blocked_reason, cleared.blocked_fingerprint) == (None, None)
    assert acks.unblock_if_changed(fresh(), "anything") == fresh()


def test_blocked_reason_is_an_enumerated_code() -> None:
    for reason in acks.BLOCK_REASONS:
        assert acks.blocked(fresh(), reason, "fp").blocked_reason == reason
    with pytest.raises(ValueError):
        acks.blocked(fresh(), f"HTTP 401: {CANARY}", "fp")
    with pytest.raises(ValueError):
        acks.record_loss(fresh(), f"bad reason {CANARY}", 1, 1, NOW)


def test_loss_is_bounded_newest_first_and_never_moves_the_ack() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 1)
    for n in range(40):
        d = acks.record_loss(d, f"spool_expired_{n}", 1, 10, f"2026-10-13T10:{n:02d}:00Z")
    assert len(d.loss_summary) == 32
    assert d.loss_summary[0] == acks.Loss("spool_expired_39", 1, 10, "2026-10-13T10:39:00Z")
    assert d.loss_summary[-1].reason == "spool_expired_8"
    assert d.lost_records == 400
    assert (d.last_acked_generation, d.last_acked_snapshot_id) == (8, SNAP8)


def test_release_moves_the_pin_to_expired_and_keeps_the_generation() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 1)
    released = acks.release_acked_snapshot(d)
    assert released.last_acked_snapshot_id is None
    assert released.expired_snapshot_id == SNAP8
    assert released.last_acked_generation == 8
    assert acks.release_acked_snapshot(fresh()) == fresh()
    renewed = ack(released, 9, SNAP9, 0, 1)
    assert (renewed.last_acked_snapshot_id, renewed.expired_snapshot_id) == (SNAP9, None)


def test_reset_generation_drops_only_that_progress_and_touches_nothing_else() -> None:
    d = ack(fresh(), 5, SNAP7, 0, 1, now=LATER)  # acknowledged generation 5
    d = acks.record_loss(d, "spool_age_expired", 1, 7, NOW)
    d = acks.outcome(
        d, generation=8, snapshot_id=SNAP8, index=0, total=3, kind="partial",
        rejected=2, identity_unavailable=True, now=NOW,
    )  # fmt: skip
    d = ack(ack(d, 9, SNAP9, 0, 2), 7, SNAP7, 0, 2)
    d = replace(d, late_acks=4)
    reset = acks.reset_generation(d, 8)
    assert reset == replace(d, progress=tuple(p for p in d.progress if p.generation != 8))
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
    assert acks.reset_generation(d, 12) == d  # no progress for that generation
    assert acks.reset_generation(d, 8) == d  # the acknowledged generation is never reset
    assert acks.reset_generation(d, 3) == d  # older than the acknowledged generation
    stale = replace(d, progress=(*d.progress, acks.Progress(7, SNAP7, 2, acked=(0,))))
    assert acks.reset_generation(stale, 7) == stale  # <= last acked: untouched
    assert acks.reset_generation(fresh(), 4) == fresh()
    with pytest.raises(ValueError):
        acks.reset_generation(d, 0)


@pytest.mark.parametrize(
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
    d = acks.record_loss(fresh(), reason, 1, 2, NOW)
    assert d.loss_summary == (acks.Loss(reason, 1, 2, NOW),)
    stored = acks.decode(acks.encode(d), DEST, Signal.METRICS)
    assert stored.loss_summary == d.loss_summary


def test_ack_vacuous_advances_exactly_like_a_full_ack() -> None:
    d = ack(fresh(), 5, SNAP7, 0, 1)
    d = ack(ack(d, 6, SNAP8, 0, 2), 9, SNAP9, 0, 2)  # progress for 6 (older) and 9 (newer)
    d = acks.retry_scheduled(acks.blocked(d, "forbidden", "fp"), NOW, "http_503")
    d = replace(d, late_acks=2, partial_batches=3, rejected_points=11)
    d = acks.record_loss(d, "spool_age_expired", 1, 5, NOW)
    done = acks.ack_vacuous(d, 8, SNAP8, LATER)
    expected = replace(
        d,
        last_acked_generation=8,
        last_acked_snapshot_id=SNAP8,
        last_acked_at=LATER,
        blocked_reason=None,
        blocked_fingerprint=None,
        next_retry_utc=None,
        last_error_category=None,
        progress=(acks.Progress(9, SNAP9, 2, acked=(0,)),),  # older entry pruned, newer kept
    )
    assert done == expected
    assert (done.late_acks, done.partial_batches, done.rejected_points) == (2, 3, 11)
    assert done.loss_summary == d.loss_summary and done.lost_records == d.lost_records
    assert acks.ack_vacuous(fresh(), 1, SNAP7, NOW).last_acked_generation == 1


def test_ack_vacuous_accepts_a_generation_whose_progress_is_all_acked() -> None:
    d = replace(fresh(), progress=(acks.Progress(8, SNAP8, 2, acked=(0, 1)),))
    done = acks.ack_vacuous(d, 8, SNAP8, LATER)
    assert (done.last_acked_generation, done.last_acked_at, done.progress) == (8, LATER, ())


def test_ack_vacuous_never_regresses_or_repeats() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 1, now=LATER)
    assert acks.ack_vacuous(d, 8, SNAP9, NOW) == d  # equal generation
    assert acks.ack_vacuous(d, 7, SNAP7, NOW) == d  # older generation
    with pytest.raises(ValueError):
        acks.ack_vacuous(d, 0, SNAP7, NOW)


@pytest.mark.parametrize("kind", ["partial", "quarantined", "lost", "discarded"])
def test_ack_vacuous_is_refused_for_a_generation_with_a_bad_batch(kind: str) -> None:
    d = acks.outcome(fresh(), generation=8, snapshot_id=SNAP8, index=0, total=2, kind=kind, now=NOW)
    d = ack(d, 8, SNAP8, 1, 2)
    assert acks.ack_vacuous(d, 8, SNAP8, LATER) == d
    assert d.last_acked_generation is None


def test_ack_vacuous_is_refused_for_a_generation_with_unacked_batches() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 3)
    assert acks.ack_vacuous(d, 8, SNAP8, LATER) == d
    other = acks.ack_vacuous(d, 9, SNAP9, LATER)  # a different generation is unaffected
    assert (other.last_acked_generation, other.progress) == (9, ())  # gen 8 pruned as older


def test_ack_vacuous_keeps_metrics_and_logs_documents_independent() -> None:
    acks.update_delivery(PROFILE, DEST, Signal.LOGS, lambda d: ack(d, 7, SNAP7, 0, 1))
    acks.update_delivery(PROFILE, DEST, Signal.METRICS, lambda d: ack(d, 8, SNAP8, 0, 2))
    acks.update_delivery(PROFILE, DEST, Signal.LOGS, lambda d: acks.ack_vacuous(d, 8, SNAP8, LATER))
    metrics = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    logs = acks.read_delivery(PROFILE, DEST, Signal.LOGS)
    assert metrics is not None and logs is not None
    assert (logs.last_acked_generation, logs.last_acked_snapshot_id) == (8, SNAP8)
    assert metrics.last_acked_generation is None  # metrics gen 8 still has index 1 outstanding
    assert metrics.progress == (acks.Progress(8, SNAP8, 2, acked=(0,)),)


def test_set_counts_are_not_derived_from_outcomes() -> None:
    d = ack(fresh(), 8, SNAP8, 0, 2)
    assert (d.pending_batches, d.unknown_batches) == (0, 0)
    d = acks.set_counts(d, 5, 2)
    assert (d.pending_batches, d.unknown_batches) == (5, 2)
    with pytest.raises(ValueError):
        acks.set_counts(d, -1, 0)


# --- durable state ----------------------------------------------------------------------------


def test_update_creates_private_files_and_roundtrips_exactly(tmp_path: Path) -> None:
    def change(d: acks.Delivery) -> acks.Delivery:
        d = acks.collected(d, 8)
        d = ack(d, 8, SNAP8, 0, 1, now=LATER, warning="receiver_warning")
        d = acks.record_loss(d, "spool_expired", 2, 9, NOW)
        d = acks.start_in_flight(d, "g0000000009-a9a9a9a9a9a9a9a9-0000", 9, NOW)
        return acks.set_counts(d, 3, 1)

    written = acks.update_delivery(PROFILE, DEST, Signal.METRICS, change)
    path = doc_path(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert acks.read_delivery(PROFILE, DEST, Signal.METRICS) == written
    assert acks.read_delivery(PROFILE, DEST, Signal.LOGS) is None
    document = json.loads(path.read_text())
    assert document["schema"] == 1 and document["accepted_means"] == "collector_acceptance"
    assert document["last_acked_snapshot_id"] == SNAP8 and document["signal"] == "metrics"
    assert document["loss_summary"] == [
        {"reason": "spool_expired", "batches": 2, "records": 9, "at": NOW}
    ]
    assert document["in_flight"]["generation"] == 9


def test_update_roundtrips_progress_sets(tmp_path: Path) -> None:
    def change(d: acks.Delivery) -> acks.Delivery:
        d = acks.outcome(
            d, generation=8, snapshot_id=SNAP8, index=1, total=4, kind="partial",
            rejected=2, identity_unavailable=True, now=NOW,
        )  # fmt: skip
        return ack(d, 8, SNAP8, 0, 4)

    written = acks.update_delivery(PROFILE, DEST, Signal.LOGS, change)
    stored = acks.read_delivery(PROFILE, DEST, Signal.LOGS)
    assert stored == written
    assert stored is not None
    assert stored.progress == (acks.Progress(8, SNAP8, 4, acked=(0,), partial=(1,)),)
    assert json.loads(doc_path(tmp_path, "logs").read_text())["progress"][0]["partial"] == [1]


def test_update_that_produces_an_invalid_state_writes_nothing(tmp_path: Path) -> None:
    acks.update_delivery(PROFILE, DEST, Signal.METRICS, lambda d: acks.collected(d, 3))
    before = doc_path(tmp_path).read_bytes()
    with pytest.raises(ValueError):
        acks.update_delivery(
            PROFILE, DEST, Signal.METRICS, lambda d: replace(d, pending_batches=-1)
        )
    with pytest.raises(ValueError):
        acks.update_delivery(
            PROFILE, DEST, Signal.METRICS, lambda d: replace(d, warning_category="Bad Text")
        )
    assert doc_path(tmp_path).read_bytes() == before
    assert not [p for p in doc_path(tmp_path).parent.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.parametrize(
    ("profile", "destination"),
    [("../evil", DEST), (PROFILE, "../evil"), (PROFILE, "a/b"), (PROFILE, ""), ("Bad", DEST)],
)
def test_ids_are_identifiers_never_paths(tmp_path: Path, profile: str, destination: str) -> None:
    with pytest.raises(ValueError):
        acks.update_delivery(profile, destination, Signal.METRICS, lambda d: d)
    with pytest.raises(ValueError):
        acks.read_delivery(profile, destination, Signal.METRICS)
    assert tree(tmp_path / "data") == []


def raw_document(**overrides: object) -> bytes:
    document: dict[str, object] = {"schema": 1, "destination_id": DEST, "signal": "metrics"}
    document.update(overrides)
    return json.dumps(document).encode()


@pytest.mark.parametrize(
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
        raw_document(loss_summary=[{"reason": "ok", "batches": 1, "records": 1, "at": NOW}] * 33),
        raw_document(
            progress=[
                {"generation": 8, "snapshot_id": SNAP8, "total": 2, "acked": [0], "partial": [0]}
            ]
        ),
        raw_document(progress=[{"generation": 8, "snapshot_id": SNAP8, "total": 2, "acked": [5]}]),
    ],
)
def test_corrupt_documents_raise_and_are_skipped_by_listing(tmp_path: Path, data: bytes) -> None:
    path = doc_path(tmp_path)
    os.close(storage.open_profile_dir(PROFILE, "delivery", DEST))
    path.write_bytes(data)
    path.chmod(0o600)
    with pytest.raises(acks.DeliveryCorrupt):
        acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    with pytest.raises(acks.DeliveryCorrupt):
        acks.update_delivery(PROFILE, DEST, Signal.METRICS, lambda d: d)
    assert path.read_bytes() == data  # a corrupt document is never silently overwritten
    assert acks.list_deliveries(PROFILE) == []


def test_listing_skips_corrupt_entries_but_returns_the_good_ones(tmp_path: Path) -> None:
    acks.update_delivery(PROFILE, DEST, Signal.METRICS, lambda d: acks.collected(d, 4))
    acks.update_delivery(PROFILE, "second", Signal.LOGS, lambda d: acks.collected(d, 6))
    doc_path(tmp_path, "logs").parent.mkdir(exist_ok=True)
    doc_path(tmp_path, "logs").write_text("{garbage")
    doc_path(tmp_path, "logs").chmod(0o600)
    (doc_path(tmp_path).parent.parent / "stray.json").write_text("{}")
    found = acks.list_deliveries(PROFILE)
    assert sorted((d.destination_id, d.signal.value) for d in found) == [
        (DEST, "metrics"),
        ("second", "logs"),
    ]


def test_symlinked_destination_directory_is_refused(tmp_path: Path) -> None:
    delivery = tmp_path / "data" / "accounting" / PROFILE / "delivery"
    os.close(storage.open_profile_dir(PROFILE, "delivery"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (elsewhere / "metrics.json").write_bytes(raw_document(late_acks=3))
    (elsewhere / "metrics.json").chmod(0o600)
    (delivery / DEST).symlink_to(elsewhere)
    with pytest.raises(acks.DeliveryCorrupt):
        acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    with pytest.raises(acks.DeliveryCorrupt):
        acks.update_delivery(PROFILE, DEST, Signal.METRICS, lambda d: acks.collected(d, 1))
    assert sorted(p.name for p in elsewhere.iterdir()) == ["metrics.json"]
    assert acks.list_deliveries(PROFILE) == []


def test_symlinked_document_is_refused_and_never_followed(tmp_path: Path) -> None:
    path = doc_path(tmp_path)
    os.close(storage.open_profile_dir(PROFILE, "delivery", DEST))
    target = tmp_path / "target.json"
    target.write_bytes(raw_document(late_acks=3))
    target.chmod(0o600)
    path.symlink_to(target)
    with pytest.raises(acks.DeliveryCorrupt):
        acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert acks.list_deliveries(PROFILE) == []


def test_reads_never_create_anything(tmp_path: Path) -> None:
    assert acks.read_delivery(PROFILE, DEST, Signal.METRICS) is None
    assert acks.list_deliveries(PROFILE) == []
    assert acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS).state == "not_acknowledged"
    assert acks.export_health(PROFILE) == []
    assert tree(tmp_path / "data") == []
    os.close(storage.open_profile_dir(PROFILE))
    before = tree(tmp_path / "data")
    acks.read_delivery(PROFILE, DEST, Signal.LOGS)
    acks.list_deliveries(PROFILE)
    acks.resolve_acknowledged(PROFILE, DEST, Signal.LOGS)
    assert tree(tmp_path / "data") == before


def _increment(profile: str, destination: str, count: int, ready: Any) -> None:
    ready.wait(30)
    for _ in range(count):
        acks.update_delivery(
            profile, destination, Signal.METRICS, lambda d: replace(d, late_acks=d.late_acks + 1)
        )


def test_two_processes_updating_concurrently_lose_no_updates(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    workers = [
        context.Process(target=_increment, args=(PROFILE, DEST, 25, ready)) for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    ready.set()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    state = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert state is not None and state.late_acks == 50
    assert json.loads(doc_path(tmp_path).read_text())["late_acks"] == 50
    assert os.path.exists(doc_path(tmp_path).parent / ".metrics.lock")


# --- independence of signals (spec scenario) --------------------------------------------------


def test_metrics_ack_does_not_move_logs_ack() -> None:
    acks.update_delivery(PROFILE, DEST, Signal.LOGS, lambda d: ack(d, 7, SNAP7, 0, 1))
    acks.update_delivery(
        PROFILE, DEST, Signal.LOGS, lambda d: ack(d, 8, SNAP8, 0, 2)
    )  # index 1 pending
    acks.update_delivery(PROFILE, DEST, Signal.METRICS, lambda d: ack(d, 8, SNAP8, 0, 1))
    metrics = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    logs = acks.read_delivery(PROFILE, DEST, Signal.LOGS)
    assert metrics is not None and logs is not None
    assert (metrics.last_acked_generation, metrics.last_acked_snapshot_id) == (8, SNAP8)
    assert (logs.last_acked_generation, logs.last_acked_snapshot_id) == (7, SNAP7)
    assert logs.progress == (acks.Progress(8, SNAP8, 2, acked=(0,)),)


# --- acknowledged resolution against real snapshot files ---------------------------------------


def acknowledge_real(generation: int, signal: Signal = Signal.METRICS) -> str:
    snapshot_id = make_envelope(generation).snapshot_id
    acks.update_delivery(PROFILE, DEST, signal, lambda d: ack(d, generation, snapshot_id, 0, 1))
    return snapshot_id


def test_resolve_returns_the_exact_older_acknowledged_snapshot_not_current(tmp_path: Path) -> None:
    older, newer = make_envelope(7), make_envelope(8)
    snapshots.publish(PROFILE, older)
    snapshots.publish(PROFILE, newer)
    acknowledge_real(7)
    result = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert result.state == "acknowledged"
    assert result.envelope == older and result.envelope is not None
    assert result.envelope.snapshot_id == older.snapshot_id != newer.snapshot_id
    assert result.envelope.generation == 7
    assert result.delivery is not None and result.delivery.last_acked_generation == 7
    assert snapshots.current_generation(PROFILE) == 8  # current is newer, and was NOT returned
    assert acks.resolve_acknowledged(PROFILE, DEST, Signal.LOGS).state == "not_acknowledged"


def test_resolve_without_an_acknowledgment_never_falls_back_to_current() -> None:
    snapshots.publish(PROFILE, make_envelope(3))
    absent = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert (absent.state, absent.delivery, absent.envelope) == ("not_acknowledged", None, None)
    acks.update_delivery(PROFILE, DEST, Signal.METRICS, lambda d: ack(d, 3, SNAP7, 0, 2))
    pending = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert pending.state == "not_acknowledged" and pending.envelope is None
    assert pending.delivery is not None and pending.delivery.last_acked_snapshot_id is None


def test_resolve_reports_expired_when_the_snapshot_file_is_gone(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(7))
    snapshots.publish(PROFILE, make_envelope(8))
    snapshot_id = acknowledge_real(7)
    gen_dir = tmp_path / "data" / "accounting" / PROFILE / "snapshots"
    (gen_dir / f"gen-0000000007-{snapshot_id}.json").unlink()
    result = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert (result.state, result.envelope) == ("acknowledged_snapshot_expired", None)
    assert result.delivery is not None and result.delivery.last_acked_generation == 7


def test_resolve_reports_expired_for_a_corrupt_snapshot_and_for_a_released_pin(
    tmp_path: Path,
) -> None:
    snapshots.publish(PROFILE, make_envelope(7))
    snapshots.publish(PROFILE, make_envelope(8))
    snapshot_id = acknowledge_real(7)
    path = tmp_path / "data" / "accounting" / PROFILE / "snapshots"
    (path / f"gen-0000000007-{snapshot_id}.json").write_bytes(b"{torn")
    corrupt = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert (corrupt.state, corrupt.envelope) == ("acknowledged_snapshot_expired", None)
    acks.update_delivery(PROFILE, DEST, Signal.METRICS, acks.release_acked_snapshot)
    released = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert (released.state, released.envelope) == ("acknowledged_snapshot_expired", None)
    assert released.delivery is not None and released.delivery.expired_snapshot_id == snapshot_id


def test_pin_is_visible_and_protects_the_acknowledged_snapshot_from_prune(
    tmp_path: Path,
) -> None:
    for generation in range(1, 6):
        snapshots.publish(PROFILE, make_envelope(generation))
    pinned_id = acknowledge_real(2)
    assert snapshots.pinned_snapshot_ids(PROFILE) == frozenset({pinned_id})
    assert snapshots.prune(PROFILE, keep=1) == [1, 3, 4]  # 2 survives: pinned; 5 is current
    assert snapshots.generations(PROFILE) == [2, 5]
    assert acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS).state == "acknowledged"
    # negative twin: after the pin is released the same prune deletes it
    acks.update_delivery(PROFILE, DEST, Signal.METRICS, acks.release_acked_snapshot)
    assert snapshots.pinned_snapshot_ids(PROFILE) == frozenset()
    assert snapshots.prune(PROFILE, keep=1) == [2]
    assert (
        acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS).state
        == "acknowledged_snapshot_expired"
    )


def test_lock_files_do_not_confuse_the_pin_scan(tmp_path: Path) -> None:
    for generation in (1, 2, 3):
        snapshots.publish(PROFILE, make_envelope(generation))
    acknowledge_real(1)
    assert (doc_path(tmp_path).parent / ".metrics.lock").exists()
    assert snapshots.pinned_snapshot_ids(PROFILE) == {make_envelope(1).snapshot_id}


# --- health and status views -------------------------------------------------------------------


def test_export_health_reports_epoch_counts_and_aggregated_losses() -> None:
    def change(d: acks.Delivery) -> acks.Delivery:
        d = ack(d, 8, SNAP8, 0, 1, now="2026-10-13T10:00:00Z")
        d = acks.outcome(
            d, generation=9, snapshot_id=SNAP9, index=0, total=2, kind="partial",
            rejected=6, identity_unavailable=True, now=NOW,
        )  # fmt: skip
        d = acks.record_loss(d, "spool_expired", 1, 4, NOW)
        d = acks.record_loss(d, "spool_expired", 2, 5, NOW)
        d = acks.record_loss(d, "tombstone_unsent", 1, 1, NOW)
        return acks.set_counts(d, 2, 0)

    acks.update_delivery(PROFILE, DEST, Signal.LOGS, change)
    acks.update_delivery(PROFILE, "other", Signal.METRICS, lambda d: acks.collected(d, 1))
    health = {(h.destination, h.signal): h for h in acks.export_health(PROFILE)}
    logs = health[(DEST, "logs")]
    assert logs.last_full_ack_epoch == 1_791_885_600  # 2026-10-13T10:00:00Z
    assert (logs.pending_batches, logs.rejected_points) == (2, 6)
    assert logs.lost == (("spool_expired", 9), ("tombstone_unsent", 1))
    other = health[("other", "metrics")]
    assert (other.last_full_ack_epoch, other.pending_batches, other.lost) == (None, 0, ())
    assert isinstance(logs.signal, str) and not isinstance(logs.signal, Signal)


def test_describe_states_and_sanitization() -> None:
    assert acks.describe(fresh())["state"] == "idle"
    d = ack(acks.collected(fresh(), 8), 8, SNAP8, 0, 1)
    assert acks.describe(d)["state"] == "acknowledged"
    assert acks.describe(acks.collected(d, 9))["state"] == "pending"
    assert acks.describe(acks.set_counts(d, 1, 0))["state"] == "pending"
    assert acks.describe(acks.set_counts(d, 0, 2))["state"] == "unknown"
    partial = acks.outcome(
        d, generation=9, snapshot_id=SNAP9, index=0, total=2, kind="partial",
        rejected=3, identity_unavailable=True, now=NOW,
    )  # fmt: skip
    assert acks.describe(partial)["state"] == "partial"
    blocked = acks.blocked(partial, "forbidden", "fp")
    view = acks.describe(blocked)
    assert view["state"] == "blocked" and view["blocked_reason"] == "forbidden"
    assert view["accepted_means"] == "collector_acceptance"
    assert view["last_acked_generation"] == 8 and view["identity_unavailable"] is True
    assert "blocked_fingerprint" not in view
    json.dumps(view)  # JSON-safe
    listing = acks.describe(acks.record_loss(blocked, "spool_expired", 1, 3, NOW))
    assert listing["loss_summary"] == [
        {"reason": "spool_expired", "batches": 1, "records": 3, "at": NOW}
    ]


def test_canary_error_message_never_reaches_any_state_or_view(tmp_path: Path) -> None:
    body = json.dumps(
        {"partialSuccess": {"rejectedLogRecords": 2, "errorMessage": f"bad token {CANARY}"}}
    ).encode()
    result = acks.parse_response(Signal.LOGS, body)

    def change(d: acks.Delivery) -> acks.Delivery:
        return acks.outcome(
            d, generation=8, snapshot_id=SNAP8, index=0, total=1, kind="partial",
            rejected=result.rejected, identity_unavailable=result.identity_unavailable,
            warning=result.warning_category, now=NOW,
        )  # fmt: skip

    state = acks.update_delivery(PROFILE, DEST, Signal.LOGS, change)
    assert CANARY not in doc_path(tmp_path, "logs").read_text()
    assert CANARY not in json.dumps(acks.describe(state))
    assert state.rejected_points == 2 and state.identity_unavailable is True
