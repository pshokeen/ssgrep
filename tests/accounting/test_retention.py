"""Export-side retention: effective numbers, spool age/size bounds, snapshot expiry.

Real spool files (``export_spool.write_batch``), real delivery documents (``acks``) and real
snapshot generations under the isolated data dir; nothing is mocked.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import (
    acks,
    export_spool,
    policy_rules,
    retention,
    retention_hooks,
    snapshots,
)
from ssgrep.accounting.export_model import Batch
from ssgrep.accounting.export_spool import Meta
from ssgrep.accounting.ledger_contracts import (
    SPOOL_MAX_BYTES,
    SPOOL_RETENTION_DAYS,
    ProfilePolicy,
    Signal,
)
from tests.accounting.snapshots_driver import make_envelope

PROFILE = "personal-local"
DEST = "org-primary"
NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
NOW_Z = "2026-10-13T12:00:00Z"
DAY = timedelta(days=1)
SCOPE = {
    "roots": [],
    "endpoint": "https://collector.example/otlp",
    "fields_version": "logs-v1",
    "policy_version": 1,
    "logs_permitted": True,
    "identity": "pseudonymous",
}


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def eff(spool_days: int = SPOOL_RETENTION_DAYS, spool_bytes: int = SPOOL_MAX_BYTES):
    return retention.Effective(
        evidence_days=30,
        receipt_days=400,
        spool_days=spool_days,
        spool_bytes=spool_bytes,
        sources={
            k: "default" for k in ("evidence_days", "receipt_days", "spool_days", "spool_bytes")
        },
    )


def snap_of(gen: int) -> str:
    return f"{gen:04x}" * 8


def put(
    gen: int,
    index: int = 0,
    *,
    state: str = "queued",
    age: timedelta = timedelta(0),
    items: int = 3,
    pad: int = 100,
    total: int = 1,
    kind: str = "data",
    signal: Signal = Signal.METRICS,
    dest: str = DEST,
) -> Meta:
    snapshot_id = snap_of(gen)
    if kind == "tombstone":
        batch_id = f"t{int((NOW - age).timestamp()):010d}-ab12cd34-{index:04d}"
    else:
        batch_id = f"g{gen:010d}-{snapshot_id[:16]}-{index:04d}"
    body = json.dumps({"pad": "x" * pad, "n": gen * 100 + index}).encode()
    batch = Batch(
        batch_id=batch_id,
        signal=signal,
        generation=gen,
        snapshot_id=snapshot_id,
        index=index,
        total=total,
        kind=kind,
        body=body,
        items=items,
        record_ids=tuple(f"{gen:08x}{index:04x}{i:020x}" for i in range(items))
        if signal is Signal.LOGS
        else (),
    )
    meta = export_spool.write_batch(
        PROFILE, dest, signal, batch, created_utc=iso(NOW - age), scope=SCOPE
    )
    if state != "queued":
        meta = export_spool.update_meta(
            PROFILE, dest, signal, batch_id, lambda m, s=state: replace(m, state=s)
        )
    return meta


def spool_dir(tmp_path: Path, signal: Signal = Signal.METRICS, dest: str = DEST) -> Path:
    return tmp_path / "data" / "accounting" / PROFILE / "spool" / dest / signal.value


def disk_bytes(tmp_path: Path, signal: Signal = Signal.METRICS) -> int:
    folder = spool_dir(tmp_path, signal)
    return sum(p.stat().st_size for p in folder.iterdir() if p.is_file() and p.suffix == ".json")


def ids(signal: Signal = Signal.METRICS) -> list[str]:
    return [m.batch_id for m in export_spool.list_batches(PROFILE, DEST, signal)]


def delivery(signal: Signal = Signal.METRICS) -> acks.Delivery:
    found = acks.read_delivery(PROFILE, DEST, signal)
    assert found is not None
    return found


def seed_delivery(change, signal: Signal = Signal.METRICS) -> acks.Delivery:
    return acks.update_delivery(PROFILE, DEST, signal, change)


def acked_through(gen: int, signal: Signal = Signal.METRICS) -> acks.Delivery:
    """Delivery whose last FULL ack is ``gen`` (a real published snapshot, one batch)."""
    snapshots.publish(PROFILE, make_envelope(gen))
    return seed_delivery(
        lambda d: acks.outcome(
            d,
            generation=gen,
            snapshot_id=make_envelope(gen).snapshot_id,
            index=0,
            total=1,
            kind="acked",
            now="2026-10-12T13:00:00Z",
        ),
        signal,
    )


def enforce(tmp_path: Path | None = None, **kw):
    kw.setdefault("effective", eff())
    return retention.enforce_spool(PROFILE, DEST, Signal.METRICS, now=NOW, **kw)


# --- effective retention ------------------------------------------------------------------


def test_defaults_are_30_400_7_days_and_100_mib() -> None:
    result = retention.effective_retention(ProfilePolicy())
    assert (result.evidence_days, result.receipt_days) == (30, 400)
    assert (result.spool_days, result.spool_bytes) == (7, 100 * 1024 * 1024)
    assert result.sources == {
        "evidence_days": "default",
        "receipt_days": "default",
        "spool_days": "default",
        "spool_bytes": "default",
    }


def test_local_narrowing_is_reported_as_local_and_wider_values_cannot_extend() -> None:
    narrowed = retention.effective_retention(
        ProfilePolicy(evidence_retention_days=10, receipt_retention_days=100)
    )
    assert (narrowed.evidence_days, narrowed.receipt_days, narrowed.spool_days) == (10, 100, 7)
    assert narrowed.sources["evidence_days"] == narrowed.sources["receipt_days"] == "local"
    assert narrowed.sources["spool_days"] == "default"  # 7 < 10: spool default binds
    wide = retention.effective_retention(
        ProfilePolicy(evidence_retention_days=90, receipt_retention_days=900)
    )
    assert (wide.evidence_days, wide.receipt_days) == (30, 400)
    assert wide.sources["evidence_days"] == wide.sources["receipt_days"] == "default"


def test_managed_ceiling_only_shortens_and_is_named_as_the_source() -> None:
    policy = ProfilePolicy(managed=True)
    ceiling = ProfilePolicy(evidence_retention_days=3, receipt_retention_days=200)
    result = retention.effective_retention(policy, ceiling)
    assert (result.evidence_days, result.receipt_days, result.spool_days) == (3, 200, 3)
    assert result.sources == {
        "evidence_days": "managed_ceiling",
        "receipt_days": "managed_ceiling",
        "spool_days": "managed_ceiling",
        "spool_bytes": "default",
    }
    looser = ProfilePolicy(evidence_retention_days=60, receipt_retention_days=900)
    again = retention.effective_retention(ProfilePolicy(), looser)
    assert (again.evidence_days, again.receipt_days) == (
        30,
        400,
    )  # a larger ceiling extends nothing
    assert again.sources["evidence_days"] == "default"


def test_profile_value_below_the_ceiling_stays_local() -> None:
    result = retention.effective_retention(
        ProfilePolicy(evidence_retention_days=2), ProfilePolicy(evidence_retention_days=5)
    )
    assert result.evidence_days == 2 and result.sources["evidence_days"] == "local"


def test_spool_age_never_outlives_the_evidence_horizon() -> None:
    assert retention.effective_retention(ProfilePolicy(evidence_retention_days=5)).spool_days == 5
    assert retention.effective_retention(ProfilePolicy(evidence_retention_days=7)).spool_days == 7
    assert retention.effective_retention(ProfilePolicy(evidence_retention_days=8)).spool_days == 7


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "30", None])
def test_invalid_policy_values_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        retention.effective_retention(replace(ProfilePolicy(), evidence_retention_days=bad))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        retention.effective_retention(replace(ProfilePolicy(), receipt_retention_days=bad))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        retention.effective_retention(
            ProfilePolicy(),
            replace(ProfilePolicy(), evidence_retention_days=bad),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["evidence_days", "receipt_days", "spool_days", "spool_bytes"])
def test_effective_rejects_non_positive_values(field: str) -> None:
    good = {"evidence_days": 30, "receipt_days": 400, "spool_days": 7, "spool_bytes": 10}
    sources = {k: "default" for k in good}
    with pytest.raises(ValueError):
        retention.Effective(**{**good, field: 0}, sources=sources)
    with pytest.raises(ValueError):
        retention.Effective(**{**good, field: True}, sources=sources)
    retention.Effective(**good, sources=sources)  # positive twin


def test_effective_rejects_unknown_or_incomplete_sources() -> None:
    good = {"evidence_days": 30, "receipt_days": 400, "spool_days": 7, "spool_bytes": 10}
    with pytest.raises(ValueError):
        retention.Effective(**good, sources={k: "because" for k in good})
    with pytest.raises(ValueError):
        retention.Effective(**good, sources={"evidence_days": "default"})


@pytest.mark.parametrize(
    ("policy", "ceiling", "expected_days"),
    [
        (ProfilePolicy(), None, 30),
        (ProfilePolicy(managed=True), ProfilePolicy(evidence_retention_days=7), 7),
        (ProfilePolicy(evidence_retention_days=12), ProfilePolicy(evidence_retention_days=20), 12),
    ],
)
def test_effective_numbers_drive_the_collector_horizons_exactly(
    policy: ProfilePolicy, ceiling: ProfilePolicy | None, expected_days: int
) -> None:
    result = retention.effective_retention(policy, ceiling)
    assert result.evidence_days == expected_days
    clamped = policy_rules.clamp_to_ceiling(policy, ceiling) if ceiling else policy
    assert clamped.evidence_retention_days == result.evidence_days
    assert clamped.receipt_retention_days == result.receipt_days
    expected_day = (NOW - timedelta(days=expected_days)).date().isoformat()
    assert retention_hooks.evidence_horizon_day(NOW, clamped) == expected_day
    receipt_day = (NOW - timedelta(days=result.receipt_days)).date().isoformat()
    assert retention_hooks.aggregate_horizon_day(NOW, clamped) == receipt_day


# --- age bound ----------------------------------------------------------------------------


def test_age_boundary_is_exact_and_one_second_more_expires() -> None:
    exactly = put(5, 0, age=timedelta(days=7))
    older = put(5, 1, age=timedelta(days=7, seconds=1), total=2)
    result = enforce()
    assert ids() == [exactly.batch_id]
    assert (result.expired_age, result.evicted_size, result.unknown_expired) == (1, 0, 0)
    assert result.records_lost == older.items == 3
    assert result.remaining == export_spool.usage(PROFILE, DEST, Signal.METRICS)
    assert result.remaining.batches == 1


def test_queued_expiry_is_an_exact_visible_loss_and_never_moves_the_ack(tmp_path: Path) -> None:
    acked_through(3)
    put(4, 0, age=timedelta(days=8), items=5)
    before = delivery()
    assert before.last_acked_generation == 3
    result = enforce()
    after = delivery()
    assert result.expired_age == 1 and result.records_lost == 5
    assert after.loss_summary == (acks.Loss("spool_age_expired", 1, 5, NOW_Z),)
    assert after.lost_records == 5
    assert (after.last_acked_generation, after.last_acked_snapshot_id) == (
        before.last_acked_generation,
        before.last_acked_snapshot_id,
    )
    assert after.last_acked_at == before.last_acked_at
    described = acks.describe(after)
    assert described["lost_records"] == 5
    assert described["loss_summary"] == [
        {"reason": "spool_age_expired", "batches": 1, "records": 5, "at": NOW_Z}
    ]
    assert [(p.generation, p.lost) for p in after.progress] == [(4, (0,))]


def test_a_lost_generation_can_never_become_fully_acknowledged_but_a_newer_one_can() -> None:
    acked_through(3)
    put(4, 0, age=timedelta(days=9), total=2)
    live = put(4, 1, total=2)
    enforce()
    snapshot_4 = snap_of(4)
    # the surviving sibling is acknowledged: generation 4 still must NOT become acknowledged
    seed_delivery(
        lambda d: acks.outcome(
            d,
            generation=4,
            snapshot_id=snapshot_4,
            index=live.index,
            total=2,
            kind="acked",
            now=NOW_Z,
        )
    )
    assert delivery().last_acked_generation == 3
    # a retransmitted copy of the lost batch cannot revive it either
    seed_delivery(
        lambda d: acks.outcome(
            d, generation=4, snapshot_id=snapshot_4, index=0, total=2, kind="acked", now=NOW_Z
        )
    )
    assert delivery().last_acked_generation == 3
    # positive twin: a later, fully acknowledged generation still advances
    snapshots.publish(PROFILE, make_envelope(5))
    seed_delivery(
        lambda d: acks.outcome(
            d,
            generation=5,
            snapshot_id=make_envelope(5).snapshot_id,
            index=0,
            total=1,
            kind="acked",
            now=NOW_Z,
        )
    )
    assert delivery().last_acked_generation == 5


def test_each_unsent_state_has_its_own_loss_reason_and_counts() -> None:
    put(1, 0, state="queued", age=timedelta(days=10), items=2, total=3)
    put(1, 1, state="in_flight", age=timedelta(days=10), items=4, total=3)
    put(1, 2, state="quarantined", age=timedelta(days=10), items=6, total=3)
    seed_delivery(lambda d: acks.start_in_flight(d, ids()[1], 1, "2026-10-13T11:00:00Z"))
    result = enforce()
    assert (result.expired_age, result.unknown_expired, result.evicted_size) == (2, 1, 0)
    assert result.records_lost == 12
    after = delivery()
    reasons = {(loss.reason, loss.batches, loss.records) for loss in after.loss_summary}
    assert reasons == {
        ("spool_age_expired", 1, 2),
        ("delivery_unknown_expired", 1, 4),
        ("quarantine_expired", 1, 6),
    }
    assert after.lost_records == 12
    assert after.in_flight is None  # the removed in-flight batch is no longer "in flight"
    assert ids() == []
    assert after.progress[0].lost == (0, 1, 2)


def test_tombstone_batches_obey_the_same_bounds_with_their_own_reason() -> None:
    old = put(2, 0, kind="tombstone", signal=Signal.LOGS, age=timedelta(days=8), items=2)
    young = put(2, 1, kind="tombstone", signal=Signal.LOGS, age=timedelta(days=6), items=1, total=2)
    result = retention.enforce_spool(PROFILE, DEST, Signal.LOGS, now=NOW, effective=eff())
    assert [m.batch_id for m in export_spool.list_batches(PROFILE, DEST, Signal.LOGS)] == [
        young.batch_id
    ]
    assert old.batch_id not in ids(Signal.LOGS)
    assert result.expired_age == 1 and result.records_lost == 2
    after = delivery(Signal.LOGS)
    assert after.loss_summary == (acks.Loss("tombstone_expired", 1, 2, NOW_Z),)
    assert after.progress == ()  # a tombstone belongs to no snapshot generation


def test_housekeeping_metas_age_out_silently() -> None:
    for index, state in enumerate(("acked", "partial", "lost", "discarded")):
        put(1, index, state=state, age=timedelta(days=9), total=4)
    kept = put(2, 0, state="acked", age=timedelta(days=1))
    calls: list[Meta] = []
    result = enforce(on_discard=calls.append)
    assert ids() == [kept.batch_id]
    assert (result.expired_age, result.evicted_size, result.unknown_expired) == (0, 0, 0)
    assert (result.records_lost, result.housekeeping_removed) == (0, 4)
    assert calls == []
    assert acks.read_delivery(PROFILE, DEST, Signal.METRICS) is None  # nothing written at all


# --- size bound ---------------------------------------------------------------------------


def test_size_eviction_removes_oldest_first_until_the_exact_bytes_fit(tmp_path: Path) -> None:
    metas = [put(g, 0, age=timedelta(hours=10 - g), pad=50 * g, items=g) for g in (1, 2, 3, 4)]
    sizes = {
        m.batch_id: sum(p.stat().st_size for p in spool_dir(tmp_path).glob(f"{m.batch_id}*"))
        for m in metas
    }
    total = sum(sizes.values())
    assert disk_bytes(tmp_path) == total == export_spool.usage(PROFILE, DEST, Signal.METRICS).bytes
    limit = total - sizes[metas[0].batch_id] - 1  # evicting only the oldest is NOT enough
    result = enforce(effective=eff(spool_bytes=limit))
    assert ids() == [metas[2].batch_id, metas[3].batch_id]
    expected_remaining = sizes[metas[2].batch_id] + sizes[metas[3].batch_id]
    assert expected_remaining <= limit
    assert disk_bytes(tmp_path) == expected_remaining
    assert result.remaining.bytes == expected_remaining and result.remaining.batches == 2
    assert result.bytes_freed == total - expected_remaining
    assert (result.evicted_size, result.expired_age) == (2, 0)
    assert result.records_lost == 1 + 2
    after = delivery()
    assert after.loss_summary == (acks.Loss("spool_size_exceeded", 2, 3, NOW_Z),)
    assert after.lost_records == 3


def test_size_limit_exactly_met_evicts_nothing(tmp_path: Path) -> None:
    put(1, 0, age=timedelta(hours=2))
    put(2, 0, age=timedelta(hours=1))
    limit = disk_bytes(tmp_path)
    result = enforce(effective=eff(spool_bytes=limit))
    assert len(ids()) == 2 and result.evicted_size == 0
    assert acks.read_delivery(PROFILE, DEST, Signal.METRICS) is None
    result = enforce(effective=eff(spool_bytes=limit - 1))
    assert len(ids()) == 1 and result.evicted_size == 1


def test_size_pressure_removes_housekeeping_before_any_unsent_batch(tmp_path: Path) -> None:
    waiting = put(1, 0, age=timedelta(hours=5), pad=500)  # OLDER than the acked one but unsent
    done = put(2, 0, state="acked", age=timedelta(hours=1), pad=500)
    calls: list[Meta] = []
    limit = disk_bytes(tmp_path) - 1
    result = enforce(effective=eff(spool_bytes=limit), on_discard=calls.append)
    assert ids() == [waiting.batch_id] and done.batch_id not in ids()
    assert (result.housekeeping_removed, result.evicted_size, result.records_lost) == (1, 0, 0)
    assert calls == []
    assert acks.read_delivery(PROFILE, DEST, Signal.METRICS) is None  # not a loss, no state change


def test_unsent_batches_follow_housekeeping_when_still_over_the_limit(tmp_path: Path) -> None:
    old = put(1, 0, age=timedelta(hours=5), pad=400)
    put(2, 0, state="acked", age=timedelta(hours=1), pad=400)
    new = put(3, 0, age=timedelta(hours=2), pad=400)
    sizes = [sum(p.stat().st_size for p in spool_dir(tmp_path).glob(f"{m}*")) for m in ids()]
    limit = sizes[2]  # only the newest batch fits
    result = enforce(effective=eff(spool_bytes=limit))
    assert ids() == [new.batch_id]
    assert (result.housekeeping_removed, result.evicted_size) == (1, 1)
    assert delivery().loss_summary == (acks.Loss("spool_size_exceeded", 1, old.items, NOW_Z),)


def test_both_bounds_apply_and_each_removal_reports_the_bound_that_won(tmp_path: Path) -> None:
    expired = put(1, 0, age=timedelta(days=8), items=2, pad=300)
    evictee = put(2, 0, age=timedelta(days=2), items=7, pad=300)
    kept = put(3, 0, age=timedelta(days=1), items=1, pad=300)
    size_of = lambda m: sum(p.stat().st_size for p in spool_dir(tmp_path).glob(f"{m.batch_id}*"))  # noqa: E731
    limit = size_of(kept)
    assert disk_bytes(tmp_path) - size_of(expired) > limit  # the age bound alone is not enough
    result = enforce(effective=eff(spool_bytes=limit))
    assert ids() == [kept.batch_id]
    assert (result.expired_age, result.evicted_size, result.records_lost) == (1, 1, 2 + 7)
    reasons = {(loss.reason, loss.batches, loss.records) for loss in delivery().loss_summary}
    assert reasons == {("spool_age_expired", 1, 2), ("spool_size_exceeded", 1, 7)}
    assert delivery().lost_records == evictee.items + expired.items


def test_in_flight_batch_survives_size_pressure_while_others_can_go(tmp_path: Path) -> None:
    sending = put(1, 0, state="in_flight", age=timedelta(hours=9), pad=200, total=3)
    put(1, 1, age=timedelta(hours=8), pad=200, total=3)
    put(1, 2, age=timedelta(hours=7), pad=200, total=3)
    seed_delivery(lambda d: acks.start_in_flight(d, sending.batch_id, 1, "2026-10-13T11:00:00Z"))
    one = sum(p.stat().st_size for p in spool_dir(tmp_path).glob(f"{sending.batch_id}*"))
    result = enforce(effective=eff(spool_bytes=one))
    assert ids() == [sending.batch_id]  # the oldest, in-flight one stays
    assert result.evicted_size == 2 and result.unknown_expired == 0
    flying = delivery().in_flight
    assert flying is not None and flying.batch_id == sending.batch_id
    # with nothing else left it is the only remaining candidate and goes last
    result = enforce(effective=eff(spool_bytes=one - 1))
    assert ids() == [] and result.evicted_size == 1
    assert delivery().in_flight is None
    assert delivery().loss_summary[0] == acks.Loss("spool_size_exceeded", 3, 9, NOW_Z)


# --- counts, callbacks, safety, idempotence ----------------------------------------------


def test_counts_reflect_the_remaining_spool_after_enforcement() -> None:
    put(1, 0, age=timedelta(days=9), total=4)
    put(1, 1, state="in_flight", age=timedelta(days=9), total=4)
    put(1, 2, state="in_flight", age=timedelta(days=1), total=4)
    put(1, 3, age=timedelta(days=1), total=4)
    seed_delivery(lambda d: acks.set_counts(d, 4, 2))
    enforce()
    after = delivery()
    assert (after.pending_batches, after.unknown_batches) == (2, 1)


def test_on_discard_runs_once_per_removed_unsent_batch_with_the_exact_meta() -> None:
    a = put(1, 0, age=timedelta(days=9), total=3)
    b = put(1, 1, state="quarantined", age=timedelta(days=9), total=3)
    put(1, 2, age=timedelta(days=1), total=3)
    seen: list[Meta] = []
    enforce(on_discard=seen.append)
    assert seen == [a, replace(b)]
    assert [m.state for m in seen] == ["queued", "quarantined"]


def test_loss_is_recorded_before_the_registry_callback_and_before_deletion() -> None:
    put(1, 0, age=timedelta(days=9))
    observed: list[tuple[int, int]] = []

    def probe(meta: Meta) -> None:
        observed.append((delivery().lost_records, len(ids())))

    enforce(on_discard=probe)
    assert observed == [(3, 1)]  # loss already durable while the batch is still on disk
    assert ids() == []


def test_a_failing_registry_callback_keeps_the_batch_and_a_rerun_does_not_double_count() -> None:
    put(1, 0, age=timedelta(days=9), items=4)

    def boom(meta: Meta) -> None:
        raise RuntimeError("registry unavailable")

    with pytest.raises(RuntimeError):
        enforce(on_discard=boom)
    assert len(ids()) == 1 and delivery().lost_records == 4
    enforce()
    assert ids() == [] and delivery().lost_records == 4  # same batch is not counted twice
    assert [loss.batches for loss in delivery().loss_summary] == [1]


def test_nothing_outside_the_spool_and_delivery_dirs_is_touched(tmp_path: Path) -> None:
    root = tmp_path / "data" / "accounting" / PROFILE
    sentinels = {
        root / "collector" / "ledger.json": b'{"ledger":1}',
        root / "snapshots" / "gen-sentinel.json": b'{"snap":1}',
        root / "export" / "records.json": b'{"records":{}}',
        tmp_path / "data" / "notes" / "notes-202610.jsonl": b'{"note":1}\n',
        tmp_path / "claude" / "projects" / "s.jsonl": b'{"transcript":1}\n',
    }
    for path, content in sentinels.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        os.utime(path, (1_700_000_000, 1_700_000_000))
    put(1, 0, age=timedelta(days=9))
    put(2, 0, age=timedelta(hours=1))
    enforce(effective=eff(spool_bytes=1))
    retention.enforce_all(PROFILE, now=NOW, policy=ProfilePolicy())
    for path, content in sentinels.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime == 1_700_000_000
    assert ids() == []


def test_symlinked_spool_directory_is_refused_and_nothing_outside_is_touched(
    tmp_path: Path,
) -> None:
    put(1, 0, age=timedelta(days=9))
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "g0000000001-0001000100010001-0000.json"
    victim.write_bytes(b'{"victim":true}')
    victim_meta = outside / "g0000000001-0001000100010001-0000.meta.json"
    victim_meta.write_bytes(b"{}")
    real = spool_dir(tmp_path)
    moved = tmp_path / "moved-away"
    real.rename(moved)
    real.symlink_to(outside, target_is_directory=True)
    with pytest.raises(export_spool.SpoolCorrupt):
        retention.enforce_spool(PROFILE, DEST, Signal.METRICS, now=NOW, effective=eff())
    assert victim.read_bytes() == b'{"victim":true}' and victim_meta.read_bytes() == b"{}"
    assert sorted(p.name for p in outside.iterdir()) == sorted([victim.name, victim_meta.name])
    assert any(moved.iterdir())  # the real spool was not swept through the link either


def test_a_second_run_is_a_no_op_and_rewrites_nothing(tmp_path: Path) -> None:
    put(1, 0, age=timedelta(days=9), items=2, total=2)
    put(1, 1, age=timedelta(days=1), total=2)
    first = enforce(effective=eff(spool_bytes=10**6))
    doc = tmp_path / "data" / "accounting" / PROFILE / "delivery" / DEST / "metrics.json"
    before = doc.read_bytes(), doc.stat().st_mtime_ns
    survivors = ids()
    second = enforce(effective=eff(spool_bytes=10**6))
    assert first.expired_age == 1 and first.records_lost == 2
    assert (second.expired_age, second.evicted_size, second.unknown_expired) == (0, 0, 0)
    assert (second.records_lost, second.bytes_freed, second.housekeeping_removed) == (0, 0, 0)
    assert (doc.read_bytes(), doc.stat().st_mtime_ns) == before
    assert ids() == survivors


def test_managed_ceiling_of_three_days_expires_a_four_day_old_batch_the_default_keeps() -> None:
    put(1, 0, age=timedelta(days=4))
    policy = ProfilePolicy(managed=True)
    kept = retention.enforce_all(PROFILE, now=NOW, policy=policy)
    assert len(ids()) == 1 and [r.expired_age for r in kept] == [0]
    shortened = retention.enforce_all(
        PROFILE, now=NOW, policy=policy, ceiling=ProfilePolicy(evidence_retention_days=3)
    )
    assert ids() == [] and [(r.destination, r.expired_age) for r in shortened] == [(DEST, 1)]


def test_enforce_all_covers_every_destination_and_signal_in_a_stable_order() -> None:
    put(1, 0, age=timedelta(days=9), dest="org-b")
    put(1, 0, age=timedelta(days=9))
    put(2, 0, kind="tombstone", signal=Signal.LOGS, age=timedelta(days=9))
    seen: list[Meta] = []
    results = retention.enforce_all(
        PROFILE, now=NOW, policy=ProfilePolicy(), on_discard=seen.append
    )
    assert [(r.destination, r.signal, r.expired_age) for r in results] == [
        ("org-b", Signal.METRICS, 1),
        (DEST, Signal.LOGS, 1),
        (DEST, Signal.METRICS, 1),
    ]
    assert len(seen) == 3


def test_naive_now_is_refused_before_anything_is_removed() -> None:
    put(1, 0, age=timedelta(days=9))
    with pytest.raises(ValueError):
        retention.enforce_spool(
            PROFILE, DEST, Signal.METRICS, now=datetime(2026, 10, 13, 12, 0), effective=eff()
        )
    assert len(ids()) == 1


# --- acknowledged snapshot expiry ---------------------------------------------------------


def test_acknowledged_snapshot_is_kept_until_receipt_days_have_passed(tmp_path: Path) -> None:
    acked_through(7)
    collected = datetime(2026, 10, 12, 12, 7, 0, tzinfo=UTC)
    assert make_envelope(7).collected_at == iso(collected)
    doc = tmp_path / "data" / "accounting" / PROFILE / "delivery" / DEST / "metrics.json"
    before = doc.read_bytes()
    at_boundary = collected + timedelta(days=400)
    assert not retention.expire_acknowledged_snapshots(
        PROFILE, DEST, Signal.METRICS, now=at_boundary, effective=eff()
    )
    assert doc.read_bytes() == before
    assert acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS).state == "acknowledged"
    assert retention.expire_acknowledged_snapshots(
        PROFILE, DEST, Signal.METRICS, now=at_boundary + timedelta(seconds=1), effective=eff()
    )
    resolved = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert resolved.state == "acknowledged_snapshot_expired" and resolved.envelope is None
    after = delivery()
    assert after.last_acked_generation == 7 and after.last_acked_snapshot_id is None
    assert after.expired_snapshot_id == make_envelope(7).snapshot_id
    assert not retention.expire_acknowledged_snapshots(
        PROFILE, DEST, Signal.METRICS, now=at_boundary + 2 * DAY, effective=eff()
    )  # idempotent


def test_a_shortened_receipt_horizon_expires_the_acknowledged_snapshot_sooner() -> None:
    acked_through(7)
    now = datetime(2026, 10, 20, 0, 0, 0, tzinfo=UTC)
    assert not retention.expire_acknowledged_snapshots(
        PROFILE, DEST, Signal.METRICS, now=now, effective=eff()
    )
    short = replace(eff(), receipt_days=3)
    assert retention.expire_acknowledged_snapshots(
        PROFILE, DEST, Signal.METRICS, now=now, effective=short
    )
    assert acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS).state == (
        "acknowledged_snapshot_expired"
    )


def test_a_missing_generation_file_releases_the_acknowledged_pin_at_once(tmp_path: Path) -> None:
    acked_through(7)
    snapshots.publish(PROFILE, make_envelope(8))  # newer data exists but is never substituted
    folder = tmp_path / "data" / "accounting" / PROFILE / "snapshots"
    gone = next(folder.glob("gen-0000000007-*.json"))
    gone.unlink()
    assert retention.expire_acknowledged_snapshots(
        PROFILE,
        DEST,
        Signal.METRICS,
        now=datetime(2026, 10, 12, 13, 0, tzinfo=UTC),
        effective=eff(),
    )
    resolved = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert resolved.state == "acknowledged_snapshot_expired" and resolved.envelope is None
    assert delivery().last_acked_generation == 7


def test_nothing_to_release_without_an_acknowledged_snapshot() -> None:
    assert not retention.expire_acknowledged_snapshots(
        PROFILE, DEST, Signal.METRICS, now=NOW, effective=eff()
    )
    assert acks.read_delivery(PROFILE, DEST, Signal.METRICS) is None


# --- status -------------------------------------------------------------------------------


def test_retention_status_is_json_safe_and_reports_limits_usage_and_losses(tmp_path: Path) -> None:
    put(1, 0, age=timedelta(days=9), items=2)
    put(2, 0, age=timedelta(days=1), state="acked")
    enforce()
    put(3, 0, age=timedelta(days=2), pad=300)
    put(4, 0, age=timedelta(days=8), pad=300)
    status = retention.retention_status(
        PROFILE, ProfilePolicy(managed=True), ProfilePolicy(evidence_retention_days=5), now=NOW
    )
    assert json.loads(json.dumps(status)) == status
    assert status["effective"] == {
        "evidence_days": 5,
        "receipt_days": 400,
        "spool_days": 5,
        "spool_bytes": SPOOL_MAX_BYTES,
        "sources": {
            "evidence_days": "managed_ceiling",
            "receipt_days": "default",
            "spool_days": "managed_ceiling",
            "spool_bytes": "default",
        },
    }
    (spool,) = status["spools"]
    assert spool["destination"] == DEST and spool["signal"] == "metrics"
    assert spool["batches"] == 3 and spool["bytes"] == disk_bytes(tmp_path)
    assert (spool["limit_bytes"], spool["limit_days"]) == (SPOOL_MAX_BYTES, 5)
    assert spool["over_age_batches"] == 1  # the 8-day-old one, against the 5-day limit
    assert spool["states"] == {"acked": 1, "queued": 2}
    (loss,) = status["losses"]
    assert loss["lost_records"] == 2
    assert loss["loss_summary"] == [
        {"reason": "spool_age_expired", "batches": 1, "records": 2, "at": NOW_Z}
    ]


def test_retention_status_with_no_spool_or_delivery_creates_nothing(tmp_path: Path) -> None:
    status = retention.retention_status(PROFILE, ProfilePolicy(), now=NOW)
    assert status["spools"] == [] and status["losses"] == []
    assert not (tmp_path / "data" / "accounting").exists()


def test_digest_of_untouched_data_is_stable_across_status_reads(tmp_path: Path) -> None:
    put(1, 0, age=timedelta(days=1))

    def tree() -> str:
        digest = hashlib.sha256()
        for path in sorted((tmp_path / "data").rglob("*")):
            digest.update(str(path).encode())
            if path.is_file():
                digest.update(path.read_bytes())
        return digest.hexdigest()

    before = tree()
    retention.retention_status(PROFILE, ProfilePolicy(), now=NOW)
    assert tree() == before
