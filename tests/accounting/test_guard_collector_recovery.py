"""Damaged committed inputs fail closed; a repaired restart counts the original work once."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.accounting import collector, collector_progress, collector_store as store
from ssgrep.accounting.collector_types import CollectError, CollectOptions, CycleBudget
from ssgrep.accounting.contracts import SCHEMA_VERSION
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE

NOW = scene.at(hour=20)


def seeded(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, scene.at(hour=9))
    assert scene.report(scene.collect(NOW)).generation == 1
    assert scene.counter_total(scene.current()) == 2


def assert_failed_without_publish(condition: str, generation: int, snapshot: bytes) -> None:
    assert (vs.accounting_dir() / "snapshots" / "current.json").read_bytes() == snapshot
    progress = collector_progress.read_progress(PROFILE)
    assert progress is not None
    assert progress["phase"] == "idle"
    assert progress["last_cycle"]["error"] == condition
    assert progress["last_cycle"]["generation"] is None
    assert scene.current().generation == generation


@pytest.mark.parametrize(
    ("damage", "condition"),
    [
        (b'{"schema":', "state_corrupt"),
        (b"[]", "state_corrupt"),
        (b'{"generation":1}', "state_corrupt"),
        (b'{"schema":1,"generation":1.5}', "state_corrupt"),
        (b'{"schema":1}', "state_corrupt"),
        (b'{"schema":true}', "unsupported_schema"),
        (json.dumps({"schema": SCHEMA_VERSION + 1}).encode(), "unsupported_schema"),
    ],
)
def test_invalid_checkpoint_never_replaces_published_totals(
    tmp_path: Path, damage: bytes, condition: str
) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / "collector" / store.CHECKPOINT
    valid = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    path.write_bytes(damage)
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == condition
    assert path.read_bytes() == damage  # no "helpful" reset to empty state
    assert_failed_without_publish(condition, 1, snapshot)
    path.write_bytes(valid)
    assert scene.report(scene.collect(NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


@pytest.mark.parametrize("target", ["checkpoint", "counters"])
def test_state_size_limits_fail_closed_and_recover_after_the_limit_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    seeded(tmp_path)
    directory = vs.accounting_dir() / "collector"
    checkpoint = scene.committed()
    path = (
        directory / store.CHECKPOINT
        if target == "checkpoint"
        else directory / "state" / checkpoint.blobs["counters"]
    )
    limit = "MAX_CHECKPOINT_BYTES" if target == "checkpoint" else "MAX_BLOB_BYTES"
    original = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(store, limit, len(original) - 1)
        with pytest.raises(CollectError) as failed:
            scene.collect(NOW)
        assert failed.value.condition == "state_corrupt"
    assert_failed_without_publish("state_corrupt", 1, snapshot)
    assert path.read_bytes() == original
    assert scene.report(scene.collect(NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


def test_missing_committed_counter_blob_does_not_start_a_new_counter_epoch(tmp_path: Path) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / "collector" / "state" / scene.committed().blobs["counters"]
    original = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    path.unlink()
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == "state_corrupt"
    assert_failed_without_publish("state_corrupt", 1, snapshot)
    path.write_bytes(original)
    path.chmod(0o600)
    resumed = scene.collect(NOW)
    assert resumed.epoch == 1 and scene.counter_total(scene.current()) == 2


@pytest.mark.parametrize("blob", ["counters", "sealed", "reduction", "ledger"])
@pytest.mark.parametrize("damage", [b"null", b"{}"])
def test_invalid_referenced_state_is_not_treated_as_an_absent_first_run(
    tmp_path: Path, blob: str, damage: bytes
) -> None:
    seeded(tmp_path)
    checkpoint = scene.committed()
    key = next(k for k in checkpoint.blobs if k.startswith("ledger-")) if blob == "ledger" else blob
    path = vs.accounting_dir() / "collector" / "state" / checkpoint.blobs[key]
    # An empty sealed-day map is valid; a list cannot be a map of sealed days.
    if blob == "sealed" and damage == b"{}":
        damage = b"[]"
    valid = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    path.write_bytes(damage)
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == "state_corrupt"
    assert scene.committed() == checkpoint  # damage must not masquerade as a scope rotation
    assert_failed_without_publish("state_corrupt", 1, snapshot)
    path.write_bytes(valid)
    assert scene.report(scene.collect(NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


@pytest.mark.parametrize("damage", ["missing", "truncated", "oversized", "malformed"])
def test_committed_evidence_damage_cannot_be_rebuilt_as_empty_history(
    tmp_path: Path, damage: str
) -> None:
    seeded(tmp_path)
    ref = scene.committed().segments[0]
    path = vs.accounting_dir() / "collector" / "evidence" / ref.name
    valid = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    if damage == "missing":
        path.unlink()
    else:
        damaged = {
            "truncated": valid[:-1],
            "oversized": valid + b"extra",
            "malformed": b"!" * len(valid),
        }
        path.write_bytes(damaged[damage])
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW, CollectOptions(recompute=True))
    assert failed.value.condition == "state_corrupt"
    assert_failed_without_publish("state_corrupt", 1, snapshot)
    path.write_bytes(valid)
    path.chmod(0o600)
    assert scene.report(scene.collect(NOW, CollectOptions(recompute=True))).generation == 2
    assert scene.counter_total(scene.current()) == 2
    assert len(vs.evidence_identities()) == len(set(vs.evidence_identities()))


def test_interrupted_epoch_commit_is_recovered_without_recounting_prior_events(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()

    def interrupt(step: str) -> None:
        if step == "checkpoint_committed":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        scene.collect(NOW, CollectOptions(new_epoch=True), on_step=interrupt)
    assert scene.committed().generation == 2
    assert_failed_without_publish("interrupted", 1, snapshot)
    recovered = scene.collect(NOW)
    assert recovered.epoch == 2
    assert scene.report(recovered).generation == 3
    assert scene.counter_total(scene.current()) == 0
    scene.journal_events(scene.execution("after-epoch", "2026-10-12T20:00:01Z"))
    scene.collect(scene.at(hour=21))
    assert scene.counter_total(scene.current()) == 1


def test_partial_backfill_resumes_its_own_offsets_after_restart(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture(day=12)
    vs.pair(vs.projects(tmp_path), 0, scene.at(day=11, hour=9))
    scene.collect(NOW)
    live_offsets = {key: row.offset for key, row in scene.committed().sources.items()}
    before = scene.counter_total(scene.current())
    options = CollectOptions(
        backfill=True,
        since="2026-10-11T00:00:00Z",
        until="2026-10-13T00:00:00Z",
        budget=CycleBudget(bytes=1),
    )
    first = scene.collect(NOW, options)
    assert first.state == "partial"
    job = scene.committed().backfill_job
    assert job["status"] == "running"
    assert any(row["state"]["offset"] > 0 for row in job["sources"].values())
    for _ in range(30):
        result = scene.collect(NOW, options)
        assert scene.counter_total(scene.current()) == before
        assert {key: row.offset for key, row in scene.committed().sources.items()} == live_offsets
        if result.state == "published":
            break
    else:
        pytest.fail("backfill did not finish its bounded resumable scan")
    checkpoint = scene.committed()
    assert not checkpoint.backfill_job
    assert len(checkpoint.backfills) == 1
    assert (
        sum(
            count
            for key, count in checkpoint.backfills[0]["counts"].items()
            if key.startswith("search:")
        )
        == 1
    )
    identities = vs.evidence_identities()
    assert identities and len(identities) == len(set(identities))
    scene.collect(NOW, CollectOptions(backfill=True, since=options.since, until=options.until))
    assert scene.committed().backfills == checkpoint.backfills
    assert vs.evidence_identities() == identities


def test_worker_failed_cycle_reports_corruption_then_recovers_without_a_new_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    monkeypatch.setattr(collector, "_now", lambda: NOW)
    path = vs.accounting_dir() / "collector" / store.CHECKPOINT
    valid = path.read_bytes()
    path.write_bytes(b"{")
    cycle = collector.run_worker_cycle(PROFILE)
    failed = cycle(lambda: False)
    assert (failed.status, failed.error, failed.generation) == ("failed", "state_corrupt", None)
    assert scene.current().generation == 1
    path.write_bytes(valid)
    succeeded = cycle(lambda: False)
    assert (succeeded.status, succeeded.generation) == ("published", 2)
    assert scene.counter_total(scene.current()) == 2
    progress = collector_progress.read_progress(PROFILE)
    assert progress is not None and progress["cycles"] == 3
    assert progress["kind"] == "serve" and progress["last_cycle"]["error"] is None
