"""Crash-at-every-publish-boundary (real subprocesses) and the torn-read hammer."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import (
    collector_progress,
    collector_store as store,
    plainfile,
    service,
    snapshots,
)
from ssgrep.accounting.collector_types import CollectError, CollectOptions
from ssgrep.accounting.contracts import SCHEMA_VERSION
from ssgrep.accounting.lease import inspect_lease
from ssgrep.accounting.ledger_contracts import SnapshotEnvelope
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.snapshots_driver import CRASH_CODE, gen_name, make_envelope
from tests.accounting.test_savings_cli import BINARY, document

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "personal-local"
FAULTS = ("before", *snapshots.STEPS)
# Faults after which gen N+1 is a complete named file / has become current.
LINKED = {"gen_linked", "current_tmp_written", "current_renamed", "published"}
SWAPPED = {"current_renamed", "published"}
LEAVES_TEMPORARY = {"gen_tmp_written", "current_tmp_written"}
OLD, NEW = 4, 5


def snap_dir(tmp_path: Path) -> Path:
    return tmp_path / "data" / "accounting" / PROFILE / "snapshots"


def child(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SSGREP_DATA_DIR": str(tmp_path / "data")}
    return subprocess.run(
        [sys.executable, "-m", "tests.accounting.snapshots_driver", *args],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def crash_publish(tmp_path: Path, generation: int, fault: str) -> None:
    done = child(tmp_path, "publish", PROFILE, str(generation), fault)
    expected = 0 if fault == "none" else CRASH_CODE
    assert done.returncode == expected, done.stderr


def temporaries(tmp_path: Path) -> list[str]:
    return [p.name for p in snap_dir(tmp_path).iterdir() if p.name.endswith(".tmp")]


def finals(tmp_path: Path) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes() for p in snap_dir(tmp_path).iterdir() if not p.name.endswith(".tmp")
    }


def assert_only_complete_finals(tmp_path: Path) -> None:
    """Every final name decodes completely and matches the canonical bytes of its generation."""
    for name, data in finals(tmp_path).items():
        envelope = snapshots.decode(data)  # raises on any partial/mixed document
        assert data == snapshots.encode(envelope)
        assert data == snapshots.encode(make_envelope(envelope.generation))
        if name != "current.json":
            assert name == gen_name(envelope.generation)


@pytest.fixture
def seeded(tmp_path: Path) -> Path:
    """A complete generation OLD published and current."""
    snapshots.publish(PROFILE, make_envelope(OLD))
    assert snapshots.current_generation(PROFILE) == OLD
    return tmp_path


@pytest.mark.parametrize("fault", FAULTS)
def test_crash_over_existing_generation_leaves_old_or_new_complete(
    seeded: Path, fault: str
) -> None:
    old_bytes, new_bytes = (
        snapshots.encode(make_envelope(OLD)),
        snapshots.encode(make_envelope(NEW)),
    )
    crash_publish(seeded, NEW, fault)

    current = (snap_dir(seeded) / "current.json").read_bytes()
    assert current == (new_bytes if fault in SWAPPED else old_bytes)
    assert snapshots.decode(current).generation == (NEW if fault in SWAPPED else OLD)
    assert_only_complete_finals(seeded)
    expected_gens = {gen_name(4)} | ({gen_name(5)} if fault in LINKED else set())
    assert {n for n in finals(seeded) if n.startswith("gen-")} == expected_gens
    assert bool(temporaries(seeded)) == (fault in LEAVES_TEMPORARY)
    # the reader facade sees exactly the same complete envelope
    envelope, failure, _ = service.read_snapshot(PROFILE)
    assert failure is None and envelope is not None
    assert snapshots.encode(envelope) == current


@pytest.mark.parametrize("fault", FAULTS)
def test_sweep_removes_only_temporaries_after_a_crash(seeded: Path, fault: str) -> None:
    crash_publish(seeded, NEW, fault)
    before = finals(seeded)
    leftovers = len(temporaries(seeded))
    assert snapshots.sweep_temporaries(PROFILE) == leftovers == int(fault in LEAVES_TEMPORARY)
    assert temporaries(seeded) == []
    assert finals(seeded) == before
    assert snapshots.sweep_temporaries(PROFILE) == 0


@pytest.mark.parametrize("fault", FAULTS)
def test_publishing_again_converges_without_duplicates(seeded: Path, fault: str) -> None:
    crash_publish(seeded, NEW, fault)
    snapshots.sweep_temporaries(PROFILE)
    snapshots.publish(PROFILE, make_envelope(NEW))
    new_bytes = snapshots.encode(make_envelope(NEW))
    assert finals(seeded) == {
        "current.json": new_bytes,
        gen_name(4): snapshots.encode(make_envelope(OLD)),
        gen_name(5): new_bytes,
    }
    assert temporaries(seeded) == []
    assert snapshots.next_generation(PROFILE) == NEW + 1


@pytest.mark.parametrize("fault", FAULTS)
def test_publishing_again_without_sweep_also_converges(seeded: Path, fault: str) -> None:
    crash_publish(seeded, NEW, fault)
    snapshots.publish(PROFILE, make_envelope(NEW))
    assert snapshots.current_generation(PROFILE) == NEW
    assert (snap_dir(seeded) / "current.json").read_bytes() == snapshots.encode(make_envelope(NEW))
    assert len([n for n in finals(seeded) if n.startswith("gen-")]) == 2
    snapshots.sweep_temporaries(PROFILE)
    assert temporaries(seeded) == []


@pytest.mark.parametrize("fault", sorted(LINKED))
def test_repair_current_after_crash_between_commit_and_swap(seeded: Path, fault: str) -> None:
    crash_publish(seeded, NEW, fault)
    needs_repair = fault not in SWAPPED
    assert snapshots.current_generation(PROFILE) == (OLD if needs_repair else NEW)
    assert snapshots.repair_current(PROFILE, NEW) is needs_repair
    assert (snap_dir(seeded) / "current.json").read_bytes() == snapshots.encode(make_envelope(NEW))
    assert snapshots.repair_current(PROFILE, NEW) is False
    assert snapshots.generations(PROFILE) == [OLD, NEW]


@pytest.mark.parametrize("fault", ["before", "gen_tmp_written"])
def test_repair_current_cannot_invent_a_generation_that_never_committed(
    seeded: Path, fault: str
) -> None:
    crash_publish(seeded, NEW, fault)
    with pytest.raises(FileNotFoundError):
        snapshots.repair_current(PROFILE, NEW)
    assert snapshots.current_generation(PROFILE) == OLD
    assert snapshots.repair_current(PROFILE, OLD) is False


@pytest.mark.parametrize("fault", FAULTS)
def test_crash_during_the_first_ever_publish(tmp_path: Path, fault: str) -> None:
    crash_publish(tmp_path, 1, fault)
    new_bytes = snapshots.encode(make_envelope(1))
    directory = snap_dir(tmp_path)
    if fault == "before":
        assert not directory.exists()  # nothing was created before the first step
        return
    current = directory / "current.json"
    if fault in SWAPPED:
        assert current.read_bytes() == new_bytes
    else:
        assert not current.exists()
    assert_only_complete_finals(tmp_path)
    assert bool(temporaries(tmp_path)) == (fault in LEAVES_TEMPORARY)
    assert service.read_snapshot(PROFILE)[0] == (make_envelope(1) if fault in SWAPPED else None)
    snapshots.sweep_temporaries(PROFILE)
    snapshots.publish(PROFILE, make_envelope(1))
    assert finals(tmp_path) == {"current.json": new_bytes, gen_name(1): new_bytes}


def test_completed_child_publish_matches_in_process_bytes(seeded: Path) -> None:
    crash_publish(seeded, NEW, "none")
    new_bytes = snapshots.encode(make_envelope(NEW))
    assert finals(seeded)["current.json"] == new_bytes
    assert finals(seeded)[gen_name(5)] == new_bytes
    assert temporaries(seeded) == []


def test_a_child_that_crashes_every_step_in_a_row_still_leaves_a_consistent_store(
    seeded: Path,
) -> None:
    for generation, fault in (
        (5, "current_tmp_written"),
        (6, "gen_linked"),
        (7, "gen_tmp_written"),
    ):
        crash_publish(seeded, generation, fault)
        assert_only_complete_finals(seeded)
        assert snapshots.current_generation(PROFILE) == OLD
    assert snapshots.generations(PROFILE) == [4, 5, 6]
    assert snapshots.next_generation(PROFILE) == 7
    assert snapshots.sweep_temporaries(PROFILE) == 2
    assert snapshots.repair_current(PROFILE, 6) is True
    assert snapshots.current_generation(PROFILE) == 6


# --- torn-read hammer ---------------------------------------------------------------------------

PUBLISHES = 200
BULK = 150_000


def _verified(envelope: SnapshotEnvelope) -> bool:
    try:
        return snapshots.snapshot_identity(envelope.__dict__) == envelope.snapshot_id
    except ValueError:
        return False


def test_readers_never_see_a_torn_or_regressing_snapshot_during_200_publishes(
    tmp_path: Path,
) -> None:
    snapshots.publish(PROFILE, make_envelope(1, BULK))
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SSGREP_DATA_DIR": str(tmp_path / "data")}
    reader = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.accounting.snapshots_driver",
            "read",
            PROFILE,
            str(ready),
            str(stop),
        ],
        env=env,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    seen: list[int] = []
    problems: list[str] = []
    halt = threading.Event()

    def hammer() -> None:
        while not halt.is_set():
            envelope, failure, why = service.read_snapshot(PROFILE)
            if envelope is None:
                problems.append(f"unreadable: {failure} {why}")
            elif not _verified(envelope):
                problems.append(f"unverifiable generation {envelope.generation}")
            elif seen and envelope.generation < seen[-1]:
                problems.append(f"regressed {seen[-1]} -> {envelope.generation}")
            else:
                seen.append(envelope.generation)

    thread = threading.Thread(target=hammer)
    thread.start()
    try:
        for _ in range(600):  # wait (bounded) for the subprocess reader's first complete read
            if ready.exists() or reader.poll() is not None:
                break
            threading.Event().wait(0.1)
        assert ready.exists(), "reader subprocess never completed a read"
        for generation in range(2, PUBLISHES + 1):
            snapshots.publish(PROFILE, make_envelope(generation, BULK + generation))
        while seen[-1] != PUBLISHES:  # the thread must observe the final generation too
            threading.Event().wait(0.01)
    finally:
        halt.set()
        thread.join(timeout=60)
        stop.write_text("stop")
        out, _ = reader.communicate(timeout=120)

    assert problems == []
    assert seen[0] <= seen[-1] == PUBLISHES and len(set(seen)) >= 2
    assert seen == sorted(seen)
    reads, first, last, bad = (int(value) for value in out.split())
    assert bad == 0 and reads > 0 and first <= last <= PUBLISHES
    assert snapshots.current_generation(PROFILE) == PUBLISHES
    assert (snap_dir(tmp_path) / "current.json").read_bytes() == snapshots.encode(
        make_envelope(PUBLISHES, BULK + PUBLISHES)
    )
    assert snapshots.generations(PROFILE) == list(range(1, PUBLISHES + 1))
    assert temporaries(tmp_path) == []


# --- discard_uncommitted: lease-holder recovery after a crash past staging -----------------------


@pytest.mark.parametrize("fault", ["gen_tmp_written", "gen_linked", "current_tmp_written"])
def test_discard_uncommitted_removes_the_staged_generation_and_keeps_old_current(
    seeded: Path, fault: str
) -> None:
    crash_publish(seeded, NEW, fault)
    old_bytes = snapshots.encode(make_envelope(OLD))
    staged = fault != "gen_tmp_written"
    assert (gen_name(NEW) in finals(seeded)) is staged
    assert snapshots.discard_uncommitted(PROFILE, OLD) == ([NEW] if staged else [])
    assert finals(seeded) == {"current.json": old_bytes, gen_name(OLD): old_bytes}
    assert temporaries(seeded) == []
    assert snapshots.discard_uncommitted(PROFILE, OLD) == []
    assert snapshots.next_generation(PROFILE) == NEW  # the discarded number is reusable
    snapshots.publish(PROFILE, make_envelope(NEW))
    assert snapshots.current_generation(PROFILE) == NEW


@pytest.mark.parametrize("fault", ["current_renamed", "published"])
def test_discard_uncommitted_refuses_when_current_is_ahead_of_the_checkpoint(
    seeded: Path, fault: str
) -> None:
    crash_publish(seeded, NEW, fault)
    before = finals(seeded)
    with pytest.raises(ValueError, match="current_ahead_of_checkpoint"):
        snapshots.discard_uncommitted(PROFILE, OLD)
    assert finals(seeded) == before
    assert before["current.json"] == snapshots.encode(make_envelope(NEW))
    assert temporaries(seeded) == []


def test_discard_uncommitted_blocks_without_touching_uniquely_pinned_generations(
    tmp_path: Path,
) -> None:
    for number in (1, 2, 3):
        snapshots.publish(PROFILE, make_envelope(number))
    for number in (4, 5, 6):
        snapshots.stage_generation(PROFILE, make_envelope(number))
    plainfile.write_private(
        tmp_path / "data",
        "accounting",
        PROFILE,
        "delivery",
        "ops",
        "metrics.json",
        data=('{"last_acked_snapshot_id": "' + make_envelope(5).snapshot_id + '"}').encode(),
    )
    leftover = snap_dir(tmp_path) / ".current.json.1.abcd.tmp"
    leftover.write_bytes(b'{"par')
    leftover.chmod(0o600)
    with pytest.raises(ValueError, match="uncommitted_pinned_generation"):
        snapshots.discard_uncommitted(PROFILE, 3)
    assert snapshots.generations(PROFILE) == [1, 2, 3, 4, 5, 6]
    assert snapshots.current_generation(PROFILE) == 3
    assert leftover.exists()  # The explicit block happens before any destructive cleanup.
    with pytest.raises(ValueError, match="uncommitted_pinned_generation"):
        snapshots.discard_uncommitted(PROFILE, 3)


def test_discard_uncommitted_without_snapshots_is_empty_and_creates_nothing(
    tmp_path: Path,
) -> None:
    assert snapshots.discard_uncommitted(PROFILE, 1) == []
    assert not (tmp_path / "data").exists()
    for hostile in (-1, 10**10, True, None):
        with pytest.raises(ValueError):
            snapshots.discard_uncommitted(PROFILE, hostile)  # ty: ignore[invalid-argument-type]


def test_discard_uncommitted_works_without_a_usable_current_pointer(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    snapshots.stage_generation(PROFILE, make_envelope(2))
    (snap_dir(tmp_path) / "current.json").write_bytes(b'{"half"')
    assert snapshots.discard_uncommitted(PROFILE, 1) == [2]
    assert snapshots.generations(PROFILE) == [1]


@pytest.mark.parametrize("fault", ["gen_linked", "current_tmp_written"])
def test_discard_everything_after_a_crash_in_the_first_ever_publish(
    tmp_path: Path, fault: str
) -> None:
    crash_publish(tmp_path, 1, fault)
    assert gen_name(1) in finals(tmp_path) and "current.json" not in finals(tmp_path)
    assert snapshots.discard_uncommitted(PROFILE, 0) == [1]
    assert finals(tmp_path) == {}
    assert temporaries(tmp_path) == []
    assert snapshots.next_generation(PROFILE) == 1
    snapshots.publish(PROFILE, make_envelope(snapshots.next_generation(PROFILE)))
    new_bytes = snapshots.encode(make_envelope(1))
    assert finals(tmp_path) == {"current.json": new_bytes, gen_name(1): new_bytes}


def test_discard_zero_deletes_everything_even_with_a_damaged_current(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    snapshots.stage_generation(PROFILE, make_envelope(2))
    (snap_dir(tmp_path) / "current.json").write_bytes(b'{"half"')
    assert snapshots.discard_uncommitted(PROFILE, 0) == [1, 2]
    assert snapshots.generations(PROFILE) == []


def test_discard_zero_refuses_when_a_usable_current_exists(seeded: Path) -> None:
    before = finals(seeded)
    with pytest.raises(ValueError, match="current_ahead_of_checkpoint"):
        snapshots.discard_uncommitted(PROFILE, 0)
    assert finals(seeded) == before


# --- real collector scenes: damaged committed inputs fail closed and recover exactly ------------

NOW = scene.at(hour=20)
NOW2 = scene.at(hour=21)
CLOCK = scene.at(hour=9)


def seeded_scene(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, CLOCK)
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
    seeded_scene(tmp_path)
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
    seeded_scene(tmp_path)
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
    seeded_scene(tmp_path)
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
    seeded_scene(tmp_path)
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
    seeded_scene(tmp_path)
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
    seeded_scene(tmp_path)
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


@pytest.mark.parametrize("component", ["worker", "snapshots"])
def test_unsafe_worker_or_snapshot_path_is_refused_without_following_it(
    tmp_path: Path, component: str
) -> None:
    seeded_scene(tmp_path)
    directory = vs.accounting_dir() / component
    saved = tmp_path / f"saved-{component}"
    current = vs.accounting_dir() / "snapshots" / "current.json"
    before = current.read_bytes()
    directory.rename(saved)
    directory.symlink_to(saved, target_is_directory=True)
    try:
        with pytest.raises(CollectError) as failed:
            scene.collect(NOW)
        assert failed.value.condition == "storage_unsafe"
        assert current.read_bytes() == before
        assert scene.committed().generation == 1
    finally:
        directory.unlink()
        saved.rename(directory)
    assert scene.report(scene.collect(NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


def test_malformed_current_pointer_is_repaired_from_the_committed_generation(
    tmp_path: Path,
) -> None:
    seeded_scene(tmp_path)
    path = vs.accounting_dir() / "snapshots" / "current.json"
    path.write_bytes(b"{")
    result = scene.collect(NOW)
    assert scene.report(result).generation == 2
    assert scene.counter_total(scene.current()) == 2
    assert scene.current().generation == 2


def test_missing_committed_snapshot_is_corruption_not_permission_to_reset(tmp_path: Path) -> None:
    seeded_scene(tmp_path)
    directory = vs.accounting_dir() / "snapshots"
    generation = next(directory.glob("gen-*.json"))
    original = generation.read_bytes()
    current = (directory / "current.json").read_bytes()
    generation.unlink()
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == "state_corrupt"
    assert (directory / "current.json").read_bytes() == current
    generation.write_bytes(original)
    generation.chmod(0o600)
    assert scene.report(scene.collect(NOW)).generation == 2


@pytest.mark.parametrize("component", ["state", "evidence"])
def test_missing_committed_input_directory_is_not_an_empty_dataset(
    tmp_path: Path, component: str
) -> None:
    seeded_scene(tmp_path)
    directory = vs.accounting_dir() / "collector" / component
    saved = tmp_path / f"saved-{component}"
    before = scene.current()
    directory.rename(saved)
    vs.filler(vs.projects(tmp_path), "new-input", NOW, 1)
    accounting_before = vs.tree(vs.accounting_dir(), skip=("worker/",))
    retained_before = vs.tree(saved)
    originals_before = vs.tree(vs.projects(tmp_path))
    try:
        with pytest.raises(CollectError) as failed:
            scene.collect(NOW)
        assert failed.value.condition == "state_corrupt"
        assert scene.current() == before
        assert not directory.exists()  # A missing committed input must never be staged anew.
        assert vs.tree(vs.accounting_dir(), skip=("worker/",)) == accounting_before
        assert vs.tree(saved) == retained_before
        assert vs.tree(vs.projects(tmp_path)) == originals_before
    finally:
        saved.rename(directory)
    assert scene.report(scene.collect(NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2
    assert vs.tree(vs.projects(tmp_path)) == originals_before


# --- SIGKILL a real worker at EVERY fault-hook step of a cycle -----------------------------------
#
# A scene runs cycle 1 (generation 1 published). New evidence arrives. A second PROCESS then runs
# cycle 2 and kills itself with SIGKILL at one hook step (no ``finally``, no cleanup). Then:
#
# * readers see exactly one COMPLETE generation (the previous or the new one): ``current.json``
#   decodes, verifies its own ``snapshot_id``, and equals the bytes of the generation file it names;
# * the committed collector state is the previous or the new complete state (ledger digest);
# * the next collect (same process type as production: no hand repair) converges to the canonical
#   ledger digest, retained evidence, live counters and ``snapshot_id`` of an UNINTERRUPTED run of
#   the same scene computed in an independent data root.

BEFORE_COMMIT = (
    "started",
    "evidence_written",
    "state_written",
    "before_stage",
    "gen_tmp_written",
    "gen_linked",
    "before_commit",
)
AFTER_COMMIT = (
    "checkpoint_committed",
    "receipts_written",
    "current_tmp_written",
    "current_renamed",
    "published",
)
CURRENT_SWAPPED = {"current_renamed", "published"}
CYCLE_STEPS = (*BEFORE_COMMIT, *AFTER_COMMIT)


facts = vs.facts
Facts = vs.Facts


def drive(root: Path, step: str, marks: Path) -> subprocess.CompletedProcess[str]:
    """One cycle in ANOTHER process that SIGKILLs itself at ``step`` (``none``: run to the end)."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.accounting.verify_driver",
            "collect",
            step,
            NOW2.isoformat(),
            "--marks",
            str(marks),
        ],
        env=vs.env_for(root),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def build_cycle_one(root: Path) -> Facts:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)
    first = scene.collect(NOW)
    assert first.state == "published" and scene.report(first).complete
    return facts()


def add_batch(root: Path, indices: tuple[int, ...]) -> None:
    for n in indices:
        vs.pair(vs.projects(root), n, CLOCK)


wipe = vs.wipe


def reference(root: Path) -> dict[str, Facts]:
    """An UNINTERRUPTED run of the scene on THIS root's paths, then wiped for the crash run.

    Witness/action locators inside the private ledger state name the transcript files, so the
    canonical ledger digest is only comparable between runs over the same paths.
    """
    one = build_cycle_one(root)
    add_batch(root, (2, 3))
    second = scene.collect(NOW2)
    assert second.state == "published" and scene.report(second).complete
    two = facts()
    add_batch(root, (4,))
    third = scene.collect(NOW2)
    assert third.state == "published" and scene.report(third).complete
    three = facts()
    # The scene really changes the ledger between the cycles; without this the old-vs-new
    # distinctions in the crash tests would be vacuous.
    assert len({one.ledger, two.ledger, three.ledger}) == 3
    assert len({one.snapshot_id, two.snapshot_id, three.snapshot_id}) == 3
    assert len(one.evidence) < len(two.evidence) < len(three.evidence)
    out = {"one": one, "two": two, "three": three}
    wipe(root)
    return out


def assert_one_complete_generation(old: Facts, new: Facts, step: str) -> Facts:
    """Readers' view right after the crash: exactly the old or the new generation, intact."""
    snap = vs.accounting_dir() / "snapshots"
    raw = (snap / "current.json").read_bytes()
    envelope = snapshots.decode(raw)  # raises on any partial or self-inconsistent document
    named = f"gen-{envelope.generation:010d}-{envelope.snapshot_id}.json"
    assert (snap / named).read_bytes() == raw, "current.json is not the bytes of its generation"
    read, failure, _ = service.read_snapshot(PROFILE)
    assert failure is None and read is not None and read.snapshot_id == envelope.snapshot_id
    if step in CURRENT_SWAPPED:
        assert envelope.generation == old.generation + 1
        assert envelope.snapshot_id == new.snapshot_id  # the NEW generation, complete
    else:
        assert raw == old.current_bytes, "readers must still see the previous generation exactly"
    for path in snap.iterdir():  # no reader-visible partial under a final name
        if not path.name.startswith("."):
            snapshots.decode(path.read_bytes())
    native_evidence, ownership = vs.evidence_semantics()
    return Facts(
        vs.ledger_digest(),
        tuple(vs.evidence_identities()),
        envelope.snapshot_id,
        "",
        envelope.generation,
        raw,
        native_evidence,
        ownership,
    )


@pytest.mark.parametrize("variant", ["plain", "late"])
@pytest.mark.parametrize("step", CYCLE_STEPS)
def test_sigkill_at_every_hook_exposes_one_complete_generation_and_recovers_exactly(
    step: str, variant: str, tmp_path: Path
) -> None:
    references = reference(tmp_path)
    ref_old, ref_crashed = references["one"], references["two"]
    ref_new = references["two" if variant == "plain" else "three"]
    old = build_cycle_one(tmp_path)
    assert (old.ledger, old.snapshot_id) == (ref_old.ledger, ref_old.snapshot_id)  # same scene
    add_batch(tmp_path, (2, 3))

    marks = tmp_path / "marks"
    died = drive(tmp_path, step, marks)
    assert died.returncode == -signal.SIGKILL, (died.returncode, died.stderr[-2000:])
    assert (marks / "died-at").read_text() == step  # the fault really struck at this step
    orphan = store.write_blob(PROFILE, "crash-unreferenced", {"canary": "not checkpointed"})
    orphan_path = vs.accounting_dir() / "collector" / "state" / orphan
    assert json.loads(orphan_path.read_bytes()) == {"canary": "not checkpointed"}
    held = inspect_lease(PROFILE)
    assert held is None or not held.alive  # the kernel dropped the lease with the process

    seen = assert_one_complete_generation(old, ref_crashed, step)
    if step in BEFORE_COMMIT:
        assert seen.ledger == old.ledger and seen.evidence == old.evidence  # old state, whole
    else:
        assert (committed := vs.checkpoint()) is not None and committed.generation == 2
        assert seen.ledger == ref_crashed.ledger  # the new state is committed, whole

    if variant == "late":
        add_batch(tmp_path, (4,))  # more evidence lands before the restart
    committed_before = scene.committed().generation
    recovered = scene.collect(NOW2)
    assert recovered.state == "published", recovered  # not already_running: dead lease recovered
    assert scene.report(recovered).complete
    final = facts()
    assert final.generation == committed_before + 1  # one new generation, never a reused number
    assert final.ledger == ref_new.ledger  # canonical rows == an uninterrupted run
    assert final.evidence == ref_new.evidence  # accepted exactly once: no loss, no duplicate
    assert final.snapshot_id == ref_new.snapshot_id
    assert final.counters == ref_new.counters  # live counters: each event once
    again = scene.collect(NOW2)  # and the system is quiescent: a rerun changes nothing durable
    assert again.state == "published"
    assert (facts().ledger, facts().snapshot_id) == (ref_new.ledger, ref_new.snapshot_id)
    assert not orphan_path.exists()  # recovery removes a proven unreferenced durable blob
    assert not [p for p in (vs.accounting_dir() / "snapshots").iterdir() if p.name.endswith(".tmp")]


def test_the_uninterrupted_driver_matches_the_in_process_reference(tmp_path: Path) -> None:
    """The positive twin of the crash runs: the same driver with no fault finishes identically."""
    references = reference(tmp_path)
    build_cycle_one(tmp_path)
    add_batch(tmp_path, (2, 3))
    done = drive(tmp_path, "none", tmp_path / "marks")
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.splitlines()[-1])["state"] == "published"
    final, ref = facts(), references["two"]
    assert (final.ledger, final.snapshot_id, final.evidence) == (
        ref.ledger,
        ref.snapshot_id,
        ref.evidence,
    )
    assert final.generation == 2


def test_a_committed_but_unpublished_generation_is_repaired_before_any_new_work(
    tmp_path: Path,
) -> None:
    """Killed after the commit, before current.json moved; the NEXT cycle itself then fails.

    Recovery must already have made the committed generation current by the time anything new
    can go wrong, so a reader never keeps the superseded generation behind a failing worker.
    """
    old = build_cycle_one(tmp_path)
    add_batch(tmp_path, (2, 3))
    marks = tmp_path / "marks"
    died = drive(tmp_path, "checkpoint_committed", marks)
    assert died.returncode == -signal.SIGKILL, died.stderr[-2000:]
    assert facts().current_bytes == old.current_bytes  # readers still see generation 1
    committed = vs.checkpoint()
    assert committed is not None and committed.generation == 2

    def fail_after_recovery(name: str) -> None:
        if name == "evidence_written":  # recovery ran at the start of the cycle, before this
            raise RuntimeError("synthetic failure after recovery")

    with pytest.raises(Exception, match="Collection failed"):
        scene.collect(NOW2, on_step=fail_after_recovery)
    repaired = snapshots.decode((vs.accounting_dir() / "snapshots" / "current.json").read_bytes())
    assert repaired.generation == 2  # the committed generation is what readers see now
    assert repaired.snapshot_id != old.snapshot_id


# --- a REAL concurrent reader while another real process sits PARKED inside a cycle ------------
#
# A reader process hammers ``service.read_snapshot`` and the real ``ssgrep savings --json`` command
# while another real process sits PARKED inside a cycle (lease held, replacement generation being
# produced). The old generation must stay complete, verified and identical the whole time; after
# release the reader sees the new generation, and never a mix.

PARK = ("evidence_written", "state_written", "before_stage", "gen_linked", "before_commit")


class Holder:
    """A real worker process parked at a hook step with the lease held."""

    def __init__(self, root: Path, step: str, *, now: datetime = NOW) -> None:
        self.hold = root / f"hold-{step}"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.accounting.verify_driver",
                "collect",
                step,
                now.isoformat(),
                "--hold",
                str(self.hold),
            ],
            env=vs.env_for(root),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 120
        while not (self.hold / "reached").exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(f"holder never parked: {self.process.communicate()}")
            time.sleep(0.02)

    def release(self) -> dict[str, Any]:
        (self.hold / "release").write_text("go")
        out, err = self.process.communicate(timeout=120)
        assert self.process.returncode == 0, err[-2000:]
        return json.loads(out.splitlines()[-1])

    def kill(self) -> None:
        self.process.send_signal(signal.SIGKILL)
        self.process.communicate(timeout=60)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate()


def seeded_journey(root: Path) -> snapshots.SnapshotEnvelope:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)
    assert scene.collect(NOW).state == "published"
    envelope = scene.current()
    vs.pair(vs.projects(root), 2, CLOCK)  # new evidence: the next cycle replaces the generation
    return envelope


def savings(root: Path) -> dict:
    done = subprocess.run(
        [str(BINARY), "savings", "--json"],
        env=vs.env_for(root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return document(done)["data"]


READER = """
import os, sys, time
from ssgrep.accounting import service, snapshots

ready, stop = sys.argv[1], sys.argv[2]
seen, bad, torn = {}, 0, 0
while not os.path.exists(stop):
    envelope, failure, _why = service.read_snapshot("personal-local")
    if envelope is None:
        bad += 1
        continue
    try:
        ok = snapshots.snapshot_identity(envelope.__dict__) == envelope.snapshot_id
    except ValueError:
        ok = False
    torn += not ok
    identity = (envelope.generation, envelope.snapshot_id)
    seen[identity] = seen.get(identity, 0) + 1
    if not os.path.exists(ready):
        open(ready, "w").close()
print(sorted(seen), bad, torn)
"""


def start_reader(root: Path) -> tuple[subprocess.Popen[str], Path]:
    stop = root / "reader-stop"
    ready = root / "reader-ready"
    process = subprocess.Popen(
        [sys.executable, "-c", READER, str(ready), str(stop)],
        env=vs.env_for(root),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 120
    while not ready.exists():
        assert process.poll() is None and time.monotonic() < deadline, process.communicate()
        time.sleep(0.02)
    return process, stop


def finish_reader(process: subprocess.Popen[str], stop: Path) -> tuple[list, int, int]:
    stop.write_text("stop")
    out, err = process.communicate(timeout=120)
    assert process.returncode == 0, err[-2000:]
    head, bad, torn = out.rsplit(" ", 2)
    return eval(head), int(bad), int(torn)  # noqa: S307 - our own literal list of int/str tuples


def test_old_generation_stays_readable_and_identical_at_every_parked_step(tmp_path: Path) -> None:
    old = seeded_journey(tmp_path)
    old_bytes = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    reader, stop = start_reader(tmp_path)
    try:
        for step in PARK:
            holder = Holder(tmp_path, step)
            try:
                # the cycle is genuinely pending: lease held by the holder, replacement in flight
                owner = inspect_lease(PROFILE)
                assert owner is not None and owner.alive and owner.pid == holder.process.pid
                for _ in range(3):
                    read, failure, _ = service.read_snapshot(PROFILE)
                    assert failure is None and read is not None
                    assert (read.generation, read.snapshot_id) == (old.generation, old.snapshot_id)
                    shown = savings(tmp_path)
                    assert shown["provenance"]["snapshot_id"] == old.snapshot_id
                assert (
                    vs.accounting_dir() / "snapshots" / "current.json"
                ).read_bytes() == old_bytes
            finally:
                holder.kill()  # SIGKILL: the pending cycle never publishes
            assert scene.collect(NOW).state == "published"  # recovery publishes a clean successor
            old = scene.current()
            old_bytes = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
            if step != PARK[-1]:
                vs.pair(vs.projects(tmp_path), 3 + PARK.index(step), CLOCK)  # next cycle's news
    finally:
        seen, bad, torn = finish_reader(reader, stop)
    assert bad == 0 and torn == 0  # the concurrent reader never saw a failure or a torn document
    generations = [g for g, _ in seen]
    assert generations == sorted(generations) and len(set(generations)) == len(seen)
    assert len(seen) >= 2  # it observed real replacement generations as they were published


def test_a_pending_cycle_that_finishes_flips_the_reader_atomically_to_the_new_generation(
    tmp_path: Path,
) -> None:
    old = seeded_journey(tmp_path)
    reader, stop = start_reader(tmp_path)
    holder = Holder(tmp_path, "before_commit")
    try:
        mid, failure, _ = service.read_snapshot(PROFILE)
        assert failure is None and mid is not None and mid.snapshot_id == old.snapshot_id
        done = holder.release()
        assert done["state"] == "published" and done["generation"] == old.generation + 1
        time.sleep(0.3)  # let the concurrent reader observe the new generation
    finally:
        holder.close()
        seen, bad, torn = finish_reader(reader, stop)
    assert bad == 0 and torn == 0
    assert [g for g, _ in seen] == [old.generation, old.generation + 1]  # old, then new: no mix
    final = scene.current()
    assert final.generation == old.generation + 1 and final.snapshot_id != old.snapshot_id
    assert savings(tmp_path)["provenance"]["snapshot_id"] == final.snapshot_id
