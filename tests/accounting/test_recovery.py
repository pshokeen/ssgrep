"""Crash-at-every-publish-boundary (real subprocesses) and the torn-read hammer."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ssgrep.accounting import plainfile, service, snapshots
from ssgrep.accounting.ledger_contracts import SnapshotEnvelope
from tests.accounting.snapshots_driver import CRASH_CODE, gen_name, make_envelope

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


def test_discard_uncommitted_never_touches_old_generations_current_or_pinned_ids(
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
    assert snapshots.discard_uncommitted(PROFILE, 3) == [4, 6]
    assert snapshots.generations(PROFILE) == [1, 2, 3, 5]
    assert snapshots.current_generation(PROFILE) == 3
    assert not leftover.exists()
    assert snapshots.discard_uncommitted(PROFILE, 3) == []


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
