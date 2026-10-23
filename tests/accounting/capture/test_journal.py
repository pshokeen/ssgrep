"""Warm journal descriptors must retain private authority without blocking lookup."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest

from ssgrep.accounting.capture import journal, journal_reader, journal_retention as retention
from ssgrep.accounting.capture.journal_reader import read_journal
from ssgrep.accounting.collect import store
from ssgrep.accounting.privacy import policy, storage
from tests.accounting.conftest import build_event

PROFILE = "personal-local"
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def fresh_journal():
    journal.close_all()
    yield
    journal.close_all()


def warm(tmp_path: Path) -> tuple[Path, bytes]:
    assert journal.append(PROFILE, build_event(execution_id="warm-private")) is True
    (path,) = (tmp_path / "data" / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    return path, path.read_bytes()


def refuse(path: Path, original: bytes) -> None:
    with pytest.raises(storage.UnsafeStorageError):
        journal.append(PROFILE, build_event(execution_id="must-not-leak"))
    assert path.read_bytes() == original
    assert PROFILE in journal._current().unsafe
    assert journal.append(PROFILE, build_event(execution_id="still-disabled")) is False
    assert path.read_bytes() == original


def test_valid_warm_descriptor_records_both_events(tmp_path: Path) -> None:
    path, original = warm(tmp_path)
    assert journal.append(PROFILE, build_event(execution_id="valid-after-warm")) is True
    assert path.read_bytes().startswith(original)
    assert {event.execution_id for event in journal_reader.read_journal(PROFILE).events} == {
        "warm-private",
        "valid-after-warm",
    }


def test_current_state_binds_root_before_exposing_cached_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = tmp_path / "first-data", tmp_path / "second-data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(first_root))
    first = journal._current()
    assert first.root == first_root
    old_nonce = first.nonce
    monkeypatch.setenv("SSGREP_DATA_DIR", str(second_root))
    second = journal._current()
    assert second.root == second_root and second.nonce != old_nonce
    assert not first_root.exists() and not second_root.exists()


def test_two_owned_data_roots_capture_separately_without_cross_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = tmp_path / "first-data", tmp_path / "second-data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(first_root))
    first_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=first_id, cwd="/private/first")) is True
    (first_path,) = (first_root / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    first_bytes = first_path.read_bytes()
    first_fd = journal._current().segments[PROFILE].fd

    monkeypatch.setenv("SSGREP_DATA_DIR", str(second_root))
    second_id = journal.next_execution_id()
    with pytest.raises(OSError) as closed:
        os.fstat(first_fd)
    assert closed.value.errno == errno.EBADF
    assert second_id.split("-")[0] != first_id.split("-")[0] and second_id.endswith("-1")
    assert (
        journal.append(PROFILE, build_event(execution_id=second_id, cwd="/private/second")) is True
    )
    assert [event.execution_id for event in journal_reader.read_journal(PROFILE).events] == [
        second_id
    ]
    (second_path,) = (second_root / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    second_bytes = second_path.read_bytes()
    assert first_path.read_bytes() == first_bytes

    monkeypatch.setenv("SSGREP_DATA_DIR", str(first_root))
    returned_id = journal.next_execution_id()
    assert returned_id.split("-")[0] not in {first_id.split("-")[0], second_id.split("-")[0]}
    assert journal.append(PROFILE, build_event(execution_id=returned_id)) is True
    assert {event.execution_id for event in journal_reader.read_journal(PROFILE).events} == {
        first_id,
        returned_id,
    }
    assert first_path.read_bytes() == first_bytes and second_path.read_bytes() == second_bytes


def test_inflight_execution_from_old_root_cannot_append_into_new_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = tmp_path / "first-data", tmp_path / "second-data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(first_root))
    old_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=old_id)) is True
    (first_path,) = (first_root / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    original = first_path.read_bytes()

    monkeypatch.setenv("SSGREP_DATA_DIR", str(second_root))
    assert (
        journal.append(PROFILE, build_event(execution_id=old_id, cwd="/private/old-root")) is False
    )
    assert not second_root.exists() and first_path.read_bytes() == original
    new_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=new_id)) is True
    assert [event.execution_id for event in journal_reader.read_journal(PROFILE).events] == [new_id]
    assert first_path.read_bytes() == original


def test_unsafe_profile_latch_survives_return_to_its_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = tmp_path / "first-data", tmp_path / "second-data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(first_root))
    first_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=first_id)) is True
    (first_path,) = (first_root / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    original = first_path.read_bytes()
    first_path.chmod(0o644)
    refuse(first_path, original)

    monkeypatch.setenv("SSGREP_DATA_DIR", str(second_root))
    second_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=second_id)) is True
    assert [event.execution_id for event in journal_reader.read_journal(PROFILE).events] == [
        second_id
    ]
    monkeypatch.setenv("SSGREP_DATA_DIR", str(first_root))
    first_path.chmod(0o600)
    returned_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=returned_id)) is False
    assert PROFILE in journal._current().unsafe and first_path.read_bytes() == original


def test_root_switch_during_append_drops_event_without_cross_tenant_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = tmp_path / "first-data", tmp_path / "second-data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(first_root))
    first_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=first_id)) is True
    (first_path,) = (first_root / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    original = first_path.read_bytes()
    real_validate = journal._validate_segment

    def validate_then_switch(segment, directory):
        result = real_validate(segment, directory)
        monkeypatch.setenv("SSGREP_DATA_DIR", str(second_root))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(journal, "_validate_segment", validate_then_switch)
        with pytest.raises(OSError) as changed:
            journal.append(PROFILE, build_event(execution_id=first_id))
    assert changed.value.errno == errno.ESTALE
    assert not isinstance(changed.value, storage.UnsafeStorageError)
    assert first_path.read_bytes() == original and not second_root.exists()
    second_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=second_id)) is True
    assert [event.execution_id for event in journal_reader.read_journal(PROFILE).events] == [
        second_id
    ]


@pytest.mark.parametrize("mode", [0o644, 0o400, 0o700, 0o2600])
def test_changed_active_file_mode_is_refused_without_repair(tmp_path: Path, mode: int) -> None:
    path, original = warm(tmp_path)
    path.chmod(mode)
    changed = path.stat()
    refuse(path, original)
    after = path.stat()
    assert (after.st_mode, after.st_ctime_ns) == (changed.st_mode, changed.st_ctime_ns)


def test_foreign_active_descriptor_owner_is_refused_when_chown_allowed(tmp_path: Path) -> None:
    path, original = warm(tmp_path)
    owner = path.stat().st_uid
    try:
        os.chown(path, owner + 1, -1)
    except PermissionError:
        pytest.skip("changing file ownership requires a privileged test process")
    try:
        refuse(path, original)
    finally:
        os.chown(path, owner, -1)


def test_foreign_descriptor_owner_is_checked_even_without_chown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, original = warm(tmp_path)
    active_fd = journal._current().segments[PROFILE].fd
    real_fstat = os.fstat

    def foreign_owner(fd: int) -> os.stat_result:
        info = real_fstat(fd)
        if fd == active_fd:
            fields = list(info)
            fields[4] = info.st_uid + 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(journal.os, "fstat", foreign_owner)
    refuse(path, original)


def test_nonregular_cached_descriptor_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, original = warm(tmp_path)
    active_fd = journal._current().segments[PROFILE].fd
    real_fstat = os.fstat

    def nonregular(fd: int) -> os.stat_result:
        info = real_fstat(fd)
        if fd == active_fd:
            fields = list(info)
            fields[0] = stat.S_IFIFO | 0o600
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(journal.os, "fstat", nonregular)
    refuse(path, original)


@pytest.mark.parametrize("component", ["accounting", "profile", "journal"])
def test_warm_append_refuses_exposed_parent_without_chmod_repair(
    tmp_path: Path,
    component: str,
) -> None:
    path, original = warm(tmp_path)
    parent = {
        "accounting": path.parents[2],
        "profile": path.parents[1],
        "journal": path.parent,
    }[component]
    parent.chmod(0o755)
    changed = parent.stat()
    refuse(path, original)
    assert (parent.stat().st_mode, parent.stat().st_ctime_ns) == (
        changed.st_mode,
        changed.st_ctime_ns,
    )


@pytest.mark.parametrize("mutation", ["exposed", "replaced"])
def test_rotation_revalidates_namespace_before_creating_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    path, original = warm(tmp_path)
    moved = path.parent.with_name("moved-journal")
    real_flock = fcntl.flock
    calls = 0

    def busy_then_mutate(fd: int, operation: int) -> None:
        nonlocal calls
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
            calls += 1
            if mutation == "exposed":
                path.parent.chmod(0o755)
            else:
                path.parent.rename(moved)
                path.parent.mkdir(mode=0o700)
            raise BlockingIOError(errno.EAGAIN, "namespace changed during compactor handoff")
        real_flock(fd, operation)

    with monkeypatch.context() as patch:
        patch.setattr(journal.fcntl, "flock", busy_then_mutate)
        refuse(moved / path.name if mutation == "replaced" else path, original)
    assert calls == 1
    if mutation == "exposed":
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o755
        assert list(path.parent.glob("seg-*.jsonl")) == [path]
    else:
        assert list(path.parent.iterdir()) == []


def test_warm_append_refuses_symlink_ancestor_without_following_it(tmp_path: Path) -> None:
    path, original = warm(tmp_path)
    profile = path.parents[1]
    moved = profile.with_name("moved-profile")
    profile.rename(moved)
    profile.symlink_to(moved, target_is_directory=True)
    refuse(moved / "journal" / path.name, original)


def test_warm_append_refuses_private_directory_replacement(tmp_path: Path) -> None:
    path, original = warm(tmp_path)
    moved = path.parent.with_name("moved-journal")
    path.parent.rename(moved)
    path.parent.mkdir(mode=0o700)
    refuse(moved / path.name, original)
    assert list(path.parent.iterdir()) == []


def test_missing_warm_namespace_is_not_recreated(tmp_path: Path) -> None:
    path, original = warm(tmp_path)
    moved = path.parent.with_name("moved-journal")
    path.parent.rename(moved)
    refuse(moved / path.name, original)
    assert not path.parent.exists()


@pytest.mark.parametrize("rotation", ["size", "utc-day"])
def test_rotation_cannot_bypass_unsafe_warm_descriptor(tmp_path: Path, rotation: str) -> None:
    path, original = warm(tmp_path)
    segment = journal._current().segments[PROFILE]
    if rotation == "size":
        segment.size = journal.SEGMENT_TARGET_BYTES
    else:
        segment.day -= 1
    path.chmod(0o644)
    refuse(path, original)
    assert list(path.parent.glob("seg-*.jsonl")) == [path]


@pytest.mark.parametrize("replacement", ["private-file", "symlink", "absent"])
def test_warm_append_refuses_renamed_leaf_instead_of_writing_moved_inode(
    tmp_path: Path,
    replacement: str,
) -> None:
    path, original = warm(tmp_path)
    moved = path.with_name("moved-private.jsonl")
    path.rename(moved)
    if replacement == "private-file":
        path.write_bytes(b"substituted\n")
        path.chmod(0o600)
    elif replacement == "symlink":
        path.symlink_to(moved)
    refuse(moved, original)
    if replacement == "private-file":
        assert path.read_bytes() == b"substituted\n"
    elif replacement == "absent":
        assert not path.exists()


def test_hardlinked_warm_descriptor_cannot_append_private_bytes_outside_journal(
    tmp_path: Path,
) -> None:
    path, original = warm(tmp_path)
    alias = tmp_path / "outside-alias.jsonl"
    os.link(path, alias)
    refuse(path, original)
    assert alias.read_bytes() == original


def test_atomic_leaf_replacement_is_not_mistaken_for_compactor_reclamation(
    tmp_path: Path,
) -> None:
    path, original = warm(tmp_path)
    active_fd = journal._current().segments[PROFILE].fd
    replacement = path.with_name("replacement.tmp")
    replacement.write_bytes(b"substituted\n")
    replacement.chmod(0o600)
    replacement.replace(path)
    assert os.fstat(active_fd).st_nlink == 0
    refuse(path, b"substituted\n")
    assert os.fstat(active_fd).st_size == len(original)
    assert list(path.parent.glob("seg-*.jsonl")) == [path]


def test_compactor_unlink_still_rotates_and_captures(tmp_path: Path) -> None:
    path, _ = warm(tmp_path)
    path.unlink()
    assert journal.append(PROFILE, build_event(execution_id="after-reclaim")) is True
    assert {event.execution_id for event in journal_reader.read_journal(PROFILE).events} == {
        "after-reclaim",
    }


def test_repeated_nonblocking_contention_is_bounded_and_next_event_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, original = warm(tmp_path)
    attempts = 0
    real_flock = fcntl.flock

    def busy(fd: int, operation: int) -> None:
        nonlocal attempts
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
            attempts += 1
            if attempts > 4:
                pytest.fail("append has an unbounded rotation loop")
            raise BlockingIOError(errno.EAGAIN, "controlled segment contention")
        real_flock(fd, operation)

    with monkeypatch.context() as patch:
        patch.setattr(journal.fcntl, "flock", busy)
        with pytest.raises(BlockingIOError) as caught:
            journal.append(PROFILE, build_event(execution_id="busy-drop"))
    assert caught.value.errno == errno.EAGAIN
    assert attempts == 2
    assert path.read_bytes() == original and PROFILE not in journal._current().unsafe
    assert journal.append(PROFILE, build_event(execution_id="restored-positive")) is True
    assert {event.execution_id for event in journal_reader.read_journal(PROFILE).events} == {
        "warm-private",
        "restored-positive",
    }


def test_repeated_reclaimed_allocations_are_bounded_and_next_event_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    real_create = storage.create_exclusive

    def reclaimed(directory: int, name: str) -> int:
        nonlocal attempts
        attempts += 1
        if attempts > 4:
            pytest.fail("append has an unbounded reclaimed-segment loop")
        fd = real_create(directory, name)
        os.unlink(name, dir_fd=directory)
        return fd

    with monkeypatch.context() as patch:
        patch.setattr(storage, "create_exclusive", reclaimed)
        with pytest.raises(FileNotFoundError) as caught:
            journal.append(PROFILE, build_event(execution_id="reclaimed-drop"))
    assert caught.value.errno == errno.ENOENT
    assert attempts == 2 and PROFILE not in journal._current().unsafe
    assert journal.append(PROFILE, build_event(execution_id="restored-positive")) is True
    assert [event.execution_id for event in journal_reader.read_journal(PROFILE).events] == [
        "restored-positive",
    ]


TRANSPORT_LOOKUP = textwrap.dedent(
    """
    import asyncio, json, os, sys
    from pathlib import Path
    from ssgrep.accounting.capture import capture, journal
    from ssgrep.accounting.ledger.contracts import Transport
    from ssgrep.services import api
    from ssgrep.utilities.types import ResultCard, SearchResponse

    card = ResultCard(ref="claude:private:ep:0", title="Lookup survives", timestamp=None,
                      score=0.9, excerpt="real transport payload", """
    """files_touched=(), is_subagent=False)
    api.search_module.search = lambda *args, **kwargs: SearchResponse([card], 1)
    transport, fault = sys.argv[1:]
    capture.arm(Transport.CLI if transport == "cli" else Transport.MCP)
    assert api.search("warm actual descriptor").results[0] == card
    root = Path(os.environ["SSGREP_DATA_DIR"]).parent
    path, = (root / "data" / "accounting" / "personal-local" / "journal").glob("seg-*.jsonl")
    (root / "before.bin").write_bytes(path.read_bytes())
    (root / "active-name.txt").write_text(path.name)
    if fault == "mode":
        path.chmod(0o644)
    elif fault == "readonly":
        path.chmod(0o400)
    elif fault == "owner":
        try:
            os.chown(path, os.geteuid() + 1, -1)
        except PermissionError:
            raise SystemExit(77)
    elif fault == "parent":
        path.parent.chmod(0o755)
    elif fault == "refile":
        path.rename(path.with_name("moved-private.jsonl"))
        path.write_bytes(b"substituted\\n")
        path.chmod(0o600)
    elif fault == "namespace":
        path.parent.rename(path.parent.with_name("moved-journal"))
        path.parent.mkdir(mode=0o700)
    elif fault == "contention":
        import errno, fcntl
        def busy(fd, operation):
            if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
                raise BlockingIOError(errno.EAGAIN, "controlled contention")
        journal.fcntl.flock = busy

    if transport == "cli":
        import usecli
        sys.argv = ["ssgrep", "search", "real lookup after warm fault"]
        usecli.main()
    else:
        from fastmcp import Client
        from ssgrep.services import mcp_server
        mcp_server._start_reconcile_background = lambda: None
        async def lookup():
            async with Client(mcp_server.get_mcp_server()) as client:
                return (await client.call_tool("search_sessions", """
    """{"query": "real lookup after warm fault"})).structured_content
        print(json.dumps(asyncio.run(lookup()), sort_keys=True))
    """
)


@pytest.mark.parametrize("transport", ["cli", "mcp"])
@pytest.mark.parametrize(
    "fault",
    ["mode", "readonly", "owner", "parent", "refile", "namespace", "contention"],
)
def test_actual_warm_fault_leaves_transport_response_unchanged_and_bytes_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
    fault: str,
) -> None:
    def invoke(name: str, condition: str) -> tuple[subprocess.CompletedProcess[bytes], Path]:
        root = tmp_path / name
        monkeypatch.setenv("SSGREP_DATA_DIR", str(root / "data"))
        policy.enable_collection(PROFILE)
        result = subprocess.run(
            [sys.executable, "-c", TRANSPORT_LOOKUP, transport, condition],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            cwd=ROOT,
            capture_output=True,
            timeout=30,
        )
        return result, root

    baseline, good = invoke("good", "none")
    changed, bad = invoke("bad", fault)
    if fault == "owner" and changed.returncode == 77:
        pytest.skip("changing file ownership requires a privileged test process")
    assert baseline.returncode == changed.returncode == 0, (baseline.stderr, changed.stderr)
    assert (changed.stdout, changed.stderr) == (baseline.stdout, baseline.stderr)
    assert b"Lookup survives" in changed.stdout
    good_dir = good / "data" / "accounting" / PROFILE / "journal"
    good_path = good_dir / (good / "active-name.txt").read_text()
    assert len(good_path.read_bytes()) > len((good / "before.bin").read_bytes())
    bad_dir = bad / "data" / "accounting" / PROFILE / "journal"
    bad_path = bad_dir / (bad / "active-name.txt").read_text()
    if fault == "refile":
        assert bad_path.read_bytes() == b"substituted\n"
        bad_path = bad_dir / "moved-private.jsonl"
    elif fault == "namespace":
        assert list(bad_dir.iterdir()) == []
        bad_path = bad_dir.with_name("moved-journal") / bad_path.name
    assert bad_path.read_bytes() == (bad / "before.bin").read_bytes()
    for candidate in bad_dir.glob("seg-*.jsonl"):
        if candidate != bad_path:
            assert candidate.read_bytes() == (b"substituted\n" if fault == "refile" else b"")


"""Journal phase identities live in existing immutable blobs, not a giant checkpoint."""


PROFILE = "personal-local"
FLOOR = "2026-11-10"
AGGREGATE = "2025-11-05"


def segment(name: str, data: bytes) -> Path:
    directory = storage.open_profile_dir(PROFILE, "journal")
    try:
        fd = storage.create_exclusive(directory, name)
        try:
            storage.write_all(fd, data)
        finally:
            os.close(fd)
    finally:
        os.close(directory)
    return storage.data_root() / "accounting" / PROFILE / "journal" / name


def event(identity: str, day: str, **kwargs):
    return build_event(execution_id=identity, ts_utc=f"{day}T12:00:00Z", **kwargs)


@pytest.fixture(autouse=True)
def fresh_writer():
    journal.close_all()
    yield
    journal.close_all()


def record(count=2000):
    keys = [hashlib.sha256(str(n).encode()).hexdigest() for n in range(count)]
    return {
        "version": 1,
        "key": "11" * 32,
        "horizon_day": FLOOR,
        "aggregate_horizon_day": AGGREGATE,
        "retired": {key: "2026-11-09" for key in keys},
        "bridges": sorted(keys[: count // 2]),
        "proofs": [],
    }


def commit(expanded, generation=1):
    checkpoint = store.Checkpoint(profile=PROFILE, generation=generation)
    checkpoint.retention["journal"] = retention.persist_journal_retention(
        PROFILE, expanded, blobs=checkpoint.blobs
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    return checkpoint


def test_checkpoint_has_bounded_handles_not_phase_id_maps_and_refs_survive_sweep():
    expanded = record()
    checkpoint = commit(expanded)
    persisted = checkpoint.retention["journal"]
    body = json.dumps(checkpoint.to_dict())
    assert "retired" not in persisted and "bridges" not in persisted
    assert len(persisted["identity_shards"]) <= 256 and len(body) < 128 * 1024
    assert all(key not in body for key in expanded["retired"])
    assert retention.journal_retention_refs(persisted) == set(checkpoint.blobs.values())
    store.sweep_orphans(PROFILE, checkpoint)
    loaded = retention.load_journal_retention(PROFILE)
    assert loaded is not None
    assert loaded["retired"] == expanded["retired"] and loaded["bridges"] == expanded["bridges"]


def test_old_inline_checkpoint_migrates_without_losing_floor_or_replay_identities():
    expanded = record(20)
    checkpoint = store.Checkpoint(profile=PROFILE, generation=1)
    checkpoint.retention["journal"] = expanded
    store.commit_checkpoint(PROFILE, checkpoint)
    old = retention.load_journal_retention(PROFILE)
    assert old is not None
    assert old["retired"] == expanded["retired"]
    migrated = commit(old, generation=2)
    assert migrated.retention["journal"]["version"] == 2
    restored = retention.load_journal_retention(PROFILE)
    assert restored is not None
    assert restored["retired"] == expanded["retired"]
    assert restored["horizon_day"] == FLOOR and restored["key"] == expanded["key"]


def test_expand_uses_supplied_checkpoint_snapshot_not_newer_mutable_current():
    first = commit(record(4))
    snapshot = deepcopy(first.retention["journal"])
    second = record(8)
    second["horizon_day"] = "2026-11-11"
    commit(second, generation=2)
    expanded = retention.expand_journal_retention(PROFILE, snapshot)
    assert expanded["horizon_day"] == FLOOR and len(expanded["retired"]) == 4


def test_clock_only_prepare_reuses_identity_map_and_existing_blob_handles():
    original = record(20)
    original["bridges"] = []
    commit(original)
    previous = retention.load_journal_retention(PROFILE)
    current = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=previous,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert current is not None and previous is not None
    assert current["retired"] is previous["retired"]
    blobs = {}
    persisted = retention.persist_journal_retention(PROFILE, current, blobs=blobs)
    assert persisted is not None
    assert persisted["identity_shards"] == previous["_identity_shards"]


def test_uncommitted_identity_blobs_cannot_authorize_physical_retirement():
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-v2-uncommitted-pointer.jsonl", serialize_event(event("old", "2026-11-09"))
    )
    expanded = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    blobs = {}
    retention.persist_journal_retention(PROFILE, expanded, blobs=blobs)
    assert retention.finish_journal_retention(PROFILE, generation=1).status == "no_checkpoint"
    assert source.exists()
    commit(expanded)
    retention.finish_journal_retention(PROFILE, generation=1)
    assert not source.exists()


def test_corrupt_or_escaping_identity_reference_is_rejected_before_source_access():
    escaped = {
        "version": 2,
        "key": "11" * 32,
        "horizon_day": FLOOR,
        "aggregate_horizon_day": AGGREGATE,
        "proofs": [],
        "identity_shards": {"00": "../../original-source.json"},
    }
    with pytest.raises(ValueError, match="identity"):
        retention.expand_journal_retention(PROFILE, escaped)


def test_bounded_proof_batch_stays_pending_then_next_accepted_batch_progresses():
    from ssgrep.accounting.capture.journal_codec import serialize_event

    for n in range(35):
        segment(f"seg-v2-batch-{n:03}.jsonl", serialize_event(event(f"old-{n}", "2026-11-09")))
    expanded = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert len(expanded["proofs"]) == 32 and expanded["queued_segments"] == 3
    commit(expanded)
    out = retention.finish_journal_retention(PROFILE, generation=1)
    assert out.status == "pending" and out.removed_segments == 32
    later = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=retention.load_journal_retention(PROFILE),
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert len(later["proofs"]) == 3 and later["queued_segments"] == 0
    commit(later, generation=2)
    assert retention.finish_journal_retention(PROFILE, generation=2).removed_segments == 3


def test_recent_malformed_segment_is_deferred_without_rewriting_private_bytes():
    from ssgrep.accounting.capture import journal_compact

    source = segment("seg-v2-recent-damage.jsonl", b"not-json\n")
    os.utime(source, (1_799_539_200, 1_799_539_200))
    original = source.read_bytes()
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert prepared is not None
    commit(prepared)

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )
    assert outcome.status == "pending"
    assert outcome.reasons["recent_damage"] == 1
    assert outcome.deferred_segments == 1
    assert source.read_bytes() == original


def test_stale_damage_compaction_publishes_retained_event_before_removing_source():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    retained = event("still-live", "2026-11-11")
    source = segment(
        "seg-v2-stale-damage.jsonl",
        serialize_event(retained) + b"not-json\n",
    )
    os.utime(source, (1, 1))
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert prepared is not None
    commit(prepared)

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )

    assert outcome.removed_segments == 1 and outcome.retained_segments == 1
    assert not source.exists()
    assert [row.execution_id for row in read_journal(PROFILE).events] == ["still-live"]


def test_stale_unterminated_tail_is_removed_without_losing_live_records():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-v2-stale-partial-tail.jsonl",
        serialize_event(event("retained-before-tail", "2026-11-11")) + b'{"partial":',
    )
    os.utime(source, (1, 1))
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert prepared is not None
    commit(prepared)

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )

    assert outcome.removed_segments == 1 and outcome.retained_segments == 1
    assert not source.exists()
    assert [row.execution_id for row in read_journal(PROFILE).events] == ["retained-before-tail"]


def test_compaction_budget_defers_then_retries_without_losing_retained_event():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-v2-budgeted-retention.jsonl",
        serialize_event(event("expired", "2026-11-09"))
        + serialize_event(event("live", "2026-11-11")),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert prepared is not None and prepared["proofs"]
    commit(prepared)
    original = source.read_bytes()

    deferred = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=prepared["proofs"][0]["size"] - 1,
        segment_budget=1,
        legacy_writers_stopped=True,
    )
    assert deferred.status == "pending"
    assert deferred.reasons["maintenance_budget"] == 1
    assert source.read_bytes() == original

    completed = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=1,
        legacy_writers_stopped=True,
    )
    assert completed.removed_segments == 1 and completed.retained_segments == 1
    assert not source.exists()
    assert [row.execution_id for row in read_journal(PROFILE).events] == ["live"]


def test_missing_committed_shard_is_not_hidden_by_warm_parse_cache():
    from ssgrep.accounting.collect.store import CollectError
    from ssgrep.accounting.privacy import storage

    checkpoint = commit(record(4))
    retained = retention.load_journal_retention(PROFILE)
    assert retained is not None and retained["retired"]
    name = next(iter(retention.journal_retention_refs(checkpoint.retention["journal"])))
    directory = storage.open_profile_dir(PROFILE, "collector", "state")
    try:
        storage.delete_file(directory, name)
    finally:
        import os

        os.close(directory)
    with pytest.raises(CollectError) as failure:
        retention.load_journal_retention(PROFILE)
    assert failure.value.condition == "state_corrupt"


def test_same_name_shard_corruption_invalidates_cached_identity_payload():
    from ssgrep.accounting.collect.store import CollectError
    from ssgrep.accounting.privacy import storage

    checkpoint = commit(record(4))
    retained = retention.load_journal_retention(PROFILE)
    assert retained is not None and retained["retired"]
    name = next(iter(retention.journal_retention_refs(checkpoint.retention["journal"])))
    path = storage.data_root() / "accounting" / PROFILE / "collector" / "state" / name
    body = json.loads(path.read_bytes())
    body["bridges"] = []
    body["retired"] = {}
    path.write_text(json.dumps(body))
    with pytest.raises(CollectError) as failure:
        retention.load_journal_retention(PROFILE)
    assert failure.value.condition == "state_corrupt"


def test_retention_load_without_checkpoint_or_retention_record_returns_none():
    assert retention.load_journal_retention(PROFILE) is None

    store.commit_checkpoint(
        PROFILE,
        store.Checkpoint(profile=PROFILE, generation=1),
    )
    assert retention.load_journal_retention(PROFILE) is None


def test_segment_name_collisions_do_not_overwrite_and_append_recovers() -> None:
    state = journal._current()
    journal.next_execution_id()
    first = journal.segment_name(state.nonce, state.segment_seq + 1)
    second = journal.segment_name(state.nonce, state.segment_seq + 2)
    directory = storage.open_profile_dir(PROFILE, "journal")
    try:
        for name in (first, second):
            fd = storage.create_exclusive(directory, name)
            try:
                storage.write_all(fd, b"preserved\n")
            finally:
                os.close(fd)
    finally:
        os.close(directory)

    with pytest.raises(FileExistsError):
        journal.append(PROFILE, build_event(execution_id="collision-1"))
    journal.append(PROFILE, build_event(execution_id="recovered-1"))

    folder = storage.data_root() / "accounting" / PROFILE / "journal"
    assert (folder / first).read_bytes() == b"preserved\n"
    assert (folder / second).read_bytes() == b"preserved\n"
    assert [row.execution_id for row in read_journal(PROFILE).events] == ["recovered-1"]


def test_reader_reports_conflicting_phase_variants_deterministically() -> None:
    import hashlib
    from dataclasses import replace

    from ssgrep.accounting.capture.journal_codec import serialize_event
    from ssgrep.accounting.ledger.contracts import canonical_bytes

    first = event("duplicate-1", "2026-11-11")
    second = replace(first, fingerprint="e" * 64)
    segment(
        "seg-v2-identical-duplicate.jsonl",
        serialize_event(first) + serialize_event(first),
    )
    segment("seg-v2-conflict.jsonl", serialize_event(second))

    result = read_journal(PROFILE)

    assert result.conflicts == [("duplicate-1", str(first.phase))]
    assert result.duplicates == 1
    assert len(result.events) == 1
    assert result.events[0] == min(
        (first, second),
        key=lambda row: hashlib.sha256(canonical_bytes(row)).hexdigest(),
    )


def test_legacy_segment_requires_offline_writer_ack_before_retirement() -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-legacy-retention.jsonl",
        serialize_event(event("legacy-expired", "2026-11-09")),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    commit(prepared)
    original = source.read_bytes()

    result = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=False,
    )

    assert result.status == "blocked_legacy"
    assert result.legacy_segments == 1
    assert result.warnings == ("journal_retention_blocked_legacy_writers",)
    assert source.read_bytes() == original


@pytest.mark.parametrize("state", ["no_checkpoint", "no_retention"])
def test_compaction_without_authority_checkpoint_is_a_noop(state: str) -> None:
    from ssgrep.accounting.capture import journal_compact

    if state == "no_retention":
        store.commit_checkpoint(
            PROFILE,
            store.Checkpoint(profile=PROFILE, generation=1),
        )

    result = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024,
        segment_budget=1,
        legacy_writers_stopped=True,
    )

    assert result.status == (
        "no_checkpoint" if state == "no_checkpoint" else "no_retention_checkpoint"
    )


def test_compaction_rejects_a_stale_checkpoint_generation_without_mutation() -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-v2-stale-generation.jsonl",
        serialize_event(event("expired-generation", "2026-11-09")),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    commit(prepared)
    original = source.read_bytes()

    result = journal_compact.compact_committed_journal(
        PROFILE,
        generation=2,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )

    assert result.status == "generation_mismatch"
    assert source.read_bytes() == original


def test_committed_maintenance_removes_expired_quarantine_and_staging_files() -> None:
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-v2-expired-for-sweep.jsonl",
        serialize_event(event("expired-for-sweep", "2026-11-09")),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    commit(prepared)

    journal_directory = storage.open_profile_dir(PROFILE, "journal")
    try:
        temporary_fd = storage.create_exclusive(journal_directory, ".retention-abandoned.tmp")
        os.close(temporary_fd)
    finally:
        os.close(journal_directory)
    quarantine_directory = storage.open_profile_dir(PROFILE, "journal-quarantine")
    try:
        expired_fd = storage.create_exclusive(quarantine_directory, "expired.json")
        storage.write_all(expired_fd, b"{")
        os.close(expired_fd)
        orphan_fd = storage.create_exclusive(quarantine_directory, "orphan.json")
        storage.write_all(orphan_fd, b'{"segment":"seg-v2-no-longer-present.jsonl"}')
        os.close(orphan_fd)
    finally:
        os.close(quarantine_directory)
    profile_dir = storage.data_root() / "accounting" / PROFILE
    expired_metadata = profile_dir / "journal-quarantine" / "expired.json"
    future = 1_800_000_000
    os.utime(expired_metadata, (1, 1))
    orphan_metadata = expired_metadata.with_name("orphan.json")
    os.utime(orphan_metadata, (future, future))

    result = retention.finish_journal_retention(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )

    quarantine = expired_metadata.parent
    cursor = profile_dir / "journal-maintenance" / "cursor.json"
    assert not source.exists()
    assert not (profile_dir / "journal" / ".retention-abandoned.tmp").exists()
    assert not (quarantine / "expired.json").exists()
    assert not (quarantine / "orphan.json").exists()
    assert result.remnants_removed == 1 and result.quarantine_removed == 2
    assert json.loads(cursor.read_text())["completed_day"] == FLOOR


def replacement_name(proof: dict[str, object], horizon: str, data: bytes) -> str:
    identity = json.dumps(
        [proof, horizon, hashlib.sha256(data).hexdigest()], sort_keys=True
    ).encode()
    return f"seg-v2-compact-{hashlib.sha256(identity).hexdigest()}.jsonl"


def test_committed_retention_bridges_live_execution_with_minimal_started_event():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import minimal_started, serialize_event
    from ssgrep.accounting.ledger.contracts import Phase

    started = event(
        "bridged-execution",
        "2026-11-09",
        phase=Phase.STARTED,
        outcome=None,
        mono_ns=None,
        rendered_bytes=None,
    )
    completed = event("bridged-execution", FLOOR)
    original = serialize_event(started) + serialize_event(completed)
    source = segment("seg-v2-bridged-execution.jsonl", original)
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert retention.phase_key(started, prepared) in prepared["bridges"]
    commit(prepared)
    assert source.read_bytes() == original
    assert read_journal(PROFILE).events == [minimal_started(started), completed]

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )

    (replacement,) = source.parent.glob("seg-v2-compact-*.jsonl")
    assert outcome.removed_segments == outcome.retained_segments == 1
    assert not source.exists()
    assert replacement.read_bytes() == serialize_event(minimal_started(started)) + serialize_event(
        completed
    )
    assert read_journal(PROFILE).events == [minimal_started(started), completed]


def test_aggregate_horizon_expires_old_retired_ids_but_keeps_boundary_id():
    expanded = record(4)
    keys = list(expanded["retired"])
    boundary_key = keys[-1]
    expanded["retired"][boundary_key] = FLOOR
    commit(expanded)
    previous = retention.load_journal_retention(PROFILE)
    assert previous is not None and len(previous["retired"]) == 4

    advanced = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=previous,
        horizon_day=FLOOR,
        aggregate_horizon_day=FLOOR,
    )
    assert advanced["retired"] == {boundary_key: FLOOR}

    checkpoint = commit(advanced, generation=2)
    restored = retention.load_journal_retention(PROFILE)
    assert restored is not None and restored["retired"] == {boundary_key: FLOOR}
    assert len(checkpoint.retention["journal"]["identity_shards"]) == 1


def test_busy_writer_does_not_block_compaction_of_another_committed_segment():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    busy_source = segment(
        "seg-v2-busy-writer.jsonl",
        serialize_event(event("busy-expired", "2026-11-09")),
    )
    free_source = segment(
        "seg-v2-free-writer.jsonl",
        serialize_event(event("free-expired", "2026-11-09")),
    )
    busy_original = busy_source.read_bytes()
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert len(prepared["proofs"]) == 2
    commit(prepared)

    fd = os.open(busy_source, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        outcome = journal_compact.compact_committed_journal(
            PROFILE,
            generation=1,
            byte_budget=1024 * 1024,
            segment_budget=32,
            legacy_writers_stopped=True,
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert outcome.status == "pending"
    assert outcome.reasons["writer_busy"] == 1
    assert outcome.removed_segments == 1
    assert busy_source.exists() and busy_source.read_bytes() == busy_original
    assert not free_source.exists()


def test_changed_checkpoint_proof_is_deferred_while_matching_proof_compacts():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    changed_source = segment(
        "seg-v2-a-changed-proof.jsonl",
        serialize_event(event("checkpoint-expired", "2026-11-09")),
    )
    matching_source = segment(
        "seg-v2-b-matching-proof.jsonl",
        serialize_event(event("matching-expired", "2026-11-09")),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert len(prepared["proofs"]) == 2
    commit(prepared)
    replacement = serialize_event(event("changed-after-checkpoint-with-new-id", FLOOR))
    changed_source.write_bytes(replacement)

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )

    assert outcome.status == "pending"
    assert outcome.reasons["changed_after_checkpoint"] == 1
    assert outcome.removed_segments == 1
    assert changed_source.read_bytes() == replacement
    assert not matching_source.exists()
    assert [row.execution_id for row in read_journal(PROFILE).events] == [
        "changed-after-checkpoint-with-new-id"
    ]


def test_missing_checkpoint_proof_does_not_starve_next_present_segment():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    segment("seg-v2-a-disappeared.jsonl", serialize_event(event("gone", "2026-11-09")))
    segment("seg-v2-b-still-present.jsonl", serialize_event(event("present", "2026-11-09")))
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert len(prepared["proofs"]) == 2
    commit(prepared)
    missing_name, remaining_name = [proof["name"] for proof in prepared["proofs"]]
    missing = storage.data_root() / "accounting" / PROFILE / "journal" / missing_name
    remaining = storage.data_root() / "accounting" / PROFILE / "journal" / remaining_name
    assert missing.exists() and remaining.exists()
    missing.unlink()
    remaining_original = remaining.read_bytes()

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=1,
        legacy_writers_stopped=True,
    )

    assert outcome.status == "complete"
    assert outcome.removed_segments == 1 and outcome.deferred_segments == 0
    assert not remaining.exists()
    assert not missing.exists() and remaining_original


def test_existing_retained_copy_defers_at_byte_boundary_then_resumes():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    live = event("existing-copy-live", FLOOR)
    source = segment(
        "seg-v2-existing-copy.jsonl",
        serialize_event(event("existing-copy-expired", "2026-11-09")) + serialize_event(live),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    (proof,) = prepared["proofs"]
    commit(prepared)
    original = source.read_bytes()
    retained_bytes = serialize_event(live)
    destination = segment(
        replacement_name(proof, FLOOR, retained_bytes),
        retained_bytes,
    )

    deferred = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=proof["size"] + len(retained_bytes) - 1,
        segment_budget=1,
        legacy_writers_stopped=True,
    )
    assert deferred.status == "pending"
    assert deferred.reasons["maintenance_budget"] == 1
    assert source.read_bytes() == original and destination.read_bytes() == retained_bytes

    completed = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=proof["size"] + len(retained_bytes),
        segment_budget=1,
        legacy_writers_stopped=True,
    )
    assert completed.removed_segments == completed.retained_segments == 1
    assert not source.exists() and destination.read_bytes() == retained_bytes
    assert read_journal(PROFILE).events == [live]


def test_conflicting_existing_retained_copy_cannot_authorize_source_deletion():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    live = event("conflicting-copy-live", FLOOR)
    source = segment(
        "seg-v2-conflicting-copy.jsonl",
        serialize_event(event("conflicting-copy-expired", "2026-11-09")) + serialize_event(live),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    (proof,) = prepared["proofs"]
    commit(prepared)
    original = source.read_bytes()
    retained_bytes = serialize_event(live)
    conflicting = bytes([retained_bytes[0] ^ 1]) + retained_bytes[1:]
    destination = segment(
        replacement_name(proof, FLOOR, retained_bytes),
        conflicting,
    )

    with pytest.raises(ValueError, match="conflicting content"):
        journal_compact.compact_committed_journal(
            PROFILE,
            generation=1,
            byte_budget=1024 * 1024,
            segment_budget=1,
            legacy_writers_stopped=True,
        )

    assert source.exists() and source.read_bytes() == original
    assert destination.read_bytes() == conflicting


def test_committed_retired_phase_identity_blocks_timestamp_mutated_replay():
    from dataclasses import replace

    from ssgrep.accounting.capture.journal_codec import serialize_event

    expired = event("replay-after-retention", "2026-11-09")
    original = serialize_event(expired)
    source = segment("seg-v2-retired-original.jsonl", original)
    initial = read_journal(PROFILE)
    assert initial.events == [expired]
    prepared = retention.prepare_journal_retention(
        initial,
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    key = retention.phase_key(expired, prepared)
    assert prepared["retired"][key] == "2026-11-09"
    commit(prepared)

    replay = replace(expired, ts_utc=f"{FLOOR}T12:00:00Z")
    replay_bytes = serialize_event(replay)
    replay_source = segment("seg-v2-retired-replay.jsonl", replay_bytes)
    filtered = read_journal(PROFILE)

    assert filtered.events == [] and filtered.retired_events == 2
    assert source.read_bytes() == original and replay_source.read_bytes() == replay_bytes


def test_committed_empty_segment_is_removed_without_publishing_a_replacement():
    from ssgrep.accounting.capture import journal_compact

    source = segment("seg-v2-empty-retention.jsonl", b"")
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert [proof["name"] for proof in prepared["proofs"]] == [source.name]
    commit(prepared)

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024,
        segment_budget=1,
        legacy_writers_stopped=True,
    )

    assert outcome.removed_segments == 1 and outcome.retained_segments == 0
    assert not source.exists()
    assert not list(source.parent.glob("seg-v2-compact-*.jsonl"))
    assert read_journal(PROFILE).events == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"version": 2, "retired": {}, "bridges": []},
            "Unsupported journal identity blob",
        ),
        (
            {"version": 1, "retired": {"bb" + "0" * 62: "2026-11-09"}, "bridges": []},
            "Journal identity key is invalid",
        ),
        (
            {"version": 1, "retired": {}, "bridges": ["bb" + "0" * 62]},
            "Journal bridge identity is invalid",
        ),
    ],
)
def test_committed_identity_shards_reject_unsupported_or_unbucketed_keys(
    payload: dict[str, object],
    message: str,
):
    from ssgrep.accounting.collect.store import CollectError

    bucket = "aa"
    name = store.write_blob(PROFILE, f"journal-retention-{bucket}", payload)
    checkpoint = store.Checkpoint(profile=PROFILE, generation=1)
    checkpoint.blobs[f"journal-retention:{bucket}"] = name
    checkpoint.retention["journal"] = {
        "version": 2,
        "key": "11" * 32,
        "horizon_day": FLOOR,
        "aggregate_horizon_day": AGGREGATE,
        "proofs": [],
        "identity_shards": {bucket: name},
    }
    store.commit_checkpoint(PROFILE, checkpoint)

    with pytest.raises(CollectError, match=message):
        retention.load_journal_retention(PROFILE)


def test_missing_identity_blob_directory_invalidates_warm_retention_snapshot():
    from ssgrep.accounting.collect.store import CollectError

    commit(record(4))
    cached = retention.load_journal_retention(PROFILE)
    assert cached is not None and cached["retired"]
    data_directory = storage.data_root() / "accounting" / PROFILE / "collector"
    state_directory = data_directory / "state"
    checkpoint_file = data_directory / store.CHECKPOINT
    checkpoint_bytes = checkpoint_file.read_bytes()
    isolated = state_directory.with_name("state-unavailable")
    state_directory.rename(isolated)

    with pytest.raises(CollectError) as failure:
        retention.load_journal_retention(PROFILE)

    assert failure.value.condition == "state_corrupt"
    assert isolated.is_dir()
    assert checkpoint_file.read_bytes() == checkpoint_bytes


def test_explicit_old_root_cannot_write_after_configured_root_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = tmp_path / "first-data", tmp_path / "second-data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(first_root))
    old_id = journal.next_execution_id()
    assert journal.append(PROFILE, build_event(execution_id=old_id)) is True
    old_authority = journal._current().authority
    assert old_authority is not None
    (first_path,) = (first_root / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    original = first_path.read_bytes()

    monkeypatch.setenv("SSGREP_DATA_DIR", str(second_root))
    assert (
        journal.append(
            PROFILE,
            build_event(execution_id="explicit-old-root"),
            root=old_authority,
        )
        is False
    )
    assert not second_root.exists()
    assert first_path.read_bytes() == original

    new_id = journal.next_execution_id()
    new_event = build_event(execution_id=new_id)
    assert journal.append(PROFILE, new_event) is True
    (second_path,) = (second_root / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    assert [row.execution_id for row in journal_reader.read_journal(PROFILE).events] == [new_id]
    assert first_path.read_bytes() == original
    from ssgrep.accounting.capture.journal_codec import serialize_event

    assert second_path.read_bytes() == serialize_event(new_event)


@pytest.mark.parametrize("boundary", ["size", "utc-day"])
def test_writer_rotates_at_exact_size_and_utc_day_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    from ssgrep.accounting.capture.journal_codec import serialize_event

    initial = build_event(execution_id="warm-boundary")
    assert journal.append(PROFILE, initial) is True
    (original_path,) = (storage.data_root() / "accounting" / PROFILE / "journal").glob(
        "seg-*.jsonl"
    )
    original = original_path.read_bytes()
    segment_state = journal._current().segments[PROFILE]
    if boundary == "size":
        monkeypatch.setattr(journal, "SEGMENT_TARGET_BYTES", segment_state.size)
    else:
        next_day = segment_state.day + 1
        monkeypatch.setattr(journal.time, "time", lambda: next_day * 86_400)

    rotated = build_event(execution_id=f"after-{boundary}-boundary")
    assert journal.append(PROFILE, rotated) is True

    paths = sorted(original_path.parent.glob("seg-*.jsonl"))
    assert len(paths) == 2
    replacement = next(path for path in paths if path != original_path)
    assert original_path.read_bytes() == original
    assert replacement.read_bytes() == serialize_event(rotated)
    assert journal_reader.read_journal(PROFILE).events == sorted(
        [initial, rotated], key=lambda row: row.execution_id
    )


def test_compaction_rejects_invalid_budgets_and_accepts_exact_byte_boundary() -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    original = serialize_event(event("exact-budget-expired", "2026-11-09"))
    source = segment("seg-v2-exact-budget.jsonl", original)
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    commit(prepared)
    checkpoint_path = storage.data_root() / "accounting" / PROFILE / "collector" / store.CHECKPOINT
    checkpoint_bytes = checkpoint_path.read_bytes()

    with pytest.raises(ValueError, match="byte_budget"):
        journal_compact.compact_committed_journal(
            PROFILE,
            generation=1,
            byte_budget=-1,
            segment_budget=1,
            legacy_writers_stopped=True,
        )
    with pytest.raises(ValueError, match="segment_budget"):
        journal_compact.compact_committed_journal(
            PROFILE,
            generation=1,
            byte_budget=len(original),
            segment_budget=True,
            legacy_writers_stopped=True,
        )
    assert source.read_bytes() == original
    assert checkpoint_path.read_bytes() == checkpoint_bytes

    completed = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=len(original),
        segment_budget=1,
        legacy_writers_stopped=True,
    )
    assert completed.removed_segments == 1 and completed.retained_segments == 0
    assert not source.exists()
    assert not list(source.parent.glob("seg-v2-compact-*.jsonl"))
    assert read_journal(PROFILE).events == []
    assert checkpoint_path.read_bytes() == checkpoint_bytes


def test_compaction_rejects_corrupt_checkpoint_segment_path_before_opening_it(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-v2-corrupt-proof.jsonl",
        serialize_event(event("corrupt-proof-expired", "2026-11-09")),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    checkpoint = commit(prepared)
    checkpoint.retention["journal"]["proofs"][0]["name"] = "../outside.jsonl"
    store.commit_checkpoint(PROFILE, checkpoint)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"not journal authority\n")
    original = source.read_bytes()

    with pytest.raises(ValueError, match="unsafe segment"):
        journal_compact.compact_committed_journal(
            PROFILE,
            generation=1,
            byte_budget=1024 * 1024,
            segment_budget=1,
            legacy_writers_stopped=True,
        )

    assert source.read_bytes() == original
    assert outside.read_bytes() == b"not journal authority\n"


def test_pending_unchanged_segment_proof_does_not_publish_or_remove_source() -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    live = event("unchanged-pending-live", FLOOR)
    original = serialize_event(live)
    source = segment("seg-v2-unchanged-pending.jsonl", original)
    observed = read_journal(PROFILE)
    observed.retention_pending.add(source.name)
    prepared = retention.prepare_journal_retention(
        observed,
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert [proof["name"] for proof in prepared["proofs"]] == [source.name]
    commit(prepared)

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=1,
        legacy_writers_stopped=True,
    )

    assert outcome.removed_segments == outcome.retained_segments == 0
    assert source.exists() and source.read_bytes() == original
    assert not list(source.parent.glob("seg-v2-compact-*.jsonl"))
    assert read_journal(PROFILE).events == [live]


def test_existing_retained_copy_with_wrong_size_cannot_authorize_source_deletion():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    live = event("wrong-sized-copy-live", FLOOR)
    original = serialize_event(event("wrong-sized-copy-expired", "2026-11-09")) + serialize_event(
        live
    )
    source = segment("seg-v2-wrong-sized-copy.jsonl", original)
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    (proof,) = prepared["proofs"]
    commit(prepared)
    conflict = b"short"
    destination = segment(replacement_name(proof, FLOOR, serialize_event(live)), conflict)

    with pytest.raises(ValueError, match="conflicting size"):
        journal_compact.compact_committed_journal(
            PROFILE,
            generation=1,
            byte_budget=1024 * 1024,
            segment_budget=1,
            legacy_writers_stopped=True,
        )

    assert source.read_bytes() == original
    assert destination.read_bytes() == conflict


def test_missing_proof_after_segment_budget_does_not_mask_later_pending_source():
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    sources = [
        segment(
            f"seg-v2-budget-order-{name}.jsonl",
            serialize_event(event(f"budget-order-{name}", "2026-11-09")),
        )
        for name in ("a", "b", "c")
    ]
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert [proof["name"] for proof in prepared["proofs"]] == [source.name for source in sources]
    commit(prepared)
    sources[1].unlink()
    third_original = sources[2].read_bytes()

    first = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=1,
        legacy_writers_stopped=True,
    )
    assert first.status == "pending"
    assert first.reasons["maintenance_budget"] == 1
    assert first.removed_segments == 1
    assert not sources[0].exists() and not sources[1].exists()
    assert sources[2].read_bytes() == third_original

    resumed = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=1,
        legacy_writers_stopped=True,
    )
    assert resumed.status == "complete"
    assert resumed.removed_segments == 1
    assert not sources[2].exists()
    assert read_journal(PROFILE).events == []


def test_remnant_sweep_defers_large_metadata_then_resumes_with_durable_cursor() -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    live = event("remnant-cursor-live", FLOOR)
    source = segment("seg-v2-remnant-cursor-live.jsonl", serialize_event(live))
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert prepared["proofs"] == []
    commit(prepared)

    quarantine = storage.open_profile_dir(PROFILE, "journal-quarantine", create=False)
    try:
        malformed_fd = storage.create_exclusive(quarantine, "a-malformed.json")
        storage.write_all(malformed_fd, b"{")
        os.close(malformed_fd)
        metadata = json.dumps(
            {"segment": source.name, "padding": "x" * 128}, sort_keys=True
        ).encode()
        large_fd = storage.create_exclusive(quarantine, "b-large.json")
        storage.write_all(large_fd, metadata)
        os.close(large_fd)
    finally:
        os.close(quarantine)
    quarantine_path = storage.data_root() / "accounting" / PROFILE / "journal-quarantine"
    for name in ("a-malformed.json", "b-large.json"):
        os.utime(quarantine_path / name, (1_800_000_000, 1_800_000_000))

    maintenance = storage.open_profile_dir(PROFILE, "journal-maintenance")
    try:
        abandoned_fd = storage.create_exclusive(maintenance, ".cursor.json.abandoned.tmp")
        os.close(abandoned_fd)
    finally:
        os.close(maintenance)
    cursor = storage.data_root() / "accounting" / PROFILE / "journal-maintenance" / "cursor.json"

    pending = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1,
        segment_budget=2,
        legacy_writers_stopped=True,
    )
    assert pending.status == "pending"
    assert pending.reasons["remnants_pending"] == 1
    assert pending.warnings[0] == "journal_retention_remnants_pending"
    assert (quarantine_path / "a-malformed.json").read_bytes() == b"{"
    assert (quarantine_path / "b-large.json").read_bytes() == metadata
    assert cursor.read_bytes() == b'{"quarantine": "a-malformed.json"}'
    assert not (cursor.parent / ".cursor.json.abandoned.tmp").exists()

    resumed = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=len(metadata) + 1,
        segment_budget=2,
        legacy_writers_stopped=True,
    )
    assert resumed.status == "complete"
    assert resumed.reasons == {}
    assert (quarantine_path / "a-malformed.json").read_bytes() == b"{"
    assert (quarantine_path / "b-large.json").read_bytes() == metadata
    assert cursor.read_bytes() == (
        f'{{"completed_day": "{FLOOR}", "quarantine": "a-malformed.json"}}'.encode()
    )
    assert read_journal(PROFILE).events == [live]


def test_remnant_sweep_completes_when_quarantine_directory_is_absent() -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    live = event("no-quarantine-live", FLOOR)
    source = segment("seg-v2-no-quarantine-live.jsonl", serialize_event(live))
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert prepared["proofs"] == []
    commit(prepared)
    quarantine = storage.data_root() / "accounting" / PROFILE / "journal-quarantine"
    quarantine.rmdir()

    outcome = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024,
        segment_budget=1,
        legacy_writers_stopped=True,
    )

    cursor = storage.data_root() / "accounting" / PROFILE / "journal-maintenance" / "cursor.json"
    assert outcome.status == "complete"
    assert outcome.warnings == ()
    assert source.read_bytes() == serialize_event(live)
    assert cursor.read_bytes() == f'{{"completed_day": "{FLOOR}"}}'.encode()
    assert read_journal(PROFILE).events == [live]


def test_retention_loader_rejects_non_object_record_without_rewriting_checkpoint() -> None:
    checkpoint = store.Checkpoint(profile=PROFILE, generation=1)
    checkpoint.retention["journal"] = ["not", "a", "record"]
    store.commit_checkpoint(PROFILE, checkpoint)
    checkpoint_path = storage.data_root() / "accounting" / PROFILE / "collector" / store.CHECKPOINT
    original = checkpoint_path.read_bytes()

    with pytest.raises(ValueError, match="unsupported journal retention checkpoint"):
        retention.load_journal_retention(PROFILE)

    assert checkpoint_path.read_bytes() == original
    loaded = store.load_checkpoint(PROFILE)
    assert loaded is not None
    assert loaded.retention["journal"] == ["not", "a", "record"]


def test_retention_prepare_rejects_unsupported_previous_version_with_initial_control() -> None:
    observed = read_journal(PROFILE)
    with pytest.raises(ValueError, match="unsupported journal retention checkpoint"):
        retention.prepare_journal_retention(
            observed,
            previous={"version": 99},
            horizon_day=FLOOR,
            aggregate_horizon_day=AGGREGATE,
        )

    initial = retention.prepare_journal_retention(
        observed,
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert initial["version"] == 1
    assert initial["horizon_day"] == FLOOR
    assert initial["aggregate_horizon_day"] == AGGREGATE


def test_apply_retention_records_only_the_inclusive_aggregate_boundary_identity() -> None:
    from ssgrep.accounting.capture.journal_codec import serialize_event

    before_aggregate = event("before-aggregate", "2025-11-04")
    on_aggregate = event("on-aggregate", AGGREGATE)
    on_horizon = event("on-horizon", FLOOR)
    original = (
        serialize_event(before_aggregate)
        + serialize_event(on_aggregate)
        + serialize_event(on_horizon)
    )
    source = segment("seg-v2-retention-boundaries.jsonl", original)
    observed = read_journal(PROFILE)
    record = {
        "version": 1,
        "key": "11" * 32,
        "horizon_day": FLOOR,
        "aggregate_horizon_day": AGGREGATE,
        "retired": {},
        "bridges": [],
    }

    retention.apply_journal_retention(observed, record)

    assert observed.events == [on_horizon]
    assert observed.retired_events == 2
    assert observed.retired == {
        retention.phase_key(on_aggregate, record): AGGREGATE,
    }
    assert source.read_bytes() == original


def test_retention_expansion_rejects_invalid_snapshot_and_reference_shapes() -> None:
    base = {
        "key": "11" * 32,
        "horizon_day": FLOOR,
        "aggregate_horizon_day": AGGREGATE,
        "proofs": [],
    }
    invalid = (
        ({"version": 3, "identity_shards": {}, **base}, "unsupported"),
        ({"version": 2, "identity_shards": [], **base}, "invalid journal identity references"),
        (
            {"version": 2, "identity_shards": {"aa": None}, **base},
            "invalid journal identity reference",
        ),
    )
    for snapshot, message in invalid:
        with pytest.raises(ValueError, match=message):
            retention.expand_journal_retention(PROFILE, snapshot)

    expanded = retention.expand_journal_retention(
        PROFILE,
        {"version": 2, "identity_shards": {}, **base},
    )
    assert expanded["version"] == 1
    assert expanded["retired"] == {} and expanded["bridges"] == []
    assert expanded["_identity_shards"] == {}


def test_retention_persist_rejects_unexpanded_and_malformed_phase_identities() -> None:
    base = {
        "version": 1,
        "key": "11" * 32,
        "horizon_day": FLOOR,
        "aggregate_horizon_day": AGGREGATE,
        "retired": {},
        "bridges": [],
        "proofs": [],
    }
    invalid = (
        ({**base, "version": 2}, "expanded journal retention"),
        ({**base, "retired": {"not-a-phase-key": "2026-11-09"}}, "invalid journal identity key"),
        ({**base, "bridges": ["not-a-phase-key"]}, "invalid journal bridge identity"),
    )
    for record, message in invalid:
        blobs = {"journal-retention:aa": "previous-shard", "unrelated": "keep"}
        original_blobs = dict(blobs)
        with pytest.raises(ValueError, match=message):
            retention.persist_journal_retention(PROFILE, record, blobs=blobs)
        assert blobs == original_blobs

    key = "ab" * 32
    blobs = {}
    persisted = retention.persist_journal_retention(
        PROFILE,
        {**base, "retired": {key: "2026-11-09"}},
        blobs=blobs,
    )
    assert persisted["version"] == 2
    assert set(blobs) == {"journal-retention:ab"}
    assert store.read_blob(PROFILE, blobs["journal-retention:ab"]) == {
        "version": 1,
        "retired": {key: "2026-11-09"},
        "bridges": [],
    }


def test_hard_linked_active_segment_is_refused_without_appending(tmp_path: Path) -> None:
    path, original = warm(tmp_path)
    alias = path.with_name("preserved-alias.jsonl")
    os.link(path, alias)

    with pytest.raises(storage.UnsafeStorageError):
        journal.append(PROFILE, build_event(execution_id="must-not-write-hardlink"))

    assert path.read_bytes() == original
    assert alias.read_bytes() == original
    assert path.stat().st_nlink == 2


def test_compaction_skips_checkpoint_segment_already_removed_by_another_maintainer() -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-v2-reclaimed-before-compact.jsonl",
        serialize_event(event("reclaimed-before-compact", "2026-11-09")),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    commit(prepared)
    source.unlink()

    result = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )

    assert result.status == "complete"
    assert result.removed_segments == 0 and result.deferred_segments == 0
    assert not source.exists()


def test_absent_root_binds_no_journal_authority_and_creates_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absent = tmp_path / "absent-data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(absent))
    state = journal._current()
    assert state.authority is None

    with pytest.raises(storage.RootChanged) as required:
        journal._require_root(state)
    with pytest.raises(storage.RootChanged) as allocated:
        journal._segment(state, PROFILE, 0)

    assert required.value.errno == errno.ESTALE and allocated.value.errno == errno.ESTALE
    assert state.segments == {} and not absent.exists()
    assert journal.append(PROFILE, build_event(execution_id="creates-root")) is True
    assert (absent / "accounting" / PROFILE / "journal").is_dir()
    assert journal._current().authority is not None


def test_root_removed_right_after_creating_append_is_a_transient_failure_not_a_latch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absent = tmp_path / "vanishing-data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(absent))
    real_snapshot = storage.snapshot

    def snapshot_then_vanish(path=None, *, create=False):
        if create:
            absent.mkdir(mode=0o700)
            absent.rmdir()  # another process removed the root before it could be pinned
        return real_snapshot(path)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "snapshot", snapshot_then_vanish)
        with pytest.raises(storage.RootChanged) as vanished:
            journal.append(PROFILE, build_event(execution_id="root-vanished"))
    assert vanished.value.errno == errno.ESTALE
    assert not absent.exists()
    assert PROFILE not in journal._current().unsafe

    assert journal.append(PROFILE, build_event(execution_id="after-vanish")) is True
    assert [row.execution_id for row in read_journal(PROFILE).events] == ["after-vanish"]


@pytest.mark.parametrize("change", ["mode", "extra-link"])
def test_active_file_changed_between_descriptor_and_name_checks_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    path, original = warm(tmp_path)
    real_stat = os.stat

    def stat_after_race(target, *args, **kwargs):
        if target == path.name and kwargs.get("dir_fd") is not None:
            if change == "mode":
                path.chmod(0o640)
            else:
                os.link(path, path.with_name("raced-alias.jsonl"))
        return real_stat(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "stat", stat_after_race)
        with pytest.raises(storage.UnsafeStorageError) as refused:
            journal.append(PROFILE, build_event(execution_id="must-not-leak"))
    assert refused.value.errno == errno.EPERM
    assert "authority changed" in str(refused.value)
    assert path.read_bytes() == original
    assert PROFILE in journal._current().unsafe
    assert journal.append(PROFILE, build_event(execution_id="still-disabled")) is False


def test_rotation_that_never_settles_fails_transiently_without_losing_later_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(journal, "SEGMENT_TARGET_BYTES", 0)  # every segment is already full
        with pytest.raises(BlockingIOError) as unsettled:
            journal.append(PROFILE, build_event(execution_id="never-settles"))
    assert unsettled.value.errno == errno.EAGAIN
    assert PROFILE not in journal._current().unsafe
    assert read_journal(PROFILE).events == []

    assert journal.append(PROFILE, build_event(execution_id="settles-after")) is True
    assert [row.execution_id for row in read_journal(PROFILE).events] == ["settles-after"]


def test_existing_retained_copy_truncated_while_verifying_cannot_authorize_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    live = event("truncated-copy-live", FLOOR)
    source = segment(
        "seg-v2-truncated-copy.jsonl",
        serialize_event(event("truncated-copy-expired", "2026-11-09")) + serialize_event(live),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    (proof,) = prepared["proofs"]
    commit(prepared)
    original = source.read_bytes()
    retained_bytes = serialize_event(live)
    destination = segment(replacement_name(proof, FLOOR, retained_bytes), retained_bytes)
    destination_inode = destination.stat().st_ino
    real_read, real_fstat = os.read, os.fstat

    def read_after_truncate(fd: int, count: int) -> bytes:
        if real_fstat(fd).st_ino == destination_inode:
            os.truncate(destination, 0)  # concurrent writer cut the file after the size check
        return real_read(fd, count)

    with monkeypatch.context() as patch:
        patch.setattr(journal_compact.os, "read", read_after_truncate)
        with pytest.raises(ValueError, match="truncated"):
            journal_compact.compact_committed_journal(
                PROFILE,
                generation=1,
                byte_budget=1024 * 1024,
                segment_budget=1,
                legacy_writers_stopped=True,
            )

    assert source.read_bytes() == original
    assert destination.read_bytes() == b""


def test_compaction_helper_rejects_unsafe_segment_name_without_opening_anything(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_reader import SegmentProof

    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"not journal authority\n")
    unsafe = SegmentProof("../outside.jsonl", 0, 0, 0, 0)
    result = journal_compact.JournalMaintenance()
    directory = storage.open_profile_dir(PROFILE, "journal")
    try:
        with pytest.raises(ValueError, match="unsafe segment"):
            journal_compact._compact_one(
                directory,
                unsafe,
                {},
                result,
                legacy_writers_stopped=True,
                byte_budget=1024,
            )
    finally:
        os.close(directory)
    assert result == journal_compact.JournalMaintenance()
    assert outside.read_bytes() == b"not journal authority\n"


def test_segment_removed_between_proof_lookup_and_open_is_skipped_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ssgrep.accounting.capture import journal_compact
    from ssgrep.accounting.capture.journal_codec import serialize_event

    source = segment(
        "seg-v2-raced-away.jsonl",
        serialize_event(event("raced-away-expired", "2026-11-09")),
    )
    prepared = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    commit(prepared)
    real_stat = os.stat

    def stat_then_unlink(target, *args, **kwargs):
        result = real_stat(target, *args, **kwargs)
        if target == source.name and kwargs.get("dir_fd") is not None and source.exists():
            source.unlink()  # another maintainer reclaimed it after the existence probe
        return result

    with monkeypatch.context() as patch:
        patch.setattr(journal_compact.os, "stat", stat_then_unlink)
        result = journal_compact.compact_committed_journal(
            PROFILE,
            generation=1,
            byte_budget=1024 * 1024,
            segment_budget=32,
            legacy_writers_stopped=True,
        )

    assert not source.exists()
    assert result.status == "complete"
    assert result.removed_segments == 0 and result.retained_segments == 0
    assert result.deferred_segments == 0 and result.bytes_read == 0


def test_committed_retention_without_a_journal_directory_is_complete_and_creates_none() -> None:
    from ssgrep.accounting.capture import journal_compact

    commit(record(4))
    journal_dir = storage.data_root() / "accounting" / PROFILE / "journal"
    assert not journal_dir.exists()

    result = journal_compact.compact_committed_journal(
        PROFILE,
        generation=1,
        byte_budget=1024 * 1024,
        segment_budget=32,
        legacy_writers_stopped=True,
    )

    assert result == journal_compact.JournalMaintenance()
    assert not journal_dir.exists()


def test_retained_event_keeps_only_horizon_events_and_bridged_started_phases() -> None:
    from ssgrep.accounting.capture.journal_codec import minimal_started
    from ssgrep.accounting.ledger.contracts import Phase

    expired_started = event("retained-started", "2026-11-09", phase=Phase.STARTED, outcome=None)
    expired_done = event("retained-done", "2026-11-09")
    current = event("retained-current", FLOOR)
    expired_retired = event("retained-retired", FLOOR)
    base = record(0)
    bridged = retention.phase_key(expired_started, base)
    retired_key = retention.phase_key(expired_retired, base)

    unbridged = {**base, "bridges": []}
    assert retention.retained_event(expired_started, unbridged) is None
    assert retention.retained_event(expired_done, unbridged) is None
    assert retention.retained_event(current, unbridged) == current

    bridging = {**base, "bridges": [bridged], "retired": {retired_key: "2026-11-09"}}
    assert retention.retained_event(expired_started, bridging) == minimal_started(expired_started)
    assert retention.retained_event(expired_done, bridging) is None
    assert retention.retained_event(expired_retired, bridging) is None


def test_identity_signatures_reject_references_that_do_not_match_their_bucket() -> None:
    checkpoint = commit(record(8))
    refs = checkpoint.retention["journal"]["identity_shards"]
    assert refs
    signatures = retention._signatures(PROFILE, refs)
    assert [(bucket, name) for bucket, name, *_ in signatures] == sorted(refs.items())

    bucket, name = next(iter(refs.items()))
    other_bucket = "00" if bucket != "00" else "01"
    for malformed in ({bucket: "not-a-blob-reference"}, {other_bucket: name}):
        with pytest.raises(ValueError, match="invalid journal identity reference"):
            retention._signatures(PROFILE, malformed)


def test_persist_replaces_stale_identity_roots_and_keeps_unrelated_blobs() -> None:
    key = "ab" * 32
    base = {**record(0), "retired": {key: "2026-11-09"}}
    blobs = {"journal-retention:zz": "stale-root", "unrelated": "keep"}

    persisted = retention.persist_journal_retention(PROFILE, base, blobs=blobs)

    assert set(blobs) == {"journal-retention:ab", "unrelated"}
    assert blobs["unrelated"] == "keep"
    assert persisted["identity_shards"] == {"ab": blobs["journal-retention:ab"]}
    shard = store.read_blob(PROFILE, blobs["journal-retention:ab"])
    assert shard is not None
    assert shard["retired"] == {key: "2026-11-09"}
