"""Warm journal descriptors must retain private authority without blocking lookup."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ssgrep.accounting import journal, journal_reader, policy, storage
from tests.accounting.conftest import build_event

PROFILE = "personal-local"
ROOT = Path(__file__).resolve().parents[2]


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
    from ssgrep.accounting import capture, journal
    from ssgrep.accounting.contracts import Transport
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
