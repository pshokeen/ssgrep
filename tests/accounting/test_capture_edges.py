"""Edge behavior of capture, journal and storage that real callers can hit."""

from __future__ import annotations

import errno
import os
import threading
from pathlib import Path

import pytest

from ssgrep.accounting import (
    capture,
    capture_policy,
    datadir,
    fingerprint,
    journal,
    journal_codec,
    journal_reader,
    policy,
    storage,
)
from ssgrep.accounting.contracts import MAX_EVENT_BYTES, Operation, Outcome, Phase, Quantity
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from tests.accounting.conftest import build_event

PROFILE = "personal-local"


@pytest.fixture(autouse=True)
def _reset():
    journal.close_all()
    fingerprint.forget_keys()
    capture._unsafe.clear()
    capture_policy._cache = ("", None, capture_policy._DISABLED)
    yield
    journal.close_all()
    capture._unsafe.clear()


def _events():
    return journal_reader.read_journal(PROFILE, quarantine=False).events


def test_rendered_records_observed_bytes_when_the_layer_knows_them() -> None:
    policy.enable_collection(PROFILE)
    execution = capture.begin(Operation.SEARCH, text="q")
    capture.rendered(execution, rendered_bytes=Quantity.observed(812, "bytes", "mcp_json"))
    rendered = [e for e in _events() if e.phase is Phase.RENDERED]
    assert rendered[0].rendered_bytes == Quantity.observed(812, "bytes", "mcp_json")


def test_unreadable_response_is_recorded_unknown_not_ok() -> None:
    policy.enable_collection(PROFILE)
    capture.finish_search(capture.begin(Operation.SEARCH, text="q"), object())
    capture.finish_show(capture.begin(Operation.SHOW, text="r"), "r", object())
    outcomes = [e.outcome for e in _events() if e.phase is Phase.COMPLETED]
    assert outcomes == [Outcome.UNKNOWN, Outcome.UNKNOWN]


def test_begin_never_raises_when_key_creation_hits_unsafe_storage(tmp_path: Path) -> None:
    policy.enable_collection(PROFILE)
    victim = tmp_path / "victim"
    victim.mkdir()
    (tmp_path / "data" / "accounting" / PROFILE).symlink_to(victim)
    assert capture.begin(Operation.SEARCH, text="q") is None
    assert (datadir.snapshot(), PROFILE) in capture._unsafe
    assert capture.begin(Operation.SEARCH, text="again") is None  # stays off for the process
    assert list(victim.iterdir()) == []


def test_unknown_scope_selects_only_local_personal_journal() -> None:
    policy.enable_collection(PROFILE)
    assert capture_policy.capture_profile(None) == PROFILE
    assert capture_policy.capture_profile("") is None


def test_explicit_workspace_assignment_beats_a_broader_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = tmp_path / "ws"
    inner = outer / "sub" / "repo"
    inner.mkdir(parents=True)
    policy.enable_collection(PROFILE)
    policy.save_profile(
        ProfilePolicy(profile_id=PROFILE, roots=(str(outer),), collection_enabled=True)
    )
    policy.save_profile(
        ProfilePolicy(profile_id="org-a", roots=(str(inner),), collection_enabled=True)
    )

    assert capture_policy.capture_profile(os.path.realpath(inner)) == "org-a"
    assert capture_policy.capture_profile(os.path.realpath(outer)) == PROFILE
    assert capture_policy.capture_profile(os.path.realpath(tmp_path)) is None


def test_workspace_assignment_resolves_overlapping_profiles(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    policy.enable_collection(PROFILE)
    file = policy.read_policy_file()
    from dataclasses import replace

    org = ProfilePolicy(profile_id="org-a", roots=(str(shared),), collection_enabled=True)
    staged = replace(
        file,
        profiles={**file.profiles, "org-a": org},
        assignments={os.path.realpath(shared): "org-a"},
    )
    assert capture_policy.policy_file() is not capture_policy._DISABLED
    capture_policy._cache = (*capture_policy._cache[:2], staged)
    assert capture_policy.capture_profile(os.path.realpath(shared)) == "org-a"


def test_event_larger_than_bound_writes_nothing_and_never_raises(tmp_path: Path) -> None:
    policy.enable_collection(PROFILE)
    huge_cwd = "/" + "d" * MAX_EVENT_BYTES
    execution = capture.begin(Operation.SEARCH, text="q", cwd=huge_cwd)
    assert execution is not None  # the operation proceeds; only the unwritable event is lost
    capture.complete(execution, Outcome.OK)
    capture.rendered(execution)
    assert _events() == []
    journal_dir = tmp_path / "data" / "accounting" / PROFILE / "journal"
    assert not journal_dir.exists() or all(p.stat().st_size == 0 for p in journal_dir.iterdir())
    assert (datadir.snapshot(), PROFILE) not in capture._unsafe  # oversize is not unsafe storage


def test_zero_progress_write_is_an_error_not_an_infinite_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal.append(PROFILE, build_event(execution_id="a-1"))
    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "write", lambda fd, data: 0)
        with pytest.raises(OSError) as caught:
            journal.append(PROFILE, build_event(execution_id="a-2"))
    assert caught.value.errno == errno.EIO
    journal.append(PROFILE, build_event(execution_id="a-3"))
    assert {e.execution_id for e in _events()} == {"a-1", "a-3"}


def test_state_is_reinitialised_when_the_pid_changes_without_the_fork_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = journal.next_execution_id()
    monkeypatch.setattr(journal.os, "getpid", lambda: journal._state.pid + 1)
    second = journal.next_execution_id()
    assert first.split("-")[0] != second.split("-")[0] and second.endswith("-1")


def test_nonce_collision_never_shares_another_processes_segment() -> None:
    state = journal._current()
    nonce = state.nonce
    taken = journal.segment_name(nonce, state.segment_seq + 1)
    original = journal.serialize_event(build_event(execution_id="previous-1"))
    directory = storage.open_profile_dir(PROFILE, "journal")
    try:
        fd = storage.create_exclusive(directory, taken)
        try:
            storage.write_all(fd, original)
        finally:
            os.close(fd)
    finally:
        os.close(directory)
    journal.append(PROFILE, build_event(execution_id="x-1"))
    root = datadir.data_root() / "accounting" / PROFILE / "journal"
    assert state.nonce == nonce
    assert (root / taken).read_bytes() == original
    assert len(list(root.glob("seg-*.jsonl"))) == 2
    assert {event.execution_id for event in _events()} == {"previous-1", "x-1"}


def test_concurrent_first_appends_create_one_segment_per_process() -> None:
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(20):
                journal.append(PROFILE, build_event(execution_id=journal.next_execution_id()))
        except BaseException as error:  # pragma: no cover - failure path
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    directory = datadir.data_root() / "accounting" / PROFILE / "journal"
    assert len(list(directory.iterdir())) == 1
    assert len({e.execution_id for e in _events()}) == 160


def test_quantity_with_bad_shape_is_quarantined_not_admitted() -> None:
    import json

    document = json.loads(journal_codec.serialize_event(build_event()))
    document["rendered_bytes"] = {"value": 5, "unit": "bytes", "basis": "not-a-basis"}
    with pytest.raises(journal_codec.MalformedEvent):
        journal_codec.parse_event(json.dumps(document).encode())
    document["rendered_bytes"] = {"value": 5}
    with pytest.raises(journal_codec.MalformedEvent):
        journal_codec.parse_event(json.dumps(document).encode())


def test_non_segment_files_in_the_journal_directory_are_ignored() -> None:
    journal.append(PROFILE, build_event(execution_id="n-1"))
    directory = storage.open_profile_dir(PROFILE, "journal")
    os.close(storage.create_exclusive(directory, "notes.txt"))
    os.close(directory)
    read = journal_reader.read_journal(PROFILE, quarantine=False)
    assert read.segments == 1 and [e.execution_id for e in read.events] == ["n-1"]


def test_oversized_line_split_across_read_chunks_is_skipped_incrementally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal.append(PROFILE, build_event(execution_id="before-1"))
    journal.close_all()
    directory = datadir.data_root() / "accounting" / PROFILE / "journal"
    (segment,) = directory.iterdir()
    follow = journal_codec.serialize_event(build_event(execution_id="after-1"))
    with open(segment, "ab") as handle:
        handle.write(b"y" * (MAX_EVENT_BYTES * 2) + b"\n" + follow)
        handle.write(b"z" * (MAX_EVENT_BYTES * 2))  # an unterminated oversize tail
    monkeypatch.setattr(journal_reader, "_CHUNK", 4096)  # force many chunk boundaries

    read = journal_reader.read_journal(PROFILE, quarantine=False)

    assert {e.execution_id for e in read.events} == {"before-1", "after-1"}
    assert len(read.quarantined) == 1 and len(read.incomplete_tails) == 1
    assert read.incomplete_tails[0].length == MAX_EVENT_BYTES * 2


def test_record_split_across_chunks_still_parses_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(30):
        journal.append(PROFILE, build_event(execution_id=f"c-{index}"))
    journal.close_all()
    monkeypatch.setattr(journal_reader, "_CHUNK", 97)
    read = journal_reader.read_journal(PROFILE, quarantine=False)
    assert len(read.events) == 30 and read.damage == []


def test_directory_race_loser_opens_the_winners_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_mkdir = os.mkdir
    calls = {"n": 0}

    def racing(name, mode=0o777, *, dir_fd=None):
        calls["n"] += 1
        real_mkdir(name, mode, dir_fd=dir_fd)  # the "other process" creates it first
        real_mkdir(name, mode, dir_fd=dir_fd)  # then ours fails with FileExistsError

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "mkdir", racing)
        os.close(storage.open_profile_dir("racer"))
    assert calls["n"] >= 1


def test_non_directory_component_and_non_regular_file_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "data" / "accounting"
    root.mkdir(parents=True, mode=0o700)
    (root / PROFILE).write_text("a file where a directory belongs")
    with pytest.raises(storage.UnsafeStorageError):
        storage.open_profile_dir(PROFILE)

    directory = storage.open_dir("plain")
    try:
        os.mkfifo("pipe", dir_fd=None) if False else os.mkfifo(root / "plain" / "pipe")
        with pytest.raises(storage.UnsafeStorageError):
            storage.open_regular(directory, "pipe")
    finally:
        os.close(directory)


def test_loose_file_permissions_are_refused_without_repair_on_read() -> None:
    directory = storage.open_profile_dir(PROFILE)
    try:
        path = datadir.data_root() / "accounting" / PROFILE / "loose"
        path.write_bytes(b"x")
        os.chmod(path, 0o644)
        before = path.stat()
        with pytest.raises(storage.UnsafeStorageError):
            storage.open_regular(directory, "loose")
        after = path.stat()
        assert (after.st_mode, after.st_ctime_ns, after.st_ino) == (
            before.st_mode,
            before.st_ctime_ns,
            before.st_ino,
        )
        assert path.read_bytes() == b"x"
    finally:
        os.close(directory)


def test_zero_progress_write_in_atomic_write_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = storage.open_profile_dir(PROFILE)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(storage.os, "write", lambda fd, data: 0)
            with pytest.raises(OSError):
                storage.atomic_write(directory, "x.json", b"data")
        assert storage.list_names(directory) == []  # no temporary left behind
    finally:
        os.close(directory)


def test_rename_failure_removes_the_temporary_and_keeps_the_old_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = storage.open_profile_dir(PROFILE)
    try:
        storage.atomic_write(directory, "x.json", b"old")
        with monkeypatch.context() as patch:

            def refuse(*_a: object, **_k: object) -> None:
                raise OSError(errno.EXDEV, "cross-device")

            patch.setattr(storage.os, "rename", refuse)
            with pytest.raises(OSError):
                storage.atomic_write(directory, "x.json", b"new")
        assert storage.read_file(directory, "x.json", 10) == b"old"
        assert storage.list_names(directory) == ["x.json"]
    finally:
        os.close(directory)


def test_fingerprint_reader_never_mints_a_key_and_missing_key_with_create_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert fingerprint.load_key(PROFILE, create=False) is None
    directory = storage.open_profile_dir(PROFILE)
    os.close(directory)
    assert fingerprint.load_key(PROFILE, create=False) is None
    assert not (tmp_path / "data" / "accounting" / PROFILE / fingerprint.KEY_FILE).exists()
    with monkeypatch.context() as patch:  # a key that vanishes between write and read
        patch.setattr(fingerprint.storage, "write_once", lambda *a, **k: False)
        with pytest.raises(storage.UnsafeStorageError) as caught:
            fingerprint.load_key(PROFILE, create=True)
        assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert not (tmp_path / "data" / "accounting" / PROFILE / fingerprint.KEY_FILE).exists()


def test_closed_segment_descriptor_is_tolerated_on_close() -> None:
    journal.append(PROFILE, build_event(execution_id="d-1"))
    os.close(journal._current().segments[PROFILE].fd)  # something else closed it
    journal.close_all()  # must not raise
    journal.append(PROFILE, build_event(execution_id="d-2"))
    assert {e.execution_id for e in _events()} == {"d-1", "d-2"}


def test_package_version_lookup_falls_back_to_unknown_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(capture, "_version_cache", [])
    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert capture._version() == "unknown"
    assert capture._version() == "unknown"  # cached, no second lookup


def test_closing_an_already_closed_parent_descriptor_during_fork_reinit_is_tolerated() -> None:
    journal.append(PROFILE, build_event(execution_id="f-1"))
    os.close(journal._current().segments[PROFILE].fd)
    journal._reinit_after_fork()  # must not raise on the stale descriptor
    assert journal._current().segments == {}


def test_personal_scope_with_roots_excludes_directories_outside_them(tmp_path: Path) -> None:
    inside = tmp_path / "work"
    inside.mkdir()
    policy.enable_collection(PROFILE)
    policy.save_profile(
        ProfilePolicy(profile_id=PROFILE, roots=(str(inside),), collection_enabled=True)
    )
    assert capture_policy.capture_profile(os.path.realpath(inside)) == PROFILE
    assert capture_policy.capture_profile(os.path.realpath(tmp_path)) is None


def test_policy_without_a_personal_profile_captures_nothing_outside_org_roots(
    tmp_path: Path,
) -> None:
    org_root = tmp_path / "org"
    other = tmp_path / "elsewhere"
    org_root.mkdir()
    other.mkdir()
    policy.save_profile(ProfilePolicy(profile_id="org-a", roots=(str(org_root),)))
    assert policy.read_policy_file().profiles.keys() == {"org-a"}
    assert capture_policy.capture_profile(os.path.realpath(other)) is None
    assert capture_policy.capture_profile(os.path.realpath(org_root)) is None  # not enabled
