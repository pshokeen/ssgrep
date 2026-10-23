"""Edge behavior of capture, journal and storage that real callers can hit."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import threading
from pathlib import Path

import pytest

import ssgrep.accounting.privacy.storage as datadir
from ssgrep.accounting.capture import (
    capture,
    capture_policy,
    fingerprint,
    journal,
    journal_codec,
    journal_reader,
)
from ssgrep.accounting.ledger import contracts
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.privacy import policy, storage, storage as plainfile
from ssgrep.services import mcp_server
from tests.accounting.conftest import build_event

MAX_EVENT_BYTES, Operation, Outcome, Phase, Quantity = (
    contracts.MAX_EVENT_BYTES,
    contracts.Operation,
    contracts.Outcome,
    contracts.Phase,
    contracts.Quantity,
)

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


def test_invalid_explicit_scope_is_not_captured_or_persisted(tmp_path: Path) -> None:
    policy.enable_collection(PROFILE)

    execution = capture.begin(Operation.SEARCH, text="private query", cwd="relative/workspace")

    assert execution is None
    profile = tmp_path / "data" / "accounting" / PROFILE
    assert not (profile / fingerprint.KEY_FILE).exists()
    assert not (profile / "journal").exists()


def test_existing_malformed_fingerprint_key_is_refused_without_replacement() -> None:
    directory = storage.open_profile_dir(PROFILE)
    try:
        storage.write_once(directory, fingerprint.KEY_FILE, b"bad-key")
    finally:
        os.close(directory)

    with pytest.raises(storage.UnsafeStorageError):
        fingerprint.load_key(PROFILE, create=False)
    assert (
        datadir.data_root() / "accounting" / PROFILE / fingerprint.KEY_FILE
    ).read_bytes() == b"bad-key"


def test_rendered_records_observed_bytes_when_the_layer_knows_them() -> None:
    policy.enable_collection(PROFILE)
    execution = capture.begin(Operation.SEARCH, text="q")
    capture.rendered(execution, rendered_bytes=Quantity.observed(812, "bytes", "mcp_json"))
    rendered = [e for e in _events() if e.phase is Phase.RENDERED]
    assert rendered[0].rendered_bytes == Quantity.observed(812, "bytes", "mcp_json")


def test_malformed_private_capture_policy_disables_selection_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    policy.save_profile(ProfilePolicy(profile_id=PROFILE, collection_enabled=True))
    path = root / "accounting" / "policy.json"
    path.write_bytes(b"{")

    assert capture_policy.policy_file() is capture_policy._DISABLED
    assert capture_policy.capture_profile(None) is None
    assert path.read_bytes() == b"{"
    assert not (root / "accounting" / PROFILE / fingerprint.KEY_FILE).exists()


def test_capture_abandons_event_when_private_root_snapshot_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy.enable_collection(PROFILE)
    monkeypatch.setattr(datadir, "snapshot", lambda *args, **kwargs: None)

    assert capture.begin(Operation.SEARCH, text="q") is None
    profile = tmp_path / "data" / "accounting" / PROFILE
    assert not (profile / fingerprint.KEY_FILE).exists()
    assert not (profile / "journal").exists()


def test_policy_lookup_with_absent_private_root_does_not_create_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "not-created"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))

    assert capture_policy.policy_file() is capture_policy._DISABLED
    assert not root.exists()


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


def test_reader_collapses_identical_events_and_reports_phase_conflicts() -> None:
    first = build_event(execution_id="conflict-1", refs=("first",))
    second = build_event(execution_id="conflict-1", refs=("second",), outcome=Outcome.ERROR)
    journal.append(PROFILE, first)
    journal.append(PROFILE, first)
    journal.append(PROFILE, second)

    read = journal_reader.read_journal(PROFILE, quarantine=False)

    expected = min(
        (first, second),
        key=lambda event: hashlib.sha256(contracts.canonical_bytes(event)).hexdigest(),
    )
    assert read.duplicates == 1
    assert read.conflicts == [("conflict-1", "completed")]
    assert read.events == [expected]


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


"""Behavior of private accounting storage (task 3.1)."""


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_directories_are_0700_and_files_0600_under_accounting_only(tmp_path: Path) -> None:
    fd = storage.open_profile_dir("personal-local", "journal")
    try:
        storage.atomic_write(fd, "state.json", b"{}")
    finally:
        os.close(fd)

    root = tmp_path / "data" / "accounting"
    assert _mode(root) == 0o700
    assert _mode(root / "personal-local") == 0o700
    assert _mode(root / "personal-local" / "journal") == 0o700
    assert _mode(root / "personal-local" / "journal" / "state.json") == 0o600
    # Never under the indexed notes tree or a repo-local .ssgrep.
    assert not (tmp_path / "data" / "notes").exists()
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["accounting"]


def test_permissive_existing_directory_is_tightened(tmp_path: Path) -> None:
    (tmp_path / "data" / "accounting" / "personal-local").mkdir(parents=True, mode=0o755)
    os.chmod(tmp_path / "data" / "accounting" / "personal-local", 0o755)
    os.close(storage.open_profile_dir("personal-local"))
    assert _mode(tmp_path / "data" / "accounting" / "personal-local") == 0o700


def test_symlinked_accounting_dir_is_refused_and_target_untouched(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("keep")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "accounting").symlink_to(victim)

    with pytest.raises(storage.UnsafeStorageError):
        storage.open_profile_dir("personal-local", "journal")

    assert sorted(p.name for p in victim.iterdir()) == ["precious.txt"]
    assert (victim / "precious.txt").read_text() == "keep"


def test_symlinked_profile_dir_is_refused(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    root = tmp_path / "data" / "accounting"
    root.mkdir(parents=True, mode=0o700)
    (root / "personal-local").symlink_to(victim)

    with pytest.raises(storage.UnsafeStorageError):
        storage.open_profile_dir("personal-local")
    assert list(victim.iterdir()) == []


def test_symlinked_file_cannot_be_read_written_or_deleted(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    fd = storage.open_profile_dir("personal-local")
    try:
        os.symlink(secret, "link", dir_fd=fd)
        with pytest.raises(storage.UnsafeStorageError):
            storage.read_file(fd, "link", 100)
        with pytest.raises(storage.UnsafeStorageError):
            storage.open_regular(fd, "link")
        with pytest.raises(storage.UnsafeStorageError):
            storage.delete_file(fd, "link")
        with pytest.raises(FileExistsError):
            storage.create_exclusive(fd, "link")
        # An atomic replace overwrites the link itself, never its target.
        storage.atomic_write(fd, "link", b"new")
    finally:
        os.close(fd)
    assert secret.read_text() == "outside"
    assert not os.path.islink(tmp_path / "data" / "accounting" / "personal-local" / "link")


@pytest.mark.parametrize("profile", ["../escape", "a/b", "", "UPPER", "x" * 33, "/abs"])
def test_profile_ids_are_identifiers_not_paths(profile: str, tmp_path: Path) -> None:
    with pytest.raises(storage.UnsafeStorageError):
        storage.open_profile_dir(profile)
    assert (
        not (tmp_path / "data" / "accounting").exists()
        or list((tmp_path / "data" / "accounting").iterdir()) == []
    )


def test_foreign_owner_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    os.close(storage.open_profile_dir("personal-local"))
    real_uid = os.geteuid()
    monkeypatch.setattr(storage.os, "geteuid", lambda: real_uid + 1)
    with pytest.raises(storage.UnsafeStorageError):
        storage.open_profile_dir("personal-local", create=False)


def test_atomic_write_replaces_whole_content_and_leaves_no_temporaries() -> None:
    fd = storage.open_profile_dir("personal-local")
    try:
        storage.atomic_write(fd, "ckpt.json", b"one")
        storage.atomic_write(fd, "ckpt.json", b"two-two")
        assert storage.read_file(fd, "ckpt.json", 100) == b"two-two"
        assert storage.list_names(fd) == ["ckpt.json"]
    finally:
        os.close(fd)


def test_atomic_write_failure_before_rename_keeps_old_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fd = storage.open_profile_dir("personal-local")
    try:
        storage.atomic_write(fd, "ckpt.json", b"good")

        def boom(*_a: object, **_k: object) -> None:
            raise OSError(28, "No space left on device")

        with monkeypatch.context() as patch:
            patch.setattr(storage.os, "fsync", boom)
            with pytest.raises(OSError):
                storage.atomic_write(fd, "ckpt.json", b"half-written-replacement")
        assert storage.read_file(fd, "ckpt.json", 100) == b"good"
        assert storage.list_names(fd) == ["ckpt.json"]
    finally:
        os.close(fd)


def test_write_once_never_overwrites() -> None:
    fd = storage.open_profile_dir("personal-local")
    try:
        assert storage.write_once(fd, "key", b"first") is True
        assert storage.write_once(fd, "key", b"second") is False
        assert storage.read_file(fd, "key", 100) == b"first"
        assert storage.list_names(fd) == ["key"]
    finally:
        os.close(fd)


def test_read_file_absent_and_oversized() -> None:
    fd = storage.open_profile_dir("personal-local")
    try:
        assert storage.read_file(fd, "missing", 10) is None
        storage.atomic_write(fd, "big", b"x" * 11)
        with pytest.raises(ValueError):
            storage.read_file(fd, "big", 10)
        assert storage.read_file(fd, "big", 11) == b"x" * 11
    finally:
        os.close(fd)


def test_delete_file_only_owned_regular_files(tmp_path: Path) -> None:
    fd = storage.open_profile_dir("personal-local")
    try:
        storage.atomic_write(fd, "gone", b"x")
        os.mkdir("sub", dir_fd=fd)
        assert storage.delete_file(fd, "gone") is True
        assert storage.delete_file(fd, "gone") is False
        with pytest.raises(storage.UnsafeStorageError):
            storage.delete_file(fd, "sub")
    finally:
        os.close(fd)


def test_component_names_cannot_traverse(tmp_path: Path) -> None:
    for bad in ("..", ".", "a/b", ""):
        with pytest.raises(storage.UnsafeStorageError):
            storage.open_dir(bad)


def test_private_directory_owned_by_another_user_is_not_read_or_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plainfile.write_private(tmp_path, "private", "state.json", data=b"original")
    directory = tmp_path / "private"
    before = directory.stat()
    uid = os.geteuid()
    with monkeypatch.context() as foreign:
        foreign.setattr(plainfile.os, "geteuid", lambda: uid + 1)
        with pytest.raises(plainfile.UnsafePathError, match="not owned"):
            plainfile.read_private(tmp_path, "private", "state.json", max_bytes=8)
    assert directory.stat() == before
    assert plainfile.read_private(tmp_path, "private", "state.json", max_bytes=8) == b"original"


@pytest.mark.parametrize("component,mode", [("directory", 0o750), ("file", 0o640), ("file", 0o700)])
def test_private_read_rejects_exposed_modes_without_mutating_original(
    tmp_path: Path, component: str, mode: int
) -> None:
    plainfile.write_private(tmp_path, "private", "state.json", data=b"private-value")
    target = tmp_path / "private" / "state.json"
    changed = target.parent if component == "directory" else target
    changed.chmod(mode)
    before = changed.stat()
    with pytest.raises(plainfile.UnsafePathError):
        plainfile.read_private(tmp_path, "private", "state.json", max_bytes=13)
    after = changed.stat()
    assert (after.st_mode, after.st_mtime_ns) == (before.st_mode, before.st_mtime_ns)
    assert target.read_bytes() == b"private-value"
    changed.chmod(0o700 if component == "directory" else 0o600)
    assert (
        plainfile.read_private(tmp_path, "private", "state.json", max_bytes=13) == b"private-value"
    )


def test_private_read_accepts_owner_readonly_modes_and_does_not_check_public_base(
    tmp_path: Path,
) -> None:
    base = tmp_path / "configured-data-root"
    base.mkdir(mode=0o755)
    plainfile.write_private(base, "private", "state.json", data=b"readonly")
    directory = base / "private"
    target = directory / "state.json"
    directory.chmod(0o500)
    target.chmod(0o400)
    before = (base.stat().st_mode, directory.stat().st_mode, target.stat().st_mode)
    try:
        assert plainfile.read_private(base, "private", "state.json", max_bytes=8) == b"readonly"
        assert (base.stat().st_mode, directory.stat().st_mode, target.stat().st_mode) == before
    finally:
        directory.chmod(0o700)
        target.chmod(0o600)


"""Capture authorization, key creation and every phase belong to one root inode."""


Operation, Outcome, Phase = contracts.Operation, contracts.Outcome, contracts.Phase

PROFILE = "personal-local"


@pytest.fixture(autouse=True)
def _reset_state():
    journal.close_all()
    fingerprint.forget_keys()
    capture._unsafe.clear()
    capture_policy._cache = ("", None, capture_policy._DISABLED)
    yield
    journal.close_all()
    fingerprint.forget_keys()
    capture._unsafe.clear()
    capture_policy._cache = ("", None, capture_policy._DISABLED)


def _select(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))


def _policy(monkeypatch: pytest.MonkeyPatch, root: Path, *, enabled: bool = True) -> None:
    _select(monkeypatch, root)
    policy.save_profile(ProfilePolicy(profile_id=PROFILE, collection_enabled=enabled))


def _private_files(root: Path) -> dict[str, bytes]:
    folder = root / "accounting" / PROFILE
    return (
        {
            str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*")
            if path.is_file()
        }
        if folder.exists()
        else {}
    )


def _completed() -> set[str]:
    return {
        event.execution_id
        for event in journal_reader.read_journal(PROFILE, quarantine=False).events
        if event.phase is Phase.COMPLETED
    }


@pytest.mark.parametrize("enabled_destination", [False, True])
@pytest.mark.parametrize("stage", ["profile", "key_create", "append"])
def test_root_switch_at_each_capture_boundary_never_mutates_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    enabled_destination: bool,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _policy(monkeypatch, second, enabled=enabled_destination)
    _select(monkeypatch, first)
    baseline = mcp_server.search_sessions("isolated-root-binding")
    assert "error" in baseline
    _policy(monkeypatch, first)
    assert not _private_files(first) and not _private_files(second)
    reached = False

    with monkeypatch.context() as fault:
        if stage == "profile":
            original = capture_policy.capture_profile

            def selected(*args, **kwargs):
                nonlocal reached
                result = original(*args, **kwargs)
                assert result == PROFILE
                reached = True
                os.environ["SSGREP_DATA_DIR"] = str(second)
                return result

            fault.setattr(capture_policy, "capture_profile", selected)
        elif stage == "key_create":
            original = storage.write_once

            def created(*args, **kwargs):
                nonlocal reached
                reached = True
                os.environ["SSGREP_DATA_DIR"] = str(second)
                return original(*args, **kwargs)

            fault.setattr(storage, "write_once", created)
        else:
            original = journal.append

            def appended(*args, **kwargs):
                nonlocal reached
                reached = True
                os.environ["SSGREP_DATA_DIR"] = str(second)
                return original(*args, **kwargs)

            fault.setattr(journal, "append", appended)
        assert mcp_server.search_sessions("isolated-root-binding") == baseline
    assert reached
    assert not _private_files(second), (
        "Authorization from the first root cannot create destination evidence"
    )
    assert not (first / "accounting" / PROFILE / "journal").exists()
    if stage == "profile":
        assert not _private_files(first)
    else:
        assert (
            len((first / "accounting" / PROFILE / fingerprint.KEY_FILE).read_bytes())
            == fingerprint.KEY_BYTES
        )
    assert mcp_server.search_sessions("isolated-root-binding") == baseline
    if enabled_destination:
        assert len(_completed()) == 1
    else:
        assert not _private_files(second)
    _select(monkeypatch, first)
    assert mcp_server.search_sessions("isolated-root-binding") == baseline
    assert len(_completed()) == 1


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_same_configured_path_cannot_authorize_a_replacement_root_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    actual, destination, configured = (
        tmp_path / "actual",
        tmp_path / "destination",
        tmp_path / "configured",
    )
    _policy(monkeypatch, destination, enabled=False)
    _policy(monkeypatch, actual)
    if replacement == "symlink":
        configured.symlink_to(actual, target_is_directory=True)
    else:
        actual.rename(configured)
    _select(monkeypatch, configured)
    baseline = mcp_server.search_sessions("isolated-inode-control")
    original_files = _private_files(configured)
    assert original_files and len(_completed()) == 1
    original = capture_policy.capture_profile
    moved = tmp_path / "original-pinned"
    reached = False

    def selected(*args, **kwargs):
        nonlocal reached
        result = original(*args, **kwargs)
        assert result == PROFILE
        reached = True
        if replacement == "symlink":
            configured.unlink()
            configured.symlink_to(destination, target_is_directory=True)
        else:
            configured.rename(moved)
            destination.rename(configured)
        return result

    try:
        with monkeypatch.context() as fault:
            fault.setattr(capture_policy, "capture_profile", selected)
            assert mcp_server.search_sessions("isolated-inode-control") == baseline
        assert reached
        assert not _private_files(configured)
        source = actual if replacement == "symlink" else moved
        assert _private_files(source) == original_files
    finally:
        if reached:
            if replacement == "symlink":
                configured.unlink()
                configured.symlink_to(actual, target_is_directory=True)
            else:
                configured.rename(destination)
                moved.rename(configured)
    assert mcp_server.search_sessions("isolated-inode-control") == baseline
    assert len(_completed()) == 2


@pytest.mark.parametrize("destination_enabled", [False, True])
def test_old_inflight_execution_drops_later_phases_after_root_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_enabled: bool,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _policy(monkeypatch, second, enabled=destination_enabled)
    _policy(monkeypatch, first)
    execution = capture.begin(Operation.SEARCH, text="isolated-inflight")
    assert execution is not None
    assert execution.root == datadir.snapshot()
    original = _private_files(first)
    assert set(event.phase for event in journal_reader.read_journal(PROFILE).events) == {
        Phase.STARTED
    }
    _select(monkeypatch, second)
    capture.complete(execution, Outcome.OK)
    capture.rendered(execution)
    assert _private_files(first) == original and not _private_files(second)
    _select(monkeypatch, first)
    capture.complete(execution, Outcome.OK)
    assert _private_files(first) == original, (
        "Observed root switching retires managed execution identity"
    )
    assert "error" in mcp_server.search_sessions("isolated-new-inflight-control")
    assert len(_completed()) == 1


def test_unsafe_capture_latch_is_root_inode_scoped_and_survives_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _policy(monkeypatch, second)
    _policy(monkeypatch, first)
    response = mcp_server.search_sessions("isolated-latch-control")
    first_authority = datadir.snapshot()
    assert first_authority is not None
    before = _private_files(first)
    key = first / "accounting" / PROFILE / fingerprint.KEY_FILE
    key.chmod(0o644)
    try:
        assert mcp_server.search_sessions("isolated-latch-control") == response
    finally:
        key.chmod(0o600)
    assert _private_files(first) == before
    assert (first_authority, PROFILE) in capture._unsafe
    _select(monkeypatch, second)
    assert mcp_server.search_sessions("isolated-latch-control") == response
    assert len(_completed()) == 1, "Unsafe first-root profile cannot suppress a healthy second root"
    _select(monkeypatch, first)
    assert mcp_server.search_sessions("isolated-latch-control") == response
    assert _private_files(first) == before


def test_replaced_unsafe_root_does_not_inherit_the_old_inode_latch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured, replacement, moved = (
        tmp_path / "configured",
        tmp_path / "replacement",
        tmp_path / "retired",
    )
    _policy(monkeypatch, replacement)
    _policy(monkeypatch, configured)
    response = mcp_server.search_sessions("isolated-inode-latch")
    key = configured / "accounting" / PROFILE / fingerprint.KEY_FILE
    key.chmod(0o644)
    assert mcp_server.search_sessions("isolated-inode-latch") == response
    key.chmod(0o600)
    original = _private_files(configured)
    configured.rename(moved)
    replacement.rename(configured)
    assert mcp_server.search_sessions("isolated-inode-latch") == response
    assert len(_completed()) == 1
    assert _private_files(moved) == original
    configured.rename(replacement)
    moved.rename(configured)
    assert mcp_server.search_sessions("isolated-inode-latch") == response
    assert _private_files(configured) == original


def test_disabled_capture_root_snapshot_never_creates_a_missing_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "disabled-missing"
    _select(monkeypatch, root)
    assert datadir.snapshot() is None
    assert "error" in mcp_server.search_sessions("isolated-disabled-root")
    assert not root.exists()
    journal.close_all()
    assert not root.exists(), "Closing private capture state is also noncreating"


def test_fork_reinitialization_preserves_root_latches_with_a_fresh_mutex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _policy(monkeypatch, second)
    _policy(monkeypatch, first)
    response = mcp_server.search_sessions("isolated-fork-root-latch")
    segment = next((first / "accounting" / PROFILE / "journal").glob("seg-*.jsonl"))
    original = _private_files(first)
    segment.chmod(0o644)
    try:
        assert mcp_server.search_sessions("isolated-fork-root-latch") == response
    finally:
        segment.chmod(0o600)
    previous = journal._current()
    assert PROFILE in previous.unsafe
    journal._reinit_after_fork()
    current = journal._current()
    assert current.lock is not previous.lock and current.segments == {}
    assert PROFILE in current.unsafe
    assert journal.append(PROFILE, build_event(execution_id="fork-root-refusal")) is False
    assert _private_files(first) == original
    _select(monkeypatch, second)
    assert mcp_server.search_sessions("isolated-fork-root-latch") == response
    assert len(_completed()) == 1
    _select(monkeypatch, first)
    assert mcp_server.search_sessions("isolated-fork-root-latch") == response
    assert _private_files(first) == original


def test_codec_rejects_unknown_phase_without_admitting_a_record() -> None:
    import json

    document = json.loads(journal_codec.serialize_event(build_event()))
    document["phase"] = "future-phase"

    with pytest.raises(journal_codec.MalformedEvent, match="unknown enum"):
        journal_codec.parse_event(json.dumps(document).encode())


def test_partial_segment_write_is_separated_from_the_next_durable_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal.append(PROFILE, build_event(execution_id="before-1"))
    real_write = os.write
    calls = 0

    def partial_then_fail(fd: int, data: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[: max(1, len(data) // 2)])
        if calls == 2:
            raise OSError(errno.ENOSPC, "simulated full journal")
        return real_write(fd, data)

    monkeypatch.setattr(journal.os, "write", partial_then_fail)
    with pytest.raises(OSError) as failure:
        journal.append(PROFILE, build_event(execution_id="partial-1"))
    assert failure.value.errno == errno.ENOSPC

    journal.append(PROFILE, build_event(execution_id="after-1"))

    events = _events()
    assert {row.execution_id for row in events} == {"before-1", "after-1"}
    assert all(
        path.read_bytes().endswith(b"\n")
        for path in (datadir.data_root() / "accounting" / PROFILE / "journal").glob("seg-*.jsonl")
    )


def test_codec_trims_only_the_reference_suffix_to_preserve_event_identity() -> None:
    refs = tuple(f"runtime:{index}:{'r' * 32}" for index in range(2500))
    scores = tuple((index, 1000) for index in range(len(refs)))

    encoded = journal_codec.serialize_event(build_event(refs=refs, ranks_scores=scores))
    restored = journal_codec.parse_event(encoded.rstrip(b"\n"))

    assert len(encoded) <= MAX_EVENT_BYTES
    assert restored.refs_omitted is True
    assert refs[: len(restored.refs)] == restored.refs
    assert len(restored.refs) < len(refs)


def test_codec_rejects_identity_that_cannot_fit_after_reference_removal() -> None:
    from dataclasses import replace

    event = replace(
        build_event(refs=(), ranks_scores=()),
        fingerprint="f" * MAX_EVENT_BYTES,
    )

    with pytest.raises(journal_codec.EventTooLarge):
        journal_codec.serialize_event(event)


def test_fingerprint_create_false_does_not_create_an_absent_profile_directory(
    tmp_path: Path,
) -> None:
    root_fd = datadir.open_dir()
    os.close(root_fd)
    root = datadir.snapshot()
    assert root is not None

    assert fingerprint.load_key("other-profile", create=False, root=root) is None
    assert not (tmp_path / "data" / "accounting" / "other-profile").exists()

    created = fingerprint.load_key("other-profile", create=True, root=root)
    assert created is not None and len(created) == fingerprint.KEY_BYTES
    assert fingerprint.load_key("other-profile", create=False, root=root) == created


def test_key_whose_metadata_changes_during_read_is_returned_but_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_fd = datadir.open_dir()
    os.close(root_fd)
    root = datadir.snapshot()
    assert root is not None
    created = fingerprint.load_key(PROFILE, create=True, root=root)
    key_path = root.path / "accounting" / PROFILE / fingerprint.KEY_FILE
    fingerprint.forget_keys()
    real_read = fingerprint.storage.read_descriptor

    def read_then_touch(descriptor: int, **kwargs: int) -> bytes:
        data = real_read(descriptor, **kwargs)
        os.utime(key_path, ns=(1_000_000_000, 1_000_000_000))  # concurrent metadata change
        return data

    with monkeypatch.context() as patch:
        patch.setattr(fingerprint.storage, "read_descriptor", read_then_touch)
        loaded = fingerprint.load_key(PROFILE, create=False, root=root)
    assert loaded == created
    assert not fingerprint._cache and not fingerprint._signatures

    assert fingerprint.load_key(PROFILE, create=False, root=root) == created
    assert fingerprint._cache == {(root, PROFILE): created}
    assert set(fingerprint._signatures) == {(root, PROFILE)}


def test_codec_rejects_non_object_rendered_quantity_and_accepts_valid_event() -> None:
    import json

    event = build_event(rendered_bytes=Quantity.observed(812, "bytes", "mcp_json"))
    encoded = journal_codec.serialize_event(event)
    assert journal_codec.parse_event(encoded.rstrip(b"\n")) == event

    document = json.loads(encoded)
    document["rendered_bytes"] = []
    with pytest.raises(journal_codec.MalformedEvent, match="rendered_bytes is not an object"):
        journal_codec.parse_event(json.dumps(document).encode())


def test_minimal_started_strips_evidence_and_rejects_other_phases() -> None:
    from dataclasses import replace

    started = build_event(
        phase=Phase.STARTED,
        outcome=None,
        cwd="/private/work",
        refs=("private:reference",),
        correlation_id="private-correlation",
    )
    minimal = journal_codec.minimal_started(started)
    assert minimal.phase is Phase.STARTED
    assert minimal.cwd is None and minimal.refs == ()
    assert minimal.fingerprint == "" and minimal.correlation_id is None
    assert minimal.refs_omitted is True

    completed = replace(started, phase=Phase.COMPLETED, outcome=Outcome.OK)
    with pytest.raises(ValueError, match="only a started phase"):
        journal_codec.minimal_started(completed)


def test_fingerprint_creation_fails_if_private_root_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(datadir, "snapshot", lambda *args, **kwargs: None)

    with pytest.raises(datadir.RootChanged):
        fingerprint.load_key(PROFILE, create=True)


def test_fingerprint_does_not_reuse_cached_key_after_descriptor_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fingerprint.load_key(PROFILE, create=True)
    assert original is not None
    real_signature = fingerprint._signature
    calls = 0

    def changed_signature(descriptor: int) -> tuple[int, ...]:
        nonlocal calls
        signature = real_signature(descriptor)
        calls += 1
        return signature if calls == 1 else (*signature, calls)

    with monkeypatch.context() as patch:
        patch.setattr(fingerprint, "_signature", changed_signature)
        assert fingerprint.load_key(PROFILE, create=False) == original

    replacement = b"r" * fingerprint.KEY_BYTES
    (datadir.data_root() / "accounting" / PROFILE / fingerprint.KEY_FILE).write_bytes(replacement)
    assert fingerprint.load_key(PROFILE, create=False) == replacement


def test_policy_changed_during_parse_is_reloaded_before_later_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    policy.save_profile(ProfilePolicy(profile_id=PROFILE, collection_enabled=True))
    path = root / "accounting" / "policy.json"
    original_load = capture_policy._load
    changed = False

    def change_after_parse(descriptor: int):
        nonlocal changed
        parsed = original_load(descriptor)
        if not changed:
            changed = True
            path.write_bytes(b"{")
        return parsed

    monkeypatch.setattr(capture_policy, "_load", change_after_parse)
    assert capture_policy.capture_profile(None) == PROFILE
    assert capture_policy.capture_profile(None) is None
    assert path.read_bytes() == b"{"


@pytest.mark.parametrize("timestamp", ["2026-13-01T07:00:00Z", "2026-11-31T07:00:00Z"])
def test_codec_rejects_impossible_calendar_timestamp_that_matches_canonical_shape(
    timestamp: str,
) -> None:
    import json

    encoded = journal_codec.serialize_event(build_event())
    document = json.loads(encoded)
    assert journal_codec.parse_event(json.dumps(document).encode()).ts_utc == document["ts_utc"]

    document["ts_utc"] = timestamp
    with pytest.raises(journal_codec.MalformedEvent, match="invalid UTC timestamp"):
        journal_codec.parse_event(json.dumps(document).encode())
