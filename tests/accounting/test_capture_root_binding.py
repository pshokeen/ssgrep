"""Capture authorization, key creation and every phase belong to one root inode."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ssgrep.accounting import (
    capture,
    capture_policy,
    datadir,
    fingerprint,
    journal,
    journal_reader,
    policy,
    storage,
)
from ssgrep.accounting.contracts import Operation, Outcome, Phase
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.services import mcp_server
from tests.accounting.conftest import build_event

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
