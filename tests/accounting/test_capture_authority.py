"""Pinned private reads and warm caches retain filesystem authority checks."""

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
    plainfile,
    policy,
    storage,
)
from ssgrep.accounting.contracts import Phase
from ssgrep.services import mcp_server

PROFILE = "personal-local"


@pytest.fixture(autouse=True)
def _reset_capture():
    journal.close_all()
    fingerprint.forget_keys()
    capture._unsafe.clear()
    capture_policy._cache = ("", None, capture_policy._DISABLED)
    yield
    journal.close_all()
    fingerprint.forget_keys()
    capture._unsafe.clear()
    capture_policy._cache = ("", None, capture_policy._DISABLED)


def _completed_ids() -> set[str]:
    return {
        event.execution_id
        for event in journal_reader.read_journal(PROFILE, quarantine=False).events
        if event.phase is Phase.COMPLETED
    }


@pytest.mark.parametrize("mutation", ["mode", "owner", "symlink"])
def test_cached_policy_parent_revocation_stops_real_lookup_capture_and_recovers(
    mutation: str,
) -> None:
    if mutation == "owner" and os.geteuid() != 0:
        pytest.skip("Changing a directory owner requires privilege")
    policy.enable_collection(PROFILE)
    response = mcp_server.search_sessions("isolated-authority-control")
    before = _completed_ids()
    assert len(before) == 1
    directory = datadir.data_root() / "accounting"
    leaf_signature = (directory / "policy.json").stat()
    moved = directory.with_name("accounting-pinned")
    try:
        if mutation == "mode":
            directory.chmod(0o750)
        elif mutation == "owner":
            os.chown(directory, 1, -1)
        else:
            directory.rename(moved)
            directory.symlink_to(moved, target_is_directory=True)
        assert (directory / "policy.json").stat() == leaf_signature
        assert capture_policy.capture_profile(None) is None
        assert mcp_server.search_sessions("isolated-authority-control") == response
    finally:
        if mutation == "mode":
            directory.chmod(0o700)
        elif mutation == "owner":
            os.chown(directory, os.geteuid(), -1)
        else:
            directory.unlink()
            moved.rename(directory)
    assert _completed_ids() == before
    assert capture_policy.capture_profile(None) == PROFILE
    assert mcp_server.search_sessions("isolated-authority-control") == response
    assert len(_completed_ids() - before) == 1


@pytest.mark.parametrize("parent", ["accounting", "profile"])
@pytest.mark.parametrize("mutation", ["mode", "owner", "symlink"])
def test_cached_key_revalidates_each_parent_without_repairing_it(
    parent: str,
    mutation: str,
) -> None:
    if mutation == "owner" and os.geteuid() != 0:
        pytest.skip("Changing a directory owner requires privilege")
    policy.enable_collection(PROFILE)
    response = mcp_server.search_sessions("isolated-key-authority")
    before = _completed_ids()
    assert len(before) == 1
    key = fingerprint.load_key(PROFILE, create=False)
    assert key is not None
    accounting = datadir.data_root() / "accounting"
    directory = accounting if parent == "accounting" else accounting / PROFILE
    leaf = accounting / PROFILE / fingerprint.KEY_FILE
    leaf_signature = leaf.stat()
    moved = directory.with_name(directory.name + "-pinned")
    try:
        if mutation == "mode":
            directory.chmod(0o750)
        elif mutation == "owner":
            os.chown(directory, 1, -1)
        else:
            directory.rename(moved)
            directory.symlink_to(moved, target_is_directory=True)
        assert leaf.stat() == leaf_signature
        for create in (False, True):
            with pytest.raises(storage.UnsafeStorageError):
                fingerprint.load_key(PROFILE, create=create)
        assert mcp_server.search_sessions("isolated-key-authority") == response
        if mutation == "mode":
            assert directory.stat().st_mode & 0o077 == 0o050
        elif mutation == "owner":
            assert directory.stat().st_uid == 1
        else:
            assert directory.is_symlink()
    finally:
        if mutation == "mode":
            directory.chmod(0o700)
        elif mutation == "owner":
            os.chown(directory, os.geteuid(), -1)
        else:
            directory.unlink()
            moved.rename(directory)
    assert _completed_ids() == before
    assert fingerprint.load_key(PROFILE, create=False) == key
    # Unsafe profile storage intentionally latches capture off until a restart.
    capture._unsafe.clear()
    assert mcp_server.search_sessions("isolated-key-authority") == response
    assert len(_completed_ids() - before) == 1


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_private_read_pins_parent_before_namespace_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    base = tmp_path / "private-root"
    base.mkdir()
    original = base / "accounting"
    original.mkdir(mode=0o700)
    (original / "policy.json").write_bytes(b"original-policy")
    (original / "policy.json").chmod(0o600)
    moved = base / "pinned-original"
    destination = tmp_path / "replacement"
    destination.mkdir(mode=0o700)
    (destination / "policy.json").write_bytes(b"replacement-policy")
    (destination / "policy.json").chmod(0o600)
    real_open = os.open
    real_check = plainfile._check_dir
    swapped = False

    def swap() -> None:
        nonlocal swapped
        if swapped:
            return
        swapped = True
        original.rename(moved)
        if replacement == "symlink":
            original.symlink_to(destination, target_is_directory=True)
        else:
            destination.rename(original)

    def checked(path: Path, *, private: bool = False) -> bool:
        valid = real_check(path, private=private)
        if path == original:
            swap()
        return valid

    def opened(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if os.fspath(path) == "accounting" and kwargs.get("dir_fd") is not None:
            swap()
        return descriptor

    # Old pathname validation and new descriptor validation reach the same race.
    with monkeypatch.context() as patch:
        patch.setattr(plainfile, "_check_dir", checked)
        patch.setattr(plainfile.os, "open", opened)
        try:
            result = plainfile.read_private(base, "accounting", "policy.json", max_bytes=64)
        except plainfile.UnsafePathError:
            result = None
    assert swapped
    assert result in (None, b"original-policy")
    assert (moved / "policy.json").read_bytes() == b"original-policy"
    if replacement == "symlink":
        original.unlink()
    else:
        original.rename(destination)
    moved.rename(original)
    assert (
        plainfile.read_private(base, "accounting", "policy.json", max_bytes=64)
        == b"original-policy"
    )


def test_policy_parse_uses_the_validated_pinned_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy.enable_collection(PROFILE)
    original = datadir.data_root() / "accounting"
    moved = original.with_name("accounting-pinned")
    replacement = original.with_name("accounting-replacement")
    replacement.mkdir(mode=0o700)
    data = (original / "policy.json").read_bytes()
    # A replacement policy selects another enabled profile, making misreads observable.
    (replacement / "policy.json").write_bytes(
        data.replace(b"personal-local", b"replacement-profile")
    )
    (replacement / "policy.json").chmod(0o600)
    real_open = os.open
    swapped = False

    def opened(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if not swapped and os.fspath(path) == "accounting" and kwargs.get("dir_fd") is not None:
            swapped = True
            original.rename(moved)
            original.symlink_to(replacement, target_is_directory=True)
        return descriptor

    try:
        with monkeypatch.context() as patch:
            patch.setattr(plainfile.os, "open", opened)
            chosen = capture_policy.capture_profile(None, profile=PROFILE)
        assert swapped
        assert chosen in (None, PROFILE)
        assert capture_policy.capture_profile(None, profile="replacement-profile") is None
    finally:
        if swapped:
            original.unlink()
            moved.rename(original)
    assert capture_policy.capture_profile(None, profile=PROFILE) == PROFILE


def test_private_read_preserves_configured_root_symlink_and_never_creates(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-root"
    assert plainfile.read_private(missing, "accounting", "policy.json", max_bytes=20) is None
    assert not missing.exists()
    actual = tmp_path / "actual-root"
    plainfile.write_private(actual, "accounting", "policy.json", data=b"operator-root")
    configured = tmp_path / "configured-root"
    configured.symlink_to(actual, target_is_directory=True)
    assert (
        plainfile.read_private(configured, "accounting", "policy.json", max_bytes=20)
        == b"operator-root"
    )
