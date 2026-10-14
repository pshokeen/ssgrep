"""Behavior of private accounting storage (task 3.1)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ssgrep.accounting import storage


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
