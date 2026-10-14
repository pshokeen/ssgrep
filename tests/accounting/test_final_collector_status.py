"""Status distinguishes unsafe, unreadable, and concurrently removed storage from zero usage."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import status_worker
from tests.accounting import verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.test_guard_collector_recovery import seeded


def test_status_refuses_a_symlinked_profile_without_counting_its_target(tmp_path: Path) -> None:
    seeded(tmp_path)
    measured = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert measured.total_bytes is not None and measured.total_bytes > 0
    directory = vs.accounting_dir()
    saved = tmp_path / "saved-profile"
    directory.rename(saved)
    directory.symlink_to(saved, target_is_directory=True)
    try:
        unknown = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
        assert unknown.total_bytes is None and unknown.reason == "unsafe_path"
        assert all(
            part.bytes is None and part.reason == "unsafe_path" for part in unknown.components
        )
    finally:
        directory.unlink()
        saved.rename(directory)
    assert (
        status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12).total_bytes
        == measured.total_bytes
    )


def test_unreadable_profile_metadata_is_not_reported_as_an_empty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    directory = vs.accounting_dir()
    original = os.lstat

    def denied(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if Path(path) == directory:
            raise PermissionError("profile metadata denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(status_worker.os, "lstat", denied)
        unknown = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
        assert unknown.total_bytes is None and unknown.reason == "unreadable"
        assert all(part.bytes is None for part in unknown.components)
    measured = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert measured.total_bytes is not None and measured.total_bytes > 0


def test_file_removed_between_directory_listing_and_stat_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    victim = vs.accounting_dir() / "history" / "vanishing.json"
    victim.write_bytes(b"123456789")
    victim.chmod(0o600)
    baseline = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert baseline.total_bytes is not None
    original = os.scandir

    class VanishingEntry:
        def __init__(self, entry: os.DirEntry[str]) -> None:
            self.entry = entry

        def __getattr__(self, name: str) -> Any:
            return getattr(self.entry, name)

        def stat(self, **kwargs: Any) -> os.stat_result:
            victim.unlink()
            return self.entry.stat(**kwargs)

    @contextmanager
    def racing_scan(path: Any):
        with original(path) as entries:
            yield (
                VanishingEntry(entry) if Path(entry.path) == victim else entry for entry in entries
            )

    monkeypatch.setattr(status_worker.os, "scandir", racing_scan)
    observed = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert not victim.exists()
    assert observed.reason is None
    assert observed.total_bytes == baseline.total_bytes - 9
