"""Tests for private application storage paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.store import paths


def test_data_dir_honors_expanded_absolute_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SSGREP_DATA_DIR", "~/relative/../private")

    assert paths.data_dir() == (home / "relative" / ".." / "private").absolute()
    assert paths.database_dir() == paths.data_dir() / "lancedb"


def test_data_dir_uses_platform_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = tmp_path / "platform-data"
    monkeypatch.delenv("SSGREP_DATA_DIR", raising=False)
    monkeypatch.setattr(paths, "user_data_path", lambda app, appauthor: expected)

    assert paths.data_dir() == expected


def test_ensure_data_dir_creates_and_restricts_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "parent" / "private"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(target))

    assert paths.ensure_data_dir() == target
    assert target.is_dir()
    assert target.stat().st_mode & 0o777 == 0o700


def test_ensure_data_dir_tolerates_chmod_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "private"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(target))
    original_chmod = Path.chmod

    def fail_only_for_target(self: Path, mode: int, *, follow_symlinks: bool = True) -> None:
        if self == target:
            raise OSError("read-only filesystem")
        original_chmod(self, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "chmod", fail_only_for_target)
    assert paths.ensure_data_dir() == target
    assert target.is_dir()
