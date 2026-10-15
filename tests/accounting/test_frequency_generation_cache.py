"""Manifest-version changes invalidate cached rarity even without timestamp changes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ssgrep.accounting import fingerprint
from ssgrep.accounting.index_frequency import read_index_view
from tests.accounting.test_frequency_generations import PROFILE, RARE, MetadataRepository, database


class VersionedRepository(MetadataRepository):
    """SQLite's real data-version mirrors the repository table-version read interface."""

    def table(self, name: str):
        assert name in ("metadata", "chunks")
        return self

    @property
    def version(self) -> int:
        return self.connection.execute("PRAGMA data_version").fetchone()[0]


def test_damaged_stats_cannot_hide_behind_unchanged_last_index_time(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = VersionedRepository(path)

    def current(cached=None):
        return read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: True,
            authority_id="approved-v1",
            cached=cached,
        )

    first = current()
    assert first.rarity_milli(RARE) == 8000
    assert "@metadata:" in first.generation and "chunks:" in first.generation
    writer = sqlite3.connect(path)
    writer.execute("UPDATE metadata SET value='{' WHERE key='lexical_stats'")
    writer.commit()
    writer.close()
    damaged = current(first)
    assert not damaged.known and damaged.unavailable_reason == "frequency_stats_damaged"
    assert damaged.rarity_milli(RARE) is None
    assert first.rarity_milli(RARE) == 8000  # frozen prior input remains intact
    repository.close()
