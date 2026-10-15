"""Expired rarity inputs remain while exact canonical replay still references them."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ssgrep.accounting import collector_frequency, fingerprint, storage
from ssgrep.accounting.frequency_retention import prune_views
from ssgrep.accounting.index_frequency import read_index_view
from tests.accounting.test_frequency_generations import PROFILE, MetadataRepository, database


def test_bounded_gc_protects_lookup_archive_references_and_current_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    views = []
    for generation in ("generation-1", "generation-2", "generation-3"):
        writer = sqlite3.connect(path)
        writer.execute("UPDATE metadata SET value=? WHERE key='last_index_time'", (generation,))
        writer.commit()
        writer.close()
        view = read_index_view(
            PROFILE, repository=repository, authority=lambda cwd: True, authority_id="approved-v1"
        )
        collector_frequency.pin(PROFILE, view)
        views.append(view)
    directory = storage.data_root() / "accounting" / PROFILE / "collector"
    names = [collector_frequency.view_file_name(view.version_id) for view in views]
    age = datetime(2025, 1, 1, tzinfo=UTC).timestamp()
    for name in names:
        os.utime(directory / name, (age, age))
    result = prune_views(
        PROFILE,
        candidates=iter(names),
        referenced=[views[0].version_id],
        before_day="2026-01-01",
        limit=2,
    )
    assert (result.scanned, result.removed, result.retained) == (2, 1, 1)
    assert (directory / names[0]).is_file()  # live assignment or inactive canonical archive
    assert not (directory / names[1]).exists()  # expired, orphan, not current
    protected = prune_views(PROFILE, candidates=[names[2]], referenced=[], before_day="2026-01-01")
    assert protected.removed == 0 and protected.retained == 1
    assert collector_frequency.load_view(PROFILE, views[0].version_id) == views[0]
    assert collector_frequency.load_pinned(PROFILE) == views[2]
    repository.close()
