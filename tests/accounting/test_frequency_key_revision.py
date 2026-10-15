"""Key replacement is an input revision, never automatic rerating of old lookups."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ssgrep.accounting import fingerprint, storage
from ssgrep.accounting.frequency_assignment import FrequencyAssignments
from ssgrep.accounting.index_frequency import read_index_view
from tests.accounting.test_frequency_generations import PROFILE, RARE, MetadataRepository, database


def test_warmed_key_rotation_preserves_old_lookup_assignment_and_rates_new_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    first_key = fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)

    def current(cached=None):
        return read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: True,
            authority_id="approved-v1",
            cached=cached,
        )

    old = current()
    original = FrequencyAssignments(old, fingerprint_key_id=fingerprint.key_id(first_key))
    original.bind("old-native-lookup", [])
    assignment = original.records()[0]
    directory = storage.open_profile_dir(PROFILE)
    try:
        storage.atomic_write(directory, fingerprint.KEY_FILE, b"r" * fingerprint.KEY_BYTES)
    finally:
        os.close(directory)
    rotated_key = fingerprint.load_key(PROFILE, create=False)
    assert rotated_key != first_key
    refreshed = current(old)
    assert refreshed.version_id != old.version_id
    resumed = FrequencyAssignments(
        refreshed,
        assignments=original.records(),
        loader=lambda _: old,
        fingerprint_key_id=fingerprint.key_id(rotated_key),
    )
    unavailable = resumed.view_for("old-native-lookup")
    assert unavailable.unavailable_reason == "fingerprint_key_revision_unavailable"
    assert resumed.unavailable["old-native-lookup"] == "fingerprint_key_revision_unavailable"
    assert resumed.records()[0] == assignment
    resumed.bind("new-native-lookup", [])
    assert resumed.view_for("new-native-lookup").rarity_milli(RARE) == 8000
    repository.close()
