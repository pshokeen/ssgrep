"""Scoped erasure requires explicit fresh authorized corpus, and remains denied later."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from ssgrep.accounting import fingerprint, storage
from ssgrep.accounting.frequency_authority import erased_view
from tests.accounting.test_frequency_generations import PROFILE, MetadataRepository, database


def test_explicit_reset_requires_fresh_generation_and_persists_erased_root_denial(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    marker = {
        "schema": 1,
        "scoped": True,
        "erased_workspaces": ["/erased"],
        "erased_generation": "generation-1",
        "authorized_scope_epoch": None,
        "authorized_generation": None,
    }
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        storage.atomic_write(directory, "frequency-purged.json", json.dumps(marker).encode())
    finally:
        os.close(directory)

    def resolve(reset=False, cached=None):
        return erased_view(
            PROFILE,
            scope_epoch=2,
            reset_authority=reset,
            repin=True,
            repository=repository,
            authority=lambda cwd: True,
            authority_id="approved-v1",
            cached=cached,
        )

    assert resolve().unavailable_reason == "frequency_scope_erased"
    assert resolve(True).unavailable_reason == "frequency_authority_reset_requires_new_generation"
    writer = sqlite3.connect(path)
    writer.execute("UPDATE metadata SET value='generation-2' WHERE key='last_index_time'")
    writer.commit()
    approved = resolve(True)
    assert approved.known
    updated = json.loads(
        (
            storage.data_root() / "accounting" / PROFILE / "collector" / "frequency-purged.json"
        ).read_text()
    )
    assert updated["authorized_scope_epoch"] == 2
    assert updated["authorized_generation"] == "generation-2"
    assert updated["erased_workspaces"] == ["/erased"]
    assert resolve(False, approved) == approved
    writer.execute("UPDATE chunks SET cwd='/erased' WHERE chunk_id='0'")
    writer.execute("UPDATE metadata SET value='generation-3' WHERE key='last_index_time'")
    writer.commit()
    rejected = resolve(False, approved)
    assert not rejected.known and rejected.unavailable_reason == "frequency_scope_unapproved"
    writer.close()
    repository.close()


def test_pending_erasure_and_unknown_legacy_scope_never_authorize_reset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        for marker in [
            {"schema": 1, "pending_purge": {"profile": PROFILE}},
            {"schema": 1, "scoped": True},
        ]:
            storage.atomic_write(directory, "frequency-purged.json", json.dumps(marker).encode())
            result = erased_view(PROFILE, scope_epoch=2, reset_authority=True, repin=True)
            assert result is not None and not result.known
            assert result.unavailable_reason in (
                "frequency_scope_erased",
                "frequency_erased_scope_unknown",
            )
    finally:
        os.close(directory)
