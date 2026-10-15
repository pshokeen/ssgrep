"""Resumable frozen-view reclamation protects actual replay references and commit order."""

import os
from datetime import UTC, datetime

from ssgrep.accounting import collector_store as store, storage
from ssgrep.accounting.collector_frequency import pin, view_file_name
from ssgrep.accounting.collector_frequency_gc import finish, stage
from ssgrep.accounting.frequency import build_view
from tests.accounting.reduce_builders import PROFILE

NOW = datetime(2026, 10, 12, tzinfo=UTC)


def view(n):
    return build_view([f"E_GC_{n}_{i}" for i in range(12)])


def old_views(count=34):
    values = [view(n) for n in range(count)]
    for value in values:
        pin(PROFILE, value)
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        for value in values:
            os.utime(view_file_name(value.version_id), ns=(1, 1), dir_fd=directory)
    finally:
        os.close(directory)
    return values


def commit_gc(checkpoint, current_refs):
    blobs = dict(checkpoint.blobs)
    stage(
        PROFILE,
        checkpoint,
        blobs,
        current_refs=current_refs,
        before_day="2026-09-12",
        today="2026-10-12",
    )
    checkpoint.blobs = blobs
    store.commit_checkpoint(PROFILE, checkpoint)


def test_protected_first_batch_does_not_starve_later_orphan_after_resume():
    values = old_views()
    # Scheduling uses deterministic filenames, not generation construction order.
    ordered = sorted(values, key=lambda value: view_file_name(value.version_id))
    keep = {value.version_id for value in ordered[:32]} | {values[-1].version_id}
    checkpoint = store.Checkpoint(PROFILE, generation=1)
    commit_gc(checkpoint, keep)
    result = finish(PROFILE, generation=1, limit=32)
    assert result.pending
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    checkpoint.generation += 1
    commit_gc(checkpoint, keep)
    result = finish(PROFILE, generation=checkpoint.generation, limit=32)
    assert not result.pending
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        names = set(storage.list_names(directory))
    finally:
        os.close(directory)
    assert all(view_file_name(value.version_id) in names for value in ordered[:32])
    assert any(
        view_file_name(value.version_id) not in names
        for value in ordered[32:]
        if value.version_id not in keep
    )


def test_uncommitted_job_cannot_erase_prior_checkpoint_replay_input():
    values = old_views(2)
    checkpoint = store.Checkpoint(PROFILE, generation=1)
    commit_gc(checkpoint, {values[0].version_id})
    staged = dict(checkpoint.blobs)
    stage(
        PROFILE, checkpoint, staged, current_refs=set(), before_day="2026-09-12", today="2026-10-13"
    )
    finish(PROFILE, generation=2)  # No accepted generation 2 exists.
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        assert view_file_name(values[0].version_id) in storage.list_names(directory)
    finally:
        os.close(directory)


def test_legacy_reference_without_index_blocks_reclamation_instead_of_guessing():
    values = old_views(2)
    legacy = store.write_blob(PROFILE, "ledger-legacy", {"state": {}, "archives": {}})
    checkpoint = store.Checkpoint(PROFILE, generation=1, blobs={"ledger-legacy": legacy})
    commit_gc(checkpoint, set())
    outcome = finish(PROFILE, generation=1)
    assert outcome.pending and "frequency_references_unknown" in outcome.warnings
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        assert all(
            view_file_name(value.version_id) in storage.list_names(directory) for value in values
        )
    finally:
        os.close(directory)
