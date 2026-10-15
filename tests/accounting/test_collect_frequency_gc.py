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


def test_new_receipt_inventory_requires_new_acceptance_before_any_view_is_pruned():
    from ssgrep.accounting import collector_publish
    from tests.accounting.test_collect_publish import ledger

    values = old_views(2)
    checkpoint = store.Checkpoint(PROFILE, generation=1)
    commit_gc(checkpoint, set())
    rows = ledger().rows()
    assert rows.receipts
    collector_publish.write_receipts(PROFILE, rows)
    directory = storage.data_root() / "accounting" / PROFILE
    receipts = {path.name: path.read_bytes() for path in (directory / "receipts").glob("*.json")}
    accepted = (directory / "collector" / store.CHECKPOINT).read_bytes()
    outcome = finish(PROFILE, generation=1)
    assert outcome.pending and outcome.removed == 0
    assert outcome.warnings == ("frequency_reference_inventory_changed",)
    assert (directory / "collector" / store.CHECKPOINT).read_bytes() == accepted
    assert all(
        (directory / "collector" / view_file_name(value.version_id)).exists() for value in values
    )
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    checkpoint.generation += 1
    commit_gc(checkpoint, set())
    outcome = finish(PROFILE, generation=2)
    assert not outcome.pending and outcome.removed == len(values) - 1
    assert {
        value.version_id
        for value in values
        if (directory / "collector" / view_file_name(value.version_id)).exists()
    } == {values[-1].version_id}
    assert {
        path.name: path.read_bytes() for path in (directory / "receipts").glob("*.json")
    } == receipts


def test_unknown_receipt_body_blocks_erasure_then_restored_receipt_allows_recovery():
    from ssgrep.accounting import collector_publish
    from tests.accounting.test_collect_publish import ledger

    values = old_views(2)
    rows = ledger().rows()
    assert rows.receipts
    collector_publish.write_receipts(PROFILE, rows)
    directory = storage.data_root() / "accounting" / PROFILE
    receipt = directory / "receipts" / (rows.receipts[0].receipt_id + ".json")
    original = receipt.read_bytes()
    receipt.write_bytes(b"null")
    checkpoint = store.Checkpoint(PROFILE, generation=1)
    commit_gc(checkpoint, set())
    outcome = finish(PROFILE, generation=1)
    assert outcome.pending and outcome.removed == 0
    assert outcome.warnings == ("frequency_references_unknown",)
    again = finish(PROFILE, generation=1)
    assert again.pending and again.removed == 0 and again.warnings == outcome.warnings
    assert all(
        (directory / "collector" / view_file_name(value.version_id)).exists() for value in values
    )
    receipt.write_bytes(original)
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    checkpoint.generation += 1
    commit_gc(checkpoint, set())
    outcome = finish(PROFILE, generation=2)
    assert not outcome.pending and outcome.removed == len(values) - 1
    assert {
        value.version_id
        for value in values
        if (directory / "collector" / view_file_name(value.version_id)).exists()
    } == {values[-1].version_id}
    assert receipt.read_bytes() == original


def test_deleted_receipt_is_not_replayed_but_its_inventory_change_still_requires_acceptance():
    from ssgrep.accounting import collector_publish
    from tests.accounting.test_collect_publish import ledger

    values = old_views(2)
    rows = ledger().rows()
    assert rows.receipts
    collector_publish.write_receipts(PROFILE, rows)
    directory = storage.data_root() / "accounting" / PROFILE
    checkpoint = store.Checkpoint(PROFILE, generation=1)
    commit_gc(checkpoint, set())
    for record in rows.receipts:
        (directory / "receipts" / (record.receipt_id + ".json")).unlink()
    outcome = finish(PROFILE, generation=1)
    assert outcome.pending and outcome.removed == 0
    assert outcome.warnings == ("frequency_reference_inventory_changed",)
    assert all(
        (directory / "collector" / view_file_name(value.version_id)).exists() for value in values
    )
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    checkpoint.generation += 1
    commit_gc(checkpoint, set())
    outcome = finish(PROFILE, generation=2)
    assert not outcome.pending and outcome.removed == len(values) - 1
    assert {
        value.version_id
        for value in values
        if (directory / "collector" / view_file_name(value.version_id)).exists()
    } == {values[-1].version_id}
    assert not list((directory / "receipts").glob("*.json"))
