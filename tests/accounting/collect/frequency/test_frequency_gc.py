# ---- Survivor source: test_frequency_gc.py ----
"""Resumable frozen-view reclamation protects actual replay references and commit order."""

import os
from datetime import UTC, datetime

import pytest

from ssgrep.accounting.collect import frequency_index, store as store
from ssgrep.accounting.collect.frequency import build_view, pin, prune_views, view_file_name
from ssgrep.accounting.collect.frequency_index import finish as frequency_gc_finish, stage
from ssgrep.accounting.privacy import storage
from tests.accounting.ledger.reduce_builders import PROFILE

NOW = datetime(2026, 10, 12, tzinfo=UTC)


def gc_view(n):
    return build_view([f"E_GC_{n}_{i}" for i in range(12)])


def old_views(count=34):
    values = [gc_view(n) for n in range(count)]
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


def test_view_pruning_only_removes_old_unreferenced_valid_views():
    values = old_views(4)
    referenced, old, recent, current = values
    recent_name = view_file_name(recent.version_id)
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        os.utime(recent_name, None, dir_fd=directory)
    finally:
        os.close(directory)

    old_name = view_file_name(old.version_id)
    missing_name = view_file_name(gc_view(99).version_id)
    result = prune_views(
        PROFILE,
        candidates=[
            "not-a-view.json",
            view_file_name(referenced.version_id),
            old_name,
            recent_name,
            view_file_name(current.version_id),
            missing_name,
        ],
        referenced=[referenced.version_id],
        before_day="2026-09-12",
    )

    assert (result.scanned, result.removed, result.retained) == (6, 1, 4)
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        names = set(storage.list_names(directory))
        assert old_name not in names
        assert view_file_name(referenced.version_id) in names
        assert recent_name in names and view_file_name(current.version_id) in names
    finally:
        os.close(directory)


def test_protected_first_batch_does_not_starve_later_orphan_after_resume():
    values = old_views()
    # Scheduling uses deterministic filenames, not generation construction order.
    ordered = sorted(values, key=lambda value: view_file_name(value.version_id))
    keep = {value.version_id for value in ordered[:32]} | {values[-1].version_id}
    checkpoint = store.Checkpoint(PROFILE, generation=1)
    commit_gc(checkpoint, keep)
    result = frequency_gc_finish(PROFILE, generation=1, limit=32)
    assert result.pending
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    checkpoint.generation += 1
    commit_gc(checkpoint, keep)
    result = frequency_gc_finish(PROFILE, generation=checkpoint.generation, limit=32)
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
    frequency_gc_finish(PROFILE, generation=2)  # No accepted generation 2 exists.
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
    outcome = frequency_gc_finish(PROFILE, generation=1)
    assert outcome.pending and "frequency_references_unknown" in outcome.warnings
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        assert all(
            view_file_name(value.version_id) in storage.list_names(directory) for value in values
        )
    finally:
        os.close(directory)


def test_new_receipt_inventory_requires_new_acceptance_before_any_view_is_pruned():
    from ssgrep.accounting.collect import cycle_publish as collector_publish
    from tests.accounting.collect.test_publication import ledger

    values = old_views(2)
    checkpoint = store.Checkpoint(PROFILE, generation=1)
    commit_gc(checkpoint, set())
    rows = ledger().rows()
    assert rows.receipts
    collector_publish.write_receipts(PROFILE, rows)
    directory = storage.data_root() / "accounting" / PROFILE
    receipts = {path.name: path.read_bytes() for path in (directory / "receipts").glob("*.json")}
    accepted = (directory / "collector" / store.CHECKPOINT).read_bytes()
    outcome = frequency_gc_finish(PROFILE, generation=1)
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
    outcome = frequency_gc_finish(PROFILE, generation=2)
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
    from ssgrep.accounting.collect import cycle_publish as collector_publish
    from tests.accounting.collect.test_publication import ledger

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
    outcome = frequency_gc_finish(PROFILE, generation=1)
    assert outcome.pending and outcome.removed == 0
    assert outcome.warnings == ("frequency_references_unknown",)
    again = frequency_gc_finish(PROFILE, generation=1)
    assert again.pending and again.removed == 0 and again.warnings == outcome.warnings
    assert all(
        (directory / "collector" / view_file_name(value.version_id)).exists() for value in values
    )
    receipt.write_bytes(original)
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    checkpoint.generation += 1
    commit_gc(checkpoint, set())
    outcome = frequency_gc_finish(PROFILE, generation=2)
    assert not outcome.pending and outcome.removed == len(values) - 1
    assert {
        value.version_id
        for value in values
        if (directory / "collector" / view_file_name(value.version_id)).exists()
    } == {values[-1].version_id}
    assert receipt.read_bytes() == original


def test_deleted_receipt_is_not_replayed_but_its_inventory_change_still_requires_acceptance():
    from ssgrep.accounting.collect import cycle_publish as collector_publish
    from tests.accounting.collect.test_publication import ledger

    values = old_views(2)
    rows = ledger().rows()
    assert rows.receipts
    collector_publish.write_receipts(PROFILE, rows)
    directory = storage.data_root() / "accounting" / PROFILE
    checkpoint = store.Checkpoint(PROFILE, generation=1)
    commit_gc(checkpoint, set())
    for record in rows.receipts:
        (directory / "receipts" / (record.receipt_id + ".json")).unlink()
    outcome = frequency_gc_finish(PROFILE, generation=1)
    assert outcome.pending and outcome.removed == 0
    assert outcome.warnings == ("frequency_reference_inventory_changed",)
    assert all(
        (directory / "collector" / view_file_name(value.version_id)).exists() for value in values
    )
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    checkpoint.generation += 1
    commit_gc(checkpoint, set())
    outcome = frequency_gc_finish(PROFILE, generation=2)
    assert not outcome.pending and outcome.removed == len(values) - 1
    assert {
        value.version_id
        for value in values
        if (directory / "collector" / view_file_name(value.version_id)).exists()
    } == {values[-1].version_id}
    assert not list((directory / "receipts").glob("*.json"))


def surviving(values):
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        names = set(storage.list_names(directory))
    finally:
        os.close(directory)
    return {value.version_id for value in values if view_file_name(value.version_id) in names}


def staged_checkpoint(blobs, current_refs=frozenset()):
    checkpoint = store.Checkpoint(PROFILE, generation=1, blobs=dict(blobs))
    commit_gc(checkpoint, set(current_refs))
    return checkpoint


def test_reference_proof_follows_ledger_blobs_and_identity_archive_children():
    values = old_views(4)
    direct, via_archive, orphan, current = values
    ledger_blob = store.write_blob(
        PROFILE,
        "ledger-a",
        {"frequency_refs": [direct.version_id], "reference_index_complete": True},
    )
    child = store.write_blob(PROFILE, "ledger-child", {"frequency_refs": [via_archive.version_id]})
    archive = store.write_blob(
        PROFILE, "identity-archive-1", {"schema": 1, "blobs": {"ledger-child": child}}
    )
    staged_checkpoint({"ledger-a": ledger_blob, "identity-archive-1": archive})
    outcome = frequency_gc_finish(PROFILE, generation=1)
    assert not outcome.pending and outcome.removed == 1 and outcome.warnings == ()
    assert surviving(values) == {direct.version_id, via_archive.version_id, current.version_id}
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    job = store.read_blob(PROFILE, checkpoint.blobs[frequency_index.BLOB])
    assert job is not None
    assert job["phase"] == "complete" and job["sources"] == [] and job["candidates"] == []
    assert orphan.version_id not in surviving(values)


@pytest.mark.parametrize(
    "name,body",
    [
        ("identity-archive-1", {"schema": 2, "blobs": {}}),
        ("ledger-a", {"reference_index_complete": False, "frequency_refs": []}),
        ("ledger-a", {"frequency_refs": [1]}),
        ("ledger-a", ["not", "an", "object"]),
    ],
    ids=["archive-schema", "index-incomplete", "non-string-refs", "non-object"],
)
def test_unprovable_reference_sources_block_reclamation_and_keep_every_view(name, body):
    values = old_views(3)
    blob = store.write_blob(PROFILE, name, body)
    staged_checkpoint({name: blob})
    outcome = frequency_gc_finish(PROFILE, generation=1)
    assert outcome.pending and outcome.warnings == ("frequency_references_unknown",)
    assert surviving(values) == {value.version_id for value in values}


def test_reference_body_over_its_size_limit_blocks_instead_of_being_read(monkeypatch):
    values = old_views(3)
    blob = store.write_blob(PROFILE, "ledger-a", {"frequency_refs": []})
    staged_checkpoint({"ledger-a": blob})
    monkeypatch.setattr(frequency_index, "MAX_REFERENCE_BYTES", 8)
    outcome = frequency_gc_finish(PROFILE, generation=1)
    assert outcome.pending and outcome.warnings == ("frequency_references_unknown",)
    assert surviving(values) == {value.version_id for value in values}


def test_reference_byte_budget_defers_remaining_sources_to_the_next_resume(monkeypatch):
    values = old_views(3)
    first = store.write_blob(PROFILE, "ledger-a", {"frequency_refs": [values[0].version_id]})
    second = store.write_blob(PROFILE, "ledger-b", {"frequency_refs": [values[1].version_id]})
    staged_checkpoint({"ledger-a": first, "ledger-b": second})
    monkeypatch.setattr(frequency_index, "REFERENCE_BYTE_BUDGET", 1)

    deferred = frequency_gc_finish(PROFILE, generation=1)
    assert deferred.pending and deferred.removed == 0
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    job = store.read_blob(PROFILE, checkpoint.blobs[frequency_index.BLOB])
    assert job is not None
    assert job["phase"] == "references" and job["source_cursor"] == 1
    assert surviving(values) == {value.version_id for value in values}

    finished = frequency_gc_finish(PROFILE, generation=1)
    assert not finished.pending and finished.removed == 0
    assert surviving(values) == {value.version_id for value in values}


def test_reference_byte_budget_defers_receipts_after_the_source_proof(monkeypatch):
    from ssgrep.accounting.collect import cycle_publish as collector_publish
    from tests.accounting.collect.test_publication import ledger

    values = old_views(3)
    rows = ledger().rows()
    assert rows.receipts
    collector_publish.write_receipts(PROFILE, rows)
    blob = store.write_blob(PROFILE, "ledger-a", {"frequency_refs": [values[0].version_id]})
    staged_checkpoint({"ledger-a": blob})
    monkeypatch.setattr(frequency_index, "REFERENCE_BYTE_BUDGET", 1)

    deferred = frequency_gc_finish(PROFILE, generation=1)
    assert deferred.pending and deferred.removed == 0
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    job = store.read_blob(PROFILE, checkpoint.blobs[frequency_index.BLOB])
    assert job is not None
    assert job["source_cursor"] == 1 and job["receipt_cursor"] == 0

    monkeypatch.setattr(frequency_index, "REFERENCE_BYTE_BUDGET", 16 * 1024 * 1024)
    finished = frequency_gc_finish(PROFILE, generation=1)
    assert not finished.pending
    assert surviving(values) == {values[0].version_id, values[-1].version_id}


def test_restaging_in_flight_job_adds_new_current_references_without_new_inventory():
    values = old_views(3)
    checkpoint = staged_checkpoint({}, {values[0].version_id})
    first = store.read_blob(PROFILE, checkpoint.blobs[frequency_index.BLOB])
    assert first is not None
    assert first["protected"] == [values[0].version_id]

    blobs = dict(checkpoint.blobs)
    pending = stage(
        PROFILE,
        checkpoint,
        blobs,
        current_refs={values[0].version_id, values[1].version_id},
        before_day="2026-09-12",
        today="2026-10-12",
    )
    assert pending
    second = store.read_blob(PROFILE, blobs[frequency_index.BLOB])
    assert first is not None and second is not None
    assert second["protected"] == sorted([values[0].version_id, values[1].version_id])
    assert second["candidates"] == first["candidates"]
    assert blobs[frequency_index.BLOB] != checkpoint.blobs[frequency_index.BLOB]


@pytest.mark.parametrize("body", [{"schema": 2}, ["not", "a", "job"]], ids=["schema", "non-object"])
def test_invalid_maintenance_job_is_state_corruption_for_stage_and_finish(body):
    old_views(2)
    blob = store.write_blob(PROFILE, frequency_index.BLOB, body)
    checkpoint = store.Checkpoint(PROFILE, generation=1, blobs={frequency_index.BLOB: blob})
    store.commit_checkpoint(PROFILE, checkpoint)
    with pytest.raises(store.CollectError) as staged:
        stage(
            PROFILE,
            checkpoint,
            dict(checkpoint.blobs),
            current_refs=set(),
            before_day="2026-09-12",
            today="2026-10-12",
        )
    assert staged.value.condition == "state_corrupt"
    with pytest.raises(store.CollectError) as resumed:
        frequency_gc_finish(PROFILE, generation=1)
    assert resumed.value.condition == "state_corrupt"


def test_collection_reports_frequency_reclamation_until_the_bounded_queue_drains():
    from tests.accounting.collect import collector_scene as scene, verify_scenes as vs

    vs.enable()
    vs.start_capture()
    values = old_views(40)
    first = scene.report(scene.collect(NOW))
    assert not first.complete and first.stopped_by == "frequency_maintenance_pending"
    for _ in range(4):
        final = scene.report(scene.collect(NOW))
        if final.complete:
            break
    assert final.complete and final.stopped_by == "complete"
    assert surviving(values) == {values[-1].version_id}
