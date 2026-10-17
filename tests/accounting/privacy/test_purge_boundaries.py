"""Purge corruption and journal boundaries with real state and recoverable file faults."""

from __future__ import annotations

import errno
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting.capture.journal_codec import serialize_event
from ssgrep.accounting.collect import store as collector_store
from ssgrep.accounting.export import records as export_records
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.privacy import (
    policy,
    purge,
    purge as purge_snapshot,
    purge_files,
    storage,
    storage as plainfile,
)
from ssgrep.accounting.privacy.purge import PurgeError
from ssgrep.accounting.serve import lease
from ssgrep.accounting.views import snapshots
from tests.accounting.collect.collector_scene import accounting_dir
from tests.accounting.conftest import build_event
from tests.accounting.export.snapshots_driver import make_envelope
from tests.accounting.privacy.test_purge import (
    ALPHA,
    CUTOFF,
    DEST,
    PROFILE,
    PURGE_AT,
    build_scene,
    configure_policy,
    current_envelope,
    export_all,
    old_record_ids,
    periods,
    select,
    tree,
)

OLD = datetime(2026, 10, 1, tzinfo=UTC).timestamp()
CUTOFF_EPOCH = datetime(2026, 10, 10, tzinfo=UTC).timestamp()


def journal_file(name: str, data: bytes, mtime: float = OLD) -> Path:
    plainfile.write_private(storage.data_root(), "accounting", PROFILE, "journal", name, data=data)
    target = accounting_dir() / "journal" / name
    os.utime(target, (mtime, mtime))
    return target


def test_scoped_purge_preserves_other_workspace_current_typed_archived_receipts(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.state import VersionedLedger, load_state, save_state
    from ssgrep.accounting.privacy.purge_state import plan_state

    build_scene(tmp_path)
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    state = load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.ledger is not None
    ledger = VersionedLedger.restore_versioned(state.ledger, state.superseded, state.archives)
    ledger.archive_current("typed-prior-publication")
    original = ledger.archives["typed-prior-publication"]
    selected_roots = {row.receipt.root for row in original.receipts if row.period == "2026-10-05"}
    other_receipts = tuple(row for row in original.receipts if row.period == "2026-10-06")
    current_receipts = tuple(row for row in original.receipts if row.period == "2026-10-15")
    assert selected_roots and other_receipts and current_receipts
    assert not selected_roots & {row.receipt.root for row in other_receipts}
    state.archives = ledger.archives_state()
    save_state(PROFILE, state, checkpoint.methodology, checkpoint.blobs)
    collector_store.commit_checkpoint(PROFILE, checkpoint)

    planned = plan_state(PROFILE, select(workspace=ALPHA), checkpoint)
    assert planned is not None
    archived = tuple(
        archive
        for namespace in planned.namespaces
        for name, archive in namespace.archives_after.items()
        if name == "typed-prior-publication"
    )
    assert len(archived) == 1
    assert archived[0].receipts == tuple(
        row for row in original.receipts if row.period != "2026-10-05"
    )
    assert all(row in archived[0].receipts for row in (*other_receipts, *current_receipts))


def test_journal_purge_keeps_incomplete_corrupt_oversized_and_other_workspace_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = serialize_event(build_event(ts_utc="2026-10-05T12:00:00Z", cwd=ALPHA))
    other = serialize_event(build_event(ts_utc="2026-10-05T12:00:00Z", cwd="/work/beta"))
    cutoff = serialize_event(build_event(ts_utc="2026-10-10T00:00:00Z", cwd=ALPHA))
    selected = journal_file("seg-old.jsonl", old)
    empty = journal_file("seg-empty.jsonl", b"\n")
    retained = {
        journal_file("seg-tail.jsonl", old[:-1]): old[:-1],
        journal_file("seg-corrupt.jsonl", b"{torn}\n"): b"{torn}\n",
        journal_file("seg-large.jsonl", old + old): old + old,
        journal_file("seg-other.jsonl", other): other,
        journal_file("seg-cutoff.jsonl", cutoff): cutoff,
        journal_file("seg-active.jsonl", old, CUTOFF_EPOCH): old,
        journal_file("notes.txt", b"unrelated"): b"unrelated",
    }
    monkeypatch.setattr(purge_files, "MAX_JOURNAL_SEGMENT", max(len(old), len(other), len(cutoff)))
    before = tree(accounting_dir())
    preview = purge.dry_run(select(workspace=ALPHA))
    assert preview.local_deleted.journal_segments == 2
    assert preview.retained.journal_segments_retained == 6
    assert tree(accounting_dir()) == before
    result = purge.execute(select(workspace=ALPHA), now=PURGE_AT)
    assert result.local_deleted.journal_segments == 2
    assert result.retained.journal_segments_retained == 6
    assert not selected.exists() and not empty.exists()
    assert {p.name: p.read_bytes() for p in (accounting_dir() / "journal").iterdir()} == {
        p.name: data for p, data in retained.items()
    }
    assert purge.execute(select(workspace=ALPHA), now=PURGE_AT).local_deleted.journal_segments == 0


@pytest.mark.parametrize(
    "damage,condition",
    [(b"{torn", "state_corrupt"), (b'{"schema":999}', "unsupported_schema")],
)
def test_corrupt_checkpoint_refuses_purge_without_deleting_journal_or_snapshot(
    tmp_path: Path, damage: bytes, condition: str
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    old = serialize_event(build_event(ts_utc="2026-10-05T00:00:00Z"))
    journal_file("seg-old.jsonl", old)
    plainfile.write_private(
        storage.data_root(), "accounting", PROFILE, "collector", "checkpoint.json", data=damage
    )
    before = tree(tmp_path / "data")
    with pytest.raises(PurgeError) as raised:
        purge.dry_run(select())
    assert raised.value.condition == condition
    assert str(tmp_path) not in str(raised.value) and "torn" not in str(raised.value)
    assert tree(tmp_path / "data") == before
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)
    assert (accounting_dir() / "journal" / "seg-old.jsonl").read_bytes() == old


@pytest.mark.parametrize("unsafe_entry", ["fifo", "depth"])
def test_preflight_refuses_special_or_excessively_nested_storage_before_any_write(
    tmp_path: Path, unsafe_entry: str
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    if unsafe_entry == "fifo":
        os.mkfifo(accounting_dir() / "unexpected-pipe", 0o600)
    else:
        directory = accounting_dir()
        for n in range(9):
            directory /= f"level-{n}"
            directory.mkdir(mode=0o700)
    before = tree(tmp_path / "data")
    for operation in (purge.dry_run, purge.execute):
        with pytest.raises(PurgeError) as raised:
            operation(select())
        assert raised.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_unreadable_journal_stops_selection_instead_of_deleting_the_readable_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = serialize_event(build_event(ts_utc="2026-10-05T00:00:00Z"))
    journal_file("seg-a.jsonl", old)
    journal_file("seg-z.jsonl", old)
    before = tree(tmp_path / "data")
    original_read = storage.read_file

    def deny_one(fd: int, name: str, max_bytes: int):
        if name == "seg-z.jsonl":
            raise PermissionError(errno.EACCES, "private journal denied")
        return original_read(fd, name, max_bytes)

    with monkeypatch.context() as fault:
        fault.setattr(storage, "read_file", deny_one)
        with pytest.raises(PurgeError) as raised:
            purge.dry_run(select())
    assert raised.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    report = purge.execute(select(), now=PURGE_AT)
    assert report.local_deleted.journal_segments == 2
    assert list((accounting_dir() / "journal").iterdir()) == []


@pytest.mark.parametrize(
    "damage", [b"{torn", b"[]", b"{}", b'{"generation":true}', b'{"generation":0}']
)
def test_corrupt_pending_marker_never_hides_unfinished_privacy_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: bytes
) -> None:
    build_scene(tmp_path)

    def before_pointer(profile: str, generation: int) -> None:
        raise OSError(errno.EIO, "interrupted before pointer publication")

    with monkeypatch.context() as fault:
        fault.setattr(purge_snapshot, "make_current", before_pointer)
        with pytest.raises(PurgeError):
            purge.execute(select(), now=PURGE_AT)
    assert periods() == {"2026-10-15"}
    committed_snapshot = snapshots.read_generation(PROFILE, 2)
    assert committed_snapshot is not None
    marker = accounting_dir() / "collector" / purge_files.MARKER
    pending = marker.read_bytes()
    marker.write_bytes(damage)
    before = tree(accounting_dir())
    with pytest.raises(PurgeError) as raised:
        purge.dry_run(select())
    assert raised.value.condition == "state_corrupt"
    assert tree(accounting_dir()) == before
    marker.write_bytes(pending)
    purge.execute(select(), now=PURGE_AT)
    assert current_envelope() == committed_snapshot
    assert snapshots.read_generation(PROFILE, 1) is None
    assert snapshots.read_generation(PROFILE, 2) == current_envelope()
    assert periods() == {"2026-10-15"} and purge_files.read_marker(PROFILE) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("before", None),
        ("profile", "other-profile"),
        ("before", "not-an-instant"),
        ("workspace", "relative/path"),
        ("workspace", "/"),
    ],
)
def test_invalid_durable_purge_selection_fails_closed_and_valid_intent_recovers(
    tmp_path: Path, field: str, value: object
) -> None:
    from ssgrep.accounting.privacy import purge_recovery

    build_scene(tmp_path)
    selected = select()
    purge_recovery.stage_erasure(selected)
    marker = accounting_dir() / "collector" / purge_recovery.MARKER
    original = json.loads(marker.read_bytes())
    broken = json.loads(marker.read_bytes())
    broken["pending_purge"][field] = value
    marker.write_text(json.dumps(broken) + "\n")
    before = tree(tmp_path / "data")

    with pytest.raises(PurgeError) as caught:
        purge_recovery.has_pending_erasure(selected)

    assert caught.value.condition == "state_corrupt"
    assert tree(tmp_path / "data") == before
    marker.write_text(json.dumps(original) + "\n")
    assert purge_recovery.has_pending_erasure(selected)


def test_unsupported_purge_marker_schema_fails_closed_without_repair(tmp_path: Path) -> None:
    from ssgrep.accounting.privacy import purge_recovery

    build_scene(tmp_path)
    selected = select()
    purge_recovery.stage_erasure(selected)
    marker = accounting_dir() / "collector" / purge_recovery.MARKER
    marker.write_text('{"schema": 2}\n')
    before = tree(tmp_path / "data")

    with pytest.raises(PurgeError) as caught:
        purge_recovery.has_pending_erasure(selected)

    assert caught.value.condition == "state_corrupt"
    assert tree(tmp_path / "data") == before


def test_noop_preserves_old_mtime_authority_and_future_temporary_content(tmp_path: Path) -> None:
    build_scene(tmp_path)
    collector = accounting_dir() / "collector"
    for path in collector.iterdir():
        if path.name == "frequency.json" or path.name.startswith("frequency-"):
            os.utime(path, (OLD - 86400, OLD - 86400))
    fd = storage.open_profile_dir(PROFILE, "collector")
    try:
        storage.atomic_write(
            fd,
            ".future-evidence.tmp",
            b'{"ts_utc":"2026-10-15T00:00:00Z","text":"future-private-canary"}',
        )
    finally:
        os.close(fd)
    before = tree(tmp_path / "data")
    assert purge.dry_run(select("2026-10-01")).local_deleted.total() == 0
    result = purge.execute(select("2026-10-01"), now=PURGE_AT)
    assert result.state == "nothing_to_purge" and result.local_deleted.total() == 0
    after = tree(tmp_path / "data")
    assert {key: value for key, value in after.items() if "lease.lock" not in key} == {
        key: value for key, value in before.items() if "lease.lock" not in key
    }


def test_pending_purge_erases_vocab_after_checkpoint_and_journal_already_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_scene(tmp_path)
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    originals = {Path(source.source_id.split(":", 1)[1]) for source in checkpoint.sources.values()}
    assert all(path.is_relative_to(tmp_path) for path in originals)
    source_bytes = {path: path.read_bytes() for path in originals}
    from ssgrep.accounting.privacy import purge_files

    real_derived = purge_files.purge_derived

    def before_derived(selection, **kwargs):
        if kwargs.get("dry_run"):
            return real_derived(selection, **kwargs)
        raise OSError(errno.EIO, "interrupted before derived erasure")

    with monkeypatch.context() as fault:
        fault.setattr(purge_files, "purge_derived", before_derived)
        with pytest.raises(PurgeError):
            purge.execute(select(), now=PURGE_AT)
    assert periods() == {"2026-10-15"}
    assert (accounting_dir() / "collector" / "frequency.json").exists()
    committed_snapshot = snapshots.read_generation(PROFILE, 2)
    purge.execute(select(), now=PURGE_AT)
    remaining = {path.name for path in (accounting_dir() / "collector").iterdir()}
    assert "frequency.json" not in remaining
    assert not any(
        name.startswith("frequency-") and name != "frequency-purged.json" for name in remaining
    )
    assert current_envelope() == committed_snapshot
    assert snapshots.read_generation(PROFILE, 1) is None
    assert {path: path.read_bytes() for path in originals} == source_bytes


def test_oversized_pending_marker_is_not_treated_as_absent(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    plainfile.write_private(
        storage.data_root(),
        "accounting",
        PROFILE,
        "collector",
        purge_files.MARKER,
        data=b" " * 4097,
    )
    before = tree(tmp_path / "data")
    with pytest.raises(PurgeError) as raised:
        purge.dry_run(select())
    assert raised.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_removed_destination_reports_remote_obligations_without_sending_or_losing_local_purge(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    export_all(DEST, ack=True)
    expected = old_record_ids(DEST)
    policy.save_profile(ProfilePolicy(collection_enabled=True))
    result = purge.execute(select(), now=PURGE_AT)
    assert result.queued_records == 0
    assert [
        (item.destination, item.reason, item.records, item.record_ids)
        for item in result.obligations
    ] == [(DEST, "unknown_destination", len(expected), tuple(expected))]
    assert result.to_dict()["backend_erased"] == "unknown_operator_responsibility"
    assert periods() == {"2026-10-15"}
    assert not (accounting_dir() / "spool").exists()


def test_purge_with_no_profile_creates_nothing_and_reports_no_deletions(tmp_path: Path) -> None:
    result = purge.execute(select(), now=PURGE_AT)
    assert result.performed and result.state == "nothing_to_purge"
    assert result.local_deleted.total() == 0 and result.obligations == ()
    assert not (tmp_path / "data").exists()


def test_backfill_horizon_is_exclusive_and_unknown_dates_are_not_guessed() -> None:
    checkpoint = collector_store.Checkpoint(
        PROFILE,
        backfills=[
            {"until": "2026-10-10T00:00:00Z", "id": "old"},
            {"until": "2026-10-10T00:00:01Z", "id": "new"},
            {"until": None, "id": "unknown"},
        ],
    )
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    preview = purge.dry_run(select())
    assert preview.local_deleted.backfill_series == 1
    assert purge.execute(select(), now=PURGE_AT).local_deleted.backfill_series == 1
    committed = collector_store.load_checkpoint(PROFILE)
    assert committed is not None
    assert committed.backfills == checkpoint.backfills[1:]
    assert committed.generation == 0 and snapshots.generations(PROFILE) == []
    assert (
        json.loads((accounting_dir() / "collector" / "checkpoint.json").read_bytes())["backfills"]
        == checkpoint.backfills[1:]
    )


@pytest.mark.parametrize("component", ["root", "snapshots"])
def test_readonly_purge_refuses_exposed_storage_without_repairing_permissions(
    tmp_path: Path, component: str
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    target = accounting_dir() if component == "root" else accounting_dir() / "snapshots"
    target.chmod(0o750)
    before = tree(tmp_path / "data")
    with pytest.raises(PurgeError) as caught:
        if component == "root":
            purge.dry_run(select())
        else:
            purge_files.generation_files(PROFILE)
    assert caught.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    target.chmod(0o700)
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


@pytest.mark.parametrize("component,mode", [("directory", 0o750), ("file", 0o644), ("file", 0o700)])
def test_preflight_refuses_exposed_unknown_entries_without_changing_anything(
    tmp_path: Path, component: str, mode: int
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    plainfile.write_private(
        storage.data_root(), "accounting", PROFILE, "extra", "original", data=b"secret"
    )
    target = accounting_dir() / "extra"
    if component == "file":
        target /= "original"
    target.chmod(mode)
    before = tree(tmp_path / "data")
    with pytest.raises(PurgeError) as caught:
        purge.dry_run(select())
    assert caught.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    target.chmod(0o700 if component == "directory" else 0o600)
    assert purge_files.preflight(PROFILE)
    assert (accounting_dir() / "extra" / "original").read_bytes() == b"secret"


def test_preflight_listing_io_failure_preserves_data_and_successful_retry_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    before = tree(tmp_path / "data")
    listdir = os.listdir

    def interrupted(path):
        if isinstance(path, int):
            raise OSError(errno.EIO, "directory read interrupted")
        return listdir(path)

    with monkeypatch.context() as fault:
        fault.setattr(purge_files.os, "listdir", interrupted)
        with pytest.raises(PurgeError) as caught:
            purge.dry_run(select())
    assert caught.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    assert purge_files.preflight(PROFILE)
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_lease_open_failure_stops_purge_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    with lease.Lease.acquire(PROFILE, "collect"):
        pass
    before = tree(tmp_path / "data")
    open_file = os.open

    def fail_lock(name, flags, mode=0o777, *, dir_fd=None):
        if name == lease.LEASE_FILE:
            raise storage.UnsafeStorageError(errno.EPERM, "foreign lock")
        return open_file(name, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as fault:
        fault.setattr(lease.os, "open", fail_lock)
        with pytest.raises(PurgeError) as caught:
            purge.execute(select(), now=PURGE_AT)
    assert caught.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    assert purge.execute(select(), now=PURGE_AT).local_deleted.total() == 0
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_dry_run_reports_unknown_worker_when_owner_record_read_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with lease.Lease.acquire(PROFILE, "serve"):
        before = tree(tmp_path / "data")

        def unreadable(fd, length, offset):
            raise OSError(errno.EIO, "owner record unreadable")

        with monkeypatch.context() as fault:
            fault.setattr(lease.os, "pread", unreadable)
            report = purge.dry_run(select())
        assert report.worker is None
        assert tree(tmp_path / "data") == before
        readable = purge.dry_run(select())
        assert readable.worker is not None and readable.worker.running


def test_missing_referenced_legacy_ledger_stops_purge_without_dropping_other_data(
    tmp_path: Path,
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    name = collector_store.write_blob(PROFILE, "ledger-legacy", {})
    checkpoint = collector_store.Checkpoint(PROFILE, blobs={"ledger-legacy": name})
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    (accounting_dir() / "collector" / collector_store.STATE_DIR / name).unlink()
    before = tree(tmp_path / "data")
    with pytest.raises(PurgeError) as caught:
        purge.dry_run(select())
    assert caught.value.condition == "state_corrupt"
    assert tree(tmp_path / "data") == before
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_preflight_reads_owner_readonly_entries_without_broadening_permissions(
    tmp_path: Path,
) -> None:
    plainfile.write_private(
        storage.data_root(), "accounting", PROFILE, "extra", "original", data=b"readonly secret"
    )
    directory = accounting_dir() / "extra"
    target = directory / "original"
    directory.chmod(0o500)
    target.chmod(0o400)
    try:
        before = tree(tmp_path / "data")
        assert purge_files.preflight(PROFILE)
        assert tree(tmp_path / "data") == before
        assert (
            plainfile.read_private(
                storage.data_root(), "accounting", PROFILE, "extra", "original", max_bytes=15
            )
            == b"readonly secret"
        )
    finally:
        directory.chmod(0o700)
        target.chmod(0o600)


def test_derived_cleanup_preserves_temp_in_dry_run_then_erases_it_with_cache(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    cache = accounting_dir() / "collector" / "cache"
    stale = cache / "stale-evidence.jsonl"
    plainfile.write_private(
        storage.data_root(),
        "accounting",
        PROFILE,
        "collector",
        "cache",
        stale.name,
        data=b"not a valid evidence line",
    )
    os.utime(stale, (OLD, OLD))
    temporary = cache / ".in-progress.tmp"
    plainfile.write_private(
        storage.data_root(),
        "accounting",
        PROFILE,
        "collector",
        "cache",
        temporary.name,
        data=b"uncommitted private content",
    )
    stale_bytes = stale.read_bytes()
    temporary_bytes = temporary.read_bytes()

    preview = purge.dry_run(select())
    assert preview.local_deleted.plaintext_cache_files >= 1
    assert preview.local_deleted.temporary_files >= 1
    assert stale.read_bytes() == stale_bytes
    assert temporary.read_bytes() == temporary_bytes

    purge.execute(select(), now=PURGE_AT)
    assert not stale.exists()
    assert not temporary.exists()


def test_temporary_cleanup_erases_cache_temp_but_preserves_frequency_and_journal_files(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    collector = accounting_dir() / "collector"
    cache_temp = collector / "cache" / ".partial.tmp"
    frequency_temp = collector / ".frequency-refresh.tmp"
    journal_file_path = accounting_dir() / "journal" / ".pending.tmp"
    plainfile.write_private(
        storage.data_root(),
        "accounting",
        PROFILE,
        "collector",
        "cache",
        cache_temp.name,
        data=b"cache",
    )
    plainfile.write_private(
        storage.data_root(),
        "accounting",
        PROFILE,
        "collector",
        frequency_temp.name,
        data=b"frequency",
    )
    plainfile.write_private(
        storage.data_root(),
        "accounting",
        PROFILE,
        "journal",
        journal_file_path.name,
        data=b"journal",
    )
    preserved = {
        frequency_temp: frequency_temp.read_bytes(),
        journal_file_path: journal_file_path.read_bytes(),
    }

    removed = purge_files._temporaries(PROFILE, dry_run=False, parts=("collector",))

    assert removed >= 1
    assert not cache_temp.exists()
    assert {path: path.read_bytes() for path in preserved} == preserved


def test_temporary_cleanup_treats_an_absent_collector_as_empty(tmp_path: Path) -> None:
    assert purge_files._temporaries(PROFILE, dry_run=False, parts=("collector", "cache")) == 0
    assert not (tmp_path / "data").exists()


def test_temporary_cleanup_depth_limit_is_inclusive_at_eight_components(tmp_path: Path) -> None:
    assert purge_files._temporaries(PROFILE, dry_run=True, parts=("deep",) * 8) == 0
    with pytest.raises(ValueError, match="depth exceeds limit"):
        purge_files._temporaries(PROFILE, dry_run=True, parts=("deep",) * 9)
    assert not (tmp_path / "data").exists()


def test_cache_entry_that_vanishes_before_it_is_read_does_not_stop_cache_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_scene(tmp_path)
    stale = accounting_dir() / "collector" / "cache" / "stale-evidence.jsonl"
    plainfile.write_private(
        storage.data_root(),
        "accounting",
        PROFILE,
        "collector",
        "cache",
        stale.name,
        data=b"not a valid evidence line",
    )
    expected = purge.dry_run(select()).local_deleted.plaintext_cache_files
    assert expected >= 1
    real_list_names = storage.list_names
    injected = []

    def list_with_vanished_entry(directory: int) -> list[str]:
        names = real_list_names(directory)
        if stale.name in names and not injected:
            injected.append(directory)  # only the cache scan sees the entry that then vanishes
            return ["aaa-vanished.jsonl", *names]
        return names

    with monkeypatch.context() as patch:
        patch.setattr(purge_files.storage, "list_names", list_with_vanished_entry)
        result = purge.execute(select(), now=PURGE_AT)

    assert injected
    assert result.local_deleted.plaintext_cache_files == expected
    assert not stale.exists()
    assert not (stale.parent / "aaa-vanished.jsonl").exists()


def add_state_archive(blobs: dict[str, str], *, share_live_state: bool = False) -> str:
    """Commit a verified-mapping legacy identity archive beside the live state."""
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    name = "identity-archive-legacy_v1"
    manifest = {
        "schema": 1,
        "methodology": checkpoint.methodology,
        "root_mapping_status": "available",
        "blobs": blobs,
        "segments": [ref.to_dict() for ref in checkpoint.segments] if share_live_state else [],
    }
    checkpoint.blobs[name] = collector_store.write_blob(PROFILE, name, manifest)
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    return name


def test_scoped_purge_keeps_unmatched_state_archive_blobs_and_manifest(tmp_path: Path) -> None:
    build_scene(tmp_path)
    note = collector_store.write_blob(PROFILE, "archived-note", {"kept": "archive-only"})
    name = add_state_archive({"archived-note": note})
    before = collector_store.load_checkpoint(PROFILE)
    assert before is not None

    purge.execute(select(workspace=ALPHA), now=PURGE_AT)

    after = collector_store.load_checkpoint(PROFILE)
    assert after is not None and after.generation > before.generation
    manifest = collector_store.read_blob(PROFILE, after.blobs[name])
    assert manifest is not None
    assert manifest["blobs"] == {"archived-note": note}
    assert collector_store.read_blob(PROFILE, note) == {"kept": "archive-only"}
    assert after.identity_migration["superseded_blobs"] == {"archived-note": note}


def test_state_archive_receipts_are_not_reported_stale_when_the_selection_keeps_them(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    ledgers = {name: ref for name, ref in checkpoint.blobs.items() if name.startswith("ledger")}
    assert ledgers
    rows = export_records.ledger_rows(PROFILE)
    assert rows is not None
    selected_old = {
        receipt.receipt_id
        for receipt in rows.receipts
        if receipt.receipt.root == "claude:a1" and receipt.period < CUTOFF
    }
    assert selected_old
    control = purge.plan(select(workspace=ALPHA)).stale_receipts

    add_state_archive(ledgers, share_live_state=True)
    archived = purge.plan(select(workspace=ALPHA)).stale_receipts

    assert control >= selected_old
    assert archived == control
    kept_public = purge_files.receipt_ids(PROFILE) - archived
    assert kept_public and not kept_public & selected_old


def test_state_archive_planned_for_whole_profile_cannot_resolve_workspace_roots(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.privacy.purge_state import plan_state

    build_scene(tmp_path)
    note = collector_store.write_blob(PROFILE, "archived-note", {"kept": "archive-only"})
    add_state_archive({"archived-note": note})
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    whole = plan_state(PROFILE, select(), checkpoint)
    assert whole is not None and whole.legacy_archives
    assert whole.legacy_archives[0].plan.scopes.selected is None

    with pytest.raises(PurgeError) as unresolved:
        purge._root_keys(whole, select(workspace=ALPHA))

    assert unresolved.value.condition == "archive_scope_unresolved"
    assert purge._root_keys(whole, select()) is None
