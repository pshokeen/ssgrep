"""Purge corruption and journal boundaries with real state and recoverable file faults."""

from __future__ import annotations

import errno
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting import (
    collector_store,
    plainfile,
    policy,
    purge,
    purge_files,
    purge_snapshot,
    snapshots,
    storage,
)
from ssgrep.accounting.journal_codec import serialize_event
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.purge_model import PurgeError
from tests.accounting.collector_scene import accounting_dir
from tests.accounting.conftest import build_event
from tests.accounting.snapshots_driver import make_envelope
from tests.accounting.test_purge import (
    ALPHA,
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
    marker = accounting_dir() / "collector" / purge_files.MARKER
    pending = marker.read_bytes()
    marker.write_bytes(damage)
    before = tree(accounting_dir())
    with pytest.raises(PurgeError) as raised:
        purge.dry_run(select())
    assert raised.value.condition == "state_corrupt"
    assert tree(accounting_dir()) == before
    marker.write_bytes(pending)
    result = purge.execute(select(), now=PURGE_AT)
    assert result.local_deleted.snapshot_generations == 1
    assert snapshots.read_generation(PROFILE, 1) is None
    assert snapshots.read_generation(PROFILE, 2) == current_envelope()
    assert periods() == {"2026-10-15"} and purge_files.read_marker(PROFILE) is None


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
