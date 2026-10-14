"""Export-owned persistence failures preserve bytes and recover through public operations."""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from ssgrep.accounting import (
    acks,
    export_files,
    export_registry,
    export_spool,
    export_spool_meta as codec,
    storage,
)
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import DEST, PROFILE, M
from tests.accounting.test_export_records import plan_revisions, project, rows_a
from tests.accounting.test_export_spool import make, put


@pytest.mark.parametrize("owner", ["acks", "files", "registry", "spool"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_lock_creation_race_is_bounded_and_recovers_without_partial_state(
    owner: str, exhausted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = plan_revisions(export_registry.Registry(), project(rows_a()))
    batch = make()
    lock_name = {
        "acks": ".metrics.lock",
        "files": ".control.lock",
        "registry": ".records.lock",
        "spool": ".spool.lock",
    }[owner]
    module = {
        "acks": acks,
        "files": export_files,
        "registry": export_registry,
        "spool": export_spool,
    }[owner]

    def persist():
        if owner == "acks":
            return acks.update_delivery(PROFILE, DEST, M, lambda d: acks.collected(d, 7))
        if owner == "registry":
            from ssgrep.accounting.export_records import mark_queued

            return mark_queued(PROFILE, DEST, planned, batch.batch_id)
        if owner == "spool":
            return put(batch)
        with export_files.locked(PROFILE, ("export", DEST), "control") as fd:
            assert fd is not None
            export_files.write_json(fd, "control.json", {"generation": 7})
        return None

    real_open = os.open
    attempts = []
    fail_count = module.LOCK_OPEN_ATTEMPTS - (0 if exhausted else 1)

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == lock_name:
            attempts.append(path)
            if len(attempts) <= fail_count:
                raise FileNotFoundError(errno.ENOENT, "concurrent lock creation")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as fault:
        fault.setattr(os, "open", racing_open)
        if exhausted:
            with pytest.raises(
                (OSError, export_registry.RegistryCorrupt, export_spool.SpoolCorrupt)
            ):
                persist()
        else:
            persist()
    assert len(attempts) == module.LOCK_OPEN_ATTEMPTS
    if exhausted:
        assert not [p for p in fx.profile_root().rglob("*.json")]
    persist()
    if owner == "acks":
        assert fx.state().last_collected_generation == 7
    elif owner == "registry":
        entry = fx.registry().entries[planned[0].record_id]
        assert [(s.revision, s.batch) for s in entry.queued] == [(1, batch.batch_id)]
    elif owner == "spool":
        assert export_spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body
        assert fx.metas()[0].attempts == 0
    else:
        path = fx.profile_root() / "export" / DEST / "control.json"
        assert json.loads(path.read_bytes()) == {"generation": 7}


@pytest.mark.parametrize("damage", ["nested", "duplicate-generation", "descending-generation"])
def test_invalid_ack_document_is_not_rewritten_and_restoration_retains_progress(
    damage: str,
) -> None:
    acks.update_delivery(PROFILE, DEST, M, lambda d: acks.collected(d, 3))
    path = fx.profile_root() / "delivery" / DEST / "metrics.json"
    original = path.read_bytes()
    if damage == "nested":
        damaged = b"[" * 2000 + b"0" + b"]" * 2000
    else:
        document = json.loads(original)
        generations = [2, 2] if damage == "duplicate-generation" else [3, 2]
        document["progress"] = [
            {"generation": g, "snapshot_id": f"{g:032x}", "total": 1, "acked": []}
            for g in generations
        ]
        damaged = json.dumps(document).encode()
    path.write_bytes(damaged)
    with pytest.raises(acks.DeliveryCorrupt):
        acks.update_delivery(PROFILE, DEST, M, lambda d: acks.collected(d, 4))
    assert path.read_bytes() == damaged
    assert acks.list_deliveries(PROFILE) == []
    path.write_bytes(original)
    recovered = acks.update_delivery(PROFILE, DEST, M, lambda d: acks.collected(d, 4))
    assert recovered.last_collected_generation == 4
    assert acks.list_deliveries(PROFILE) == [recovered]


def test_ack_update_refuses_symlink_without_touching_target_then_recovers(tmp_path: Path) -> None:
    acks.update_delivery(PROFILE, DEST, M, lambda d: acks.collected(d, 1))
    path = fx.profile_root() / "delivery" / DEST / "metrics.json"
    target = tmp_path / "saved-delivery"
    path.rename(target)
    original = target.read_bytes()
    path.symlink_to(target)
    with pytest.raises(acks.DeliveryCorrupt):
        acks.update_delivery(PROFILE, DEST, M, lambda d: acks.collected(d, 2))
    assert target.read_bytes() == original and path.is_symlink()
    path.unlink()
    target.rename(path)
    assert (
        acks.update_delivery(
            PROFILE, DEST, M, lambda d: acks.collected(d, 2)
        ).last_collected_generation
        == 2
    )


@pytest.mark.parametrize("owner", ["acks", "spool"])
def test_directory_listing_failure_is_read_only_and_next_listing_recovers(
    owner: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = put(make())
    saved = acks.update_delivery(PROFILE, DEST, M, lambda d: acks.collected(d, 1))
    before = fx.tree()
    real_list = storage.list_names
    wanted = fx.profile_root() / ("delivery" if owner == "acks" else "spool")
    identity = wanted.stat()

    def unreadable(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            raise OSError(errno.EIO, "directory unavailable")
        return real_list(fd)

    with monkeypatch.context() as fault:
        fault.setattr(storage, "list_names", unreadable)
        found = (
            acks.list_deliveries(PROFILE) if owner == "acks" else export_spool.iter_spools(PROFILE)
        )
        assert found == []
    assert fx.tree() == before
    assert acks.list_deliveries(PROFILE) == [saved]
    assert export_spool.iter_spools(PROFILE) == [(DEST, M)]
    assert fx.body_of(meta) == make().body


@pytest.mark.parametrize(
    "damage",
    [
        {"destination_id": "../escape"},
        {"record_ids": "record"},
        {"scope": []},
        {"scope": {"count": 1 << 63}},
        {"scope": {"nested": [[[[[[[[[0]]]]]]]]]}},
        {"purge_requested": 1},
        {"warning": 4},
        {"warning": "x" * 513},
    ],
    ids=[
        "destination",
        "record-list",
        "scope-object",
        "scope-int",
        "scope-depth",
        "purge-bool",
        "warning-type",
        "warning-length",
    ],
)
def test_invalid_spool_metadata_is_never_sendable_and_original_bytes_recover(damage: dict) -> None:
    batch = make()
    meta = put(batch)
    path = fx.profile_root() / "spool" / DEST / M.value / f"{meta.batch_id}.meta.json"
    original = path.read_bytes()
    damaged = json.dumps({**json.loads(original), **damage}).encode()
    path.write_bytes(damaged)
    with pytest.raises(export_spool.SpoolCorrupt):
        export_spool.read_body(PROFILE, DEST, M, meta.batch_id)
    assert export_spool.corrupt_batches(PROFILE, DEST, M) == [meta.batch_id]
    assert export_spool.list_batches(PROFILE, DEST, M) == []
    assert path.read_bytes() == damaged
    path.write_bytes(original)
    assert fx.body_of(meta) == batch.body and fx.metas() == [meta]


def test_metadata_encoding_cap_refuses_write_without_replacing_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = put(make())
    path = fx.profile_root() / "spool" / DEST / M.value / f"{meta.batch_id}.meta.json"
    original = path.read_bytes()
    with monkeypatch.context() as fault:
        fault.setattr(codec, "MAX_META_BYTES", len(original) - 1)
        with pytest.raises(ValueError):
            export_spool.update_meta(
                PROFILE, DEST, M, meta.batch_id, lambda m: replace(m, attempts=1)
            )
    assert path.read_bytes() == original and fx.body_of(meta) == make().body
    assert (
        export_spool.update_meta(
            PROFILE, DEST, M, meta.batch_id, lambda m: replace(m, attempts=1)
        ).attempts
        == 1
    )


def test_invalid_update_result_cannot_replace_a_valid_spool_meta() -> None:
    meta = put(make())
    before = fx.tree()
    with pytest.raises(ValueError):
        invalid = cast(Callable[[codec.Meta], codec.Meta], lambda _meta: None)
        export_spool.update_meta(PROFILE, DEST, M, meta.batch_id, invalid)
    assert fx.tree() == before
    assert (
        export_spool.update_meta(
            PROFILE, DEST, M, meta.batch_id, lambda m: replace(m, state="acked")
        ).state
        == "acked"
    )


def test_privacy_erase_requires_marker_and_cannot_erase_deletion_requests() -> None:
    batch = make()
    assert not export_spool.erase_marked_batch(PROFILE, DEST, M, batch.batch_id)
    meta = put(batch)
    before = fx.tree()
    with pytest.raises(ValueError):
        export_spool.erase_marked_batch(PROFILE, DEST, M, batch.batch_id)
    assert fx.tree() == before and fx.body_of(meta) == batch.body
    tomb = put(make(signal=fx.L, kind="tombstone"))
    with pytest.raises(ValueError):
        export_spool.update_meta(
            PROFILE, DEST, fx.L, tomb.batch_id, lambda m: replace(m, purge_requested=True)
        )
    assert fx.body_of(tomb) == make(signal=fx.L, kind="tombstone").body
    export_spool.update_meta(
        PROFILE, DEST, M, batch.batch_id, lambda m: replace(m, purge_requested=True)
    )
    assert export_spool.erase_marked_batch(PROFILE, DEST, M, batch.batch_id)
    assert not export_spool.erase_marked_batch(PROFILE, DEST, M, batch.batch_id)
    assert fx.metas() == [] and fx.metas(fx.L) == [tomb]


@pytest.mark.parametrize("operation", ["corrupt", "usage"])
def test_spool_file_removed_after_listing_is_not_reported_as_existing(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = put(make())
    real_list = storage.list_names
    folder = fx.profile_root() / "spool" / DEST / M.value
    removed = []

    def concurrent_delete(fd):
        names = real_list(fd)
        if f"{meta.batch_id}.meta.json" in names and not removed:
            removed.append(meta.batch_id)
            for path in folder.glob(f"{meta.batch_id}.*"):
                path.unlink()
        return names

    with monkeypatch.context() as race:
        race.setattr(storage, "list_names", concurrent_delete)
        if operation == "corrupt":
            assert export_spool.corrupt_batches(PROFILE, DEST, M) == []
        else:
            assert export_spool.usage(PROFILE, DEST, M) == export_spool.Usage(0, 0)
    assert removed == [meta.batch_id] and fx.metas() == []
    put(make())
    assert fx.body_of(meta) == make().body
    assert export_spool.usage(PROFILE, DEST, M).batches == 1


def test_invalid_state_path_is_refused_before_creating_anything() -> None:
    before = fx.tree()
    with pytest.raises(ValueError):
        with export_files.locked(PROFILE, ("../escape",), "control"):
            pytest.fail("invalid path acquired a lock")
    assert fx.tree() == before
    with export_files.locked(PROFILE, ("export", DEST), "control") as fd:
        assert fd is not None
        export_files.write_json(fd, "control.json", {"ok": True})
    assert json.loads((fx.profile_root() / "export" / DEST / "control.json").read_bytes()) == {
        "ok": True
    }


def test_spool_codec_rejects_invalid_namespace_even_when_expected_binding_matches() -> None:
    meta = put(make())
    original = codec.encode_meta(meta)
    document = json.loads(original)
    document["destination_id"] = "../escape"
    with pytest.raises(codec.SpoolCorrupt, match="destination id"):
        codec.decode_meta(json.dumps(document).encode(), meta.batch_id, "../escape", M)
    restored = codec.decode_meta(original, meta.batch_id, DEST, M)
    assert restored == meta and fx.body_of(meta) == make().body


def test_spool_codec_bounds_record_count_independently_of_identifier_shape() -> None:
    meta = put(make())
    original = codec.encode_meta(meta)
    document = json.loads(original)
    document["record_ids"] = ["r"] * (codec.MAX_RECORD_IDS + 1)
    with pytest.raises(codec.SpoolCorrupt, match="record_ids must be a list of at most"):
        codec.decode_meta(json.dumps(document).encode(), meta.batch_id, DEST, M)
    restored = codec.decode_meta(original, meta.batch_id, DEST, M)
    assert restored == meta and fx.body_of(meta) == make().body
