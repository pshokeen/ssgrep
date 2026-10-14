"""Durable immutable outbound spool: exact bytes, crash safety, integrity, strict meta codec."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import export_spool as spool, storage
from ssgrep.accounting.export_model import MAX_BATCH_BYTES, Batch
from ssgrep.accounting.export_spool import BatchConflict, Meta, SpoolCorrupt, Usage
from ssgrep.accounting.ledger_contracts import Signal

PROFILE = "personal-local"
DEST = "org-primary"
NOW_Z = "2026-10-13T12:00:00Z"
SCOPE: dict[str, object] = {
    "roots": ["a", "b"],
    "endpoint": "https://collector.example/otlp",
    "fields_version": "logs-v1",
    "policy_version": 3,
    "logs_permitted": True,
    "identity": None,
}
M, L = Signal.METRICS, Signal.LOGS


def snap(gen: int) -> str:
    return f"{gen:04x}" * 8


def make(
    gen: int = 1,
    index: int = 0,
    *,
    total: int = 1,
    signal: Signal = M,
    kind: str = "data",
    body: bytes | None = None,
    items: int = 2,
    batch_id: str | None = None,
) -> Batch:
    if batch_id is None:
        batch_id = (
            f"t1760000000-ab12cd34-{index:04d}"
            if kind == "tombstone"
            else f"g{gen:010d}-{snap(gen)[:16]}-{index:04d}"
        )
    payload = (
        body if body is not None else json.dumps({"n": gen * 100 + index, "pad": "x" * 40}).encode()
    )
    ids = tuple(f"{gen:08x}{index:04x}{i:020x}" for i in range(items)) if signal is L else ()
    return Batch(batch_id, signal, gen, snap(gen), index, total, kind, payload, items, ids)


def put(batch: Batch, *, dest: str = DEST, scope: dict[str, object] | None = None) -> Meta:
    return spool.write_batch(
        PROFILE,
        dest,
        batch.signal,
        batch,
        created_utc=NOW_Z,
        scope=SCOPE if scope is None else scope,
    )


def folder(tmp_path: Path, signal: Signal = M, dest: str = DEST) -> Path:
    return tmp_path / "data" / "accounting" / PROFILE / "spool" / dest / signal.value


def tree(root: Path) -> dict[str, tuple[int, int, int]]:
    """Every entry under ``root`` as (mode, size, mtime_ns); empty when ``root`` is absent."""
    if not root.exists():
        return {}
    found = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        found[str(path.relative_to(root))] = (info.st_mode, info.st_size, info.st_mtime_ns)
    return found


def names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name != ".spool.lock")


def ids(signal: Signal = M) -> list[str]:
    return [m.batch_id for m in spool.list_batches(PROFILE, DEST, signal)]


# --- write: exact bytes, idempotency, conflict, validation -----------------------------------


def test_write_batch_persists_exact_body_and_meta_derived_from_the_bytes(tmp_path: Path) -> None:
    batch = make(7, 2, total=3, signal=L, items=2)
    meta = put(batch)
    body = folder(tmp_path, L) / f"{batch.batch_id}.json"
    assert body.read_bytes() == batch.body
    expected = Meta(
        batch_id=batch.batch_id,
        destination_id=DEST,
        signal=L,
        snapshot_id=snap(7),
        generation=7,
        index=2,
        total=3,
        kind="data",
        created_utc=NOW_Z,
        bytes=len(batch.body),
        sha256=hashlib.sha256(batch.body).hexdigest(),
        items=2,
        record_ids=batch.record_ids,
        scope=SCOPE,
    )
    assert meta == expected
    assert spool.read_meta(PROFILE, DEST, L, batch.batch_id) == expected
    on_disk = json.loads((folder(tmp_path, L) / f"{batch.batch_id}.meta.json").read_text())
    assert on_disk["schema"] == 1 and on_disk["state"] == "queued" and on_disk["attempts"] == 0
    assert on_disk["sha256"] == hashlib.sha256(batch.body).hexdigest()
    assert on_disk["record_ids"] == list(batch.record_ids)
    assert spool.read_body(PROFILE, DEST, L, batch.batch_id) == batch.body


def test_identical_rewrite_returns_the_existing_meta_without_resetting_it(tmp_path: Path) -> None:
    batch = make()
    put(batch)
    advanced = spool.update_meta(
        PROFILE, DEST, M, batch.batch_id, lambda m: replace(m, state="acked", attempts=3)
    )
    meta_file = folder(tmp_path) / f"{batch.batch_id}.meta.json"
    before = meta_file.read_bytes(), tree(folder(tmp_path))
    again = spool.write_batch(
        PROFILE, DEST, M, batch, created_utc="2027-01-01T00:00:00Z", scope={"other": 1}
    )
    assert again == advanced
    assert (again.created_utc, again.state, again.attempts) == (NOW_Z, "acked", 3)
    assert (meta_file.read_bytes(), tree(folder(tmp_path))) == before


def test_different_bytes_for_the_same_id_conflict_and_change_nothing(tmp_path: Path) -> None:
    batch = make()
    put(batch)
    before = tree(folder(tmp_path))
    changed = replace(batch, body=b'{"n":999}')
    with pytest.raises(BatchConflict):
        put(changed)
    assert tree(folder(tmp_path)) == before
    assert spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body


def test_orphan_body_with_different_bytes_conflicts_and_is_left_for_the_sweep(
    tmp_path: Path,
) -> None:
    batch = make()
    directory = folder(tmp_path)
    directory.parent.mkdir(parents=True, mode=0o700)
    directory.mkdir(mode=0o700)
    orphan = directory / f"{batch.batch_id}.json"
    orphan.write_bytes(b'{"older":true}')
    orphan.chmod(0o600)
    with pytest.raises(BatchConflict):
        put(batch)
    assert orphan.read_bytes() == b'{"older":true}' and ids() == []
    assert spool.sweep_orphans(PROFILE, DEST, M) == 1
    put(batch)
    assert spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(make(body=b"x" * (MAX_BATCH_BYTES + 1)), id="too-large"),
        pytest.param(make(body=b""), id="empty"),
        pytest.param(make(body=b"not json"), id="not-json"),
        pytest.param(make(body=b"[1,2]"), id="json-array"),
        pytest.param(make(body=b'"text"'), id="json-string"),
        pytest.param(make(index=1, total=1), id="index-not-below-total"),
        pytest.param(make(kind="weird"), id="bad-kind"),
        pytest.param(make(kind="tombstone"), id="tombstone-for-metrics"),
        pytest.param(make(batch_id="../evil-0000"), id="path-id"),
        pytest.param(make(batch_id="g1-abc-0"), id="short-id"),
        pytest.param(make(batch_id="G0000000001-0001000100010001-0000"), id="uppercase-id"),
        pytest.param(make(signal=L), id="signal-mismatch"),
    ],
)
def test_invalid_batches_are_rejected_and_nothing_is_written(tmp_path: Path, bad: Batch) -> None:
    with pytest.raises(ValueError):
        spool.write_batch(PROFILE, DEST, M, bad, created_utc=NOW_Z, scope=SCOPE)
    assert tree(tmp_path / "data") == {}


@pytest.mark.parametrize(
    "scope",
    [
        {"ratio": 0.5},
        {"nested": [{"deep": [1.0]}]},
        {"blob": "x" * (16 * 1024)},
        {"obj": object()},
        {1: "int key"},
    ],
    ids=["float", "nested-float", "over-16KiB", "non-json", "non-str-key"],
)
def test_invalid_scope_is_rejected_before_anything_is_written(
    tmp_path: Path, scope: dict[Any, Any]
) -> None:
    with pytest.raises(ValueError):
        put(make(), scope=scope)
    assert tree(tmp_path / "data") == {}


def test_scope_is_capped_at_exactly_16_kib_canonical_bytes_and_is_copied(tmp_path: Path) -> None:
    def canonical(blob: str) -> int:
        return len(
            json.dumps({"blob": blob, "n": [1, None, True]}, separators=(",", ":"), sort_keys=True)
        )

    fill = 16 * 1024 - canonical("")
    scope: dict[str, object] = {"blob": "y" * fill, "n": [1, None, True]}
    assert canonical("y" * fill) == 16 * 1024
    meta = put(make(), scope=scope)
    scope["blob"] = "mutated"
    assert meta.scope["blob"] == "y" * fill
    assert spool.read_meta(PROFILE, DEST, M, make().batch_id) == meta
    with pytest.raises(ValueError):
        put(make(2), scope={"blob": "y" * (fill + 1), "n": [1, None, True]})
    assert ids() == [make().batch_id]


@pytest.mark.parametrize(
    ("profile", "destination"),
    [("../x", DEST), (PROFILE, "../x"), (PROFILE, "UPPER"), ("a/b", DEST), (PROFILE, "")],
)
def test_profile_and_destination_are_identifiers_never_paths(
    tmp_path: Path, profile: str, destination: str
) -> None:
    with pytest.raises(ValueError):
        spool.write_batch(profile, destination, M, make(), created_utc=NOW_Z, scope=SCOPE)
    with pytest.raises(ValueError):
        spool.list_batches(profile, destination, M)
    with pytest.raises(ValueError):
        spool.usage(profile, destination)
    assert tree(tmp_path / "data") == {}


@pytest.mark.parametrize("batch_id", ["../../victim", "g1", "x" * 64, "g0000000001-zz-0000", ""])
def test_only_well_formed_batch_ids_ever_become_filenames(tmp_path: Path, batch_id: str) -> None:
    sentinel = tmp_path / "data" / "accounting" / PROFILE / "spool" / "victim.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"keep")
    put(make())
    for call in (spool.read_meta, spool.read_body, spool.delete_batch):
        with pytest.raises(ValueError):
            call(PROFILE, DEST, M, batch_id)
    with pytest.raises(ValueError):
        spool.update_meta(PROFILE, DEST, M, batch_id, lambda m: m)
    assert sentinel.read_bytes() == b"keep"


# --- crash safety and durability ordering -----------------------------------------------------


def test_crash_between_body_and_meta_leaves_an_orphan_that_the_sweep_removes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = make()
    real = storage.atomic_write

    def crashing(dir_fd: int, name: str, data: bytes) -> None:
        if name.endswith(".meta.json"):
            raise OSError("power cut")
        real(dir_fd, name, data)

    monkeypatch.setattr(storage, "atomic_write", crashing)
    with pytest.raises(OSError, match="power cut"):
        put(batch)
    monkeypatch.setattr(storage, "atomic_write", real)
    assert names(folder(tmp_path)) == [f"{batch.batch_id}.json"]
    assert ids() == [] and spool.read_meta(PROFILE, DEST, M, batch.batch_id) is None
    assert spool.sweep_orphans(PROFILE, DEST, M) == 1
    assert names(folder(tmp_path)) == []
    meta = put(batch)
    assert spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body
    assert ids() == [meta.batch_id]
    assert spool.sweep_orphans(PROFILE, DEST, M) == 0  # positive twin: a healthy batch survives


def test_orphan_body_with_identical_bytes_is_adopted_by_a_retry_without_a_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = make()
    real = storage.atomic_write
    monkeypatch.setattr(
        storage,
        "atomic_write",
        lambda *a: (_ for _ in ()).throw(OSError("power cut")),
    )
    with pytest.raises(OSError):
        put(batch)
    monkeypatch.setattr(storage, "atomic_write", real)
    put(batch)
    assert ids() == [batch.batch_id]
    assert spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body


def test_failure_while_writing_the_body_leaves_no_final_named_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = make()
    real = storage.write_all

    def failing(descriptor: int, data: bytes) -> None:
        real(descriptor, data[:5])
        raise OSError(28, "disk full")

    monkeypatch.setattr(storage, "write_all", failing)
    with pytest.raises(OSError, match="disk full"):
        put(batch)
    monkeypatch.setattr(storage, "write_all", real)
    assert names(folder(tmp_path)) == []  # no body, no meta, no leaked temporary
    assert ids() == [] and spool.corrupt_batches(PROFILE, DEST, M) == []
    put(batch)
    assert spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body


def test_body_is_fsynced_and_linked_and_its_directory_fsynced_before_the_meta_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = make()
    events: list[tuple[Any, ...]] = []
    real_fsync, real_link, real_rename = os.fsync, os.link, os.rename

    def fsync(fd: int) -> None:
        info = os.fstat(fd)
        events.append(
            ("fsync-dir",) if stat.S_ISDIR(info.st_mode) else ("fsync-file", info.st_size)
        )
        real_fsync(fd)

    def link(src: Any, dst: Any, **kwargs: Any) -> None:
        events.append(("link", dst))
        real_link(src, dst, **kwargs)

    def rename(src: Any, dst: Any, **kwargs: Any) -> None:
        events.append(("rename", dst))
        real_rename(src, dst, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(os, "rename", rename)
    put(batch)
    meta_name = f"{batch.batch_id}.meta.json"
    meta_size = (folder(tmp_path) / meta_name).stat().st_size
    assert events == [
        ("fsync-file", len(batch.body)),  # temporary body durable ...
        ("link", f"{batch.batch_id}.json"),  # ... then published under its final name ...
        ("fsync-dir",),  # ... and the directory entry durable ...
        ("fsync-file", meta_size),  # ... BEFORE the meta even starts
        ("rename", meta_name),
        ("fsync-dir",),
    ]


# --- integrity --------------------------------------------------------------------------------


def test_read_body_returns_the_exact_bytes_and_refuses_a_flipped_byte(tmp_path: Path) -> None:
    batch = make(body=b'{"resourceMetrics":[{"v":"abcdef"}]}')
    put(batch)
    path = folder(tmp_path) / f"{batch.batch_id}.json"
    assert spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body
    flipped = bytearray(batch.body)
    flipped[-5] ^= 0x01  # same size, different content
    path.write_bytes(bytes(flipped))
    with pytest.raises(SpoolCorrupt):
        spool.read_body(PROFILE, DEST, M, batch.batch_id)
    path.write_bytes(batch.body)  # restored: readable again (the meta was never the problem)
    assert spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body


@pytest.mark.parametrize("damage", ["truncate", "append", "empty", "delete"])
def test_a_damaged_body_is_never_returned_and_is_reported_corrupt(
    tmp_path: Path, damage: str
) -> None:
    batch = make()
    put(batch)
    healthy = make(2)
    put(healthy)
    path = folder(tmp_path) / f"{batch.batch_id}.json"
    if damage == "truncate":
        path.write_bytes(batch.body[:-3])
    elif damage == "append":
        path.write_bytes(batch.body + b" ")
    elif damage == "empty":
        path.write_bytes(b"")
    else:
        path.unlink()
    with pytest.raises(SpoolCorrupt):
        spool.read_body(PROFILE, DEST, M, batch.batch_id)
    assert spool.corrupt_batches(PROFILE, DEST, M) == [batch.batch_id]
    assert spool.read_body(PROFILE, DEST, M, healthy.batch_id) == healthy.body


def test_read_body_of_an_unknown_batch_is_file_not_found_not_corruption() -> None:
    with pytest.raises(FileNotFoundError):
        spool.read_body(PROFILE, DEST, M, make().batch_id)
    put(make(2))
    with pytest.raises(FileNotFoundError):
        spool.read_body(PROFILE, DEST, M, make().batch_id)


def test_symlinked_body_is_refused_and_its_target_is_never_read_or_removed(
    tmp_path: Path,
) -> None:
    batch = make()
    put(batch)
    target = tmp_path / "outside.json"
    target.write_bytes(batch.body)  # identical bytes: only the symlink itself can betray it
    path = folder(tmp_path) / f"{batch.batch_id}.json"
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(SpoolCorrupt):
        spool.read_body(PROFILE, DEST, M, batch.batch_id)
    assert spool.corrupt_batches(PROFILE, DEST, M) == [batch.batch_id]
    with pytest.raises(SpoolCorrupt):
        spool.delete_batch(PROFILE, DEST, M, batch.batch_id)
    with pytest.raises(SpoolCorrupt):
        put(batch)
    assert target.read_bytes() == batch.body and path.is_symlink()


def test_symlinked_spool_directory_is_refused_and_nothing_outside_is_touched(
    tmp_path: Path,
) -> None:
    batch = make()
    put(batch)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / f"{batch.batch_id}.json"
    victim_meta = outside / f"{batch.batch_id}.meta.json"
    stray = outside / ".g.tmp"
    victim.write_bytes(b'{"victim":true}')
    victim_meta.write_bytes(b"{}")
    stray.write_bytes(b"x")
    real = folder(tmp_path)
    moved = tmp_path / "moved-away"
    real.rename(moved)
    real.symlink_to(outside, target_is_directory=True)
    before = tree(outside)
    calls: list[Callable[[], object]] = [
        lambda: put(make(2)),
        lambda: spool.list_batches(PROFILE, DEST, M),
        lambda: spool.read_meta(PROFILE, DEST, M, batch.batch_id),
        lambda: spool.read_body(PROFILE, DEST, M, batch.batch_id),
        lambda: spool.delete_batch(PROFILE, DEST, M, batch.batch_id),
        lambda: spool.sweep_orphans(PROFILE, DEST, M),
        lambda: spool.drop_destination(PROFILE, DEST, M),
        lambda: spool.update_meta(PROFILE, DEST, M, batch.batch_id, lambda m: m),
        lambda: spool.corrupt_batches(PROFILE, DEST, M),
    ]
    for call in calls:
        with pytest.raises(SpoolCorrupt):
            call()
    assert tree(outside) == before
    assert victim.read_bytes() == b'{"victim":true}' and victim_meta.read_bytes() == b"{}"
    assert (moved / f"{batch.batch_id}.json").read_bytes() == batch.body  # not swept through it
    assert spool.iter_spools(PROFILE) == []  # a symlinked spool is not a spool


# --- update_meta ------------------------------------------------------------------------------


def _bump(profile: str, destination: str, batch_id: str, count: int, ready: Any) -> None:
    ready.wait(30)
    for _ in range(count):
        spool.update_meta(
            profile, destination, M, batch_id, lambda m: replace(m, attempts=m.attempts + 1)
        )


def test_two_processes_updating_concurrently_lose_no_updates(tmp_path: Path) -> None:
    batch = make()
    put(batch)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    workers = [
        context.Process(target=_bump, args=(PROFILE, DEST, batch.batch_id, 25, ready))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    ready.set()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    meta = spool.read_meta(PROFILE, DEST, M, batch.batch_id)
    assert meta is not None and meta.attempts == 50
    on_disk = json.loads((folder(tmp_path) / f"{batch.batch_id}.meta.json").read_text())
    assert on_disk["attempts"] == 50 and on_disk["state"] == "queued"
    assert spool.read_body(PROFILE, DEST, M, batch.batch_id) == batch.body


def test_mutable_progress_fields_update_and_persist() -> None:
    batch = make()
    put(batch)
    updated = spool.update_meta(
        PROFILE,
        DEST,
        M,
        batch.batch_id,
        lambda m: replace(
            m,
            state="partial",
            attempts=2,
            last_attempt_utc="2026-10-13T12:01:00Z",
            next_attempt_utc="2026-10-13T12:06:00Z",
            last_status=429,
            last_category="rate_limited",
            rejected=3,
            warning="collector rejected 3 points",
        ),
    )
    assert spool.read_meta(PROFILE, DEST, M, batch.batch_id) == updated
    assert (updated.state, updated.attempts, updated.last_status, updated.rejected) == (
        "partial",
        2,
        429,
        3,
    )
    assert updated.warning == "collector rejected 3 points"


IDENTITY_CHANGES: dict[str, Any] = {
    "batch_id": "g0000000002-0002000200020002-0000",
    "destination_id": "other-dest",
    "signal": M,
    "snapshot_id": "ff" * 32,
    "generation": 99,
    "index": 1,
    "total": 4,
    "kind": "tombstone",
    "created_utc": "2026-01-01T00:00:00Z",
    "bytes": 5,
    "sha256": "0" * 64,
    "items": 9,
    "record_ids": ("b" * 32,),
    "scope": {"changed": 1},
}


@pytest.mark.parametrize("field", sorted(IDENTITY_CHANGES))
def test_identity_fields_are_immutable(tmp_path: Path, field: str) -> None:
    batch = make(3, 0, total=3, signal=L, items=1)
    put(batch)
    path = folder(tmp_path, L) / f"{batch.batch_id}.meta.json"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        spool.update_meta(
            PROFILE,
            DEST,
            L,
            batch.batch_id,
            lambda m: replace(m, **{field: IDENTITY_CHANGES[field]}),
        )
    assert path.read_bytes() == before


def test_every_identity_field_is_covered_by_the_immutability_test() -> None:
    from ssgrep.accounting.export_spool_meta import IDENTITY

    assert sorted(IDENTITY) == sorted(IDENTITY_CHANGES)


@pytest.mark.parametrize("state", ["sent", "", "QUEUED", "done"])
def test_invalid_state_is_rejected_and_every_valid_state_is_accepted(
    tmp_path: Path, state: str
) -> None:
    batch = make()
    put(batch)
    path = folder(tmp_path) / f"{batch.batch_id}.meta.json"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        spool.update_meta(PROFILE, DEST, M, batch.batch_id, lambda m: replace(m, state=state))
    assert path.read_bytes() == before
    for good in spool.STATES:
        moved = spool.update_meta(
            PROFILE, DEST, M, batch.batch_id, lambda m, s=good: replace(m, state=s)
        )
        assert moved.state == good


def test_update_of_an_absent_batch_or_spool_is_file_not_found_and_creates_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        spool.update_meta(PROFILE, DEST, M, make().batch_id, lambda m: m)
    assert tree(tmp_path / "data") == {}
    put(make(2))
    with pytest.raises(FileNotFoundError):
        spool.update_meta(PROFILE, DEST, M, make().batch_id, lambda m: m)


# --- strict stored-meta codec -----------------------------------------------------------------

OTHER_ID = "g0000000002-0002000200020002-0000"


def _doc(mutate: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], bytes]:
    def build(doc: dict[str, Any]) -> bytes:
        mutate(doc)
        return json.dumps(doc).encode()

    return build


def _set(key: str, value: Any) -> Callable[[dict[str, Any]], bytes]:
    return _doc(lambda doc: doc.__setitem__(key, value))


def _drop(key: str) -> Callable[[dict[str, Any]], bytes]:
    return _doc(lambda doc: doc.pop(key))


CODEC_CASES: dict[str, Callable[[dict[str, Any]], bytes]] = {
    "unknown-key": _set("extra", 1),
    "float": _set("attempts", 1.5),
    "integral-float": _set("attempts", 1.0),
    "nan": _set("attempts", float("nan")),
    "float-in-scope": _set("scope", {"ratio": 0.5}),
    "bool-attempts": _set("attempts", True),
    "bool-generation": _set("generation", True),
    "bool-rejected": _set("rejected", False),
    "bool-schema": _set("schema", True),
    "newer-schema": _set("schema", 2),
    "older-schema": _set("schema", 0),
    "missing-identity": _drop("sha256"),
    "wrong-batch-id": _set("batch_id", OTHER_ID),
    "wrong-destination": _set("destination_id", "elsewhere"),
    "wrong-signal": _set("signal", "logs"),
    "bad-state": _set("state", "sent"),
    "bad-time": _set("created_utc", "2026-02-30T00:00:00Z"),
    "bad-sha": _set("sha256", "XYZ"),
    "negative-attempts": _set("attempts", -1),
    "string-attempts": _set("attempts", "1"),
    "record-ids-not-list": _set("record_ids", "abc"),
    "tombstone-for-metrics": _set("kind", "tombstone"),
    "index-past-total": _set("index", 5),
    "not-object": lambda doc: b"[1]",
    "not-json": lambda doc: b"{nope",
    "empty": lambda doc: b"",
    "duplicate-key": lambda doc: json.dumps(doc)[:-1].encode() + b',"attempts":7}',
}


@pytest.mark.parametrize("case", sorted(CODEC_CASES))
def test_a_stored_meta_outside_schema_1_is_corrupt_and_never_rewritten(
    tmp_path: Path, case: str
) -> None:
    batch = make()
    meta = put(batch)
    healthy = put(make(2))
    path = folder(tmp_path) / f"{batch.batch_id}.meta.json"
    good = json.loads(path.read_text())
    assert good["schema"] == 1
    path.write_bytes(CODEC_CASES[case](good))
    broken = path.read_bytes()
    with pytest.raises(SpoolCorrupt):
        spool.read_meta(PROFILE, DEST, M, batch.batch_id)
    with pytest.raises(SpoolCorrupt):
        spool.read_body(PROFILE, DEST, M, batch.batch_id)
    with pytest.raises(SpoolCorrupt):
        spool.update_meta(PROFILE, DEST, M, batch.batch_id, lambda m: replace(m, attempts=1))
    assert path.read_bytes() == broken  # left untouched for a newer/older reader or an operator
    assert spool.list_batches(PROFILE, DEST, M) == [healthy]  # skipped, not fatal
    assert spool.corrupt_batches(PROFILE, DEST, M) == [batch.batch_id]
    assert meta != healthy


def test_a_meta_round_trips_byte_for_byte_through_the_codec(tmp_path: Path) -> None:
    batch = make(4, 1, total=2, signal=L, items=3)
    put(batch)
    path = folder(tmp_path, L) / f"{batch.batch_id}.meta.json"
    before = path.read_bytes()
    spool.update_meta(PROFILE, DEST, L, batch.batch_id, lambda m: replace(m))
    assert path.read_bytes() == before
    spool.update_meta(PROFILE, DEST, L, batch.batch_id, lambda m: replace(m, attempts=1))
    after = json.loads(path.read_text())
    assert after == {**json.loads(before), "attempts": 1}
    assert spool.corrupt_batches(PROFILE, DEST, L) == []


# --- listing, usage, deletion -----------------------------------------------------------------


def test_list_batches_is_ordered_by_generation_then_data_before_tombstone_then_index(
    tmp_path: Path,
) -> None:
    tombstone = make(10, kind="tombstone", signal=L)
    for batch in (
        make(11, 0, signal=L),
        make(10, 1, total=2, signal=L),
        tombstone,
        make(9, 0, signal=L),
        make(10, 0, total=2, signal=L),
    ):
        put(batch)
    assert ids(L) == [
        f"g{9:010d}-{snap(9)[:16]}-0000",
        f"g{10:010d}-{snap(10)[:16]}-0000",
        f"g{10:010d}-{snap(10)[:16]}-0001",
        tombstone.batch_id,
        f"g{11:010d}-{snap(11)[:16]}-0000",
    ]
    assert ids(M) == []  # the other signal's spool is separate (and was never created)
    assert not folder(tmp_path, M).exists()


def test_usage_equals_the_bytes_and_batches_actually_on_disk(tmp_path: Path) -> None:
    assert spool.usage(PROFILE, DEST) == Usage(0, 0)
    for gen in (1, 2, 3):
        put(make(gen, body=json.dumps({"gen": gen, "pad": "z" * (gen * 300)}).encode()))
    put(make(4, signal=L))
    put(make(1, signal=L), dest="other-dest")

    def disk(signal: Signal, dest: str = DEST) -> int:
        return sum(
            os.stat(p).st_size
            for p in folder(tmp_path, signal, dest).iterdir()
            if p.name != ".spool.lock"
        )

    assert spool.usage(PROFILE, DEST, M) == Usage(3, disk(M))
    assert spool.usage(PROFILE, DEST, L) == Usage(1, disk(L))
    assert spool.usage(PROFILE, DEST) == Usage(4, disk(M) + disk(L))
    assert spool.usage(PROFILE, "other-dest") == Usage(1, disk(L, "other-dest"))
    assert spool.usage(PROFILE, "unknown-dest") == Usage(0, 0)
    bodies = sum(len(spool.read_body(PROFILE, DEST, M, i)) for i in ids(M))
    metas = sum((folder(tmp_path) / f"{i}.meta.json").stat().st_size for i in ids(M))
    assert metas > 0 and spool.usage(PROFILE, DEST, M).bytes == bodies + metas
    spool.delete_batch(PROFILE, DEST, M, ids(M)[0])
    assert spool.usage(PROFILE, DEST, M) == Usage(2, disk(M))


def test_delete_batch_removes_meta_then_body_and_fsyncs_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keep, gone = make(1), make(2)
    put(keep)
    put(gone)
    events: list[str] = []
    real_unlink, real_fsync = os.unlink, os.fsync

    def unlink(path: str, **kwargs: Any) -> None:
        events.append(str(path))
        real_unlink(path, **kwargs)

    def fsync(fd: int) -> None:
        events.append("fsync-dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync-file")
        real_fsync(fd)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "fsync", fsync)
    assert spool.delete_batch(PROFILE, DEST, M, gone.batch_id) is True
    monkeypatch.setattr(os, "unlink", real_unlink)
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert events == [f"{gone.batch_id}.meta.json", f"{gone.batch_id}.json", "fsync-dir"]
    assert names(folder(tmp_path)) == [f"{keep.batch_id}.json", f"{keep.batch_id}.meta.json"]
    assert spool.delete_batch(PROFILE, DEST, M, gone.batch_id) is False
    assert spool.delete_batch(PROFILE, "unknown-dest", M, gone.batch_id) is False
    assert spool.read_body(PROFILE, DEST, M, keep.batch_id) == keep.body


# --- sweep, purge, discovery, permissions -----------------------------------------------------


def test_sweep_removes_orphan_bodies_leaked_temporaries_and_bodyless_metas_only(
    tmp_path: Path,
) -> None:
    healthy, orphan_body, orphan_meta = make(1), make(2), make(3)
    put(healthy)
    put(orphan_body)
    put(orphan_meta)
    directory = folder(tmp_path)
    (directory / f"{orphan_body.batch_id}.meta.json").unlink()
    (directory / f"{orphan_meta.batch_id}.json").unlink()
    leaked = directory / f".{healthy.batch_id}.json.123.abcd1234.tmp"
    leaked.write_bytes(b"half written")
    unrelated = [directory / "README.txt", directory / "notes.json", directory / "g1.json"]
    for path in unrelated:
        path.write_bytes(b"not ours")
    assert spool.sweep_orphans(PROFILE, DEST, M) == 3
    assert names(directory) == sorted(
        [f"{healthy.batch_id}.json", f"{healthy.batch_id}.meta.json", *(p.name for p in unrelated)]
    )
    assert all(p.read_bytes() == b"not ours" for p in unrelated)
    assert spool.read_body(PROFILE, DEST, M, healthy.batch_id) == healthy.body
    assert spool.sweep_orphans(PROFILE, DEST, M) == 0
    assert spool.sweep_orphans(PROFILE, "unknown-dest", M) == 0
    assert not folder(tmp_path, M, "unknown-dest").exists()


def test_drop_destination_purges_every_batch_of_that_destination_only(tmp_path: Path) -> None:
    for gen in (1, 2):
        put(make(gen))
    put(make(1, signal=L))
    put(make(5), dest="other-dest")
    (folder(tmp_path) / f".{make(1).batch_id}.json.9.deadbeef.tmp").write_bytes(b"leak")
    orphan = make(9)
    put(orphan)
    (folder(tmp_path) / f"{orphan.batch_id}.meta.json").unlink()  # an orphan body counts as a batch
    assert spool.drop_destination(PROFILE, DEST, L) == 1
    assert ids(M) == [make(1).batch_id, make(2).batch_id] and ids(L) == []
    assert spool.drop_destination(PROFILE, DEST) == 3
    assert names(folder(tmp_path)) == [] and names(folder(tmp_path, L)) == []
    assert [m.batch_id for m in spool.list_batches(PROFILE, "other-dest", M)] == [make(5).batch_id]
    assert spool.drop_destination(PROFILE, DEST) == 0
    assert spool.drop_destination(PROFILE, "unknown-dest") == 0


def test_iter_spools_lists_existing_spools_and_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "data"
    assert spool.iter_spools(PROFILE) == [] and tree(root) == {}
    put(make(1))
    put(make(1, signal=L), dest="dest-b")
    put(make(2), dest="dest-b")
    spools = tmp_path / "data" / "accounting" / PROFILE / "spool"
    (spools / "stray-file").write_bytes(b"x")
    (spools / "BAD NAME").mkdir(mode=0o700)
    (spools / "not-a-signal-dest").mkdir(mode=0o700)
    (spools / "not-a-signal-dest" / "traces").mkdir(mode=0o700)
    (spools / "linked").symlink_to(tmp_path, target_is_directory=True)
    before = tree(root)
    assert sorted(spool.iter_spools(PROFILE), key=lambda t: (t[0], t[1].value)) == [
        ("dest-b", L),
        ("dest-b", M),
        (DEST, M),
    ]
    assert spool.iter_spools("unknown-profile") == [] and spool.iter_spools("../x") == []
    for call in (
        lambda: spool.list_batches(PROFILE, "ghost", M),
        lambda: spool.read_meta(PROFILE, "ghost", M, make().batch_id),
        lambda: spool.corrupt_batches(PROFILE, "ghost", L),
        lambda: spool.usage(PROFILE, "ghost"),
        lambda: spool.sweep_orphans(PROFILE, "ghost", M),
        lambda: spool.drop_destination(PROFILE, "ghost"),
        lambda: spool.delete_batch(PROFILE, "ghost", M, make().batch_id),
    ):
        call()
    assert tree(root) == before


def test_files_are_0600_and_directories_0700(tmp_path: Path) -> None:
    batch = make(1, signal=L)
    put(batch)
    spool.update_meta(PROFILE, DEST, L, batch.batch_id, lambda m: replace(m, attempts=1))
    base = tmp_path / "data" / "accounting"
    entries = [base, *base.rglob("*")]
    assert len(entries) >= 8
    for path in entries:
        mode = stat.S_IMODE(path.lstat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), path
    assert {p.name for p in folder(tmp_path, L).iterdir()} == {
        f"{batch.batch_id}.json",
        f"{batch.batch_id}.meta.json",
        ".spool.lock",
    }
