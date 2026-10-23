"""File-level steps of a purge, all through ``ssgrep.accounting.privacy.storage`` descriptors.

Only names this product writes are ever touched: receipts ``<id>.json``, snapshot
``gen-*.json``/``current.json``, journal ``seg-*.jsonl`` and the purge marker. Every directory is
opened component by component with ``O_NOFOLLOW`` and ownership checks; ``preflight`` refuses the
whole run when anything under the profile is a symlink, foreign-owned or special, so a hostile
link can never redirect a deletion outside ``accounting/<profile>/``.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting.capture import fingerprint
from ssgrep.accounting.capture.journal import SEGMENT_PREFIX, SEGMENT_SUFFIX
from ssgrep.accounting.capture.journal_codec import MalformedEvent, parse_event
from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.cycle_publish import history_document, write_history, write_receipts
from ssgrep.accounting.collect.replay import evidence_from_line
from ssgrep.accounting.collect.replay_evidence import MAX_CACHE_BYTES
from ssgrep.accounting.collect.state import VersionedLedger, load_state
from ssgrep.accounting.collect.steps import archive_retired_rows
from ssgrep.accounting.collect.store import dumps, loads
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.published import ARCHIVE_PREFIX
from ssgrep.accounting.privacy.purge import PurgeError, Scoper, Selection, evidence_cwd
from ssgrep.accounting.privacy.purge_spool import release_pins
from ssgrep.accounting.views import snapshots

MARKER = "purge-pending.json"
MAX_JOURNAL_SEGMENT = 256 * 1024 * 1024
_MAX_DEPTH = 8
_GEN = re.compile(r"gen-(\d{10})-([0-9a-f]{16,64})\.json")


def unsafe() -> PurgeError:
    return PurgeError(
        "storage_unsafe",
        "Accounting storage is not private to this user (a symlink, a foreign-owned or a "
        "special entry was found). Nothing was changed.",
    )


def _walk(fd: int, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise unsafe()
    for name in os.listdir(fd):
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode) or info.st_uid != os.geteuid():
            raise unsafe()
        if stat.S_ISDIR(info.st_mode):
            if info.st_mode & 0o077:
                raise unsafe()
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            child = os.open(name, flags, dir_fd=fd)
            try:
                _walk(child, depth + 1)
            finally:
                os.close(child)
        elif not stat.S_ISREG(info.st_mode) or info.st_mode & 0o177:
            raise unsafe()


def preflight(profile: str) -> bool:
    """Read-only: False when the profile has no accounting directory; else prove it is safe."""
    try:
        root = storage.open_profile_dir(profile, create=False)
    except FileNotFoundError:
        return False
    except (storage.UnsafeStorageError, OSError):
        raise unsafe() from None
    try:
        _walk(root, 0)
    except OSError:
        raise unsafe() from None
    finally:
        os.close(root)
    return True


def _dir(profile: str, *parts: str) -> int | None:
    try:
        return storage.open_profile_dir(profile, *parts, create=False)
    except FileNotFoundError:
        return None
    except (storage.UnsafeStorageError, OSError):
        raise unsafe() from None


# ---- receipts ---------------------------------------------------------------------------------


def receipt_ids(profile: str) -> set[str]:
    fd = _dir(profile, "receipts")
    if fd is None:
        return set()
    try:
        return {
            n[:-5] for n in storage.list_names(fd) if n.endswith(".json") and not n.startswith(".")
        }
    finally:
        os.close(fd)


def delete_receipts(profile: str, ids: set[str]) -> int:
    fd = _dir(profile, "receipts")
    if fd is None:
        return 0
    try:
        return sum(storage.delete_file(fd, f"{rid}.json") for rid in sorted(ids))
    finally:
        os.close(fd)


# ---- snapshots --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    number: int
    snapshot_id: str
    name: str


def generation_files(profile: str) -> list[Generation]:
    fd = _dir(profile, "snapshots")
    if fd is None:
        return []
    try:
        found = (_GEN.fullmatch(n) for n in storage.list_names(fd))
        return sorted(
            (Generation(int(m.group(1)), m.group(2), m.group(0)) for m in found if m),
            key=lambda g: (g.number, g.name),
        )
    finally:
        os.close(fd)


def delete_generations(profile: str, doomed: list[Generation]) -> int:
    """Delete the named generation files (never ``current.json``)."""
    fd = _dir(profile, "snapshots")
    if fd is None:
        return 0
    try:
        removed = sum(storage.delete_file(fd, g.name) for g in doomed)
        if removed:
            storage.fsync_dir(fd)
        return removed
    finally:
        os.close(fd)


# ---- the pending marker (crash safety) --------------------------------------------------------


def read_marker(profile: str) -> int | None:
    """The generation a purge was about to commit, None when no purge is unfinished."""
    fd = _dir(profile, "collector")
    if fd is None:
        return None
    try:
        data = storage.read_file(fd, MARKER, 4096)
    except (ValueError, OSError):
        raise unsafe() from None
    finally:
        os.close(fd)
    if data is None:
        return None
    try:
        generation = loads(data)["generation"]
        if type(generation) is not int or not 1 <= generation <= snapshots.MAX_GENERATION:
            raise ValueError("invalid purge generation")
    except (ValueError, KeyError, TypeError, RecursionError):
        raise PurgeError(
            "state_corrupt", "Purge recovery marker is unreadable; purge stopped safely."
        ) from None
    return generation


def write_marker(profile: str, generation: int) -> None:
    fd = storage.open_profile_dir(profile, "collector")
    try:
        storage.atomic_write(fd, MARKER, dumps({"generation": generation}) + b"\n")
    finally:
        os.close(fd)


def clear_marker(profile: str) -> None:
    fd = _dir(profile, "collector")
    if fd is None:
        return
    try:
        storage.delete_file(fd, MARKER)
    finally:
        os.close(fd)


# ---- journal segments -------------------------------------------------------------------------


def _segment_old(
    data: bytes, mtime: float, cutoff_day: str, cutoff_epoch: float, scoper: Scoper
) -> bool:
    """Every event strictly before the cutoff day (and inside the workspace) and a quiet file."""
    if mtime >= cutoff_epoch:
        return False
    if data and not data.endswith(b"\n"):
        return False  # an unterminated tail: a writer may still be appending
    for line in data.split(b"\n"):
        if not line:
            continue
        try:
            event = parse_event(line)
        except (MalformedEvent, ValueError):
            return False
        if event.ts_utc[:10] >= cutoff_day or not scoper.inside(event.cwd):
            return False
    return True


def plan_journal(profile: str, cutoff_day: str, scoper: Scoper) -> tuple[tuple[str, ...], int]:
    """(segment names to delete, segments retained because they hold newer/other events)."""
    fd = _dir(profile, "journal")
    if fd is None:
        return (), 0
    cutoff_epoch = (
        datetime.fromisoformat(f"{cutoff_day}T00:00:00+00:00").astimezone(UTC).timestamp()
    )
    doomed: list[str] = []
    retained = 0
    try:
        for name in storage.list_names(fd):
            if not (name.startswith(SEGMENT_PREFIX) and name.endswith(SEGMENT_SUFFIX)):
                continue
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                data = storage.read_file(fd, name, MAX_JOURNAL_SEGMENT)
            except ValueError:
                retained += 1
                continue
            except OSError:
                raise unsafe() from None
            if data is not None and _segment_old(
                data, info.st_mtime, cutoff_day, cutoff_epoch, scoper
            ):
                doomed.append(name)
            else:
                retained += 1
    finally:
        os.close(fd)
    return tuple(doomed), retained


def delete_journal(profile: str, names: tuple[str, ...]) -> int:
    fd = _dir(profile, "journal")
    if fd is None:
        return 0
    try:
        removed = sum(storage.delete_file(fd, name) for name in names)
        if removed:
            storage.fsync_dir(fd)
        return removed
    finally:
        os.close(fd)


@dataclass
class LegacyArchive:
    name: str
    raw: dict[str, Any]
    plan: Any


def plan_archives(
    profile: str, checkpoint: store.Checkpoint, sel: Selection
) -> list[LegacyArchive]:
    from ssgrep.accounting.privacy.purge_state import plan_state

    found = []
    for name, filename in checkpoint.blobs.items():
        if not name.startswith(ARCHIVE_PREFIX):
            continue
        raw = store.read_blob(profile, filename)
        if not isinstance(raw, dict) or raw.get("schema") != 1:
            raise PurgeError("state_corrupt", "Identity archive manifest is invalid.")
        if any(key.startswith(ARCHIVE_PREFIX) for key in raw.get("blobs", {})):
            raise PurgeError(
                "archive_scope_unresolved",
                "Nested identity archives require recovery before purge.",
            )
        if sel.workspace is not None and raw.get("root_mapping_status") != "available":
            raise PurgeError(
                "archive_scope_unresolved",
                "Legacy identity archive has no verified workspace/root mapping. "
                "Recover its mapping or request a whole-profile purge; nothing was changed.",
            )
        if raw.get("published_documents") is not None:
            from ssgrep.accounting.privacy.purge_published import plan_published

            found.append(LegacyArchive(name, raw, plan_published(profile, raw, sel)))
            continue
        blobs = dict(raw.get("blobs", {}))
        archived = store.Checkpoint(
            profile,
            methodology=str(raw.get("methodology", "")),
            blobs=blobs,
            segments=[store.SegmentRef.from_dict(ref) for ref in raw.get("segments", [])],
            collection_scope=dict(raw.get("collection_scope", {})),
        )
        planned = plan_state(profile, sel, archived)
        if planned is not None:
            found.append(LegacyArchive(name, raw, planned))
    return found


def write_archives(
    profile: str, archives: list[LegacyArchive], sel: Selection, blobs: dict[str, str]
) -> dict[str, str]:
    from ssgrep.accounting.privacy.purge_state import write_purged_state

    retained = {}
    for archive in archives:
        from ssgrep.accounting.privacy.purge_published import PublishedPlan, write_published

        if isinstance(archive.plan, PublishedPlan):
            retained.update(write_published(profile, archive.name, archive.plan, sel, blobs))
            continue
        if not archive.plan.changed:
            retained.update(archive.raw.get("blobs", {}))
            continue
        new = write_purged_state(profile, archive.plan, sel)
        has_rows = any(
            namespace.rows_after.periods
            or namespace.rows_after.receipts
            or namespace.archives_after
            for namespace in archive.plan.namespaces
        )
        has_invocations = bool(
            archive.plan.reduction_after and archive.plan.reduction_after.invocations
        )
        if not new.segments and not has_rows and not has_invocations:
            blobs.pop(archive.name, None)
            continue
        document = {
            **archive.raw,
            "blobs": new.blobs,
            "segments": [ref.to_dict() for ref in new.segments],
        }
        if archive.plan.reduction_after is not None:
            document["session_roots"] = list(archive.plan.reduction_after.session_roots)
            document["session_cwd"] = list(archive.plan.reduction_after.session_cwd)
        blobs[archive.name] = store.write_blob(profile, archive.name, document)
        retained.update(new.blobs)
    return retained


def row_receipt_ids(rows) -> set[str]:
    return {receipt.receipt_id for receipt in rows.receipts} | {
        receipt.receipt_id for archive in rows.archives for receipt in archive.receipts
    }


def retained_receipt_ids(archives: list[LegacyArchive]) -> set[str]:
    """Only original or canonical receipts surviving the selection remain public."""
    from ssgrep.accounting.privacy.purge_published import PublishedPlan

    kept = set()
    for archive in archives:
        if isinstance(archive.plan, PublishedPlan):
            kept.update(archive.plan.documents["receipt_blobs"])
        else:
            for namespace in archive.plan.namespaces:
                kept.update(row_receipt_ids(namespace.rows_after))
    return kept


@dataclass(frozen=True)
class Cleaned:
    receipt_files: int = 0
    snapshot_generations: int = 0
    delivery_pins_released: int = 0
    orphans_swept: int = 0


def _committed(profile: str, checkpoint: store.Checkpoint) -> VersionedLedger | None:
    state = load_state(profile, checkpoint, checkpoint.methodology)
    if state.ledger is None:
        return None
    ledger = VersionedLedger.restore_versioned(state.ledger, state.superseded, state.archives)
    archive_retired_rows(profile, ledger, checkpoint.blobs)
    return ledger


def older_generations(profile: str, committed: int) -> list:
    return [g for g in generation_files(profile) if g.number < committed]


def cleanup(
    profile: str,
    *,
    today: str,
    publish: bool,
    generations: bool,
    stale_ids: frozenset[str],
) -> Cleaned:
    """Remove stale derived files; ``publish`` also rewrites receipts/history from the ledger.

    ``generations`` deletes every snapshot older than the committed one (set after a purge
    committed a fresh generation, or when an unfinished purge is being completed).
    """
    checkpoint = store.load_checkpoint(profile)
    if checkpoint is None:
        return Cleaned()
    ledger = _committed(profile, checkpoint)
    from ssgrep.accounting.privacy.published import merge_history, restore_receipts

    # ``stale_ids`` is the purge's complete precommit deletion proof across namespaces; this
    # function never derives its own, so an absent active ledger cannot erase inactive history.
    receipts = delete_receipts(profile, set(stale_ids))
    if publish:
        rows = ledger.rows() if ledger else None
        current = (
            history_document(rows, ledger.methodology.units[0], today)
            if ledger is not None and rows is not None
            else []
        )
        document, _ = merge_history(profile, current, checkpoint.blobs)
        if rows is not None:
            write_receipts(profile, rows)
        restore_receipts(
            profile, checkpoint.blobs, {row.receipt_id for row in rows.receipts} if rows else set()
        )
        write_history(profile, document)
    removed = 0
    released = 0
    if ledger is not None and generations:
        removed = delete_generations(profile, older_generations(profile, checkpoint.generation))
        surviving = frozenset(g.snapshot_id for g in generation_files(profile))
        released = release_pins(profile, surviving)
    swept = store.sweep_orphans(profile, checkpoint) if publish else 0
    return Cleaned(receipts, removed, released, swept)


AUTHORITY_TOMBSTONE = "frequency-purged.json"
__all__ = ["DerivedDeleted", "purge_derived"]


@dataclass(frozen=True)
class DerivedDeleted:
    cache_files: int = 0
    frequency_files: int = 0
    temporary_files: int = 0
    keys: int = 0


def _directory(profile: str, *parts: str) -> int | None:
    try:
        return storage.open_profile_dir(profile, *parts, create=False)
    except FileNotFoundError:
        return None


def _temporaries(profile: str, *, dry_run: bool, parts: tuple[str, ...] = ()) -> int:
    if len(parts) > 8:
        raise ValueError("accounting temporary directory depth exceeds limit")
    fd = _directory(profile, *parts)
    if fd is None:
        return 0
    removed = 0
    try:
        for name in storage.list_names(fd):
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                if name != "journal":
                    removed += _temporaries(profile, dry_run=dry_run, parts=(*parts, name))
            elif (name.startswith(".") or parts == ("collector", "cache")) and name.endswith(
                ".tmp"
            ):
                if parts == ("collector",) and "frequency" in name:
                    continue
                removed += 1 if dry_run else storage.delete_file(fd, name)
        if removed and not dry_run:
            storage.fsync_dir(fd)
    finally:
        os.close(fd)
    return removed


def purge_derived(
    sel: Selection, *, dry_run: bool = False, invalidate_frequency: bool = False
) -> DerivedDeleted:
    """Lease holder only. Shared vocabulary is invalidated, not other workspace ledgers.

    The durable tombstone is written BEFORE deleting shared views. An interrupted
    cleanup therefore cannot restore erased authority on the next index refresh.
    """
    from ssgrep.accounting.privacy.purge_recovery import has_pending_erasure

    cache = frequency = keys = 0
    temporary = 0

    def remove(fd: int, name: str) -> int:
        if not dry_run:
            return int(storage.delete_file(fd, name))
        try:
            os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return 0
        return 1

    checkpoint = store.load_checkpoint(sel.profile)
    from ssgrep.accounting.collect.store import referenced_files

    retained_segments = referenced_files(sel.profile, checkpoint)[0] if checkpoint else set()
    journal_empty = False
    if sel.workspace is None and not retained_segments:
        journal = _directory(sel.profile, "journal")
        try:
            journal_empty = journal is None or not storage.list_names(journal)
        finally:
            if journal is not None:
                os.close(journal)
    empty_profile = journal_empty and (checkpoint is None or not checkpoint.blobs)
    erasure_in_progress = invalidate_frequency or has_pending_erasure(sel)
    fd = _directory(sel.profile, "collector", "cache")
    scoper = Scoper(sel.workspace)
    if fd is not None:
        try:
            for name in storage.list_names(fd):
                if name.endswith(".tmp"):
                    # Uncommitted content has no trusted semantic date;
                    # clean it only with a real erasure.
                    continue
                raw = storage.read_file(fd, name, MAX_CACHE_BYTES)
                if raw is None:
                    continue
                try:
                    item = evidence_from_line(raw)
                except (ValueError, KeyError, TypeError):
                    cache += remove(fd, name)
                    continue
                if (item.ts_utc is None or item.ts_utc[:10] < sel.cutoff_day) and scoper.inside(
                    evidence_cwd(item.payload)
                ):
                    cache += remove(fd, name)
            if not dry_run:
                storage.fsync_dir(fd)
        finally:
            os.close(fd)
    # Filesystem mtime is not an accounting event clock. Collateral state follows
    # selected evidence or durable recovery intent, not mere metadata presence.
    if empty_profile:
        temporary = _temporaries(sel.profile, dry_run=True)
    erase = erasure_in_progress or cache > 0 or temporary > 0
    if erase and not empty_profile:
        temporary = _temporaries(sel.profile, dry_run=True)
    fd = _directory(sel.profile, "collector")
    if fd is not None:
        try:
            names = storage.list_names(fd)
            if empty_profile and any(
                name != AUTHORITY_TOMBSTONE
                and (
                    name == "frequency.json"
                    or name.startswith("frequency-")
                    or name.startswith(".")
                    and "frequency" in name
                )
                for name in names
            ):
                erase = True
            if erase and not dry_run:
                from ssgrep.accounting.collect.frequency import erasure_document

                storage.atomic_write(
                    fd, AUTHORITY_TOMBSTONE, store.dumps(erasure_document(fd, sel)) + b"\n"
                )
            for name in names:
                if name == AUTHORITY_TOMBSTONE:
                    continue
                if (
                    name == "frequency.json"
                    or name.startswith("frequency-")
                    or (name.startswith(".") and "frequency" in name)
                ):
                    if erase:
                        frequency += remove(fd, name)
                elif erase and name.startswith(".") and name.endswith(".tmp"):
                    remove(fd, name)
            if not dry_run:
                storage.fsync_dir(fd)
        finally:
            os.close(fd)
    if checkpoint is not None and erase and not dry_run:
        store.sweep_orphans(sel.profile, checkpoint)
    # A shared request key cannot be rotated for one workspace or while later
    # retained evidence/journal entries still depend on it.
    if sel.workspace is None and not retained_segments and journal_empty:
        root = _directory(sel.profile)
        if root is not None:
            try:
                keys += remove(root, fingerprint.KEY_FILE)
                if not dry_run:
                    storage.fsync_dir(root)
            finally:
                os.close(root)
        if keys and not dry_run:
            fingerprint.forget_keys()
    if erase and not dry_run:
        _temporaries(sel.profile, dry_run=False)
    return DerivedDeleted(cache, frequency, temporary, keys)
