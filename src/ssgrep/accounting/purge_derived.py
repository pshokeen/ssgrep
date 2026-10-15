"""Post-checkpoint erasure of plaintext caches and unsplittable rarity authority."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass

from ssgrep.accounting import collector_store as store, fingerprint, storage
from ssgrep.accounting.evidence_cache import MAX_CACHE_BYTES
from ssgrep.accounting.evidence_codec import evidence_from_line
from ssgrep.accounting.purge_recovery import (
    has_pending_erasure,
    recover_pending_purge,
    stage_erasure,
)
from ssgrep.accounting.purge_select import Scoper, Selection, evidence_cwd

AUTHORITY_TOMBSTONE = "frequency-purged.json"
__all__ = ["DerivedDeleted", "purge_derived", "recover_pending_purge", "stage_erasure"]


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
    from ssgrep.accounting.evidence_references import referenced_files

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
                from ssgrep.accounting.frequency_purge_policy import erasure_document

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
