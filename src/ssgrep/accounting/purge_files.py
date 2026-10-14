"""File-level steps of a purge, all through ``ssgrep.accounting.storage`` descriptors.

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

from ssgrep.accounting import snapshots, storage
from ssgrep.accounting.collector_store import dumps, loads
from ssgrep.accounting.journal import SEGMENT_PREFIX, SEGMENT_SUFFIX
from ssgrep.accounting.journal_codec import MalformedEvent, parse_event
from ssgrep.accounting.purge_model import PurgeError
from ssgrep.accounting.purge_select import Scoper

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


def has_current(profile: str) -> bool:
    fd = _dir(profile, "snapshots")
    if fd is None:
        return False
    try:
        return snapshots.CURRENT_NAME in storage.list_names(fd)
    finally:
        os.close(fd)


def delete_generations(profile: str, doomed: list[Generation]) -> int:
    """Delete the named generation files (never ``current.json``) and leaked temporaries."""
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


def delete_current(profile: str) -> bool:
    fd = _dir(profile, "snapshots")
    if fd is None:
        return False
    try:
        return storage.delete_file(fd, snapshots.CURRENT_NAME)
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
