"""Bounded reclamation of unreferenced frozen rarity inputs; no corpus mutations."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice

from ssgrep.accounting import storage
from ssgrep.accounting.collector_frequency import _LOADED, MAX_POINTER_BYTES, POINTER
from ssgrep.accounting.collector_store import loads

_VIEW = re.compile(r"frequency-(freq-[0-9a-f]{24})\.json")


@dataclass(frozen=True)
class ViewPruneOutcome:
    scanned: int = 0
    removed: int = 0
    retained: int = 0


def prune_views(
    profile: str,
    *,
    candidates: Iterable[str],
    referenced: Iterable[str],
    before_day: str,
    limit: int = 32,
) -> ViewPruneOutcome:
    """Reclaim only an explicitly scheduled bounded batch of expired orphan views.

    The collector's resumable directory scheduler supplies candidate filenames;
    this primitive never enumerates/reloads all retained state. References MUST
    include live lookup assignments, inactive canonical archives and surviving
    receipt revisions. The current pointer is independently protected. Expiry
    alone never deletes an input needed for exact historical replay.
    """
    datetime.strptime(before_day, "%Y-%m-%d")
    if limit < 1 or limit > 128:
        raise ValueError("frequency maintenance batch must be 1..128")
    protected = set(referenced)
    try:
        directory = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError:
        return ViewPruneOutcome()
    scanned = removed = retained = 0
    try:
        pointer = storage.read_file(directory, POINTER, MAX_POINTER_BYTES)
        if pointer is not None:
            protected.add(loads(pointer)["view_id"])
        for name in islice(candidates, limit):
            scanned += 1
            match = _VIEW.fullmatch(name)
            if match is None or match[1] in protected:
                retained += 1
                continue
            try:
                descriptor = storage.open_regular(directory, name)
            except FileNotFoundError:
                continue
            try:
                info = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if datetime.fromtimestamp(info.st_mtime, UTC).date().isoformat() >= before_day:
                retained += 1
                continue
            # Recheck the exact file before deletion; raced replacements stay intact.
            current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ) != (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ):
                retained += 1
                continue
            os.unlink(name, dir_fd=directory)
            _LOADED.pop((str(storage.data_root()), profile, match[1]), None)
            removed += 1
        if removed:
            os.fsync(directory)
    finally:
        os.close(directory)
    return ViewPruneOutcome(scanned, removed, retained)
