"""Snapshot ids pinned by delivery state (read-only, tolerant of absent or garbage files).

``delivery/<destination_id>/<signal>.json`` is written only by the export module; the
``last_acked_snapshot_id`` it records names a snapshot generation file that retention
must keep so an acknowledged snapshot can still be looked up by digest.
"""

from __future__ import annotations

import json
import os

from ssgrep.accounting import storage

DELIVERY_DIR = "delivery"
MAX_DELIVERY_BYTES = 1024 * 1024


def pinned_snapshot_ids(profile: str) -> frozenset[str]:
    """Every ``last_acked_snapshot_id`` under ``delivery/*/*.json``; tolerant, read-only.

    A missing directory, an unreadable or garbage file, or a foreign shape pins
    nothing and never raises or creates anything.
    """
    pinned: set[str] = set()
    try:
        root = storage.open_profile_dir(profile, DELIVERY_DIR, create=False)
    except (OSError, ValueError):
        return frozenset()
    try:
        destinations = storage.list_names(root)
    except OSError:
        destinations = []
    finally:
        os.close(root)
    for destination in destinations:
        pinned.update(_pins_of(profile, destination))
    return frozenset(pinned)


def _pins_of(profile: str, destination: str) -> set[str]:
    found: set[str] = set()
    try:
        dir_fd = storage.open_profile_dir(profile, DELIVERY_DIR, destination, create=False)
    except (OSError, ValueError):
        return found
    try:
        for name in storage.list_names(dir_fd):
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                raw = storage.read_file(dir_fd, name, MAX_DELIVERY_BYTES)
                state = None if raw is None else json.loads(raw)
            except (OSError, ValueError, RecursionError):
                continue
            acked = state.get("last_acked_snapshot_id") if isinstance(state, dict) else None
            if isinstance(acked, str) and 0 < len(acked) <= 128:
                found.add(acked)
    except OSError:
        pass
    finally:
        os.close(dir_fd)
    return found
