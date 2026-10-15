"""Expire quarantine and interrupted retention staging without source authority.

A small cursor is operational progress only, never an acceptance record. Each
pass opens at most the supplied sensitive-file budget and reads at most the
sensitive-byte budget, plus a bounded 4 KiB progress record. Filename inventory
uses the existing directory helper; no raw journal segment is opened here.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ssgrep.accounting import storage

if TYPE_CHECKING:
    from ssgrep.accounting.journal_compact import JournalMaintenance

_STATE = "cursor.json"
_MAX_METADATA = 512 * 1024
_MAX_CURSOR = 4096


def _page(names: list[str], cursor: str, limit: int) -> list[str]:
    after = [name for name in names if name > cursor]
    before = [name for name in names if name <= cursor]
    return (after + before)[:limit]


def sweep_remnants(
    profile: str,
    journal: int,
    record: dict[str, Any],
    result: JournalMaintenance,
    *,
    file_budget: int,
    byte_budget: int,
) -> None:
    if file_budget <= 0 or byte_budget <= 0:
        return
    try:
        state_dir = storage.open_profile_dir(profile, "journal-maintenance", create=False)
    except FileNotFoundError:
        state_dir = None
    state = {}
    try:
        if state_dir is not None:
            raw = storage.read_file(state_dir, _STATE, _MAX_CURSOR)
            state = json.loads(raw) if raw else {}
        floor = record["horizon_day"]
        # An unchanged day with no pending cursor needs no extra inventory sweep.
        if state.get("completed_day") == floor and not record["proofs"]:
            return
        original = dict(state)
        temporaries = [
            name
            for name in storage.list_names(journal)
            if name.startswith(".retention-") and name.endswith(".tmp")
        ]
        visited = _page(temporaries, state.get("temporary", ""), file_budget)
        for name in visited:
            if storage.delete_file(journal, name):
                result.remnants_removed += 1
        file_budget -= len(visited)
        if visited:
            state["temporary"] = visited[-1]
            storage.fsync_dir(journal)
        pending = len(temporaries) > len(visited)
        try:
            quarantine = storage.open_profile_dir(profile, "journal-quarantine", create=False)
        except FileNotFoundError:
            quarantine = None
        if quarantine is not None:
            try:
                names = [name for name in storage.list_names(quarantine) if name.endswith(".json")]
                selected = _page(names, state.get("quarantine", ""), file_budget)
                processed = []
                for name in selected:
                    fd = storage.open_regular(quarantine, name)
                    try:
                        info = os.fstat(fd)
                        day = (
                            datetime.fromtimestamp(info.st_mtime_ns // 1_000_000_000, UTC)
                            .date()
                            .isoformat()
                        )
                        if day < floor:
                            if storage.delete_file(quarantine, name):
                                result.quarantine_removed += 1
                            processed.append(name)
                            continue
                        if info.st_size > byte_budget or info.st_size > _MAX_METADATA:
                            pending = True
                            continue
                        raw = os.read(fd, info.st_size)
                    finally:
                        os.close(fd)
                    result.bytes_read += len(raw)
                    byte_budget -= len(raw)
                    processed.append(name)
                    absent = False
                    try:
                        source = json.loads(raw).get("segment")
                        if (
                            isinstance(source, str)
                            and source.startswith("seg-")
                            and "/" not in source
                            and "\0" not in source
                        ):
                            try:
                                os.stat(source, dir_fd=journal, follow_symlinks=False)
                            except FileNotFoundError:
                                absent = True
                    except (ValueError, AttributeError):
                        pass  # malformed metadata expires by its own age, never as evidence
                    if absent or day < floor:
                        if storage.delete_file(quarantine, name):
                            result.quarantine_removed += 1
                if processed:
                    state["quarantine"] = processed[-1]
                    storage.fsync_dir(quarantine)
                pending = pending or len(names) > len(processed)
            finally:
                os.close(quarantine)
        if pending:
            state.pop("completed_day", None)
            result.reasons["remnants_pending"] = 1
            if not result.warnings:
                result.warnings = ("journal_retention_remnants_pending",)
        else:
            state["completed_day"] = floor
        if state != original:
            if state_dir is None:
                state_dir = storage.open_profile_dir(profile, "journal-maintenance")
            # Abandoned cursor temporaries contain only keyed metadata filenames.
            for name in storage.list_names(state_dir)[: max(1, file_budget)]:
                if name.startswith(".cursor.json.") and name.endswith(".tmp"):
                    storage.delete_file(state_dir, name)
            storage.atomic_write(state_dir, _STATE, json.dumps(state, sort_keys=True).encode())
    finally:
        if state_dir is not None:
            os.close(state_dir)
