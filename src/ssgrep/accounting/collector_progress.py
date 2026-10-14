"""``worker/progress.json``: the lease owner's live progress, readable by anyone.

Only the lease holder writes it (atomically, so a reader sees an old or a new complete
document). Readers (``metrics status``, ``metrics collect`` reporting ``already_running``)
treat an absent, oversized, or malformed file as *unknown*, never as an error and never
as zero progress.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting import plainfile, storage
from ssgrep.accounting.collector_store import dumps, loads
from ssgrep.accounting.collector_types import CycleReport

PROGRESS_FILE = "progress.json"
MAX_PROGRESS_BYTES = 64 * 1024
_UTC = "%Y-%m-%dT%H:%M:%SZ"


def utc_stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(_UTC)


def last_cycle_document(report: CycleReport, finished_at: datetime) -> dict[str, Any]:
    return {
        "finished_at": utc_stamp(finished_at),
        "status": report.status,
        "stopped_by": report.stopped_by,
        "error": report.error,
        "generation": report.generation,
        "elapsed_ms": report.elapsed_ms,
        "bytes_read": report.bytes_read,
        "complete": report.complete,
    }


@dataclass
class Progress:
    """Mutable progress of the cycle in flight plus the last finished cycle."""

    profile: str
    kind: str
    pid: int = field(default_factory=os.getpid)
    phase: str = "idle"
    mode: str = "live"
    cycle_started_at: str | None = None
    updated_at: str = ""
    cycles: int = 0
    bytes_read: int = 0
    sources_done: int = 0
    sources_total: int = 0
    lag_bytes: int = 0
    last_cycle: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "profile": self.profile,
            "pid": self.pid,
            "kind": self.kind,
            "phase": self.phase,
            "mode": self.mode,
            "cycle_started_at": self.cycle_started_at,
            "updated_at": self.updated_at,
            "cycles": self.cycles,
            "bytes_read": self.bytes_read,
            "sources_done": self.sources_done,
            "sources_total": self.sources_total,
            "lag_bytes": self.lag_bytes,
            "last_cycle": self.last_cycle,
        }

    def write(self, now: datetime) -> None:
        """Persist atomically; progress is advisory, so a failed write never fails a cycle."""
        self.updated_at = utc_stamp(now)
        try:
            directory = storage.open_profile_dir(self.profile, "worker")
        except OSError:
            return
        try:
            storage.atomic_write(directory, PROGRESS_FILE, dumps(self.to_dict()) + b"\n")
        except OSError:
            pass
        finally:
            os.close(directory)


def load_previous(profile: str) -> Progress | None:
    """Carry cycle counters and ``last_cycle`` over when the next owner starts."""
    raw = read_progress(profile)
    if raw is None:
        return None
    last = raw.get("last_cycle")
    try:
        return Progress(
            profile=profile,
            kind=str(raw.get("kind", "collect")),
            cycles=int(raw.get("cycles", 0)),
            last_cycle=last if isinstance(last, dict) else None,
        )
    except (TypeError, ValueError):
        return None


def read_progress(profile: str) -> dict[str, Any] | None:
    """The progress document, or None when absent/unreadable/not an object (never raises)."""
    try:
        data = plainfile.read_private(
            storage.data_root(),
            "accounting",
            profile,
            "worker",
            PROGRESS_FILE,
            max_bytes=MAX_PROGRESS_BYTES,
        )
        if data is None:
            return None
        raw = loads(data)
    except (plainfile.UnsafePathError, ValueError, OSError):
        return None
    return raw if isinstance(raw, dict) else None
