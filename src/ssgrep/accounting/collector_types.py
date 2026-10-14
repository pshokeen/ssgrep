"""Pure data shared by the collection worker's modules (no I/O, no clocks, no floats).

The worker is split so each piece is testable alone: ``lease`` (exclusive per-profile
lease), ``collector`` (one bounded cycle), ``snapshots`` (immutable generations),
``windows`` (event-time aggregates), ``counters`` (live captured counters),
``retention_hooks`` (expiry as consumed by a cycle) and ``serve_loop`` (scheduling).
"""

from __future__ import annotations

from dataclasses import dataclass

MIB = 1024 * 1024
DEFAULT_CYCLE_SECONDS = 10
DEFAULT_CYCLE_BYTES = 32 * MIB


@dataclass(frozen=True)
class CycleBudget:
    """Cooperative per-cycle limits; an already-started bounded record may overshoot."""

    seconds: int = DEFAULT_CYCLE_SECONDS
    bytes: int = DEFAULT_CYCLE_BYTES


@dataclass(frozen=True)
class CycleReport:
    """What one finished collection cycle did (also the serve loop's view of it)."""

    status: str  # published | failed
    generation: int | None
    snapshot_id: str | None
    complete: bool  # every source fully processed: no lag, no pending tail, none missing
    stopped_by: str  # complete | byte_budget | time_budget
    bytes_read: int = 0
    lag_bytes: int = 0
    new_evidence: int = 0
    sources: int = 0
    elapsed_ms: int = 0
    error: str | None = None  # stable condition, never raw exception text


@dataclass(frozen=True)
class RetentionCoverage:
    """What retention has already removed (drives ``lifetime`` coverage statements)."""

    evidence_expired_before: str | None = None  # UTC day; evidence older than this is gone
    aggregates_expired_before: str | None = None  # UTC day; aggregates older than this are gone
    frozen_roots: int = 0  # roots whose totals are kept but are non-recomputable


@dataclass(frozen=True)
class CollectOptions:
    """Validated arguments of one ``metrics collect`` run (``since``/``until`` normalized UTC)."""

    since: str | None = None
    until: str | None = None
    backfill: bool = False
    recompute: bool = False
    new_epoch: bool = False
    repin_frequency: bool = False
    budget: CycleBudget = CycleBudget()

    @property
    def mode(self) -> str:
        return "backfill" if self.backfill else ("recompute" if self.recompute else "live")


@dataclass(frozen=True)
class CollectResult:
    """Outcome of ``collect_once``: a published cycle or the live worker's progress."""

    state: str  # published | partial | already_running
    mode: str  # live | backfill | recompute
    report: CycleReport | None = None
    progress: dict[str, object] | None = None  # the running owner's progress (already_running)
    epoch: int | None = None
    unavailable_periods: tuple[str, ...] = ()
    backfill: dict[str, object] | None = None  # {since, until, complete, ...}: separate series
    warnings: tuple[str, ...] = ()


class CollectError(Exception):
    """A collection failure with a stable ``condition`` (never raw exception text)."""

    def __init__(
        self, condition: str, message: str, command: str = "ssgrep metrics status"
    ) -> None:
        super().__init__(message)
        self.condition = condition
        self.command = command
