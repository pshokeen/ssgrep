"""Types and small pure helpers shared by the delivery modules (no I/O except ``reason_of``).

``Deps`` is the injectable seam (production defaults), ``CycleResult``/``CheckResult`` are the
public verdicts, ``Cycle`` is the mutable context of ONE ``run_destination`` call and ``Tally``
its counters. Every code that can reach a result, a delivery document or ``last_error.json`` is a
stable enumerated string: ``reason_of`` maps an exception to one WITHOUT ever reading its text.
"""

from __future__ import annotations

import random
import re
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ssgrep.accounting import export_http
from ssgrep.accounting.export_policy import Gate, ScopeState
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal

STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# a dispatched request whose fate the sender cannot know (the receiver may have processed it)
UNRESOLVED_CATEGORIES = frozenset(
    {"network", "timeout", "oversize", "malformed", "send_interrupted"}
)
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CLASS = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


class ExportPaused(Exception):
    """Sending/checking was refused because the profile (or an invalid managed update) paused it."""

    condition = "export_paused"

    def __init__(self) -> None:
        super().__init__(self.condition)


class CycleError(Exception):
    """A cycle step failed for a stable, text-free reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Deps:
    """Injectable seams; the defaults are the production behaviour."""

    post: Callable[..., export_http.Outcome] = export_http.post
    rand: Callable[[], float] = random.random
    environ: Mapping[str, str] | None = None
    context: ssl.SSLContext | None = None
    connect_timeout: float = 10.0
    read_timeout: float = 10.0
    max_batches: int = 64


DEFAULT_DEPS = Deps()


@dataclass(frozen=True)
class CycleResult:
    destination: str
    signal: Signal
    state: str  # sent|idle|waiting|paused|not_enabled|blocked|revoked|busy|no_snapshot|failed
    reason: str | None = None
    enqueued_batches: int = 0
    invalidated_batches: int = 0
    sent: int = 0
    acked: int = 0
    partial: int = 0
    quarantined: int = 0
    retry_scheduled_utc: str | None = None
    blocked_reason: str | None = None
    nonrecallable_batch: str | None = None
    warnings: tuple[str, ...] = ()
    error_category: str | None = None  # last transport/refusal category (deviation: extra field)


@dataclass(frozen=True)
class CheckResult:
    destination: str
    signal: Signal
    state: str  # ok|unauthorized|forbidden|rate_limited|unreachable|timeout|tls_error|...
    ok: bool
    status: int | None
    category: str


@dataclass
class Tally:
    enqueued: int = 0
    invalidated: int = 0
    sent: int = 0
    acked: int = 0
    partial: int = 0
    quarantined: int = 0
    retry_utc: str | None = None
    blocked_reason: str | None = None
    nonrecallable: str | None = None
    warnings: list[str] = field(default_factory=list)
    stop: tuple[str, str] | None = None  # (state, reason) that ended the cycle
    failure: str | None = None  # category of the last retry/refusal
    enqueue_error: str | None = None
    idle: str | None = None  # why nothing was enqueued (no_snapshot|ineligible_scope)

    def warn(self, code: str) -> None:
        if code not in self.warnings:
            self.warnings.append(code)


@dataclass
class Cycle:
    """Everything one locked cycle knows; filled in step by step by ``delivery``."""

    profile: str
    destination: str
    signal: Signal
    now: datetime
    fixed_clock: bool
    deps: Deps
    enqueue: bool
    tally: Tally = field(default_factory=Tally)
    gate: Gate | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    scope_state: ScopeState | None = None
    key: bytes | None = None
    skip: set[int] = field(default_factory=set)  # generations whose build failed this cycle

    @property
    def dest(self) -> DestinationPolicy:
        assert self.gate is not None and self.gate.destination is not None
        return self.gate.destination

    @property
    def policy(self) -> ProfilePolicy:
        assert self.gate is not None
        return self.gate.policy

    def at(self) -> datetime:
        """The instant to stamp events with: injected ``now`` verbatim, else the wall clock."""
        return self.now if self.fixed_clock else resolve_now(None)

    def stamp(self) -> str:
        return stamp_of(self.at())


def resolve_now(now: datetime | None) -> datetime:
    """A whole-second UTC instant; naive datetimes are a caller bug."""
    moment = datetime.now(UTC) if now is None else now
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return moment.astimezone(UTC).replace(microsecond=0)


def stamp_of(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(STAMP_FORMAT)


def plus_seconds(moment: datetime, seconds: int) -> str:
    return stamp_of(moment + timedelta(seconds=seconds))


def reason_of(error: BaseException) -> str:
    """A stable code for ``error``; its message text is NEVER read."""
    from ssgrep.accounting import acks_model, export_files, export_spool, otlp, storage

    if isinstance(error, CycleError):
        return error.reason
    condition = getattr(error, "condition", None)
    if isinstance(condition, str) and _CODE.fullmatch(condition):
        return condition
    if isinstance(error, export_files.StateCorrupt | acks_model.DeliveryCorrupt):
        return "state_corrupt"
    if isinstance(error, export_spool.SpoolCorrupt):
        return "spool_corrupt"
    if isinstance(error, export_spool.BatchConflict):
        return "batch_conflict"
    if isinstance(error, otlp.ExportTooLarge):
        return "export_too_large"
    if isinstance(error, storage.UnsafeStorageError):
        return "storage_unsafe"
    if isinstance(error, OSError):
        return "storage_error"
    if isinstance(error, ValueError):
        return "invalid_state"
    return "internal_error"


def class_code(error: BaseException) -> str:
    """The exception CLASS name when it is a plain identifier (never the message)."""
    name = type(error).__name__
    return name if _CLASS.fullmatch(name) else "Exception"
