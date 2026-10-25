"""Derived-state contracts: policy, receipts, snapshots, and delivery state.

Everything here is *derived* from evidence and recomputable; none of it is
authoritative for what happened. Canonical serialization excludes collection
clocks so replay equality holds (design D7/D8).
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.ledger.contracts import (
    PERSONAL_PROFILE,
    SCHEMA_VERSION,
    ExecutionEvent,
    Phase,
    Quantity,
    Scenario,
)
from ssgrep.accounting.ledger.units import UnitSpec

if TYPE_CHECKING:
    from ssgrep.accounting.ledger.reduce_records import InvocationRecord

PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_PROFILES = 8
MAX_DESTINATIONS = 8
MAX_SERIES = 5000
WINDOWS = ("24h", "7d", "30d", "lifetime")
DEFAULT_STALE_INTERVALS = 3
DEFAULT_COLLECT_SECONDS = 60
DEFAULT_EXPORT_SECONDS = 300
EVIDENCE_RETENTION_DAYS = 30
RECEIPT_RETENTION_DAYS = 400
SPOOL_RETENTION_DAYS = 7
SPOOL_MAX_BYTES = 100 * 1024 * 1024


class Signal(StrEnum):
    METRICS = "metrics"
    LOGS = "logs"


class SnapshotState(StrEnum):
    DISABLED = "disabled"
    NO_SNAPSHOT = "no_snapshot"
    FRESH = "fresh"
    STALE = "stale"
    PAUSED = "paused"
    CORRUPT = "corrupt"
    UNSUPPORTED_SCHEMA = "unsupported_schema"


@dataclass(frozen=True)
class DestinationPolicy:
    destination_id: str
    endpoint: str
    signals: tuple[Signal, ...] = (Signal.METRICS,)
    logs_permitted: bool = False
    credential_ref: str | None = None
    proxy: str | None = None


@dataclass(frozen=True)
class ProfilePolicy:
    """What one profile may collect and export; narrowing-only locally."""

    profile_id: str = PERSONAL_PROFILE
    roots: tuple[str, ...] = ()
    managed: bool = False
    policy_version: int = 1
    policy_source: str = "local"
    evidence_retention_days: int = EVIDENCE_RETENTION_DAYS
    receipt_retention_days: int = RECEIPT_RETENTION_DAYS
    destinations: tuple[DestinationPolicy, ...] = ()
    collection_enabled: bool = False
    paused: bool = False
    paused_at: str | None = None
    enabled_exports: tuple[tuple[str, str], ...] = ()
    evidence_cache_enabled: bool = False
    evidence_cache_retention_days: int = 0
    evidence_cache_max_bytes: int = 0


@dataclass(frozen=True)
class SourceAtom:
    """Immutable source span; ``digest`` is verified against resolved content."""

    atom_id: str
    origin_event: str
    digest: str
    chars: int
    seconds: int | None = None


@dataclass(frozen=True)
class FrequencyAssignment:
    """Frozen frequency identity for one attributed lookup."""

    invocation_id: str
    view_id: str
    unavailable_reason: str = ""
    profile_scope: str = ""
    revision_name: str = ""
    unavailable_provenance: tuple[tuple[str, str], ...] = ()
    fingerprint_key_id: str | None = None


@dataclass(frozen=True)
class EstimateBounds:
    """Deterministic interval bounds, not statistical confidence intervals."""

    credit_lower: Quantity
    credit_upper: Quantity
    debit_lower: Quantity
    debit_upper: Quantity
    net_lower: Quantity
    net_upper: Quantity
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Receipt:
    """Explains one candidate reuse; allocations are per scenario."""

    receipt_id: str
    revision: int
    root: str
    task_turn: str | None
    execution_ids: tuple[str, ...]
    exposure_ids: tuple[str, ...]
    candidate_fingerprint: str
    candidate_type: str
    atoms: tuple[str, ...]
    allocations: tuple[tuple[str, tuple[str, ...]], ...]
    exclusions: tuple[str, ...]
    costs: tuple[tuple[str, Quantity], ...]
    methodology: str
    evidence_available: bool = True
    schema: int = SCHEMA_VERSION
    eligibility: str = "unavailable"
    eligibility_reasons: tuple[str, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    scenarios: tuple[ScenarioEstimate, ...] = ()


@dataclass(frozen=True)
class ScenarioEstimate:
    scenario: Scenario
    credit: Quantity
    debit: Quantity
    net: Quantity
    known_net: int | None = None  # known subtotal, NEVER a complete-net substitute
    complete: bool = False
    credit_missing: tuple[tuple[str, str], ...] = ()
    debit_missing: tuple[tuple[str, str], ...] = ()
    known_atoms: int = 0
    known_debits: int = 0
    bounds: EstimateBounds | None = None
    eligibility: tuple[tuple[str, str], ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    unit: UnitSpec | None = None  # full method/tokenizer/currency compatibility identity


@dataclass(frozen=True)
class SnapshotEnvelope:
    """Complete immutable snapshot; ``collected_at`` is outside ledger identity."""

    schema_version: int
    snapshot_id: str
    generation: int
    counter_epoch: int
    methodology_id: str
    policy_version: int
    collected_at: str
    coverage_through: str
    source_manifest_digest: str
    windows: dict[str, dict[str, object]] = field(default_factory=dict)
    coverage: dict[str, object] = field(default_factory=dict)
    observations: dict[str, object] = field(default_factory=dict)
    estimates: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeliveryState:
    """Per destination and signal; the three generations never merge."""

    destination_id: str
    signal: Signal
    last_collected_generation: int | None = None
    last_attempted_generation: int | None = None
    last_acked_generation: int | None = None
    last_acked_snapshot_id: str | None = None
    pending_batches: int = 0
    rejected_points: int = 0
    unknown_batches: int = 0
    blocked_reason: str | None = None


def valid_profile_id(profile_id: str) -> bool:
    return bool(PROFILE_ID_RE.fullmatch(profile_id))


BUCKET_BOUNDS_US = (
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    30_000_000,
)
BUCKET_COUNT = len(BUCKET_BOUNDS_US) + 1

_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def bucket_index(duration_us: int) -> int:
    """Index of the first bucket whose inclusive bound holds ``duration_us``."""
    return bisect_left(BUCKET_BOUNDS_US, duration_us)


def as_int(value: Any, name: str) -> int:
    """A non-negative real int (bool and float are garbage)."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: expected a non-negative integer")
    return value


def as_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}: expected a non-empty string")
    return value


def as_list(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected a list")
    return value


def as_map(value: Any, name: str, fields: set[str]) -> Mapping[str, Any]:
    """A dict with EXACTLY ``fields`` as keys."""
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name}: expected an object with fields {sorted(fields)}")
    return value


def as_day(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DAY.fullmatch(value):
        raise ValueError(f"{name}: expected YYYY-MM-DD")
    return value


def as_utc(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        raise ValueError(f"{name}: expected YYYY-MM-DDTHH:MM:SSZ")
    return value


@dataclass(frozen=True)
class Histogram:
    buckets: tuple[int, ...] = (0,) * BUCKET_COUNT
    count: int = 0
    sum_us: int = 0

    def __post_init__(self) -> None:
        if len(self.buckets) != BUCKET_COUNT or any(b < 0 for b in self.buckets):
            raise ValueError(f"a histogram has {BUCKET_COUNT} non-negative buckets")
        if self.count != sum(self.buckets) or self.sum_us < 0:
            raise ValueError("histogram count must equal the bucket total and sum_us be >= 0")

    def add(self, duration_us: int) -> Histogram:
        index = bucket_index(duration_us)
        buckets = tuple(b + (i == index) for i, b in enumerate(self.buckets))
        return Histogram(buckets, self.count + 1, self.sum_us + duration_us)

    def cumulative(self) -> tuple[int, ...]:
        """Prometheus ``le`` counts: bucket i holds every observation <= bound i."""
        total, out = 0, []
        for bucket in self.buckets:
            total += bucket
            out.append(total)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"buckets": list(self.buckets), "count": self.count, "sum_us": self.sum_us}

    @classmethod
    def from_dict(cls, raw: Any) -> Histogram:
        data = as_map(raw, "histogram", {"buckets", "count", "sum_us"})
        buckets = tuple(as_int(b, "bucket") for b in as_list(data["buckets"], "buckets"))
        return cls(buckets, as_int(data["count"], "count"), as_int(data["sum_us"], "sum_us"))


__all__ = [
    "BUCKET_BOUNDS_US",
    "UNKNOWN_RUNTIME",
    "BackfillSeries",
    "ExecKey",
    "Histogram",
    "LiveCounters",
    "OpKey",
    "backfill_series",
    "bucket_index",
    "empty_counters",
    "observe",
    "prune",
    "start_epoch",
    "view",
]

UNKNOWN_RUNTIME = "unknown"
UNKNOWN_CARDINALITY = "unknown"
_PHASES = frozenset(p.value for p in Phase)
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True, order=True)
class ExecKey:
    operation: str
    transport: str
    outcome: str
    runtime: str


@dataclass(frozen=True, order=True)
class OpKey:
    operation: str
    transport: str


def _sorted(mapping: Mapping[Any, Any]) -> tuple[tuple[Any, Any], ...]:
    return tuple(sorted(mapping.items(), key=lambda item: item[0]))


def _op_rows(pairs: tuple[tuple[OpKey, int], ...], field: str) -> list[dict[str, Any]]:
    return [{"operation": k.operation, "transport": k.transport, field: n} for k, n in pairs]


def _read_ops(raw: Any, name: str, field: str) -> tuple[tuple[OpKey, int], ...]:
    out: dict[OpKey, int] = {}
    for item in as_list(raw, name):
        row = as_map(item, name, {"operation", "transport", field})
        key = OpKey(as_str(row["operation"], name), as_str(row["transport"], name))
        if key in out:
            raise ValueError(f"{name}: duplicate series")
        out[key] = as_int(row[field], name)
    return _sorted(out)


@dataclass(frozen=True)
class LiveCounters:
    epoch: int
    epoch_started_at: str
    executions: tuple[tuple[ExecKey, int], ...] = ()
    rendered_bytes: tuple[tuple[OpKey, int], ...] = ()
    rendered_unknown: tuple[tuple[OpKey, int], ...] = ()
    durations: tuple[tuple[OpKey, Histogram], ...] = ()
    duration_unknown: tuple[tuple[OpKey, int], ...] = ()
    counted: tuple[tuple[str, str], ...] = ()  # ("<execution_id>|<phase>", "YYYY-MM-DD")
    skipped_expired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "epoch_started_at": self.epoch_started_at,
            "executions": [
                {
                    "operation": k.operation,
                    "transport": k.transport,
                    "outcome": k.outcome,
                    "runtime": k.runtime,
                    "count": n,
                }
                for k, n in self.executions
            ],
            "rendered_bytes": _op_rows(self.rendered_bytes, "bytes"),
            "rendered_unknown": _op_rows(self.rendered_unknown, "count"),
            "durations": [
                {"operation": k.operation, "transport": k.transport, "histogram": h.to_dict()}
                for k, h in self.durations
            ],
            "duration_unknown": _op_rows(self.duration_unknown, "count"),
            "counted": [[key, day] for key, day in self.counted],
            "skipped_expired": self.skipped_expired,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> LiveCounters:
        names = {
            "epoch",
            "epoch_started_at",
            "executions",
            "rendered_bytes",
            "rendered_unknown",
            "durations",
            "duration_unknown",
            "counted",
            "skipped_expired",
        }
        data = as_map(raw, "counters", names)
        epoch = as_int(data["epoch"], "epoch")
        if epoch < 1:
            raise ValueError("epoch: must be >= 1")
        return cls(
            epoch=epoch,
            epoch_started_at=as_utc(data["epoch_started_at"], "epoch_started_at"),
            executions=_read_executions(data["executions"]),
            rendered_bytes=_read_ops(data["rendered_bytes"], "rendered_bytes", "bytes"),
            rendered_unknown=_read_ops(data["rendered_unknown"], "rendered_unknown", "count"),
            durations=_read_durations(data["durations"]),
            duration_unknown=_read_ops(data["duration_unknown"], "duration_unknown", "count"),
            counted=_read_counted(data["counted"]),
            skipped_expired=as_int(data["skipped_expired"], "skipped_expired"),
        )


def _read_executions(raw: Any) -> tuple[tuple[ExecKey, int], ...]:
    out: dict[ExecKey, int] = {}
    for item in as_list(raw, "executions"):
        row = as_map(item, "executions", {"operation", "transport", "outcome", "runtime", "count"})
        key = ExecKey(
            *(
                as_str(row[f], "executions")
                for f in ("operation", "transport", "outcome", "runtime")
            )
        )
        if key in out:
            raise ValueError("executions: duplicate series")
        out[key] = as_int(row["count"], "executions")
    return _sorted(out)


def _read_durations(raw: Any) -> tuple[tuple[OpKey, Histogram], ...]:
    out: dict[OpKey, Histogram] = {}
    for item in as_list(raw, "durations"):
        row = as_map(item, "durations", {"operation", "transport", "histogram"})
        key = OpKey(as_str(row["operation"], "durations"), as_str(row["transport"], "durations"))
        if key in out:
            raise ValueError("durations: duplicate series")
        out[key] = Histogram.from_dict(row["histogram"])
    return _sorted(out)


def _read_counted(raw: Any) -> tuple[tuple[str, str], ...]:
    out: dict[str, str] = {}
    for item in as_list(raw, "counted"):
        pair = as_list(item, "counted")
        if len(pair) != 2:
            raise ValueError("counted: expected [key, day]")
        key = as_str(pair[0], "counted")
        if "|" not in key or key.rpartition("|")[2] not in _PHASES or not key.rpartition("|")[0]:
            raise ValueError("counted: key must be <execution_id>|<phase>")
        if key in out:
            raise ValueError("counted: duplicate key")
        out[key] = as_day(pair[1], "counted")
    return _sorted(out)


def empty_counters(epoch: int, started_at: str) -> LiveCounters:
    if as_int(epoch, "epoch") < 1:
        raise ValueError("epoch: must be >= 1")
    return LiveCounters(epoch, as_utc(started_at, "started_at"))


def _key(event: ExecutionEvent) -> str:
    return f"{event.execution_id}|{event.phase.value}"


def _day(event: ExecutionEvent) -> str:
    return as_day(event.ts_utc[:10], "ts_utc")


def observe(
    counters: LiveCounters, events: Iterable[ExecutionEvent], *, horizon_day: str | None
) -> LiveCounters:
    """Count every not-yet-counted ``completed``/``rendered`` event once; return a new value.

    Events older than ``horizon_day`` are skipped (``skipped_expired`` rises once per
    distinct expired event per call) and are NOT remembered.
    """
    counted = dict(counters.counted)
    executions, durations = dict(counters.executions), dict(counters.durations)
    rendered, rendered_unknown = dict(counters.rendered_bytes), dict(counters.rendered_unknown)
    duration_unknown = dict(counters.duration_unknown)
    expired: set[str] = set()
    for event in events:
        key, day = _key(event), _day(event)
        if key in counted or event.phase is Phase.STARTED:
            continue
        if horizon_day is not None and day < horizon_day:
            expired.add(key)
            continue
        counted[key] = day
        op = OpKey(event.operation.value, event.transport.value)
        if event.phase is Phase.COMPLETED:
            outcome = event.outcome.value if event.outcome is not None else "unknown"
            label = ExecKey(op.operation, op.transport, outcome, UNKNOWN_RUNTIME)
            executions[label] = executions.get(label, 0) + 1
            if event.mono_ns is None or event.mono_ns < 0:
                duration_unknown[op] = duration_unknown.get(op, 0) + 1
            else:
                durations[op] = durations.get(op, Histogram()).add(event.mono_ns // 1000)
        else:
            size = event.rendered_bytes.value if event.rendered_bytes is not None else None
            if size is None or size < 0:
                rendered_unknown[op] = rendered_unknown.get(op, 0) + 1
            else:
                rendered[op] = rendered.get(op, 0) + size
    return LiveCounters(
        epoch=counters.epoch,
        epoch_started_at=counters.epoch_started_at,
        executions=_sorted(executions),
        rendered_bytes=_sorted(rendered),
        rendered_unknown=_sorted(rendered_unknown),
        durations=_sorted(durations),
        duration_unknown=_sorted(duration_unknown),
        counted=_sorted(counted),
        skipped_expired=counters.skipped_expired + len(expired),
    )


def start_epoch(
    previous: LiveCounters | None, known_events: Iterable[ExecutionEvent], now: datetime
) -> LiveCounters:
    """A new epoch: every series zero; events captured before now never count in it."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    counted = dict(previous.counted) if previous is not None else {}
    for event in known_events:
        counted.setdefault(_key(event), _day(event))
    return LiveCounters(
        epoch=previous.epoch + 1 if previous is not None else 1,
        epoch_started_at=now.astimezone(UTC).strftime(_UTC_FORMAT),
        counted=_sorted(counted),
    )


def prune(counters: LiveCounters, horizon_day: str) -> LiveCounters:
    """Forget once-only keys older than the horizon; every total stays as it was."""
    kept = tuple((key, day) for key, day in counters.counted if day >= horizon_day)
    return LiveCounters(
        epoch=counters.epoch,
        epoch_started_at=counters.epoch_started_at,
        executions=counters.executions,
        rendered_bytes=counters.rendered_bytes,
        rendered_unknown=counters.rendered_unknown,
        durations=counters.durations,
        duration_unknown=counters.duration_unknown,
        counted=kept,
        skipped_expired=counters.skipped_expired,
    )


def view(counters: LiveCounters) -> dict[str, Any]:
    """Integer/string-only document for ``observations["counters"]``; histogram is cumulative."""
    duration = []
    for key, hist in counters.durations:
        bounds = [str(bound) for bound in BUCKET_BOUNDS_US] + ["+Inf"]
        duration.append(
            {
                "operation": key.operation,
                "transport": key.transport,
                "count": hist.count,
                "sum_us": hist.sum_us,
                "buckets": [
                    {"le_us": le, "count": n}
                    for le, n in zip(bounds, hist.cumulative(), strict=True)
                ],
            }
        )
    document = counters.to_dict()
    return {
        "epoch": counters.epoch,
        "epoch_started_at": counters.epoch_started_at,
        "executions": document["executions"],
        "rendered_bytes": document["rendered_bytes"],
        "rendered_unknown": document["rendered_unknown"],
        "duration": duration,
        "duration_unknown": document["duration_unknown"],
        "skipped_expired": counters.skipped_expired,
    }


@dataclass(frozen=True)
class BackfillSeries:
    """Imported history for an explicit interval; never merged with live counters."""

    since: str
    until: str
    completed_at: str
    counts: tuple[tuple[str, int], ...]  # ("<operation>:<class>", n) sorted
    unknown_cardinality: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "backfill",
            "since": self.since,
            "until": self.until,
            "completed_at": self.completed_at,
            "counts": dict(self.counts),
            "unknown_cardinality": self.unknown_cardinality,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> BackfillSeries:
        names = {"kind", "since", "until", "completed_at", "counts", "unknown_cardinality"}
        data = as_map(raw, "backfill", names)
        if data["kind"] != "backfill" or not isinstance(data["counts"], dict):
            raise ValueError("backfill: wrong kind or counts")
        counts = {as_str(k, "counts"): as_int(v, "counts") for k, v in data["counts"].items()}
        return cls(
            since=as_utc(data["since"], "since"),
            until=as_utc(data["until"], "until"),
            completed_at=as_utc(data["completed_at"], "completed_at"),
            counts=_sorted(counts),
            unknown_cardinality=as_int(data["unknown_cardinality"], "unknown_cardinality"),
        )


def backfill_series(
    invocations: Iterable[InvocationRecord], since: str, until: str, completed_at: str
) -> BackfillSeries:
    """Count recognised invocations with ``since <= ts_utc < until``, once per invocation id."""
    if as_utc(since, "since") >= as_utc(until, "until"):
        raise ValueError("backfill interval must satisfy since < until")
    as_utc(completed_at, "completed_at")
    counts: dict[str, int] = {}
    seen: set[str] = set()
    unknown = 0
    for record in invocations:
        if (
            record.ts_utc is None
            or not since <= record.ts_utc < until
            or record.invocation_id in seen
        ):
            continue
        seen.add(record.invocation_id)
        unknown += record.cardinality == UNKNOWN_CARDINALITY
        for operation in record.operations:
            key = f"{operation}:{record.cls.value}"
            counts[key] = counts.get(key, 0) + 1
    return BackfillSeries(since, until, completed_at, _sorted(counts), unknown)
