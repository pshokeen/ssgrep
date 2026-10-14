"""Live captured counters (design D8/D9; pure data, no I/O, no clocks, no floats).

Counters are keyed by labels known AT CAPTURE time. Runtime is therefore always
``unknown``: a later attribution is a revisable ledger/gauge fact and never moves a
persisted count between label sets. Each ``(execution_id, phase)`` is counted at most
once per epoch (the ``counted`` key set), so duplicates, reordering, repeated batches
and restarts cannot double count. Historical backfill is a different type
(:class:`BackfillSeries`) that can never be merged into :class:`LiveCounters`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting.contracts import ExecutionEvent, Phase
from ssgrep.accounting.counters_hist import (
    BUCKET_BOUNDS_US,
    Histogram,
    as_day,
    as_int,
    as_list,
    as_map,
    as_str,
    as_utc,
    bucket_index,
)
from ssgrep.accounting.reduce_types import InvocationRecord

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
