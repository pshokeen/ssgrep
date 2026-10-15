"""Bounded metric catalog and snapshot-to-sample projection (design D9; integers only).

``snapshot_samples`` is a pure function of a ``SnapshotEnvelope``; ``export_samples`` of
delivery health. Both yield :class:`Sample` values whose label names/values are checked
against the fixed catalog, so no task/employee/path/ref/date/digest/error-text label can
exist. A number the snapshot does not hold produces NO sample (never a zero); the
``ssgrep_coverage_records`` gauge says how many were omitted. Samples carry no timestamps:
freshness is the ``ssgrep_snapshot_collected_timestamp_seconds`` gauge.

Histogram ``_sum`` samples carry integer MICROSECONDS (``kind="histogram_sum"``); the
OpenMetrics renderer prints seconds. Counter samples are named ``*_total``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting.contracts import (
    RUNTIMES,
    Attribution,
    Operation,
    Outcome,
    Reuse,
    Scenario,
    Transport as TransportEnum,
)
from ssgrep.accounting.counters_hist import BUCKET_BOUNDS_US
from ssgrep.accounting.ledger_contracts import (
    MAX_DESTINATIONS,
    MAX_SERIES,
    WINDOWS,
    Signal,
    SnapshotEnvelope,
    valid_profile_id,
)
from ssgrep.accounting.metric_estimates import ADDITIONAL_FAMILIES, METHODS, estimate_samples
from ssgrep.accounting.metrics_render import Sample, ValidationError

__all__ = [
    "FAMILIES",
    "ExportHealth",
    "MetricsValidationError",
    "Sample",
    "all_samples",
    "export_samples",
    "snapshot_samples",
    "validate_dimensions",
    "validate_samples",
]

MetricsValidationError = ValidationError


@dataclass(frozen=True)
class Family:
    name: str  # public name as in the design: ssgrep_x_total / ssgrep_x / ssgrep_x_info
    kind: str  # counter | gauge | histogram | info
    labels: tuple[str, ...]


FAMILIES: tuple[Family, ...] = (
    Family("ssgrep_executions_total", "counter", ("operation", "transport", "outcome", "runtime")),
    Family("ssgrep_rendered_bytes_total", "counter", ("operation", "transport")),
    Family("ssgrep_export_rejected_points_total", "counter", ("destination", "signal")),
    Family("ssgrep_export_lost_records_total", "counter", ("destination", "reason")),
    Family("ssgrep_operation_duration_seconds", "histogram", ("operation", "transport")),
    Family("ssgrep_reconciliation_records", "gauge", ("attribution",)),
    Family("ssgrep_reuse_evidence_records", "gauge", ("evidence",)),
    Family("ssgrep_estimated_credit_tokens", "gauge", ("scenario", "window", "method")),
    Family("ssgrep_debit_tokens", "gauge", ("window", "method")),
    Family("ssgrep_estimated_net_tokens", "gauge", ("scenario", "window", "method")),
    Family("ssgrep_estimated_net_work_seconds", "gauge", ("scenario", "window", "method")),
    Family("ssgrep_coverage_records", "gauge", ("kind", "status")),
    Family("ssgrep_snapshot_collected_timestamp_seconds", "gauge", ()),
    Family("ssgrep_snapshot_available", "gauge", ()),
    Family("ssgrep_export_last_full_ack_timestamp_seconds", "gauge", ("destination", "signal")),
    Family("ssgrep_export_pending_batches", "gauge", ("destination", "signal")),
    Family("ssgrep_accounting_info", "info", ("schema_version", "methodology_id")),
    *(Family(name, "gauge", labels) for name, labels in ADDITIONAL_FAMILIES),
)
_BY_NAME = {family.name: family for family in FAMILIES}
_SAMPLE_FAMILY = {
    **_BY_NAME,
    **{
        f"ssgrep_operation_duration_seconds{s}": _BY_NAME["ssgrep_operation_duration_seconds"]
        for s in ("_bucket", "_sum", "_count")
    },
}
FORBIDDEN_LABELS = frozenset(
    {
        "employee",
        "user",
        "task",
        "task_id",
        "query",
        "ref",
        "path",
        "fact",
        "error",
        "message",
        "date",
        "day",
        "receipt",
        "receipt_id",
        "digest",
        "snapshot_id",
        "revision",
        "session",
    }
)
_ID = re.compile(r"[a-z0-9][a-z0-9_.:+-]{0,127}")
_DOMAINS: dict[str, frozenset[str]] = {
    "operation": frozenset(o.value for o in Operation),
    "transport": frozenset(t.value for t in TransportEnum),
    "outcome": frozenset(o.value for o in Outcome),
    "runtime": frozenset((*RUNTIMES, "unknown")),
    "attribution": frozenset(a.value for a in Attribution),
    "evidence": frozenset(r.value for r in Reuse),
    "scenario": frozenset(s.value for s in Scenario),
    "window": frozenset(WINDOWS),
    "signal": frozenset(s.value for s in Signal),
    "method": METHODS | {"unknown"},
    "quantity": frozenset({"credit", "debit", "net"}),
    "bound": frozenset({"lower", "upper"}),
}
LE = "le"


def validate_dimensions(family: str, labels: Iterable[str]) -> None:
    """Reject a requested label set that is not within the family's bounded catalog labels."""
    known = _BY_NAME.get(family)
    if known is None:
        raise MetricsValidationError("unknown_family", f"{family!r} is not a catalog family")
    names = set(labels)
    forbidden = sorted(names & FORBIDDEN_LABELS)
    if forbidden:
        raise MetricsValidationError("forbidden_label", f"labels {forbidden} are private")
    extra = sorted(names - set(known.labels))
    if extra:
        raise MetricsValidationError(
            "unbounded_label", f"labels {extra} are not catalog dimensions"
        )


def _check_value(name: str, value: str) -> None:
    if name in _DOMAINS:
        ok = value in _DOMAINS[name]
    elif name == "destination":
        ok = valid_profile_id(value)
    elif name == "schema_version":
        ok = value.isdigit() and len(value) <= 6
    else:
        # Names have already passed the catalog dimension check. The remaining
        # catalog labels (method, reason, kind, status, methodology_id) are identifiers.
        ok = bool(_ID.fullmatch(value))
    if not ok:
        raise MetricsValidationError("invalid_label_value", f"label {name} has an unbounded value")


def _kind_of(name: str, family: Family) -> str:
    if family.kind != "histogram":
        return family.kind
    return {"_bucket": "histogram_bucket", "_count": "histogram_count", "_sum": "histogram_sum"}[
        name[len(family.name) :]
    ]


def _sample(name: str, labels: Mapping[str, str], value: int, kind: str | None = None) -> Sample:
    """Build a catalog-shaped point; callers fix names/keys, while values remain untrusted."""
    family = _SAMPLE_FAMILY[name]
    kind = kind or _kind_of(name, family)
    validate_dimensions(family.name, [n for n in labels if n != LE])
    for label, text in labels.items():
        if label != LE:
            _check_value(label, text)
    if type(value) is not int or (kind in ("counter", "histogram_bucket") and value < 0):
        raise MetricsValidationError("invalid_value", f"{name} needs an integer (counters >= 0)")
    ordered = tuple((n, labels[n]) for n in (*family.labels, *((LE,) if LE in labels else ())))
    return Sample(name, kind, ordered, value)


def validate_samples(samples: Sequence[Sample]) -> None:
    """Series and destination limits; raises instead of silently truncating."""
    series = {(s.family, s.labels) for s in samples}
    if len(series) != len(samples):
        raise MetricsValidationError("duplicate_series", "two samples share one series identity")
    if len(series) > MAX_SERIES:
        raise MetricsValidationError("series_limit", f"{len(series)} series exceeds {MAX_SERIES}")
    destinations = {v for s in samples for n, v in s.labels if n == "destination"}
    if len(destinations) > MAX_DESTINATIONS:
        raise MetricsValidationError(
            "destination_limit", f"{len(destinations)} destinations exceeds {MAX_DESTINATIONS}"
        )


def _int(value: Any) -> int | None:
    return value if type(value) is int else None


def _rows(raw: Any) -> list[dict[str, Any]]:
    return [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []


def _le_seconds(le_us: str) -> str:
    """Prometheus-client style ``le`` text: ``0.001``, ``1.0``, ``30.0``, ``+Inf``."""
    if le_us == "+Inf":
        return le_us
    seconds, rest = divmod(int(le_us), 10**6)
    return f"{seconds}.{rest:06d}".rstrip("0").rstrip(".") + (".0" if not rest else "")


_LE_EXPECTED = [*(str(b) for b in BUCKET_BOUNDS_US), "+Inf"]


def _histogram(row: dict[str, Any]) -> list[Sample]:
    labels = {"operation": str(row.get("operation")), "transport": str(row.get("transport"))}
    count, total = _int(row.get("count")), _int(row.get("sum_us"))
    buckets = _rows(row.get("buckets"))
    if count is None or total is None or [str(b.get("le_us")) for b in buckets] != _LE_EXPECTED:
        raise MetricsValidationError("invalid_histogram", "duration histogram is malformed")
    raw_values = [_int(b.get("count")) for b in buckets]
    values = [v for v in raw_values if v is not None]
    if len(values) != len(raw_values) or values != sorted(values) or values[-1] != count:
        raise MetricsValidationError("invalid_histogram", "histogram buckets are not cumulative")
    name = "ssgrep_operation_duration_seconds"
    out = [
        _sample(f"{name}_bucket", {**labels, LE: _le_seconds(str(b["le_us"]))}, v)
        for b, v in zip(buckets, values, strict=True)
    ]
    out.append(_sample(f"{name}_count", labels, count))
    out.append(_sample(f"{name}_sum", labels, total))
    return out


def _counter_samples(counters: Mapping[str, Any]) -> list[Sample]:
    out: list[Sample] = []
    for row in _rows(counters.get("executions")):
        count = _int(row.get("count"))
        if count is not None:
            keys = ("operation", "transport", "outcome", "runtime")
            out.append(
                _sample("ssgrep_executions_total", {k: str(row.get(k)) for k in keys}, count)
            )
    for row in _rows(counters.get("rendered_bytes")):
        size = _int(row.get("bytes"))
        if size is not None:
            keys = ("operation", "transport")
            out.append(
                _sample("ssgrep_rendered_bytes_total", {k: str(row.get(k)) for k in keys}, size)
            )
    for row in _rows(counters.get("duration")):
        out.extend(_histogram(row))
    return out


def _gauge_block(raw: Any, family: str, label: str, domain: Iterable[str]) -> list[Sample]:
    block = raw if isinstance(raw, dict) else {}
    return [
        _sample(family, {label: name}, value)
        for name in sorted(domain)
        if (value := _int(block.get(name))) is not None
    ]


def _coverage(
    envelope: SnapshotEnvelope, counters: Mapping[str, Any] | None, available: int, missing: int
) -> list[Sample]:
    cov = envelope.coverage
    pairs: list[tuple[str, str, int | None]] = []
    recorded, attributed = _int(cov.get("recorded_lookups")), _int(cov.get("attributed_lookups"))
    pairs += [("lookups", "recorded", recorded), ("lookups", "attributed", attributed)]
    if recorded is not None and attributed is not None and recorded >= attributed:
        pairs.append(("lookups", "unattributed", recorded - attributed))
    pairs += [
        ("source_costs", "available", _int(cov.get("source_costs_available"))),
        ("source_costs", "unavailable", _int(cov.get("unavailable_sources"))),
        ("candidates", "receipts", _int(cov.get("candidate_receipts"))),
        ("candidates", "ambiguous", _int(cov.get("ambiguous_joins"))),
        ("replay", "sessions_unavailable", _int(cov.get("replay_unavailable_sessions"))),
        ("replay", "frequency_unavailable", _int(cov.get("frequency_revision_unavailable"))),
    ]
    sources = cov.get("sources")
    for status in ("total", "complete", "missing", "unreadable"):
        pairs.append(
            ("sources", status, _int(sources.get(status)) if isinstance(sources, dict) else None)
        )
    if counters is not None:
        for kind, key in (
            ("rendered_bytes", "rendered_unknown"),
            ("durations", "duration_unknown"),
        ):
            raw = counters.get(key)
            counts = [_int(row.get("count")) for row in _rows(raw)]
            value = (
                sum(count for count in counts if count is not None)
                if isinstance(raw, list)
                and len(counts) == len(raw)
                and all(count is not None and count >= 0 for count in counts)
                else None
            )
            pairs.append((kind, "unknown", value))
    pairs += [("estimates", "available", available), ("estimates", "missing", missing)]
    return [
        _sample("ssgrep_coverage_records", {"kind": k, "status": s}, v)
        for k, s, v in pairs
        if v is not None
    ]


def snapshot_samples(envelope: SnapshotEnvelope) -> list[Sample]:
    """Every sample the snapshot can honestly support (missing numbers omitted)."""
    counters = envelope.observations.get("counters")
    counters = counters if isinstance(counters, dict) else None
    estimates, available, missing = estimate_samples(envelope.windows, _sample)
    out = _counter_samples(counters) if counters else []
    out += _gauge_block(
        envelope.coverage.get("attribution"),
        "ssgrep_reconciliation_records",
        "attribution",
        _DOMAINS["attribution"],
    )
    out += _gauge_block(
        envelope.coverage.get("reuse_evidence"),
        "ssgrep_reuse_evidence_records",
        "evidence",
        _DOMAINS["evidence"],
    )
    out += estimates
    out += _coverage(envelope, counters, available, missing)
    try:
        collected = int(
            datetime.strptime(envelope.collected_at, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=UTC)
            .timestamp()
        )
    except ValueError:
        pass
    else:
        out.append(_sample("ssgrep_snapshot_collected_timestamp_seconds", {}, collected))
    out.append(_sample("ssgrep_snapshot_available", {}, 1))
    info = {
        "schema_version": str(envelope.schema_version),
        "methodology_id": envelope.methodology_id,
    }
    out.append(_sample("ssgrep_accounting_info", info, 1))
    return out


@dataclass(frozen=True)
class ExportHealth:
    """One destination+signal as exporter health; ``None`` = not known (sample omitted)."""

    destination: str
    signal: str
    last_full_ack_epoch: int | None = None
    pending_batches: int | None = None
    rejected_points: int | None = None
    lost: tuple[tuple[str, int], ...] = ()  # (reason, records)


def export_samples(health: Sequence[ExportHealth]) -> list[Sample]:
    out: list[Sample] = []
    lost: dict[tuple[str, str], int] = {}
    for item in health:
        labels = {"destination": item.destination, "signal": item.signal}
        for family, value in (
            ("ssgrep_export_rejected_points_total", item.rejected_points),
            ("ssgrep_export_last_full_ack_timestamp_seconds", item.last_full_ack_epoch),
            ("ssgrep_export_pending_batches", item.pending_batches),
        ):
            if value is not None:
                out.append(_sample(family, labels, value))
        for reason, records in item.lost:
            key = (item.destination, reason)
            lost[key] = lost.get(key, 0) + records
    for (destination, reason), records in sorted(lost.items()):
        out.append(
            _sample(
                "ssgrep_export_lost_records_total",
                {"destination": destination, "reason": reason},
                records,
            )
        )
    return out


def all_samples(envelope: SnapshotEnvelope | None, health: Sequence[ExportHealth]) -> list[Sample]:
    """Snapshot samples plus exporter health; without a snapshot ONLY availability + health."""
    samples = (
        snapshot_samples(envelope)
        if envelope is not None
        else [_sample("ssgrep_snapshot_available", {}, 0)]
    )
    samples += export_samples(health)
    validate_samples(samples)
    return samples
