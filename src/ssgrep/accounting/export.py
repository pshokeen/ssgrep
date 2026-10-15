"""Positive per-signal export projection: snapshot aggregates -> sanitized OTLP data points.

Nothing is "stripped from" the snapshot. Each exported datum is *constructed* from an
explicit family table, and every attribute value must pass a closed validator; whatever does
not match is counted (never copied) in ``Projection.omitted``. Consequently a query, path,
fact, title, ref, transcript text, receipt content, raw error body, token, or arbitrary
``warnings``/``coverage``/``windows`` string planted anywhere in the snapshot cannot reach a
payload: the only strings that can appear are the catalog's fixed family names, the bounded
enumerations of ``metrics``/``contracts``, a method label that matches the method allowlist,
and a lowercase-hex methodology digest. Numbers are integers (histogram seconds are exact
decimal literals derived from integer microseconds).

Metrics identity: resource attributes are pseudonymous service/version/installation/policy
(+ the persisted counter epoch). No subject, tenant, device name, host, or path is ever
asserted; the collector derives authoritative identity from the enrolled credential.
Timestamps are the snapshot's own ``collected_at`` and the persisted counter-epoch start, so
re-projecting the same snapshot yields the same bytes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting.export_model import FIELDS_VERSION, Hist, Point, Resource
from ssgrep.accounting.ledger_contracts import WINDOWS, Signal, SnapshotEnvelope
from ssgrep.accounting.metric_estimates import ADDITIONAL_FAMILIES
from ssgrep.accounting.metrics import Sample, snapshot_samples

METRICS_FIELDS = FIELDS_VERSION[Signal.METRICS.value]
_METHOD = re.compile(
    r"(estimate:)?(chars_div4_v1|capped_gap_v1|transcript_intervals_v1|monotonic_v1|"
    r"provider_usage_v1)|estimate:[0-9a-f]{16,64}|exact:[a-z0-9][a-z0-9_.-]{0,47}|unknown"
)
_HEX = re.compile(r"[0-9a-f]{16,64}")
OTHER = "other"
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def safe_method(value: object) -> str:
    """A measurement-method label from the allowlist; anything else is the constant ``other``."""
    return value if isinstance(value, str) and _METHOD.fullmatch(value) else OTHER


# Closed vocabulary of the reasons the reducers/valuation can attach to an unavailable quantity.
# Shape alone is never enough (a token-like string would pass any lexical test), so a reason is
# exported only when it IS one of these codes. Codes that are built with a free value suffix
# (``no_applicable_tariff:<category>``, ``tokenizer_*:<library/version/error>``) lose the suffix.
UNSPECIFIED = "unspecified"
REASON_CODES = frozenset(
    {
        UNSPECIFIED,
        "debit_chars_unavailable",
        "conflicting_duplicate_debit",
        "downstream_exposure_unobserved",
        "evidence_expired",
        "exact_tokens_unavailable",
        "history_unreadable",
        "invalid_timing_in_gaps",
        "lookup_work_seconds_not_modeled",
        "malformed_quantity",
        "mixed_shell_output_unattributable",
        "monotonic_clock_reversed",
        "no_closed_periods",
        "no_dependency_evidence",
        "no_journal_event_in_capture_span",
        "no_render_event",
        "no_runtime_exposure_joined",
        "no_tariff",
        "not_estimated",
        "output_shared_by_several_executions",
        "provider_cost_missing",
        "rendered_bytes_missing_for_some_executions",
        "scenario_missing",
        "source_costs_missing",
        "source_seconds_unavailable",
        "tokenizer_file_digest_mismatch",
        "tokenizer_file_missing",
        "unit_not_in_period",
        "visible_bytes_missing_for_some_executions",
        "journal_monotonic_duration_missing",
        "conflicting_journal_timing",
        "ambiguous_journal_timing",
        "journal_cardinality_unbounded",
        "no_linked_journal_timing",
        "source_missing_clock",
        "source_clock_reversed",
        "source_terminal_gap",
        "source_task_boundary_gap",
        "source_work_method_incompatible",
        "source_ownership_unknown",
        "rarity_unknown",
        "prior_context_unavailable",
        "legacy_valuation_uncertainty_unavailable",
        "credit_upper_unbounded",
        "debit_upper_unbounded",
        "net_lower_unbounded_debit",
        "net_upper_unbounded_credit",
        "measured_service_ms_missing_for_some_executions",
        "lookup_work_nanoseconds_unavailable",
        "journal_nanosecond_precision_unavailable",
        "nanosecond_precision_unavailable",
        "time_precision_namespace_migration_required",
    }
)
_SUFFIXED_CODES = frozenset(
    {
        "no_applicable_tariff",
        "tokenizer_library_not_installed",
        "tokenizer_load_failed",
        "tokenizer_version_mismatch",
        "unsupported_tokenizer_library",
    }
)
_INCOMPLETE = "incomplete:"


def _reason_code(member: str) -> str | None:
    head, _, _ = member.partition(":")
    if head in _SUFFIXED_CODES:
        return head
    return member if member in REASON_CODES else None


def safe_reason(value: object) -> str:
    """A reason from the closed vocabulary (or ``incomplete:`` plus members of it).

    Anything else, including a single unknown member inside ``incomplete:``, collapses the whole
    reason to the constant ``unspecified``: free text never reaches a payload.
    """
    if not isinstance(value, str):
        return UNSPECIFIED
    if value.startswith(_INCOMPLETE):
        members = [_reason_code(m) for m in value[len(_INCOMPLETE) :].split(",")]
        if not members or any(m is None for m in members):
            return UNSPECIFIED
        return _INCOMPLETE + ",".join(sorted({m for m in members if m is not None}))
    return _reason_code(value) or UNSPECIFIED


def unix_nano(stamp: str) -> int:
    """UTC ``YYYY-MM-DDTHH:MM:SSZ`` -> nanoseconds since the epoch (ValueError otherwise)."""
    try:
        moment = datetime.strptime(stamp, _UTC_FORMAT).replace(tzinfo=UTC)
    except (TypeError, ValueError):
        raise ValueError("invalid_timestamp") from None  # never quote the offending value
    return int(moment.timestamp()) * 1_000_000_000


# family -> (OTLP name, kind, unit, description). Sums drop the Prometheus ``_total`` suffix:
# the collector's Prometheus exporter re-adds it, and OTLP names stay idiomatic.
_TABLE: dict[str, tuple[str, str, str, str]] = {
    "ssgrep_executions_total": (
        "ssgrep_executions",
        "sum",
        "{execution}",
        "Captured search/show/note executions",
    ),
    "ssgrep_rendered_bytes_total": (
        "ssgrep_rendered_bytes",
        "sum",
        "By",
        "Rendered response bytes where observable",
    ),
    "ssgrep_operation_duration_seconds": (
        "ssgrep_operation_duration_seconds",
        "histogram",
        "s",
        "Live captured operation duration",
    ),
    "ssgrep_reconciliation_records": (
        "ssgrep_reconciliation_records",
        "gauge",
        "{record}",
        "Lookups by attribution state",
    ),
    "ssgrep_reuse_evidence_records": (
        "ssgrep_reuse_evidence_records",
        "gauge",
        "{record}",
        "Candidates by reuse evidence",
    ),
    "ssgrep_estimated_credit_tokens": (
        "ssgrep_estimated_credit_tokens",
        "gauge",
        "{token}",
        "Modeled credit (not causal proof)",
    ),
    "ssgrep_debit_tokens": ("ssgrep_debit_tokens", "gauge", "{token}", "Known lookup debit"),
    "ssgrep_estimated_net_tokens": (
        "ssgrep_estimated_net_tokens",
        "gauge",
        "{token}",
        "Modeled net context tokens; may be < 0",
    ),
    "ssgrep_estimated_net_work_seconds": (
        "ssgrep_estimated_net_work_seconds",
        "gauge",
        "s",
        "Modeled net agent work; may be < 0",
    ),
    "ssgrep_coverage_records": (
        "ssgrep_coverage_records",
        "gauge",
        "{record}",
        "Evidence coverage counts",
    ),
    "ssgrep_snapshot_collected_timestamp_seconds": (
        "ssgrep_snapshot_collected_timestamp_seconds",
        "gauge",
        "s",
        "Snapshot collection time",
    ),
    "ssgrep_snapshot_available": (
        "ssgrep_snapshot_available",
        "gauge",
        "{state}",
        "Snapshot exists",
    ),
    "ssgrep_accounting_info": (
        "ssgrep_accounting_info",
        "gauge",
        "{info}",
        "Schema and methodology identifiers (value 1)",
    ),
}
for _name, _labels in ADDITIONAL_FAMILIES:
    _unit = (
        "ms"
        if _name.endswith("_milliseconds")
        else "s"
        if _name.endswith("_work_seconds")
        else "{record}"
        if _name.endswith("_records")
        else "{token}"
    )
    _TABLE[_name] = (
        _name,
        "gauge",
        _unit,
        "Known subtotal, interval bound or evidence coverage; not causal or human time savings",
    )
_ENUM_LABELS = frozenset(
    {"operation", "transport", "outcome", "runtime", "attribution", "evidence", "scenario",
     "window", "kind", "status", "quantity", "bound"}
)  # fmt: skip
_BUCKET = re.compile(r"ssgrep_operation_duration_seconds_(bucket|count|sum)")


@dataclass(frozen=True)
class Projection:
    """The sanitized candidate for one snapshot/signal; ``omitted`` counts, never quotes."""

    points: tuple[Point, ...]
    omitted: int
    notes: tuple[str, ...] = ()


def _label_ok(name: str, value: str) -> bool:
    if name == "window":
        return value in WINDOWS
    if name in _ENUM_LABELS:
        return bool(re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value))
    if name == "method":
        return value == safe_method(value)
    if name == "schema_version":
        return value.isdigit() and len(value) <= 6
    assert name == "methodology_id"  # snapshot_samples enforces the fixed label catalog
    return bool(_HEX.fullmatch(value))


def _attributes(sample: Sample) -> tuple[tuple[str, str], ...] | None:
    out = []
    for name, value in sample.labels:
        if name == "le":
            continue
        if not _label_ok(name, value):
            return None
        out.append((name, value))
    return tuple(sorted(out))


def _decimal_seconds(micros: int) -> str:
    seconds, rest = divmod(micros, 1_000_000)
    return str(seconds) if not rest else f"{seconds}.{rest:06d}".rstrip("0")


def _epoch_start(envelope: SnapshotEnvelope) -> int | None:
    counters = envelope.observations.get("counters")
    stamp = counters.get("epoch_started_at") if isinstance(counters, dict) else None
    if not isinstance(stamp, str):
        epoch = envelope.observations.get("epoch")
        stamp = epoch.get("started_at") if isinstance(epoch, dict) else None
    try:
        return unix_nano(stamp) if isinstance(stamp, str) else None
    except ValueError:
        return None


def _histograms(
    groups: dict[tuple[tuple[str, str], ...], list[Sample]], now: int, start: int | None
) -> tuple[list[Point], int]:
    points: list[Point] = []
    omitted = 0
    for attrs, samples in sorted(groups.items()):
        buckets = [s for s in samples if s.family.endswith("_bucket")]
        count = next((s.value for s in samples if s.family.endswith("_count")), None)
        total = next((s.value for s in samples if s.family.endswith("_sum")), None)
        if start is None or count is None or total is None or not buckets:
            omitted += 1
            continue
        cumulative = [b.value for b in buckets]
        bounds = [dict(b.labels)["le"] for b in buckets][:-1]
        counts = [c - p for c, p in zip(cumulative, [0, *cumulative[:-1]], strict=True)]
        hist = Hist(tuple(bounds), tuple(counts), count, _decimal_seconds(total))
        name, _, unit, description = _TABLE["ssgrep_operation_duration_seconds"]
        points.append(Point(name, "histogram", unit, description, attrs, now, None, hist, start))
    return points, omitted


def project_metrics(envelope: SnapshotEnvelope) -> Projection:
    """Every datum the snapshot supports, built only from the family table above."""
    try:
        now = unix_nano(envelope.collected_at)
    except ValueError:
        return Projection((), 1, ("invalid_collected_at",))
    start = _epoch_start(envelope)
    points: list[Point] = []
    groups: dict[tuple[tuple[str, str], ...], list[Sample]] = {}
    omitted = 0
    for sample in snapshot_samples(envelope):
        entry = _TABLE.get(sample.family) or _TABLE.get(
            _BUCKET.sub("ssgrep_operation_duration_seconds", sample.family)
        )
        attrs = _attributes(sample)
        if entry is None or attrs is None or type(sample.value) is not int:
            omitted += 1
            continue
        if entry[1] == "histogram":
            groups.setdefault(attrs, []).append(sample)
            continue
        name, kind, unit, description = entry
        if kind == "sum" and start is None:
            omitted += 1  # a cumulative point without a persisted start time is never invented
            continue
        points.append(
            Point(name, kind, unit, description, attrs, now, sample.value, None,
                  start if kind == "sum" else None)
        )  # fmt: skip
    hist_points, hist_omitted = _histograms(groups, now, start)
    ordered = sorted(
        [*points, *hist_points], key=lambda p: (p.name, p.attributes, p.time_unix_nano)
    )
    notes = ("counters_omitted_no_epoch_start",) if start is None and omitted else ()
    return Projection(tuple(ordered), omitted + hist_omitted, notes)


def metrics_resource(
    *, version: str, installation: str, policy_version: int, envelope: SnapshotEnvelope
) -> Resource:
    """Metrics resource: pseudonymous identity plus the persisted counter epoch, nothing else."""
    return Resource(version, installation, policy_version, counter_epoch=envelope.counter_epoch)


def installed_version() -> str:
    """The ssgrep package version as ``service.version`` (the string ``unknown`` if absent)."""
    try:
        from importlib.metadata import version

        return version("ssgrep")
    except Exception:  # noqa: BLE001 - metadata lookup must never block an export preview
        return "unknown"


def describe_omitted(projection: Projection) -> dict[str, Any]:
    return {"omitted_data_points": projection.omitted, "notes": list(projection.notes)}
