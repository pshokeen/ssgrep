"""Neutral, wire-format-free export model shared by projection, OTLP serialization, and delivery.

Everything here is already sanitized: only values produced by the positive allowlist
projections in ``export.py`` / ``export_records.py`` may be put into these types. All numbers
are ``int``; decimal fractions that OTLP requires (histogram bounds and sums, in seconds) are
carried as exact DECIMAL LITERAL STRINGS so no Python float ever touches accounting data.
"""

from __future__ import annotations

from dataclasses import dataclass

from ssgrep.accounting.ledger_contracts import Signal

MAX_BATCH_BYTES = 1024 * 1024
SERVICE_NAME = "ssgrep"
SCOPE_NAME = "ssgrep.accounting"


@dataclass(frozen=True)
class Resource:
    """Approved pseudonymous resource attributes (design D10).

    service name/version, a destination-scoped installation pseudonym, the policy version, and
    (metrics only) the persisted counter epoch. No organization subject, tenant, or device name
    is ever asserted by the client: the collector derives authoritative identity from the
    enrolled credential and overwrites or rejects anything the payload claims.
    """

    service_version: str
    installation_id: str
    policy_version: int
    counter_epoch: int | None = None


@dataclass(frozen=True)
class Hist:
    """Explicit-bounds histogram data point. ``counts`` are PER-BUCKET (not cumulative)."""

    bounds: tuple[str, ...]  # ascending finite upper bounds as decimal literals (seconds)
    counts: tuple[int, ...]  # len(bounds) + 1; the last bucket is +Inf
    count: int
    sum: str  # exact decimal literal in the metric's unit (seconds)


@dataclass(frozen=True)
class Point:
    """One OTLP data point (sum, gauge or histogram) with its metric metadata."""

    name: str
    kind: str  # "sum" (monotonic, cumulative) | "gauge" | "histogram" (cumulative)
    unit: str  # UCUM: "s", "By", "1", "{execution}" ...
    description: str
    attributes: tuple[tuple[str, str], ...]
    time_unix_nano: int
    value: int | None = None  # sum / gauge
    hist: Hist | None = None
    start_unix_nano: int | None = None  # REQUIRED for sum/histogram: persisted epoch start


@dataclass(frozen=True)
class LogRecord:
    """One OTLP log record; ``body`` is the canonical JSON text of a ledger record body."""

    time_unix_nano: int
    body: str
    attributes: tuple[tuple[str, str | int], ...] = ()  # int -> AnyValue.intValue (string)
    severity_number: int = 9  # INFO


@dataclass(frozen=True)
class Batch:
    """One immutable request: exact bytes plus the identities needed to track it."""

    batch_id: str
    signal: Signal
    generation: int
    snapshot_id: str
    index: int
    total: int
    kind: str  # "data" | "tombstone"
    body: bytes
    items: int  # data points or log records in the request
    record_ids: tuple[str, ...] = ()


FIELDS_VERSION = {Signal.METRICS.value: "metrics-v1", Signal.LOGS.value: "logs-v1"}
