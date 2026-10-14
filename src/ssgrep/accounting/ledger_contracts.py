"""Derived-state contracts: policy, receipts, snapshots, and delivery state.

Everything here is *derived* from evidence and recomputable; none of it is
authoritative for what happened. Canonical serialization excludes collection
clocks so replay equality holds (design D7/D8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from ssgrep.accounting.contracts import PERSONAL_PROFILE, SCHEMA_VERSION, Quantity, Scenario

PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_PROFILES = 8
MAX_DESTINATIONS = 8
MAX_SERIES = 5000
WINDOWS = ("24h", "7d", "30d", "lifetime")
DEFAULT_STALE_INTERVALS = 3
DEFAULT_COLLECT_SECONDS = 60
DEFAULT_EXPORT_SECONDS = 300
DEFAULT_PORT = 9469
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


@dataclass(frozen=True)
class SourceAtom:
    """Immutable source span; ``digest`` is verified against resolved content."""

    atom_id: str
    origin_event: str
    digest: str
    chars: int
    seconds: int | None = None


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


@dataclass(frozen=True)
class ScenarioEstimate:
    scenario: Scenario
    credit: Quantity
    debit: Quantity
    net: Quantity


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
