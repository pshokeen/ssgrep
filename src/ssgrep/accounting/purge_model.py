"""Types of a privacy purge: stable errors and the four-tier report (no paths, no contents).

The report keeps four things apart, because only the first is something this product can do:

* ``local_deleted``: exact counts of what was (or, in a dry run, would be) removed here;
* ``queued_tombstones``: deletion requests spooled for the next export cycle (bounded retention);
* ``accepted_by_collector``: ALWAYS 0 at purge time, a tombstone is only queued;
* ``backend_erased``: ALWAYS ``unknown_operator_responsibility``; erasure belongs to the operator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

BACKEND_ERASED = "unknown_operator_responsibility"
MAX_OBLIGATION_IDS = 1000
_QUEUING = ("queued", "would_queue")
METRICS_NOTE = "aggregate_metrics_already_accepted_cannot_be_recalled"
NOTES = (
    "source_transcripts_untouched",
    "search_index_untouched",
    "other_profiles_untouched",
    "live_counters_retained",
    "source_read_offsets_retained",
    "pinned_frequency_view_retained",
)


class PurgeError(Exception):
    """A purge refusal or failure with a stable condition (never raw filesystem text)."""

    def __init__(
        self,
        condition: str,
        message: str,
        *,
        exit_code: int = 1,
        command: str = "ssgrep metrics status",
    ) -> None:
        super().__init__(message)
        self.condition = condition
        self.exit_code = exit_code
        self.command = command


@dataclass(frozen=True)
class LocalDeleted:
    evidence_records: int = 0
    evidence_segments_rewritten: int = 0
    ledger_items: int = 0
    ledger_period_rows: int = 0
    ledger_receipt_rows: int = 0
    executions: int = 0
    sealed_days: int = 0
    backfill_series: int = 0
    receipt_files: int = 0
    history_rows: int = 0
    snapshot_generations: int = 0
    journal_segments: int = 0
    spool_batches: int = 0
    delivery_pins_released: int = 0

    def total(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class TombstoneLine:
    """What happened (or would happen) to one destination's deletion requests."""

    destination: str
    records: int
    batches: int | None  # None: a dry run could not size the batches
    status: str  # queued | would_queue | not_permitted | skipped
    reason: str | None


@dataclass(frozen=True)
class Obligation:
    """Pseudonymous record ids the OPERATOR must erase: sending was not permitted."""

    destination: str
    reason: str | None
    records: int
    record_ids: tuple[str, ...]  # at most MAX_OBLIGATION_IDS


@dataclass(frozen=True)
class Retained:
    unknown_scope_conversations: int = 0
    mixed_scope_conversations: int = 0
    unknown_scope_executions: int = 0
    sealed_days_not_splittable: int = 0
    backfill_series_not_splittable: int = 0
    journal_segments_retained: int = 0
    in_flight_batches_not_recallable: int = 0


@dataclass(frozen=True)
class WorkerView:
    running: bool
    kind: str | None


@dataclass(frozen=True)
class PurgeReport:
    profile: str
    performed: bool
    before: str
    cutoff_day: str
    workspace: str | None
    local_deleted: LocalDeleted
    tombstones: tuple[TombstoneLine, ...] = ()
    obligations: tuple[Obligation, ...] = ()
    non_reversible_destinations: tuple[str, ...] = ()
    retained: Retained = field(default_factory=Retained)
    pending_retention_days: int = 7
    notes: tuple[str, ...] = NOTES
    worker: WorkerView | None = None

    @property
    def queued_records(self) -> int:
        return sum(t.records for t in self.tombstones if t.status in _QUEUING)

    @property
    def queued_batches(self) -> int:
        return sum(t.batches or 0 for t in self.tombstones if t.status in _QUEUING)

    @property
    def empty(self) -> bool:
        return (
            self.local_deleted.total() == 0
            and not self.tombstones
            and not self.obligations
            and not self.non_reversible_destinations
        )

    @property
    def state(self) -> str:
        if self.empty:
            return "nothing_to_purge"
        return "purged" if self.performed else "dry_run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "performed": self.performed,
            "scope": {
                "before": self.before,
                "cutoff_day": self.cutoff_day,
                "workspace": self.workspace,
            },
            "local_deleted": asdict(self.local_deleted),
            "queued_tombstones": {
                "records": self.queued_records,
                "batches": self.queued_batches,
                "pending_retention_days": self.pending_retention_days,
                "destinations": [asdict(t) for t in self.tombstones],
            },
            "accepted_by_collector": {
                "count": 0,
                "note": "tombstones_are_queued_and_delivered_by_the_next_export_cycle",
            },
            "backend_erased": BACKEND_ERASED,
            "operator_obligations": [asdict(o) for o in self.obligations],
            "non_reversible_metrics": [
                {"destination": d, "note": METRICS_NOTE} for d in self.non_reversible_destinations
            ],
            "retained": {**asdict(self.retained), "notes": list(self.notes)},
            "worker": None if self.worker is None else asdict(self.worker),
        }
