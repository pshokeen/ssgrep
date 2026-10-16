"""Privacy purge of one profile's accounting data (framework-free; the CLI only renders it).

``plan`` is strictly read-only (a dry run is exactly this and nothing else). ``execute`` takes the
profile worker lease and applies the plan in a crash-safe, privacy-first order; every step is
idempotent, so a run that died anywhere is completed by running it again:

1. select (workspace by the cwd of retained evidence, cutoff by UTC day);
2. record ids of the selected, previously exported ledger records (export registry);
3. erase selected DATA bodies, including settled payloads, without rewriting acknowledgment
   history; mark active requests non-recallable and forbid their resend after settlement;
4. queue deletion tombstones while the registry linkage still exists;
5. commit the purged collector state (evidence, ledger, executions, sealed days) with a fresh
   snapshot generation in one checkpoint commit, then delete the old generations, stale
   receipts and wholly-old journal segments.

Only ``accounting/<profile>/`` is ever touched, always through ``storage`` descriptors. Nothing
here claims backend erasure: the report keeps local deletion, queued tombstones, collector
acceptance (always 0 at purge time) and backend erasure (always the operator's) apart.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.capture.journal_reader import read_journal
from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.cycle_publish import SnapshotInputs, build_snapshot
from ssgrep.accounting.collect.read import journal_last_ts, journal_stats
from ssgrep.accounting.collect.replay import LedgerState
from ssgrep.accounting.collect.state import ReductionDoc
from ssgrep.accounting.collect.steps import previous_coverage, source_gaps
from ssgrep.accounting.collect.store import Checkpoint, CollectError
from ssgrep.accounting.export import (
    records as export_records,
    registry as export_registry,
    spool as export_spool,
)
from ssgrep.accounting.export.acks import DeliveryCorrupt
from ssgrep.accounting.export.registry import RegistryCorrupt
from ssgrep.accounting.export.spool import SpoolCorrupt
from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE
from ssgrep.accounting.ledger.ledger_contracts import (
    ProfilePolicy,
    Signal,
    SnapshotEnvelope,
    empty_counters,
)
from ssgrep.accounting.privacy import (
    policy as policy_module,
    published as tombstones,
    purge_spool,
    storage,
)
from ssgrep.accounting.privacy.policy import (
    PolicyError,
    canonical_root,
    contains,
    load_profile_policy,
)
from ssgrep.accounting.serve import lease as lease_module
from ssgrep.accounting.views import snapshots
from ssgrep.accounting.views.types import check_profile, parse_instant

if TYPE_CHECKING:
    from ssgrep.accounting.privacy import purge_files, purge_state

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
    "shared_frequency_authority_invalidated_requires_explicit_rebuild",
)


class PurgeError(Exception):
    """A purge refusal or failure with a stable condition (never raw filesystem text)."""

    def __init__(
        self,
        condition: str,
        message: str,
        *,
        exit_code: int = 1,
        command: str = "ssgrep savings status",
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
    invocations: int = 0
    sealed_days: int = 0
    backfill_series: int = 0
    receipt_files: int = 0
    history_rows: int = 0
    snapshot_generations: int = 0
    journal_segments: int = 0
    spool_batches: int = 0
    delivery_pins_released: int = 0
    plaintext_cache_files: int = 0
    frequency_files: int = 0
    temporary_files: int = 0
    keys: int = 0
    derived_caches: int = 0
    archive_metadata_records: int = 0

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


USAGE = "ssgrep savings purge BEFORE_DATE [--workspace PATH] [--dry-run]"


@dataclass(frozen=True)
class Selection:
    profile: str
    before: str  # normalized UTC instant
    workspace: str | None  # canonical root path, None = the whole profile

    @property
    def cutoff_day(self) -> str:
        return self.before[:10]

    @property
    def cutoff_midnight(self) -> str:
        return f"{self.cutoff_day}T00:00:00Z"


def _usage(error: PolicyError, condition: str | None = None) -> PurgeError:
    return PurgeError(condition or error.condition, str(error), exit_code=2, command=USAGE)


def parse_selection(
    *, profile: str, before: str | None, workspace: str | None, policy: ProfilePolicy
) -> Selection:
    """Validate every argument; nothing has been read or written when this raises."""
    try:
        check_profile(profile)
    except PolicyError as error:
        raise _usage(error) from None
    if before is None or not before.strip():
        raise PurgeError(
            "invalid_interval",
            "--before is required: say which data to delete (a day or an RFC3339 instant). "
            "Nothing was changed.",
            exit_code=2,
            command=USAGE,
        )
    try:
        instant = parse_instant(before.strip(), "--before")
    except PolicyError as error:
        raise _usage(error) from None
    return Selection(profile, instant, _workspace(profile, workspace, policy))


def _workspace(profile: str, raw: str | None, policy: ProfilePolicy) -> str | None:
    if raw is None:
        return None
    try:
        chosen = canonical_root(raw)
    except PolicyError:
        raise PurgeError(
            "invalid_workspace",
            "--workspace must be an absolute path (or ~/...) other than '/'. Nothing was changed.",
            exit_code=2,
            command=USAGE,
        ) from None
    if not policy.roots:
        if profile == PERSONAL_PROFILE:
            return chosen
    else:
        for root in policy.roots:
            try:
                if contains(canonical_root(root), chosen):
                    return chosen
            except PolicyError:
                continue
    raise PurgeError(
        "workspace_not_in_scope",
        f"That workspace is not inside an approved root of profile {profile}. Nothing was changed.",
        exit_code=2,
        command="ssgrep savings status",
    )


class Scoper:
    """Whether a working directory lies inside the selected workspace (realpath cached)."""

    def __init__(self, workspace: str | None) -> None:
        self.workspace = workspace
        self._cache: dict[str, bool] = {}

    def inside(self, cwd: str | None) -> bool:
        if self.workspace is None:
            return True
        if not isinstance(cwd, str) or not cwd or "\0" in cwd or not os.path.isabs(cwd):
            return False
        known = self._cache.get(cwd)
        if known is None:
            known = self._cache[cwd] = contains(self.workspace, os.path.realpath(cwd))
        return known


def evidence_cwd(payload: dict) -> str | None:
    for name in ("cwd", "directory"):
        value = payload.get(name)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass(frozen=True)
class Scopes:
    """Which conversation roots are selected; ``selected`` None means every root."""

    selected: frozenset[str] | None
    root_of: dict[tuple[str, str], str] = field(default_factory=dict)
    unknown: int = 0
    mixed: int = 0

    def root(self, runtime: str | None, session: str | None) -> str | None:
        if runtime is None or session is None:
            return None
        return self.root_of.get((runtime, session), f"{runtime}:{session}")

    def has_root(self, root: str | None) -> bool:
        return self.selected is None or (root is not None and root in self.selected)

    def has_session(self, runtime: str | None, session: str | None) -> bool:
        return self.has_root(self.root(runtime, session))


def ledger_roots(states: Iterable[LedgerState]) -> set[str]:
    """Every conversation root the ledgers know (live evidence and frozen)."""
    roots: set[str] = set()
    for state in states:
        roots.update(c.task.root for c in state.candidates)
        roots.update(d.task.root for d in state.debits)
        roots.update(f.root for f in state.frozen)
    return roots


def classify(
    profile: str,
    checkpoint: Checkpoint | None,
    doc: ReductionDoc | None,
    known_roots: Iterable[str],
    scoper: Scoper,
) -> Scopes:
    """Select conversation roots by the working directories of their retained evidence."""
    root_of = {(rt, sid): root for rt, sid, root in (doc.session_roots if doc else ())}
    if scoper.workspace is None:
        return Scopes(None, root_of)
    cwds: dict[str, set[str | None]] = {}
    for runtime, session, cwd in doc.session_cwd if doc else ():
        root = root_of.get((runtime, session), f"{runtime}:{session}")
        cwds.setdefault(root, set()).add(cwd)
    if checkpoint is not None:
        for item in store.iter_evidence(profile, checkpoint.segments):
            root = root_of.get((item.runtime, item.session_id), f"{item.runtime}:{item.session_id}")
            cwds.setdefault(root, set()).add(evidence_cwd(item.payload))
    universe = set(cwds) | set(known_roots) | set(root_of.values())
    selected: set[str] = set()
    unknown = mixed = 0
    for root in sorted(universe):
        known = {c for c in cwds.get(root, ()) if c is not None}
        if not known:
            unknown += 1
        elif all(scoper.inside(c) for c in known):
            selected.add(root)
        elif any(scoper.inside(c) for c in known):
            mixed += 1
    return Scopes(frozenset(selected), root_of, unknown, mixed)


__all__ = ["Plan", "PurgeError", "PurgeReport", "Selection", "execute", "plan"]

_KNOWN_CONDITIONS = frozenset({"storage_unsafe", "unsupported_schema", "state_corrupt"})


def _empty_derived() -> purge_files.DerivedDeleted:
    from ssgrep.accounting.privacy import purge_files

    return purge_files.DerivedDeleted()


@dataclass
class Plan:
    """Everything a real run would do, computed without writing anything."""

    selection: Selection
    policy: ProfilePolicy
    present: bool = False
    state: purge_state.StatePlan | None = None
    selected_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    tomb_plans: list[tombstones.TombstonePlan] = field(default_factory=list)
    unlisted: list[Obligation] = field(default_factory=list)
    spool: purge_spool.SpoolPlan = field(default_factory=purge_spool.SpoolPlan)
    journal: tuple[str, ...] = ()
    journal_retained: int = 0
    stale_receipts: frozenset[str] = frozenset()
    old_generations: list[purge_files.Generation] = field(default_factory=list)
    unfinished: bool = False
    derived: purge_files.DerivedDeleted = field(default_factory=_empty_derived)

    @property
    def purges_something(self) -> bool:
        return bool(
            (self.state is not None and self.state.changed) or any(self.selected_ids.values())
        )


def _condition(error: BaseException) -> str:
    code = getattr(error, "condition", None)
    return code if code in _KNOWN_CONDITIONS else "state_corrupt"


def _failure(error: Exception, *, resumable: bool = False) -> PurgeError:
    """Map any step failure to a stable condition; raw filesystem/parser text never leaks."""
    from ssgrep.accounting.privacy import purge_files

    if isinstance(error, PurgeError):
        return error
    if isinstance(error, CollectError):
        message = (
            "Collector state could not be read. Purge may have committed changes; re-run to resume."
            if resumable
            else "Collector state could not be read. Nothing changed."
        )
        return PurgeError(_condition(error), message)
    if isinstance(error, storage.UnsafeStorageError):
        return purge_files.unsafe()
    if isinstance(error, RegistryCorrupt | SpoolCorrupt | DeliveryCorrupt):
        return PurgeError(
            "export_state_unreadable", "Export bookkeeping is unreadable; purge stopped safely."
        )
    return PurgeError("purge_failed", f"Purge failed ({type(error).__name__}); re-run to resume.")


def _destinations(profile: str, policy: ProfilePolicy) -> list[str]:
    known = {d.destination_id for d in policy.destinations}
    return sorted(known | set(purge_spool.export_destinations(profile)))


def _ids(profile: str, destination: str, sel: Selection, keys: frozenset[str] | None) -> list[str]:
    return export_registry.records_before(profile, destination, sel.cutoff_day, keys)


def _unlisted(profile: str, policy: ProfilePolicy, selected: dict[str, frozenset[str]]) -> list:
    """Destinations that left the policy: nothing may be sent, so the operator owes erasure."""
    known = {d.destination_id for d in policy.destinations}
    found: list[Obligation] = []
    for destination, ids in sorted(selected.items()):
        if destination in known or not ids:
            continue
        registry = export_registry.load_registry(profile, destination)
        in_flight = {
            rid
            for meta in export_spool.list_batches(profile, destination, Signal.LOGS)
            if meta.kind == "data" and meta.state == "in_flight"
            for rid in meta.record_ids
        }
        sent = sorted(
            i
            for i in ids
            if (e := registry.entries.get(i)) is not None
            and (e.acked is not None or e.unknown is not None or i in in_flight)
        )
        if sent:
            found.append(
                Obligation(
                    destination, "unknown_destination", len(sent), tuple(sent[:MAX_OBLIGATION_IDS])
                )
            )
    return found


def _root_keys(state: purge_state.StatePlan | None, sel: Selection) -> frozenset[str] | None:
    if sel.workspace is None:
        return None
    if state is None:
        return frozenset()
    roots = set(state.scopes.selected or ())
    for archive in state.legacy_archives:
        if archive.plan.scopes.selected is None:
            raise PurgeError("archive_scope_unresolved", "Archived workspace scope is unresolved.")
        roots.update(archive.plan.scopes.selected)
    return frozenset(export_records.root_key_of(root) for root in roots)


def plan(sel: Selection, *, policy: ProfilePolicy | None = None) -> Plan:
    """Read-only plan; raises ``PurgeError`` (never writes) when storage is unsafe/unreadable."""
    from ssgrep.accounting.privacy import purge_files, purge_state

    current = policy or policy_module.load_profile_policy(sel.profile)
    result = Plan(sel, current)
    try:
        result.present = purge_files.preflight(sel.profile)
        if not result.present:
            return result
        checkpoint = store.load_checkpoint(sel.profile)
        result.state = purge_state.plan_state(sel.profile, sel, checkpoint)
        keys = _root_keys(result.state, sel)
        for destination in _destinations(sel.profile, current):
            result.selected_ids[destination] = frozenset(_ids(sel.profile, destination, sel, keys))
        result.tomb_plans = tombstones.plan_tombstones(
            sel.profile, current, before_day=sel.cutoff_day, root_keys=keys
        )
        result.unlisted = _unlisted(sel.profile, current, result.selected_ids)
        result.spool = purge_spool.plan_spool(
            sel.profile, result.selected_ids, purge_metrics=result.purges_something
        )
        scoper = Scoper(sel.workspace)
        result.journal, result.journal_retained = purge_files.plan_journal(
            sel.profile, sel.cutoff_day, scoper
        )
        result.derived = purge_files.purge_derived(
            sel,
            dry_run=True,
            invalidate_frequency=bool(result.journal)
            or result.state is not None
            and result.state.changed,
        )
        result.unfinished = purge_files.read_marker(sel.profile) is not None
        state = result.state
        if state is not None and state.current is not None:
            committed = state.checkpoint.generation + (1 if state.changed else 0)
            if state.changed or result.unfinished:
                result.old_generations = purge_files.older_generations(sel.profile, committed)
        if state is not None:
            from ssgrep.accounting.privacy.purge_files import retained_receipt_ids, row_receipt_ids

            kept = set().union(
                *(row_receipt_ids(namespace.rows_after) for namespace in state.namespaces)
            )
            kept.update(retained_receipt_ids(state.legacy_archives))
            result.stale_receipts = frozenset(purge_files.receipt_ids(sel.profile) - kept)
    except Exception as error:  # noqa: BLE001 - mapped to a stable condition
        raise _failure(error) from error
    return result


# ---- report -----------------------------------------------------------------------------------


def _worker(profile: str) -> WorkerView | None:
    try:
        info = lease_module.inspect_lease(profile)
    except (storage.UnsafeStorageError, OSError):
        return None
    if info is None or not info.alive:
        return WorkerView(False, None)
    return WorkerView(True, info.kind)


def _local(p: Plan, spool_removed: int) -> LocalDeleted:
    state = p.state
    current = state.current if state is not None else None
    states = [state, *(a.plan for a in state.legacy_archives)] if state else []
    return LocalDeleted(
        evidence_records=sum(s.evidence_records for s in states),
        evidence_segments_rewritten=sum(len(s.rewrite) for s in states),
        ledger_items=sum(s.ledger_items for s in states),
        ledger_period_rows=sum(n.period_rows_removed for s in states for n in s.namespaces),
        ledger_receipt_rows=sum(n.receipt_rows_removed for s in states for n in s.namespaces),
        executions=sum(s.executions_removed for s in states),
        invocations=sum(s.invocations_removed for s in states),
        sealed_days=sum(len(s.sealed_removed) for s in states),
        backfill_series=sum(s.backfills_removed for s in states),
        receipt_files=len(p.stale_receipts),
        history_rows=current.period_rows_removed if current else 0,
        snapshot_generations=len(p.old_generations),
        journal_segments=len(p.journal),
        spool_batches=spool_removed,
        delivery_pins_released=0,
        plaintext_cache_files=p.derived.cache_files,
        frequency_files=p.derived.frequency_files,
        temporary_files=p.derived.temporary_files,
        keys=p.derived.keys,
        derived_caches=sum(len(s.derived_caches) for s in states),
        archive_metadata_records=sum(getattr(s, "archive_metadata_records", 0) for s in states),
    )


def _retained(p: Plan) -> Retained:
    state = p.state
    scopes = state.scopes if state else None
    return Retained(
        unknown_scope_conversations=scopes.unknown if scopes else 0,
        mixed_scope_conversations=scopes.mixed if scopes else 0,
        unknown_scope_executions=state.unknown_scope_executions if state else 0,
        sealed_days_not_splittable=state.sealed_not_splittable if state else 0,
        backfill_series_not_splittable=state.backfills_not_splittable if state else 0,
        journal_segments_retained=p.journal_retained,
        in_flight_batches_not_recallable=p.spool.not_recallable,
    )


def _lines(
    p: Plan, results: list[tombstones.TombstoneResult] | None
) -> tuple[tuple[TombstoneLine, ...], tuple[Obligation, ...]]:
    lines: list[TombstoneLine] = []
    owed: list[Obligation] = list(p.unlisted)
    by_dest = {r.destination_id: r for r in results or ()}
    for item in p.tomb_plans:
        if not item.record_count:
            continue
        dest, count = item.destination_id, item.record_count
        result = by_dest.get(dest)
        if not item.permitted:
            lines.append(TombstoneLine(dest, count, 0, "not_permitted", item.reason))
            owed.append(Obligation(dest, item.reason, count, item.obligations[:MAX_OBLIGATION_IDS]))
        elif result is None:
            lines.append(TombstoneLine(dest, count, None, "would_queue", None))
        elif result.skipped_reason is not None:
            lines.append(TombstoneLine(dest, count, 0, "skipped", result.skipped_reason))
            ids = result.obligations[:MAX_OBLIGATION_IDS]
            owed.append(Obligation(dest, result.skipped_reason, count, ids))
        else:
            lines.append(TombstoneLine(dest, result.queued_records, result.batches, "queued", None))
    return tuple(lines), tuple(owed)


def report(
    p: Plan,
    *,
    performed: bool,
    spool_removed: int,
    results: list[tombstones.TombstoneResult] | None = None,
    pins: int = 0,
    observe_worker: bool = True,
) -> PurgeReport:
    lines, owed = _lines(p, results)
    local = _local(p, spool_removed)
    local = replace(local, delivery_pins_released=pins)
    from ssgrep.accounting.privacy.retention import effective_retention

    return PurgeReport(
        profile=p.selection.profile,
        performed=performed,
        before=p.selection.before,
        cutoff_day=p.selection.cutoff_day,
        workspace=p.selection.workspace,
        local_deleted=local,
        tombstones=lines,
        obligations=owed,
        non_reversible_destinations=tuple(
            sorted(t.destination_id for t in p.tomb_plans if t.non_reversible_metrics)
        ),
        retained=_retained(p),
        pending_retention_days=effective_retention(p.policy).spool_days,
        worker=_worker(p.selection.profile) if observe_worker else None,
    )


def dry_run(sel: Selection) -> PurgeReport:
    """What a real run would do right now; creates and modifies nothing."""
    p = plan(sel)
    return report(p, performed=False, spool_removed=p.spool.count)


# ---- execution --------------------------------------------------------------------------------


def _commit(p: Plan, now: datetime) -> None:
    """Purged state + new snapshot generation in one checkpoint commit (crash-safe order)."""
    from ssgrep.accounting.privacy import purge_files, purge_state

    profile, state = p.selection.profile, p.state
    assert state is not None
    snapshots.discard_uncommitted(profile, state.checkpoint.generation)
    new = purge_state.write_purged_state(profile, state, p.selection)
    ledger_known = state.current is not None
    if ledger_known:
        new.generation = state.checkpoint.generation + 1
        envelope = build(profile, state, new, now)
        stage(profile, envelope)
        purge_files.write_marker(profile, new.generation)
    store.commit_checkpoint(profile, new)
    if ledger_known:
        make_current(profile, new.generation)


def execute(
    sel: Selection, *, now: datetime | None = None, _lease_held: bool = False
) -> PurgeReport:
    """Take the profile lease and purge; ``already_running`` leaves everything unchanged."""
    from ssgrep.accounting.privacy import purge_files, purge_state

    moment = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    from ssgrep.accounting.privacy.purge_recovery import (
        complete_erasure,
        recover_pending_purge,
        stage_erasure,
    )

    profile = sel.profile
    held = None
    try:
        if not purge_files.preflight(profile):
            return report(
                Plan(sel, policy_module.load_profile_policy(profile)),
                performed=True,
                spool_removed=0,
            )
        if not _lease_held:
            held = lease_module.Lease.acquire(profile, "purge")
    except lease_module.AlreadyRunning:
        raise PurgeError(
            "already_running",
            "A collection or serve worker holds this profile. Nothing was changed; stop it "
            "or wait, then run purge again.",
        ) from None
    except (storage.UnsafeStorageError, OSError) as error:
        raise _failure(error) from error
    try:
        if not _lease_held:
            recover_pending_purge(profile)
        p = plan(sel)  # again, under the lease: what we delete is what we just verified
        stamp = purge_state.stamp_of(moment)
        try:
            if p.purges_something or p.journal or p.derived != purge_files.DerivedDeleted():
                stage_erasure(sel)
            removed, nonrecallable = purge_spool.apply_spool(profile, p.spool, stamp)
            p.spool = replace(p.spool, not_recallable=nonrecallable)
            # A request may have started after planning but before the durable purge mark.
            # Re-select remote obligations now that no selected batch can start a new send.
            p.tomb_plans = tombstones.plan_tombstones(
                profile, p.policy, before_day=sel.cutoff_day, root_keys=_root_keys(p.state, sel)
            )
            p.unlisted = _unlisted(profile, p.policy, p.selected_ids)
            results = tombstones.enqueue_tombstones(profile, p.policy, p.tomb_plans, now=moment)
            if p.state is not None and p.state.changed:
                _commit(p, moment)
            purge_files.delete_journal(profile, p.journal)
            p.derived = purge_files.purge_derived(
                sel, invalidate_frequency=bool(p.journal) or p.state is not None and p.state.changed
            )
            pins = _finish(p, moment)
            complete_erasure(profile)
        except Exception as error:  # noqa: BLE001 - mapped to a stable condition
            raise _failure(error, resumable=True) from error
        return report(
            p,
            performed=True,
            spool_removed=removed,
            results=results,
            pins=pins,
            observe_worker=False,
        )
    finally:
        if held is not None:
            held.release()


def _finish(p: Plan, moment: datetime) -> int:
    """Idempotent post-commit cleanup; clears the unfinished marker last."""
    from ssgrep.accounting.privacy import purge_files

    profile = p.selection.profile
    from ssgrep.accounting.privacy.purge_recovery import has_pending_erasure

    marker = purge_files.read_marker(profile) is not None
    committed = store.load_checkpoint(profile)
    if marker and committed is not None and committed.generation:
        snapshots.repair_current(profile, committed.generation)  # crash before make_current
    # The durable intent also obliges publication repair after an archive-only
    # checkpoint commit. Such a commit has no active snapshot generation marker,
    # and retrying its already-filtered archive legitimately plans no more deletions.
    changed = (
        marker or has_pending_erasure(p.selection) or (p.state is not None and p.state.changed)
    )
    cleaned = purge_files.cleanup(
        profile,
        today=moment.date().isoformat(),
        generations=changed,
        publish=changed,
        stale_ids=p.stale_receipts,
    )
    if marker:
        purge_files.clear_marker(profile)
    return cleaned.delivery_pins_released


def build(
    profile: str, plan: purge_state.StatePlan, checkpoint: Checkpoint, now: datetime
) -> SnapshotEnvelope:
    """The purged state as the next generation (``checkpoint`` is the NEW checkpoint value)."""
    from ssgrep.accounting.privacy import purge_state

    current: purge_state.Namespace | None = plan.current
    if current is None:
        raise ValueError("no ledger of the current methodology")
    journal = read_journal(profile, quarantine=False)
    doc: ReductionDoc | None = plan.reduction_after
    counters = plan.state.counters or empty_counters(1, purge_state.stamp_of(now.astimezone(UTC)))
    return build_snapshot(
        SnapshotInputs(
            profile=profile,
            now=now,
            generation=checkpoint.generation,
            policy=load_profile_policy(profile),
            methodology=current.after.methodology,
            executions=doc.executions if doc else (),
            sealed=purge_state.sealed_after(plan),
            rows=current.rows_after,
            counters=counters,
            backfills=checkpoint.backfills,
            reduction_coverage=doc.coverage if doc else {},
            warnings=(*(doc.warnings if doc else ()), "purged: older data was deleted on request"),
            sources=checkpoint.sources,
            journal=journal_stats(journal),
            journal_last_ts=journal_last_ts(journal),
            read_gaps=source_gaps(checkpoint.sources),
            dropped=checkpoint.dropped,
            stopped_by="complete",
            retention=previous_coverage(checkpoint.retention),
            since_day=checkpoint.since_day,
            baseline_ts=checkpoint.baseline_ts,
            collection_scope=checkpoint.collection_scope,
            scope_epoch=checkpoint.scope_epoch,
        )
    )


def stage(profile: str, envelope: SnapshotEnvelope) -> None:
    snapshots.stage_generation(profile, envelope)


def make_current(profile: str, generation: int) -> None:
    snapshots.make_current(profile, generation)
