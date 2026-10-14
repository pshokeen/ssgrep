"""The collector-state half of a purge: compute what goes (read-only), then commit it once.

``plan_state`` reads the committed checkpoint and returns exactly what a real run removes; it
never writes. ``commit_state`` writes the purged blobs and evidence segments (unreferenced until
the checkpoint names them), stages a fresh snapshot generation built from the purged state, and
commits the checkpoint (the single commit point, exactly as a collection cycle does). Only then
are the superseded files deleted, so a crash earlier leaves the old, complete state and a re-run
finishes the job.

Ledger purge uses ``replay.Ledger.purge``: with a workspace it is applied to the selected
conversations only and the rest is merged back untouched. Live counters hold counts and no
content and are kept; sealed days and backfill series cannot be split per workspace, so a
workspace-scoped purge keeps and reports them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_state import (
    LEDGER_PREFIX,
    REDUCTION,
    SEALED,
    ReductionDoc,
    State,
    load_state,
)
from ssgrep.accounting.collector_store import Checkpoint, SegmentRef
from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.ledger_codec import ledger_state_from_dict, ledger_state_to_dict
from ssgrep.accounting.purge_select import Scoper, Scopes, Selection, classify, ledger_roots
from ssgrep.accounting.replay import Ledger, LedgerRows, LedgerState
from ssgrep.accounting.windows import SealedDay


@dataclass
class Namespace:
    """One methodology's ledger before and after the purge (rows are canonical and sorted)."""

    blob: str
    before: LedgerState
    after: LedgerState
    superseded_after: dict[str, str]
    rows_after: LedgerRows
    items_removed: int
    period_rows_removed: int
    receipt_rows_removed: int

    @property
    def changed(self) -> bool:
        return self.items_removed > 0 or self.period_rows_removed > 0


@dataclass
class StatePlan:
    checkpoint: Checkpoint
    state: State
    scopes: Scopes
    namespaces: list[Namespace] = field(default_factory=list)
    rewrite: frozenset[str] = frozenset()  # evidence segment names that lose records
    evidence_records: int = 0
    gone_sessions: frozenset[tuple[str, str]] = frozenset()
    late_sessions: frozenset[tuple[str, str]] = frozenset()
    executions_removed: int = 0
    unknown_scope_executions: int = 0
    sealed_removed: tuple[str, ...] = ()
    sealed_not_splittable: int = 0
    backfills_removed: int = 0
    backfills_not_splittable: int = 0
    reduction_after: ReductionDoc | None = None

    @property
    def ledger_items(self) -> int:
        return sum(n.items_removed for n in self.namespaces)

    @property
    def changed(self) -> bool:
        return bool(
            self.evidence_records
            or self.executions_removed
            or self.sealed_removed
            or self.backfills_removed
            or any(n.changed for n in self.namespaces)
        )

    @property
    def current(self) -> Namespace | None:
        name = f"{LEDGER_PREFIX}{self.checkpoint.methodology[:12]}"
        return next((n for n in self.namespaces if n.blob == name), None)


# ---- ledgers ----------------------------------------------------------------------------------


def _purged(
    before: LedgerState, superseded: dict[str, str], day: str, selected: frozenset[str] | None
) -> tuple[LedgerState, VersionedLedger]:
    ledger = VersionedLedger.restore_versioned(before, superseded)
    if selected is None:
        ledger.purge(before_day=day)
        return ledger.state(), ledger
    mine = LedgerState(
        before.methodology,
        tuple(c for c in before.candidates if c.task.root in selected),
        tuple(d for d in before.debits if d.task.root in selected),
        tuple(f for f in before.frozen if f.root in selected),
    )
    part = Ledger.restore(mine)
    part.purge(before_day=day)
    rest = (
        tuple(c for c in before.candidates if c.task.root not in selected),
        tuple(d for d in before.debits if d.task.root not in selected),
        tuple(f for f in before.frozen if f.root not in selected),
    )
    kept = part.state()
    merged = LedgerState(
        before.methodology,
        (*rest[0], *kept.candidates),
        (*rest[1], *kept.debits),
        (*rest[2], *kept.frozen),
    )
    final = VersionedLedger.restore_versioned(merged, superseded)
    return final.state(), final


def _items(state: LedgerState) -> int:
    return len(state.candidates) + len(state.debits) + sum(len(f.leaves) for f in state.frozen)


def _namespace(
    blob: str, before: LedgerState, superseded: dict[str, str], day: str, sel: frozenset[str] | None
) -> Namespace:
    after, ledger = _purged(before, superseded, day, sel)
    rows_before = VersionedLedger.restore_versioned(before, superseded).rows()
    rows_after = ledger.rows()
    return Namespace(
        blob,
        before,
        after,
        ledger.superseded_state(),
        rows_after,
        _items(before) - _items(after),
        len(rows_before.periods) - len(rows_after.periods),
        len(rows_before.receipts) - len(rows_after.receipts),
    )


def _read_namespaces(
    profile: str, checkpoint: Checkpoint
) -> list[tuple[str, LedgerState, dict[str, str]]]:
    found = []
    for name, file_name in sorted(checkpoint.blobs.items()):
        if not name.startswith(LEDGER_PREFIX):
            continue
        raw = store.read_blob(profile, file_name)
        assert raw is not None  # A committed string reference cannot mean an absent blob.
        superseded = {str(k): str(v) for k, v in raw.get("superseded", {}).items()}
        found.append((name, ledger_state_from_dict(raw["state"]), superseded))
    return found


# ---- evidence ---------------------------------------------------------------------------------


def _selected_record(
    item: RawEvidence, day: str, scopes: Scopes, late: frozenset[tuple[str, str]]
) -> bool:
    """Dated before the cutoff, or undated structure of a conversation with nothing newer."""
    if not scopes.has_session(item.runtime, item.session_id):
        return False
    ts = item.ts_utc
    if isinstance(ts, str):
        return ts[:10] < day
    return (item.runtime, item.session_id) not in late


def _scan_evidence(
    profile: str, checkpoint: Checkpoint, day: str, scopes: Scopes
) -> tuple[frozenset[str], int, frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
    """(segments to rewrite, records removed, emptied sessions, sessions with newer records)."""
    late: set[tuple[str, str]] = set()
    for ref in checkpoint.segments:
        for item in store.read_segment(profile, ref):
            if isinstance(item.ts_utc, str) and item.ts_utc[:10] >= day:
                late.add((item.runtime, item.session_id))
    frozen_late = frozenset(late)
    rewrite: set[str] = set()
    removed = 0
    total: dict[tuple[str, str], int] = {}
    gone: dict[tuple[str, str], int] = {}
    for ref in checkpoint.segments:
        for item in store.read_segment(profile, ref):
            key = (item.runtime, item.session_id)
            total[key] = total.get(key, 0) + 1
            if _selected_record(item, day, scopes, frozen_late):
                rewrite.add(ref.name)
                removed += 1
                gone[key] = gone.get(key, 0) + 1
    emptied = frozenset(k for k, n in gone.items() if n == total[k])
    return frozenset(rewrite), removed, emptied, frozen_late


# ---- executions, sealed days, backfills -------------------------------------------------------


def _filter_executions(
    doc: ReductionDoc | None, sel: Selection, scopes: Scopes, scoper: Scoper
) -> tuple[tuple, int, int]:
    """(kept executions, removed, retained because their workspace is unknown)."""
    if doc is None:
        return (), 0, 0
    kept, removed, unknown = [], 0, 0
    for e in doc.executions:
        if e.day >= sel.cutoff_day:
            kept.append(e)
        elif sel.workspace is None:
            removed += 1
        elif e.cwd:
            if scoper.inside(e.cwd):
                removed += 1
            else:
                kept.append(e)
        elif e.session_id is not None and scopes.has_session(e.runtime, e.session_id):
            removed += 1
        else:
            kept.append(e)
            unknown += 1
    return tuple(kept), removed, unknown


def _reduction_after(
    doc: ReductionDoc | None, kept: tuple, gone: frozenset[tuple[str, str]]
) -> ReductionDoc | None:
    if doc is None:
        return None
    return replace(
        doc,
        executions=kept,
        session_roots=tuple(t for t in doc.session_roots if (t[0], t[1]) not in gone),
        session_last=tuple(t for t in doc.session_last if (t[0], t[1]) not in gone),
    )


def _series_until(series: dict) -> str | None:
    value = series.get("until")
    return value if isinstance(value, str) else None


# ---- plan -------------------------------------------------------------------------------------


def plan_state(profile: str, sel: Selection, checkpoint: Checkpoint | None) -> StatePlan | None:
    """Everything a real run would remove from the collector state; reads only."""
    if checkpoint is None:
        return None
    state = load_state(profile, checkpoint, checkpoint.methodology)
    scoper = Scoper(sel.workspace)
    namespaces = _read_namespaces(profile, checkpoint)
    scopes = classify(
        profile,
        checkpoint,
        state.reduction,
        ledger_roots(ns[1] for ns in namespaces),
        scoper,
    )
    plan = StatePlan(checkpoint, state, scopes)
    plan.namespaces = [
        _namespace(name, before, superseded, sel.cutoff_day, scopes.selected)
        for name, before, superseded in namespaces
    ]
    scan = _scan_evidence(profile, checkpoint, sel.cutoff_day, scopes)
    plan.rewrite, plan.evidence_records, plan.gone_sessions, plan.late_sessions = scan
    kept, plan.executions_removed, plan.unknown_scope_executions = _filter_executions(
        state.reduction, sel, scopes, scoper
    )
    plan.reduction_after = _reduction_after(state.reduction, kept, plan.gone_sessions)
    older = tuple(day for day in sorted(state.sealed) if day < sel.cutoff_day)
    old_series = [
        s for s in checkpoint.backfills if (_series_until(s) or "9") <= sel.cutoff_midnight
    ]
    if sel.workspace is None:
        plan.sealed_removed, plan.backfills_removed = older, len(old_series)
    else:
        plan.sealed_not_splittable, plan.backfills_not_splittable = len(older), len(old_series)
    return plan


# ---- commit -----------------------------------------------------------------------------------


def _write_segments(profile: str, plan: StatePlan, sel: Selection) -> list[SegmentRef]:
    out: list[SegmentRef] = []
    for ref in plan.checkpoint.segments:
        if ref.name not in plan.rewrite:
            out.append(ref)
            continue
        kept = (
            item
            for item in store.read_segment(profile, ref)
            if not _selected_record(item, sel.cutoff_day, plan.scopes, plan.late_sessions)
        )
        out.extend(store.write_segments(profile, kept, origin=ref.origin))
    return out


def write_purged_state(profile: str, plan: StatePlan, sel: Selection) -> Checkpoint:
    """Write purged blobs/segments (not yet referenced) and return the NEW checkpoint value."""
    old = plan.checkpoint
    blobs = dict(old.blobs)
    for ns in plan.namespaces:
        if ns.changed:
            prior = store.read_blob(profile, old.blobs[ns.blob])
            document = {
                "state": ledger_state_to_dict(ns.after),
                "superseded": ns.superseded_after,
                "collection_scope": prior.get("collection_scope", {}) if prior else {},
            }
            blobs[ns.blob] = store.write_blob(profile, ns.blob, document)
    if plan.sealed_removed:
        sealed = {d: v for d, v in plan.state.sealed.items() if d not in plan.sealed_removed}
        blobs[SEALED] = store.write_blob(
            profile, SEALED, {d: v.to_dict() for d, v in sorted(sealed.items())}
        )
    if plan.reduction_after is not None and (plan.executions_removed or plan.gone_sessions):
        blobs[REDUCTION] = store.write_blob(profile, REDUCTION, plan.reduction_after.to_dict())
    segments = _write_segments(profile, plan, sel) if plan.rewrite else list(old.segments)
    series = [
        s
        for s in old.backfills
        if sel.workspace is not None or (_series_until(s) or "9") > sel.cutoff_midnight
    ]
    return replace(old, blobs=blobs, segments=segments, backfills=series)


def sealed_after(plan: StatePlan) -> dict[str, SealedDay]:
    return {d: v for d, v in plan.state.sealed.items() if d not in plan.sealed_removed}


def stamp_of(now: datetime) -> str:
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")
