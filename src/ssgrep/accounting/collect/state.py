"""The collector's derived state, persisted as content-addressed blobs named by the checkpoint.

Everything a *clock-only* cycle needs to republish without re-reading sources or re-reducing:
the ledger of the CURRENT methodology (frozen roots plus live candidates/debits and the
superseded-version map), live counters, sealed UTC days, and the last reduction's executions
and coverage. A methodology change is a new namespace: its ledger blob is separate, and the
old namespace's blob is kept untouched (never summed with the new one).

Each blob is written before the checkpoint that names it, so a crash can never leave the
checkpoint pointing at a blob that is missing.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeVar

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.partitions import (
    expand_reduction_document as expand_reduction,
    save_reduction_incremental as save_reduction,
)
from ssgrep.accounting.collect.replay import ArchivedRows, Ledger, LedgerRows, LedgerState
from ssgrep.accounting.collect.store import Checkpoint, CollectError
from ssgrep.accounting.ledger.contracts import digest_of
from ssgrep.accounting.ledger.ledger_contracts import LiveCounters
from ssgrep.accounting.ledger.reduce_records import ExecutionRecord, InvocationRecord
from ssgrep.accounting.ledger.revisions import (
    build_period_rows,
    estimates_of,
    revision_points,
    seal_day,
)
from ssgrep.accounting.ledger.valuation import Leaf
from ssgrep.accounting.ledger.valuation_inputs import (
    RETIRED_VALUATION_VERSION,
    Candidate,
    Debit,
    EvidenceBundle,
)
from ssgrep.accounting.privacy.storage import from_plain, to_plain
from ssgrep.accounting.views.records import ReceiptRecord

if TYPE_CHECKING:
    from ssgrep.accounting.collect.frequency import FrequencyView
    from ssgrep.accounting.collect.replay import ArchivedRows, LedgerState
    from ssgrep.accounting.ledger.ledger_contracts import FrequencyAssignment
    from ssgrep.accounting.views.history import SealedDay

COUNTERS, SEALED, REDUCTION = "counters", "sealed", "reduction"
FREQUENCIES = "frequency-assignments"
LEDGER_PREFIX = "ledger-"
NAMESPACE_CHARS = 12


def ledger_blob_name(methodology_hash: str) -> str:
    return f"{LEDGER_PREFIX}{methodology_hash[:NAMESPACE_CHARS]}"


@dataclass
class ReductionDoc:
    """What a clock-only cycle reuses of the last reduction (plain data, no sources)."""

    executions: tuple[ExecutionRecord, ...] = ()
    coverage: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    digest: str = ""
    session_roots: tuple[tuple[str, str, str], ...] = ()  # (runtime, session_id, root)
    session_last: tuple[tuple[str, str, str], ...] = ()  # (runtime, session_id, last evidence ts)
    # additive (readers tolerate absence): lineage parent and working directory per session
    session_parents: tuple[tuple[str, str, str | None], ...] = ()  # (runtime, session_id, parent)
    session_cwd: tuple[tuple[str, str, str], ...] = ()  # (runtime, session_id, cwd)
    # (invocation_id, runtime, session_id): what Candidate.exposure_ids / Debit.exposure_id name
    invocation_sessions: tuple[tuple[str, str, str], ...] = ()
    invocations: tuple[InvocationRecord, ...] = ()
    lookup_schema: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "executions": [to_plain(e) for e in self.executions],
            "coverage": dict(sorted(self.coverage.items())),
            "warnings": list(self.warnings),
            "digest": self.digest,
            "session_roots": [list(t) for t in self.session_roots],
            "session_last": [list(t) for t in self.session_last],
            "session_parents": [list(t) for t in self.session_parents],
            "session_cwd": [list(t) for t in self.session_cwd],
            "invocation_sessions": [list(t) for t in self.invocation_sessions],
            "invocations": [to_plain(row) for row in self.invocations],
            "lookup_schema": self.lookup_schema,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ReductionDoc:
        return cls(
            tuple(from_plain(ExecutionRecord, e) for e in raw["executions"]),
            {str(k): int(v) for k, v in raw["coverage"].items()},
            tuple(str(w) for w in raw.get("warnings", [])),
            str(raw.get("digest", "")),
            tuple((str(a), str(b), str(c)) for a, b, c in raw.get("session_roots", [])),
            tuple((str(a), str(b), str(c)) for a, b, c in raw.get("session_last", [])),
            tuple(
                (str(a), str(b), None if c is None else str(c))
                for a, b, c in raw.get("session_parents", [])
            ),
            tuple((str(a), str(b), str(c)) for a, b, c in raw.get("session_cwd", [])),
            tuple((str(a), str(b), str(c)) for a, b, c in raw.get("invocation_sessions", [])),
            tuple(from_plain(InvocationRecord, row) for row in raw.get("invocations", [])),
            str(raw.get("lookup_schema", "")),
        )


@dataclass
class State:
    """Loaded derived state; ``None`` fields mean the blob does not exist yet."""

    ledger: LedgerState | None = None
    superseded: dict[str, str] = field(default_factory=dict)
    counters: LiveCounters | None = None
    sealed: dict[str, SealedDay] = field(default_factory=dict)
    reduction: ReductionDoc | None = None
    collection_scope: dict[str, Any] = field(default_factory=dict)
    frequency_assignments: tuple[FrequencyAssignment, ...] = ()
    archives: dict[str, ArchivedRows] = field(default_factory=dict)
    frequency_probe: FrequencyView | None = None
    root_migration_pending: int = 0


def _decode(name: str, build: Any, raw: Any) -> Any:
    try:
        return build(raw)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise CollectError("state_corrupt", f"Collector state blob {name!r} is invalid.") from error


def decode_ledger(raw: Any, *, profile: str | None = None) -> tuple[LedgerState, dict[str, str]]:
    """Validate a referenced ledger before its provenance can trigger a scope rotation."""
    from ssgrep.accounting.ledger.ledger_codec import ledger_state_from_dict

    try:
        if raw.get("format") is not None:
            if profile is None:
                raise ValueError("a root ledger manifest needs its profile")
            raw = expand_ledger_document(profile, raw)
        state = ledger_state_from_dict(raw["state"])
        return state, {str(k): str(v) for k, v in raw.get("superseded", {}).items()}
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise CollectError("state_corrupt", "Collector ledger state is invalid.") from error


def decode_frequencies(raw: Any) -> tuple[FrequencyAssignment, ...]:
    from ssgrep.accounting.ledger.ledger_contracts import FrequencyAssignment

    if raw.get("schema") != 1:
        raise ValueError("unsupported frequency assignment schema")
    rows = tuple(from_plain(FrequencyAssignment, row) for row in raw["assignments"])
    if len({row.invocation_id for row in rows}) != len(rows):
        raise ValueError("duplicate frequency assignments")
    return rows


def load_state(profile: str, checkpoint: Checkpoint, methodology_hash: str) -> State:
    """Every blob the checkpoint names (corrupt or missing committed blob => state_corrupt)."""
    from ssgrep.accounting.collect.replay import ArchivedRows
    from ssgrep.accounting.views.history import SealedDay

    blobs = checkpoint.blobs

    def blob(name: str) -> Any | None:
        return store.read_blob(profile, blobs.get(name))

    state = State()
    name = ledger_blob_name(methodology_hash)
    raw = blob(name)
    if raw is not None:
        state.root_migration_pending = int(raw.get("migration_pending_roots", 0))
        raw = _decode(name, lambda value: expand_ledger_document(profile, value), raw)
        state.ledger, state.superseded = decode_ledger(raw)
        state.collection_scope = raw.get("collection_scope", {})
        state.archives = _decode(
            name,
            lambda doc: {key: from_plain(ArchivedRows, value) for key, value in doc.items()},
            raw.get("archives", {}),
        )
    raw = blob(COUNTERS)
    if raw is not None:
        state.counters = _decode(COUNTERS, LiveCounters.from_dict, raw)
    raw = blob(SEALED)
    if raw is not None:
        state.sealed = _decode(
            SEALED, lambda r: {str(k): SealedDay.from_dict(v) for k, v in r.items()}, raw
        )
    raw = blob(REDUCTION)
    if raw is not None:
        raw = _decode(REDUCTION, lambda value: expand_reduction(profile, value), raw)
        state.reduction = _decode(REDUCTION, ReductionDoc.from_dict, raw)
    raw = blob(FREQUENCIES)
    if raw is not None:
        raw = _decode(FREQUENCIES, lambda value: expand_assignments_document(profile, value), raw)
        state.frequency_assignments = _decode(FREQUENCIES, decode_frequencies, raw)
        probe = raw.get("unavailable_probe")
        if probe is not None:
            from ssgrep.accounting.collect.frequency import FrequencyView

            state.frequency_probe = _decode(
                FREQUENCIES, lambda value: from_plain(FrequencyView, value), probe
            )
            if state.frequency_probe.known or state.frequency_probe.df:
                raise CollectError(
                    "state_corrupt", "Unavailable frequency probe contains corpus data."
                )
    return state


def save_state(
    profile: str,
    state: State,
    methodology_hash: str,
    blobs: dict[str, str],
    *,
    previous: State | None = None,
    clock=time.monotonic,
    deadline: float = float("inf"),
    dirty_roots: set[str] | None = None,
) -> None:
    """Write changed immutable blobs; reuse unchanged canonical state after restart."""
    ledger_changed = (
        previous is None
        or (
            previous.ledger != state.ledger
            or previous.superseded != state.superseded
            or previous.archives != state.archives
            or previous.collection_scope != state.collection_scope
        )
        or ledger_blob_name(methodology_hash) not in blobs
    )
    if state.ledger is not None and (ledger_changed or state.root_migration_pending):
        state.root_migration_pending = save_ledger_incremental(
            profile,
            ledger_blob_name(methodology_hash),
            state,
            blobs,
            previous=previous,
            clock=clock,
            deadline=deadline,
            dirty_roots=dirty_roots,
        )
    if state.counters is not None and (
        previous is None or previous.counters != state.counters or COUNTERS not in blobs
    ):
        blobs[COUNTERS] = store.write_blob(profile, COUNTERS, state.counters.to_dict())
    if previous is None or previous.sealed != state.sealed or SEALED not in blobs:
        sealed = {day: value.to_dict() for day, value in state.sealed.items()}
        blobs[SEALED] = store.write_blob(profile, SEALED, sealed)
    if state.reduction is not None and (
        previous is None or previous.reduction != state.reduction or REDUCTION not in blobs
    ):
        save_reduction(
            profile,
            REDUCTION,
            state.reduction,
            blobs,
            previous=previous.reduction if previous else None,
        )
    if (
        previous is None
        or previous.frequency_assignments != state.frequency_assignments
        or previous.frequency_probe != state.frequency_probe
        or FREQUENCIES not in blobs
    ):
        save_assignments_incremental(profile, FREQUENCIES, state, blobs, previous=previous)


"""Versioned evidence reconciliation on top of the append-only ``replay.Ledger`` (D7).

The collector re-reduces affected dependency closures, so one item identity can
reappear with refined content (a debit first unavailable, later with ``chars``). The
base ledger keeps evidence as a set keyed by content digest, which would keep both
versions and turn the debit into ``conflicting_duplicate_debit`` forever. ``reconcile``
keeps exactly one *active* version per identity:

* inside the provisional window a refinement or retraction rewrites the item (the old
  version is deleted; a refinement keeps the old acceptance stamp, so no revision);
* once the period is sealed nothing is rewritten: the new version is stamped with the
  reconcile day and the old digest is *superseded* from that day. Evaluation at a point
  ignores items superseded at or before it, and supersession days are extra revision
  points, so a sealed refinement or retraction yields a superseding revision while
  every earlier revision stays byte-identical.

Content is compared with ``accepted_day`` cleared: the stamp is ledger bookkeeping, not
evidence. No clocks and no I/O; ``today`` is a UTC ``YYYY-MM-DD`` parameter.
"""


Item = TypeVar("Item", Candidate, Debit)

_EXPIRED = "evidence_expired_root"
CANDIDATE_IDENTITY_METHOD = "lookup_exposure_v1"
RETIRED_ARCHIVE_PREFIX = "methodology-retired:"


def candidate_identity(cand: Candidate) -> tuple[str, str, str, str, tuple[str, ...]]:
    """Refine one lookup's observation, not every exposure of its task fingerprint.

    Separate lookups can legitimately disagree about novelty/eligibility. Retain
    those observations independently; valuation owns the per-task source union,
    so repeated supported exposures still cannot claim the same source twice.
    """
    return (
        cand.task.root,
        cand.task.turn or "",
        cand.task.branch,
        cand.candidate_id,
        tuple(sorted(set(cand.exposure_ids))),
    )


def debit_identity(debit: Debit) -> str:
    return debit.debit_id


def _content_digest(item: Candidate | Debit) -> str:
    return digest_of(replace(item, accepted_day=""))


@dataclass(frozen=True)
class ReconcileResult:
    added: int
    replaced: int
    retracted: int
    unchanged: int
    superseded: int
    rejected: tuple[tuple[str, str], ...]  # (item id, reason)


@dataclass
class _Tally:
    added: int = 0
    replaced: int = 0
    retracted: int = 0
    unchanged: int = 0
    superseded: int = 0


@dataclass
class VersionedLedger(Ledger):
    """``Ledger`` with one active version per evidence identity (see module doc)."""

    superseded: dict[str, str] = field(default_factory=dict)  # item digest -> UTC day
    archives: dict[str, ArchivedRows] = field(default_factory=dict)
    _by_root: tuple[dict, dict] | None = field(default=None, repr=False)
    _root_cache_files: dict[str, str] = field(default_factory=dict, repr=False)
    _root_cache_keys: dict[str, list[tuple[str, ...]]] = field(default_factory=dict, repr=False)

    def _root_items(self, store: dict, root: str) -> dict:
        if self._by_root is None:
            groups = ({}, {})
            for items, grouped in zip((self._cands, self._debits), groups, strict=True):
                for ident, item in items.items():
                    grouped.setdefault(item.task.root, {})[ident] = item
            self._by_root = groups
        grouped = self._by_root[0 if store is self._cands else 1]
        return grouped.setdefault(root, {})

    def ingest(self, batch: EvidenceBundle):
        self._by_root = None
        return super().ingest(batch)

    @classmethod
    def restore_versioned(
        cls,
        state: LedgerState,
        superseded: Mapping[str, str],
        archives: Mapping[str, ArchivedRows] | None = None,
    ) -> VersionedLedger:
        base = Ledger.restore(state)
        ledger = cls(
            state.methodology, _cands=base._cands, _debits=base._debits, _frozen=base._frozen
        )
        ledger.superseded = dict(superseded)
        ledger.superseded = ledger.superseded_state()
        ledger.archives = dict(archives or {})
        return ledger

    def superseded_state(self) -> dict[str, str]:
        present = self._cands.keys() | self._debits.keys()
        return {k: self.superseded[k] for k in sorted(self.superseded) if k in present}

    def archive_current(self, revision_name: str) -> None:
        """Freeze prior canonical rows once before an explicit named replacement.

        The archive is separate from active evidence and period/receipt selection.
        In particular a same-day provisional correction cannot overwrite it or
        charge its old debit again. The caller persists archives with the checkpoint.
        A retry of one committed replacement name is idempotent.
        """
        if not revision_name:
            raise ValueError("a replacement requires a nonempty revision name")
        if revision_name not in self.archives:
            rows = self.rows()
            self.archives[revision_name] = ArchivedRows(
                revision_name, rows.namespace, rows.periods, rows.receipts
            )

    def archives_state(self) -> dict[str, ArchivedRows]:
        """Named immutable prior rows, ordered for deterministic checkpoint encoding."""
        return {name: self.archives[name] for name in sorted(self.archives)}

    # ---- reconciliation ------------------------------------------------------

    def reconcile(
        self, desired: Mapping[str, EvidenceBundle], *, today: str, exhaustive: bool = False
    ) -> ReconcileResult:
        """Make each mentioned root's active evidence equal its current projection.

        ``exhaustive`` says ``desired`` is the COMPLETE projection of all retained evidence, so a
        live root it no longer mentions (its conversation was merged under another root once a
        late parent arrived, or fell out of scope) is retracted like an emptied one. Frozen
        roots are never touched either way.
        """
        tally = _Tally()
        rejected: list[tuple[str, str]] = []
        absent = EvidenceBundle((), ())
        live = {i.task.root for store in (self._cands, self._debits) for i in store.values()}
        roots = set(desired) | (live - set(self._frozen) if exhaustive else set())
        for root in sorted(roots):
            bundle = desired.get(root, absent)
            if root in self._frozen:
                rejected += [(c.candidate_id, _EXPIRED) for c in bundle.candidates]
                rejected += [(d.debit_id, _EXPIRED) for d in bundle.debits]
                continue
            want_c, bad_c = _select(bundle.candidates, root, candidate_identity, _cid)
            want_d, bad_d = _select(bundle.debits, root, debit_identity, _did)
            rejected += bad_c + bad_d
            self._reconcile_kind(self._cands, want_c, root, today, candidate_identity, tally)
            self._reconcile_kind(self._debits, want_d, root, today, debit_identity, tally)
        result = (tally.added, tally.replaced, tally.retracted, tally.unchanged, tally.superseded)
        return ReconcileResult(*result, tuple(sorted(rejected)))

    def _reconcile_kind(
        self,
        store: dict[str, Item],
        wanted: Mapping[Hashable, Item],
        root: str,
        today: str,
        identity: Callable[[Item], Hashable],
        tally: _Tally,
    ) -> None:
        active: dict[Hashable, list[tuple[str, Item]]] = {}
        roots = self._root_items(store, root)
        for key in sorted(roots):
            if key not in self.superseded:
                active.setdefault(identity(roots[key]), []).append((key, roots[key]))
        for ident in sorted(active.keys() | wanted.keys(), key=repr):
            olds = active.get(ident, [])
            new = wanted.get(ident)
            if new is None:
                for key, old in olds:
                    self._retire(store, key, old, today)
                tally.retracted += 1
                continue
            digest = _content_digest(new)
            same = [o for o in olds if _content_digest(o[1]) == digest]
            if same:
                stale = [o for o in olds if o is not same[0]]
                for key, old in stale:
                    self._retire(store, key, old, today)
                tally.replaced += bool(stale)
                tally.unchanged += not stale
                continue
            gone = [old for key, old in olds if self._retire(store, key, old, today)]
            sealed = len(olds) - len(gone)
            inherit = bool(gone) and not sealed and today < seal_day(new.day)
            stamp = max(old.accepted_day for old in gone) if inherit else today
            stamped = replace(new, accepted_day=stamp)
            key = digest_of(stamped)
            store[key] = stamped
            self._root_items(store, root)[key] = stamped
            if olds:
                tally.replaced += 1
                tally.superseded += sealed
            else:
                tally.added += 1

    def _retire(self, store: dict[str, Item], key: str, old: Item, today: str) -> bool:
        """Delete a provisional (or created-today) version; supersede a sealed one.

        Returns True when deleted.
        """
        if today < seal_day(old.day) or old.accepted_day == today:
            del store[key]
            if self._by_root is not None:
                self._root_items(store, old.task.root).pop(key, None)
            self.superseded.pop(key, None)
            return True
        self.superseded[key] = today
        return False

    # ---- evaluation ----------------------------------------------------------

    def _root_subset(self, root: str, point: str) -> tuple[tuple[str, ...], EvidenceBundle]:
        def live(key: str, accepted: str) -> bool:
            return accepted <= point and (
                key not in self.superseded or self.superseded[key] > point
            )

        cands = {
            k: c for k, c in self._root_items(self._cands, root).items() if live(k, c.accepted)
        }
        debits = {
            k: d for k, d in self._root_items(self._debits, root).items() if live(k, d.accepted)
        }
        key = tuple(sorted(cands)) + ("|",) + tuple(sorted(debits))
        bundle = EvidenceBundle(
            tuple(cands[k] for k in sorted(cands)), tuple(debits[k] for k in sorted(debits))
        )
        return key, bundle

    def _accepted_days(self) -> list[str]:
        """Acceptance days plus supersession days: each can open a revision."""
        days = [c.accepted for c in self._cands.values()]
        days += [d.accepted for d in self._debits.values()]
        return days + list(self.superseded.values())

    def _points(self, periods: Iterable[str]) -> list[str]:
        days = self._accepted_days()
        points: set[str] = set()
        for period in periods:
            points.update(revision_points(period, days))
        return sorted(points)

    def rows(self) -> LedgerRows:
        """``Ledger.rows`` with supersession days as additional revision points."""
        retired = self.archives.get(RETIRED_ARCHIVE_PREFIX + self.namespace)
        if retired is not None:
            if (
                self.methodology.valuation_version != RETIRED_VALUATION_VERSION
                or retired.namespace != self.namespace
            ):
                raise ValueError("retired_methodology_namespace_mismatch")
            return LedgerRows(
                self.namespace,
                tuple(replace(row, recomputable=False) for row in retired.periods),
                retired.receipts,
                tuple(self.archives[name] for name in sorted(self.archives)),
            )
        rows = self._canonical_rows()
        if self.methodology.valuation_version == RETIRED_VALUATION_VERSION:
            return replace(
                rows, periods=tuple(replace(row, recomputable=False) for row in rows.periods)
            )
        return rows

    def _canonical_rows(self) -> LedgerRows:
        """Assemble accepted rows with their original metadata before retirement."""
        roots = self._roots()
        periods = {c.day for c in self._cands.values()} | {d.day for d in self._debits.values()}
        frozen_leaves = [leaf for f in self._frozen.values() for leaf in f.leaves]
        periods |= {leaf.day for leaf in frozen_leaves}

        def leaves_at(point: str, period: str) -> list[Leaf]:
            live = [
                leaf for r in roots for leaf in self._evaluate(r, point)[0] if leaf.day == period
            ]
            return live + [leaf for leaf in frozen_leaves if leaf.day == period]

        period_rows = build_period_rows(
            self.namespace, periods, self._accepted_days(), leaves_at, frozenset(self._frozen)
        )
        history: dict[str, list[ReceiptRecord]] = {}
        for point in self._points(periods):
            for root in roots:
                for rid, record in self._evaluate(root, point)[1].items():
                    seen = history.setdefault(rid, [])
                    if not seen or seen[-1].digest != record.digest:
                        seen.append(replace(record, receipt=replace(record.receipt, revision=1)))
        receipts: list[ReceiptRecord] = []
        for rid in sorted(history):
            for n, record in enumerate(history[rid], start=1):
                receipts.append(replace(record, receipt=replace(record.receipt, revision=n)))
        for f in sorted(self._frozen.values(), key=lambda f: f.root):
            receipts.extend(f.receipts)
        receipts.sort(key=lambda r: (r.receipt_id, r.receipt.revision))
        return LedgerRows(
            self.namespace,
            tuple(period_rows),
            tuple(receipts),
            tuple(self.archives[name] for name in sorted(self.archives)),
        )

    # ---- retention -----------------------------------------------------------

    def has_before_day(self, day: str) -> bool:
        """Avoid destroying reusable evaluations when an unchanged floor removes nothing."""
        return (
            any(item.day < day for items in (self._cands, self._debits) for item in items.values())
            or any(leaf.day < day for frozen in self._frozen.values() for leaf in frozen.leaves)
            or any(
                row.period < day for archive in self.archives.values() for row in archive.periods
            )
            or any(
                row.period < day for archive in self.archives.values() for row in archive.receipts
            )
        )

    def expire_roots(self, roots: Iterable[str]) -> list[str]:
        frozen = super().expire_roots(roots)
        self._by_root = None
        self.superseded = self.superseded_state()
        return frozen

    def _purge(self, purged: Callable[[str, str], bool]) -> None:
        super()._purge(purged)
        self._by_root = None
        self.superseded = self.superseded_state()

        retained: dict[str, ArchivedRows] = {}
        for name, archive in self.archives.items():
            periods = []
            for row in archive.periods:
                leaves = tuple(leaf for leaf in row.leaves if not purged(leaf.day, leaf.task.root))
                if leaves:
                    periods.append(
                        row
                        if leaves == row.leaves
                        else replace(row, leaves=leaves, estimates=estimates_of(leaves))
                    )
            receipts = tuple(
                record
                for record in archive.receipts
                if not purged(record.period, record.receipt.root)
            )
            if periods or receipts:
                retained[name] = ArchivedRows(name, archive.namespace, tuple(periods), receipts)
        self.archives = retained


def _cid(cand: Candidate) -> str:
    return cand.candidate_id


def _did(debit: Debit) -> str:
    return debit.debit_id


def _select(
    items: Iterable[Item],
    root: str,
    identity: Callable[[Item], Hashable],
    label: Callable[[Item], str],
) -> tuple[dict[Hashable, Item], list[tuple[str, str]]]:
    """One item per identity (smallest content digest wins); the rest are rejected."""
    wanted: dict[Hashable, Item] = {}
    rejected: list[tuple[str, str]] = []
    for item in sorted(items, key=_content_digest):
        if item.task.root != root:
            rejected.append((label(item), "root_mismatch"))
        elif identity(item) in wanted:
            rejected.append((label(item), "conflicting_desired_variant"))
        else:
            wanted[identity(item)] = item
    return wanted, rejected


"""Small cycle integration helpers for partition queues and immutable state reuse."""


LEDGER_FORMAT = "root-ledger-v1"


def frequency_refs(value):
    """Actual frozen input IDs, not text tokens or methodology identities."""
    found = set()
    if isinstance(value, dict):
        for key, row in value.items():
            if (
                key in {"view_id", "frequency_view_id"}
                and isinstance(row, str)
                and row.startswith("freq-")
            ):
                found.add(row)
            found.update(frequency_refs(row))
    elif isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and value[0] in {"frequency", "frequency_view"}
            and isinstance(value[1], str)
            and value[1].startswith("freq-")
        ):
            found.add(value[1])
        for row in value:
            found.update(frequency_refs(row))
    return found


def root_states(state):
    from ssgrep.accounting.collect.replay import LedgerState

    grouped = defaultdict(lambda: [[], [], []])
    for row in state.candidates:
        grouped[row.task.root][0].append(row)
    for row in state.debits:
        grouped[row.task.root][1].append(row)
    for row in state.frozen:
        grouped[row.root][2].append(row)
    return {
        root: LedgerState(state.methodology, tuple(rows[0]), tuple(rows[1]), tuple(rows[2]))
        for root, rows in grouped.items()
    }


def expand_ledger_document(profile, raw, *, depth=0):
    if raw.get("format") is None:
        return raw
    if raw.get("format") != LEDGER_FORMAT:
        raise CollectError("unsupported_schema", "Collector ledger format is unsupported.")
    if depth > 1:
        raise CollectError("state_corrupt", "Collector ledger base chain is invalid.")
    base = raw.get("base_blob")
    document = (
        expand_ledger_document(profile, store.read_blob(profile, base), depth=depth + 1)
        if base
        else {
            "state": raw["state"],
            "superseded": {},
            "archives": {},
        }
    )
    roots = set(raw["roots"])
    state = document["state"]
    candidates = [row for row in state["candidates"] if row["task"]["root"] not in roots]
    debits = [row for row in state["debits"] if row["task"]["root"] not in roots]
    frozen = [row for row in state["frozen"] if row["root"] not in roots]
    superseded = dict(document.get("superseded", {}))
    for root, entry in sorted(raw["roots"].items()):
        if entry is None:
            continue
        child = store.read_blob(profile, entry["blob"])
        if not isinstance(child, dict):
            raise CollectError("state_corrupt", "Collector root document is invalid.")
        if child["state"]["methodology"] != raw["state"]["methodology"]:
            raise CollectError("state_corrupt", "Collector root method differs from its manifest.")
        if any(
            row["task"]["root"] != root
            for kind in ("candidates", "debits")
            for row in child["state"][kind]
        ):
            raise CollectError(
                "state_corrupt", "Collector root ownership differs from its manifest."
            )
        candidates.extend(child["state"]["candidates"])
        debits.extend(child["state"]["debits"])
        frozen.extend(child["state"]["frozen"])
        superseded.update(child.get("superseded", {}))
    archives = {
        name: store.read_blob(profile, filename)
        for name, filename in raw.get("archive_blobs", {}).items()
    }
    return {
        "state": {
            "methodology": raw["state"]["methodology"],
            "candidates": sorted(candidates, key=digest_of),
            "debits": sorted(debits, key=digest_of),
            "frozen": sorted(frozen, key=lambda row: row["root"]),
        },
        "superseded": superseded,
        "archives": archives,
        "collection_scope": raw.get("collection_scope", {}),
    }


def save_ledger_incremental(
    profile,
    name,
    state,
    blobs,
    *,
    previous=None,
    clock=time.monotonic,
    deadline=float("inf"),
    migration_limit=32,
    dirty_roots=None,
):
    from ssgrep.accounting.collect.replay import LedgerState
    from ssgrep.accounting.ledger.ledger_codec import ledger_state_to_dict

    before = store.read_blob(profile, blobs.get(name))
    if before is not None and not isinstance(before, dict):
        raise CollectError("state_corrupt", "Collector ledger base document is invalid.")
    current = root_states(state.ledger)
    old = root_states(previous.ledger) if previous and previous.ledger else {}
    if before and before.get("format") == LEDGER_FORMAT and previous is not None:
        roots = dict(before["roots"])
        base = before.get("base_blob")
        base_roots = set(before.get("base_roots", ()))
        old_archives = before.get("archive_blobs", {})
    else:
        roots, old_archives = {}, {}
        base = blobs.get(name) if before and previous is not None else None
        base_roots = set(old) if base else set()
    changed = {root for root in current.keys() | old.keys() if current.get(root) != old.get(root)}
    if previous is None:
        changed = set(current)
    if previous and state.superseded != previous.superseded:
        changed.update(dirty_roots if dirty_roots is not None else current.keys() | old.keys())
    unresolved = sorted(base_roots - roots.keys() - changed)
    scheduled = []
    for root in unresolved[:migration_limit]:
        if clock() >= deadline:
            break
        scheduled.append(root)
    for root in sorted(changed | set(scheduled)):
        local = current.get(root)
        if local is None:
            roots[root] = None
            continue
        identities = {digest_of(row) for row in (*local.candidates, *local.debits)}
        document = {
            "state": ledger_state_to_dict(local),
            "superseded": {
                key: value for key, value in state.superseded.items() if key in identities
            },
        }
        refs = sorted(frequency_refs(document))
        roots[root] = {
            "blob": store.write_blob(profile, "ledger-root", document),
            "frequency_refs": refs,
        }
    remaining = len(base_roots - roots.keys())
    if not remaining:
        base, base_roots = None, set()
        roots = {
            root: entry for root, entry in roots.items() if entry is not None and root in current
        }
    archives = {}
    archive_refs = set()
    for key, archive in sorted(state.archives.items()):
        if previous and previous.archives.get(key) == archive and key in old_archives:
            archives[key] = old_archives[key]
        else:
            archives[key] = store.write_blob(profile, "ledger-archive-rows", to_plain(archive))
        archive_refs.update(frequency_refs(to_plain(archive)))
    refs = archive_refs | {
        view for entry in roots.values() if entry for view in entry["frequency_refs"]
    }
    if base and before is not None:
        refs.update(
            before.get("frequency_refs", ())
            if before.get("format") == LEDGER_FORMAT
            else frequency_refs(before)
        )
    raw = {
        "format": LEDGER_FORMAT,
        "state": ledger_state_to_dict(LedgerState(state.ledger.methodology)),
        "roots": roots,
        "base_blob": base,
        "base_roots": sorted(base_roots),
        "collection_scope": state.collection_scope,
        "archive_blobs": archives,
        "frequency_refs": sorted(refs),
        "migration_pending_roots": remaining,
    }
    blobs[name] = store.write_blob(profile, name, raw)
    return remaining


def ledger_references(raw):
    if raw.get("format") != LEDGER_FORMAT:
        return set()
    return (
        ({raw["base_blob"]} if raw.get("base_blob") else set())
        | {entry["blob"] for entry in raw["roots"].values() if entry is not None}
        | set(raw.get("archive_blobs", {}).values())
    )


def write_ledger_document(profile, name, document):
    """Complete logical-document writer used by explicit privacy purge."""
    from ssgrep.accounting.collect.replay import ArchivedRows
    from ssgrep.accounting.ledger.ledger_codec import ledger_state_from_dict

    state = State(
        ledger=ledger_state_from_dict(document["state"]),
        superseded=dict(document.get("superseded", {})),
        collection_scope=document.get("collection_scope", {}),
        archives={
            key: from_plain(ArchivedRows, row) for key, row in document.get("archives", {}).items()
        },
    )
    blobs = {}
    save_ledger_incremental(profile, name, state, blobs)
    return blobs[name]


ASSIGNMENT_FORMAT = "frequency-assignments-v2"


def expand_assignments_document(profile, raw):
    if raw.get("format") is None:
        return raw
    if raw.get("format") != ASSIGNMENT_FORMAT:
        raise CollectError("unsupported_schema", "Collector assignment format is unsupported.")
    rows = []
    for filename in raw["shards"].values():
        shard = store.read_blob(profile, filename)
        if not isinstance(shard, list):
            raise CollectError("state_corrupt", "Collector assignment shard is invalid.")
        rows.extend(shard)
    return {
        "schema": 1,
        "assignments": sorted(rows, key=lambda row: row["invocation_id"]),
        "unavailable_probe": raw.get("unavailable_probe"),
    }


def save_assignments_incremental(profile, name, state, blobs, *, previous=None):
    def groups(rows):
        result = defaultdict(list)
        for row in rows:
            shard = hashlib.sha256(row.invocation_id.encode()).hexdigest()[:2]
            result[shard].append(row)
        return result

    current = groups(state.frequency_assignments)
    old = groups(previous.frequency_assignments) if previous else {}
    before = store.read_blob(profile, blobs.get(name))
    files = before.get("shards", {}) if before and before.get("format") == ASSIGNMENT_FORMAT else {}
    shards = {}
    for shard, rows in sorted(current.items()):
        if shard in files and rows == old.get(shard):
            shards[shard] = files[shard]
        else:
            shards[shard] = store.write_blob(profile, "assignment-shard", to_plain(rows))
    blobs[name] = store.write_blob(
        profile,
        name,
        {
            "schema": 1,
            "format": ASSIGNMENT_FORMAT,
            "shards": shards,
            "unavailable_probe": to_plain(state.frequency_probe) if state.frequency_probe else None,
            "frequency_refs": sorted({row.view_id for row in state.frequency_assignments}),
        },
    )


def assignment_references(raw):
    return set(raw.get("shards", {}).values()) if raw.get("format") == ASSIGNMENT_FORMAT else set()
