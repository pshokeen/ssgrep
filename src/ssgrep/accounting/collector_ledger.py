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

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TypeVar

from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.receipts import ReceiptRecord
from ssgrep.accounting.replay import ArchivedRows, Ledger, LedgerRows, LedgerState
from ssgrep.accounting.revisions import build_period_rows, estimates_of, revision_points, seal_day
from ssgrep.accounting.valuation import Leaf
from ssgrep.accounting.valuation_inputs import Candidate, Debit, EvidenceBundle

Item = TypeVar("Item", Candidate, Debit)

_EXPIRED = "evidence_expired_root"
CANDIDATE_IDENTITY_METHOD = "lookup_exposure_v1"


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
