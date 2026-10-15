"""Deterministic reduction of retained evidence into canonical ledger rows (D7/D8).

``Ledger`` is a pure reducer over frozen evidence (no clocks, no file I/O). Evidence
is a *set* keyed by content digest, so full replay, restarted incremental batches,
and any batch order converge on the same rows and digest under one methodology.
A different methodology is a different namespace whose rows are never summed with
another's. Expired roots are frozen (non-recomputable) rather than rebuilt from
changed sources; an explicit purge removes them entirely.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

from ssgrep.accounting.contracts import canonical_bytes, digest_of
from ssgrep.accounting.receipts import ReceiptRecord, build_receipts
from ssgrep.accounting.revisions import (
    PeriodRow,
    build_period_rows,
    revision_points,
)
from ssgrep.accounting.valuation import Leaf, value_evidence
from ssgrep.accounting.valuation_inputs import Candidate, Debit, EvidenceBundle, Methodology

Evaluated = tuple[tuple[Leaf, ...], Mapping[str, ReceiptRecord]]


@dataclass(frozen=True)
class FrozenRoot:
    """A root whose evidence expired: totals and receipts kept, recomputation gone."""

    root: str
    leaves: tuple[Leaf, ...]
    receipts: tuple[ReceiptRecord, ...]


@dataclass(frozen=True)
class LedgerState:
    """Everything retained between restarts (persisted by the caller, not here)."""

    methodology: Methodology
    candidates: tuple[Candidate, ...] = ()
    debits: tuple[Debit, ...] = ()
    frozen: tuple[FrozenRoot, ...] = ()


@dataclass(frozen=True)
class ArchivedRows:
    """An immutable named replacement's prior rows; never part of current totals."""

    revision_name: str
    namespace: str
    periods: tuple[PeriodRow, ...]
    receipts: tuple[ReceiptRecord, ...]

    def as_rows(self) -> LedgerRows:
        return LedgerRows(self.namespace, self.periods, self.receipts)


@dataclass(frozen=True)
class LedgerRows:
    """Canonical, ordered ledger content of ONE methodology namespace."""

    namespace: str
    periods: tuple[PeriodRow, ...]
    receipts: tuple[ReceiptRecord, ...]
    archives: tuple[ArchivedRows, ...] = ()

    def canonical_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = [
            {"kind": "period", "digest": p.digest, "row": p} for p in self.periods
        ]
        rows += [
            {"kind": "receipt", "digest": r.digest, "row": r.canonical_row()} for r in self.receipts
        ]
        return rows

    def _document(self) -> dict[str, object]:
        document: dict[str, object] = {"namespace": self.namespace, "rows": self.canonical_rows()}
        if self.archives:
            document["archives"] = self.archives
        return document

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self._document())

    @property
    def digest(self) -> str:
        return digest_of(self._document())


@dataclass(frozen=True)
class IngestResult:
    accepted: int
    duplicates: int
    rejected: tuple[tuple[str, str], ...]  # (evidence id, reason)


@dataclass
class Ledger:
    """Mutable holder of one namespace; ``state()``/``restore`` survive restarts."""

    methodology: Methodology
    _cands: dict[str, Candidate] = field(default_factory=dict)
    _debits: dict[str, Debit] = field(default_factory=dict)
    _frozen: dict[str, FrozenRoot] = field(default_factory=dict)
    _eval_cache: dict[
        tuple[str, tuple[str, ...]], tuple[tuple[Leaf, ...], Mapping[str, ReceiptRecord]]
    ] = field(default_factory=dict)

    @property
    def namespace(self) -> str:
        return self.methodology.hash

    @classmethod
    def restore(cls, state: LedgerState) -> Ledger:
        ledger = cls(state.methodology)
        ledger._cands = {digest_of(c): c for c in state.candidates}
        ledger._debits = {digest_of(d): d for d in state.debits}
        ledger._frozen = {f.root: f for f in state.frozen}
        return ledger

    def state(self) -> LedgerState:
        return LedgerState(
            self.methodology,
            tuple(self._cands[k] for k in sorted(self._cands)),
            tuple(self._debits[k] for k in sorted(self._debits)),
            tuple(self._frozen[k] for k in sorted(self._frozen)),
        )

    def ingest(self, batch: EvidenceBundle) -> IngestResult:
        """Add evidence idempotently; evidence for an expired root is never rebuilt."""
        accepted = duplicates = 0
        rejected: list[tuple[str, str]] = []
        for cand in batch.candidates:
            key = digest_of(cand)
            if cand.task.root in self._frozen:
                rejected.append((cand.candidate_id, "evidence_expired_root"))
            elif key in self._cands:
                duplicates += 1
            else:
                self._cands[key] = cand
                accepted += 1
        for debit in batch.debits:
            key = digest_of(debit)
            if debit.task.root in self._frozen:
                rejected.append((debit.debit_id, "evidence_expired_root"))
            elif key in self._debits:
                duplicates += 1
            else:
                self._debits[key] = debit
                accepted += 1
        return IngestResult(accepted, duplicates, tuple(sorted(rejected)))

    # ---- evaluation ------------------------------------------------------------

    def _root_subset(self, root: str, point: str) -> tuple[tuple[str, ...], EvidenceBundle]:
        cands = {
            k: c for k, c in self._cands.items() if c.task.root == root and c.accepted <= point
        }
        debits = {
            k: d for k, d in self._debits.items() if d.task.root == root and d.accepted <= point
        }
        key = tuple(sorted(cands)) + ("|",) + tuple(sorted(debits))
        bundle = EvidenceBundle(
            tuple(cands[k] for k in sorted(cands)), tuple(debits[k] for k in sorted(debits))
        )
        return key, bundle

    def _evaluate(
        self, root: str, point: str
    ) -> tuple[tuple[Leaf, ...], Mapping[str, ReceiptRecord]]:
        key, bundle = self._root_subset(root, point)
        cached = self._eval_cache.get((root, key))
        if cached is None:
            valuation = value_evidence(bundle, self.methodology)
            receipts = build_receipts(
                bundle.candidates, valuation.allocations, bundle.debits, self.methodology
            )
            cached = (valuation.leaves, receipts)
            self._eval_cache[(root, key)] = cached
        return cached

    def _roots(self) -> list[str]:
        roots = {c.task.root for c in self._cands.values()}
        return sorted(roots | {d.task.root for d in self._debits.values()})

    def _points(self, periods: Iterable[str]) -> list[str]:
        accepted = [c.accepted for c in self._cands.values()]
        accepted += [d.accepted for d in self._debits.values()]
        points: set[str] = set()
        for period in periods:
            points.update(revision_points(period, accepted))
        return sorted(points)

    def rows(self) -> LedgerRows:
        """Canonical rows of every period revision and receipt revision."""
        roots = self._roots()
        periods = {c.day for c in self._cands.values()} | {d.day for d in self._debits.values()}
        frozen_leaves = [leaf for f in self._frozen.values() for leaf in f.leaves]
        periods |= {leaf.day for leaf in frozen_leaves}
        frozen_roots = frozenset(self._frozen)

        def leaves_at(point: str, period: str) -> list[Leaf]:
            live = [
                leaf for r in roots for leaf in self._evaluate(r, point)[0] if leaf.day == period
            ]
            return live + [leaf for leaf in frozen_leaves if leaf.day == period]

        period_rows = build_period_rows(
            self.namespace,
            periods,
            [c.accepted for c in self._cands.values()]
            + [d.accepted for d in self._debits.values()],
            leaves_at,
            frozen_roots,
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
        return LedgerRows(self.namespace, tuple(period_rows), tuple(receipts))

    # ---- retention ---------------------------------------------------------------

    def expirable_roots(self, evidence_before_day: str) -> list[str]:
        """Roots whose every evidence item was accepted before the given UTC day."""
        latest: dict[str, str] = {}
        for item in (*self._cands.values(), *self._debits.values()):
            latest[item.task.root] = max(latest.get(item.task.root, ""), item.accepted)
        return sorted(r for r, day in latest.items() if day < evidence_before_day)

    def expire_roots(self, roots: Iterable[str]) -> list[str]:
        """Freeze roots at their current totals; drop their evidence; mark receipts.

        Returns the roots frozen. Frozen leaves keep contributing to closed totals
        but are non-recomputable; nothing is re-derived from changed sources.
        """
        frozen_now: list[str] = []
        final = "9999-12-31"
        for root in sorted(set(roots) & set(self._roots())):
            leaves, receipts = self._evaluate(root, final)
            history = [r for r in self.rows().receipts if r.receipt.root == root]
            self._frozen[root] = FrozenRoot(
                root,
                leaves,
                tuple(r.mark_expired() for r in (history or list(receipts.values()))),
            )
            self._cands = {k: c for k, c in self._cands.items() if c.task.root != root}
            self._debits = {k: d for k, d in self._debits.items() if d.task.root != root}
            self._eval_cache = {k: v for k, v in self._eval_cache.items() if k[0] != root}
            frozen_now.append(root)
        return frozen_now

    def purge(self, before_day: str | None = None, roots: Iterable[str] = ()) -> None:
        """Explicit privacy purge: remove evidence, frozen totals, and receipts.

        Privacy outranks reproducibility. ``before_day`` removes everything dated
        earlier (remaining evidence is recomputed on its own); ``roots`` removes whole
        roots. Frozen leaves are removed per day, and a frozen root with no leaves
        left disappears with its receipts.
        """
        doomed = frozenset(roots)
        self._purge(
            lambda day, root: root in doomed or (before_day is not None and day < before_day)
        )

    def purge_scoped(self, *, before_day: str, roots: Iterable[str]) -> None:
        """Remove only records before the cutoff AND belonging to selected roots."""
        doomed = frozenset(roots)
        self._purge(lambda day, root: root in doomed and day < before_day)

    def _purge(self, purged: Callable[[str, str], bool]) -> None:

        self._cands = {k: c for k, c in self._cands.items() if not purged(c.day, c.task.root)}
        self._debits = {k: d for k, d in self._debits.items() if not purged(d.day, d.task.root)}
        kept: dict[str, FrozenRoot] = {}
        for root, frozen in self._frozen.items():
            leaves = tuple(leaf for leaf in frozen.leaves if not purged(leaf.day, root))
            receipts = tuple(r for r in frozen.receipts if not purged(r.period, root))
            if leaves:
                kept[root] = FrozenRoot(root, leaves, receipts)
        self._frozen = kept
        self._eval_cache.clear()


def full_replay(state_or_bundle: EvidenceBundle, methodology: Methodology) -> LedgerRows:
    """Reduce all retained evidence in one pass (the reference for equivalence)."""
    ledger = Ledger(methodology)
    ledger.ingest(state_or_bundle)
    return ledger.rows()


def select_namespace(views: Iterable[LedgerRows], methodology: Methodology) -> LedgerRows:
    """Pick exactly one methodology's rows; other namespaces are never merged in."""
    for rows in views:
        if rows.namespace == methodology.hash:
            return rows
    raise KeyError(f"no ledger rows for methodology {methodology.hash}")
