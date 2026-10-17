"""Replay compact evidence and preserve accepted measurements on unavailable inputs."""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.collect.store import loads as store_loads
from ssgrep.accounting.ledger.contracts import (
    MAX_RAW_RECORD_BYTES,
    SCHEMA_VERSION,
    RawEvidence,
    canonical_bytes,
    digest_of,
)
from ssgrep.accounting.ledger.reduce_records import Reduction
from ssgrep.accounting.ledger.revisions import PeriodRow, build_period_rows, revision_points
from ssgrep.accounting.ledger.valuation import Leaf, value_evidence
from ssgrep.accounting.ledger.valuation_inputs import (
    AtomStatus,
    Candidate,
    Debit,
    EvidenceBundle,
    Methodology,
)
from ssgrep.accounting.readers.base import JsonlReader, loads
from ssgrep.accounting.readers.native_source import (
    MAX_SOURCE_BYTES,
    CodexReplayContext,
    NativeRecordResult,
    NativeSourceStatus as Status,
    ReadProof,
    SourceReadError,
    decode_source_record,
    open_source,
)
from ssgrep.accounting.views.records import ReceiptRecord, build_receipts

if TYPE_CHECKING:
    from ssgrep.accounting.collect.state import ReductionDoc, VersionedLedger


TEXT_KINDS = frozenset({"user_turn", "tool_call", "tool_result"})
OFFSET = re.compile(r"^(?:offset|byte):(\d+)$")


@dataclass(frozen=True)
class ReplayUnavailable:
    runtime: str
    session_id: str
    source_id: str
    reason: str
    native_identity: str


class EvidenceReplay:
    def __init__(self, profile: str, *, adapter: Any = None, env: Any = None) -> None:
        self.profile = profile
        self.adapter = adapter
        self.env = env
        self.unavailable: dict[tuple[str, str], str] = {}
        self.failures: list[ReplayUnavailable] = []

    def hydrate(self, item: RawEvidence) -> RawEvidence:
        if item.payload.get("_evidence_format") != COMPACT or item.kind not in TEXT_KINDS:
            return item
        from ssgrep.accounting.collect.replay_evidence import cached_evidence

        cached = cached_evidence(self.profile, item)
        if cached is not None:
            return replace(
                cached, payload={k: v for k, v in cached.payload.items() if k != "_evidence_format"}
            )
        if self.adapter is None:
            from ssgrep.accounting.ledger.reduce import native_source_adapter
            from ssgrep.accounting.privacy.policy import load_profile_policy

            policy = load_profile_policy(self.profile)
            if not policy.collection_enabled or policy.paused:
                return self._missing(item, "source_unapproved")
            self.adapter = native_source_adapter(
                policy, self.env, now_ms=time.time_ns() // 1_000_000
            )
        from ssgrep.accounting.readers.native_source import NativeSourceReference

        prefix = item.runtime + ":"
        if not item.source_id.startswith(prefix):
            return self._missing(item, "source_unresolved")
        reference = NativeSourceReference(
            item.source_id,
            item.runtime,
            item.session_id,
            Path(item.source_id[len(prefix) :]),
            record_digests=((item.native_identity, item.digest),),
            cwd=item.payload.get("cwd"),
        )
        return self._accepted(
            item,
            self.adapter.read_record(
                reference, item.native_identity, item.digest, locator=item.locator
            ),
        )

    def _accepted(self, item, result):
        if result.evidence is not None and result.evidence.identity == item.identity:
            if result.evidence.ts_utc != item.ts_utc:
                return self._missing(item, "source_timestamp_drift")
            if (result.evidence.runtime, result.evidence.session_id, result.evidence.branch_id) != (
                item.runtime,
                item.session_id,
                item.branch_id,
            ):
                return self._missing(item, "source_ownership_drift")
            for field in ("cwd", "parent_session"):
                if (
                    field in item.payload
                    and result.evidence.payload.get(field) != item.payload[field]
                ):
                    return self._missing(item, "source_ownership_drift")
            payload = dict(result.evidence.payload)
            for name in ("call_record_id", "call_link_status", "occurrence"):
                if name in item.payload:
                    payload[name] = item.payload[name]
            return replace(result.evidence, payload=payload, ts_utc=item.ts_utc)
        reason = getattr(result.status, "value", str(result.status))
        return self._missing(item, "source_identity_mismatch" if reason == "ok" else reason)

    def hydrate_many(self, items) -> list[RawEvidence]:
        from ssgrep.accounting.ledger.reduce import native_source_adapter
        from ssgrep.accounting.privacy.policy import load_profile_policy
        from ssgrep.accounting.readers.native_source import NativeSourceReference

        items = list(items)
        result = list(items)
        policy = load_profile_policy(self.profile)
        if self.adapter is None and (not policy.collection_enabled or policy.paused):
            return [
                self._missing(item, "source_unapproved")
                if item.payload.get("_evidence_format") == COMPACT and item.kind in TEXT_KINDS
                else item
                for item in items
            ]
        groups = defaultdict(list)
        for index, item in enumerate(items):
            if item.payload.get("_evidence_format") != COMPACT or item.kind not in TEXT_KINDS:
                continue
            from ssgrep.accounting.collect.replay_evidence import cached_evidence

            cached = cached_evidence(self.profile, item) if policy.evidence_cache_enabled else None
            if cached is not None:
                result[index] = replace(
                    cached,
                    payload={k: v for k, v in cached.payload.items() if k != "_evidence_format"},
                )
                continue
            if not item.source_id.startswith(item.runtime + ":"):
                result[index] = self._missing(item, "source_unresolved")
                continue
            groups[(item.runtime, item.session_id, item.source_id)].append((index, item))
        if not groups:
            return result
        if self.adapter is None:
            self.adapter = native_source_adapter(
                policy, self.env, now_ms=time.time_ns() // 1_000_000
            )
        for (runtime, session, source), rows in sorted(groups.items()):
            selected = [item for _, item in rows]
            reference = NativeSourceReference(
                source,
                runtime,
                session,
                Path(source[len(runtime) + 1 :]),
                cwd=selected[0].payload.get("cwd"),
            )
            hydrated = read_batch(self.adapter, reference, selected)
            for (index, item), native in zip(rows, hydrated, strict=True):
                result[index] = self._accepted(item, native)
        return result

    def _missing(self, item: RawEvidence, reason: str) -> RawEvidence:
        self.unavailable[(item.runtime, item.session_id)] = reason
        self.failures.append(
            ReplayUnavailable(
                item.runtime, item.session_id, item.source_id, reason, item.native_identity
            )
        )
        return replace(item, payload={**item.payload, "_replay_unavailable": reason})


def preserve_unavailable(
    ledger: VersionedLedger,
    reduction: Reduction,
    previous: ReductionDoc | None,
    unavailable_sessions: Mapping[tuple[str, str], str],
    unavailable_frequencies: Mapping[str, str],
) -> tuple[Reduction, dict[str, EvidenceBundle]]:
    from ssgrep.accounting.collect.state import candidate_identity

    desired = dict(zip(reduction.bundle_roots, reduction.bundles, strict=True))
    if not unavailable_sessions and not unavailable_frequencies:
        return reduction, desired
    old_invocations = previous.invocations if previous else ()
    blocked = set(unavailable_frequencies)
    blocked.update(
        row.invocation_id
        for row in old_invocations
        if (row.runtime, row.session_id) in unavailable_sessions
    )
    source_actors = {f"{runtime}:{session}" for runtime, session in unavailable_sessions}
    state = ledger.state()
    kept_candidates = [
        row
        for row in state.candidates
        if set(row.exposure_ids) & blocked
        or any(
            atom.actor in source_actors for atom in (*row.immediate, *row.discovery, *row.episode)
        )
    ]
    kept_candidates = [
        row for row in kept_candidates if ledger.superseded.get(_digest(row)) is None
    ]
    kept_debits = [
        row
        for row in state.debits
        if row.exposure_id in blocked and ledger.superseded.get(_digest(row)) is None
    ]
    candidate_keys = {candidate_identity(row) for row in kept_candidates}
    debit_keys = {row.debit_id for row in kept_debits}
    frozen_candidate_exposures = {
        exposure for row in kept_candidates for exposure in row.exposure_ids if exposure in blocked
    }
    roots = set(desired) | {row.task.root for row in (*kept_candidates, *kept_debits)}
    merged = {}
    for root in roots:
        current = desired.get(root, EvidenceBundle())
        merged[root] = EvidenceBundle(
            tuple(
                row
                for row in current.candidates
                if candidate_identity(row) not in candidate_keys
                and not set(row.exposure_ids) & frozen_candidate_exposures
            )
            + tuple(row for row in kept_candidates if row.task.root == root),
            tuple(
                row
                for row in current.debits
                if row.debit_id not in debit_keys and row.exposure_id not in blocked
            )
            + tuple(row for row in kept_debits if row.task.root == root),
        )
    kept_invocations = [row for row in old_invocations if row.invocation_id in blocked]
    kept_executions = [
        row for row in (previous.executions if previous else ()) if row.invocation_id in blocked
    ]
    execution_ids = {row.execution_id for row in kept_executions}
    coverage = dict(reduction.coverage)
    coverage["replay_unavailable_sessions"] = len(unavailable_sessions)
    coverage["frequency_revision_unavailable"] = len(unavailable_frequencies)
    updated = replace(
        reduction,
        invocations=tuple(
            sorted(
                (
                    *[row for row in reduction.invocations if row.invocation_id not in blocked],
                    *kept_invocations,
                ),
                key=lambda row: row.invocation_id,
            )
        ),
        executions=tuple(
            sorted(
                (
                    *[row for row in reduction.executions if row.execution_id not in execution_ids],
                    *kept_executions,
                ),
                key=lambda row: (row.ts_utc, row.execution_id),
            )
        ),
        coverage=coverage,
        warnings=tuple(sorted(set(reduction.warnings) | {"historical_replay_unavailable"})),
    )
    return updated, merged


def _digest(row) -> str:
    from ssgrep.accounting.ledger.contracts import digest_of

    return digest_of(row)


def read_batch(adapter, reference, items):
    """Read exact selected native records and verified intervening context."""
    reader = adapter.readers.get(reference.runtime)
    if not isinstance(reader, JsonlReader):
        return [
            adapter.read_record(reference, item.native_identity, item.digest, locator=item.locator)
            for item in items
        ]
    if not items:
        return []
    if reference.expires_at_ms is not None and reference.expires_at_ms <= adapter.now_ms:
        return [NativeRecordResult(Status.EXPIRED) for _ in items]
    if reference.cwd is not None:
        scope, why = adapter.scoper.source_status(reference.cwd)
        if scope is not AtomStatus.OK:
            return [NativeRecordResult(Status(why)) for _ in items]
    budget = min(MAX_SOURCE_BYTES, adapter.byte_budget) - adapter.bytes_read
    if budget <= 0:
        return [NativeRecordResult(Status.BOUNDED) for _ in items]
    grouped = {}
    for index, item in enumerate(items):
        found = OFFSET.fullmatch(item.locator)
        if found is None:
            return [NativeRecordResult(Status.UNRESOLVED) for _ in items]
        grouped.setdefault(int(found.group(1)), []).append((index, item))
    results = [NativeRecordResult(Status.INCOMPLETE) for _ in items]
    count = 0
    descriptor = before = None
    proof = ReadProof()
    codex = CodexReplayContext(reference, adapter.scoper) if reference.runtime == "codex" else None
    context = codex.values if codex is not None else {"session": reference.session_id}
    source = reader.source_for(reference.path)
    header_seen = reference.runtime == "claude"

    def charge(size, *, started_record=False):
        nonlocal count
        count += size
        adapter.bytes_read += size
        if count > budget and not started_record:
            raise SourceReadError(Status.BOUNDED)

    def decode(raw, position):
        nonlocal header_seen
        decoded = decode_source_record(
            reader, source, loads(raw), context, f"offset:{position}", codex
        )
        headers = [row for row in decoded if row.kind == "session_header"]
        if headers:
            if reference.session_id not in {row.session_id for row in headers}:
                raise SourceReadError(Status.IDENTITY_MISMATCH)
            header_seen = True
        return decoded

    def remaining(status):
        return [row if row.evidence is not None else NativeRecordResult(status) for row in results]

    try:
        descriptor = open_source(reference.path, adapter.approved_roots)
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            position = 0
            for offset, wanted in sorted(grouped.items()):
                if offset >= before.st_size:
                    for index, _ in wanted:
                        results[index] = NativeRecordResult(Status.MISSING)
                    continue
                while position < offset:
                    if count >= budget:
                        raise SourceReadError(Status.BOUNDED)
                    stream.seek(position)
                    raw = stream.readline(min(MAX_RAW_RECORD_BYTES + 1, offset - position))
                    charge(len(raw), started_record=True)
                    proof.add(position, raw)
                    if len(raw) > MAX_RAW_RECORD_BYTES:
                        raise SourceReadError(Status.BOUNDED)
                    if not raw.endswith(b"\n"):
                        raise SourceReadError(Status.INCOMPLETE)
                    decode(raw, position)
                    position += len(raw)
                stream.seek(offset)
                if count >= budget:
                    raise SourceReadError(Status.BOUNDED)
                raw = stream.readline(MAX_RAW_RECORD_BYTES + 1)
                charge(len(raw), started_record=True)
                proof.add(offset, raw)
                if len(raw) > MAX_RAW_RECORD_BYTES or not raw.endswith(b"\n"):
                    raise SourceReadError(
                        Status.BOUNDED if len(raw) > MAX_RAW_RECORD_BYTES else Status.INCOMPLETE
                    )
                decoded = decode(raw, offset)
                position = offset + len(raw)
                if not (codex.header_seen if codex is not None else header_seen):
                    raise SourceReadError(Status.IDENTITY_MISMATCH)
                native_cwd = codex.source_context().cwd if codex is not None else context.get("cwd")
                for index, expected in wanted:
                    matches = [
                        row for row in decoded if row.native_identity == expected.native_identity
                    ]
                    if len(matches) != 1:
                        results[index] = NativeRecordResult(Status.IDENTITY_MISMATCH)
                        continue
                    actual = matches[0]
                    if actual.identity != expected.identity or actual.ts_utc != expected.ts_utc:
                        results[index] = NativeRecordResult(Status.CHANGED)
                        continue
                    if (actual.runtime, actual.session_id) != (
                        expected.runtime,
                        expected.session_id,
                    ):
                        results[index] = NativeRecordResult(Status.IDENTITY_MISMATCH)
                        continue
                    cwd = actual.payload.get("cwd") or actual.payload.get("directory") or native_cwd
                    scope, why = adapter.scoper.source_status(cwd if isinstance(cwd, str) else None)
                    results[index] = (
                        NativeRecordResult(Status(why))
                        if scope is not AtomStatus.OK
                        else NativeRecordResult(Status.OK, actual)
                    )
    except SourceReadError as error:
        results = remaining(error.status)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        results = remaining(Status.UNREADABLE)
    finally:
        if descriptor is not None:
            if before is not None and any(row.evidence is not None for row in results):
                try:
                    proof.verify(descriptor, before, charge)
                except SourceReadError as error:
                    results = [NativeRecordResult(error.status) for _ in items]
                except OSError:  # pragma: no cover - I/O fault on an open regular file
                    pass
            os.close(descriptor)
    if results:
        results[0] = replace(results[0], bytes_read=count)
    return results


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


COMPACT = "compact-v1"
PLAINTEXT = "plaintext-cache-v1"
LEGACY = "legacy-plaintext"
_IDENTIFIERS = frozenset(
    {
        "cwd",
        "directory",
        "call_id",
        "call_record_id",
        "call_link_status",
        "turn_id",
        "parent_session",
        "parent_session_id",
        "parent_id",
        "parent_session_path",
        "session_file",
        "child_id",
        "id",
        "message_id",
        "session_id",
        "status",
        "reason",
        "original_kind",
        "source_id",
        "branch_id",
        "parent",
        "claimant_source_id",
        "claimant_session_id",
        "claimant_branch_id",
        "claimant_cwd",
        "claimant_parent_session",
        "model",
        "provider",
        "counter",
        "unavailable",
        "target_digest",
        "target_record_id",
        "originating_turn_record_id",
        "task_context_basis",
    }
)
_NUMBERS = frozenset(
    {
        "exit_code",
        "is_error",
        "duration_ms",
        "duration_ns",
        "time_updated",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "updates",
        "stream_updates",
        "streaming_updates",
        "rlm_depth",
        "version",
        "reset",
        "occurrence",
        "completed_ms",
        "origin_conflict",
        "baseline_prior_context_unavailable",
    }
)


def _measurements(value: Any) -> Any:
    if value is None or isinstance(value, (int, bool)):
        return value
    names = {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "input",
        "output",
        "cacheRead",
        "cacheWrite",
        "total",
    }
    if isinstance(value, dict):
        return {
            k: v
            for k, v in value.items()
            if k in names and isinstance(v, int) and not isinstance(v, bool)
        }
    return None


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        k: v
        for k, v in payload.items()
        if k in _IDENTIFIERS and (v is None or isinstance(v, str) and len(v) <= 4096)
    }
    out.update(
        {
            k: v
            for k, v in payload.items()
            if k in _NUMBERS and (v is None or isinstance(v, (int, bool)))
        }
    )
    for key in ("tokens", "delta", "usage", "cumulative", "cost_nano_usd"):
        if key in payload:
            out[key] = _measurements(payload[key])
    if isinstance(payload.get("session_context"), dict):
        out["session_context"] = {
            k: v
            for k, v in compact_payload(payload["session_context"]).items()
            if k != "_evidence_format"
        }
    target = payload.get("target_native_key")
    if (
        isinstance(target, (list, tuple))
        and len(target) == 3
        and all(isinstance(v, str) for v in target)
    ):
        out["target_native_key"] = list(target)
    out["_evidence_format"] = COMPACT
    return out


def evidence_line(item: RawEvidence, *, plaintext: bool = False) -> bytes:
    raw = {
        name: getattr(item, name)
        for name in (
            "runtime",
            "source_id",
            "branch_id",
            "session_id",
            "record_id",
            "locator",
            "digest",
            "ts_utc",
            "kind",
            "schema",
            "origin_domain",
            "origin_id",
            "origin_status",
        )
    }
    raw["evidence_format"] = PLAINTEXT if plaintext else COMPACT
    raw["payload"] = item.payload if plaintext else compact_payload(item.payload)
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def evidence_from_line(line: bytes) -> RawEvidence:
    raw = store_loads(line)
    form = raw.get("evidence_format", LEGACY)
    if form not in {COMPACT, PLAINTEXT, LEGACY}:
        raise ValueError("unsupported retained evidence format")
    payload = dict(raw["payload"])
    if form == COMPACT:
        payload = compact_payload(payload)
    else:
        payload["_evidence_format"] = form
    return RawEvidence(
        runtime=str(raw["runtime"]),
        source_id=str(raw["source_id"]),
        branch_id=str(raw["branch_id"]),
        session_id=str(raw["session_id"]),
        record_id=str(raw["record_id"]),
        locator=str(raw["locator"]),
        digest=str(raw["digest"]),
        ts_utc=raw["ts_utc"],
        kind=str(raw["kind"]),
        payload=payload,
        schema=int(raw.get("schema", SCHEMA_VERSION)),
        origin_domain=raw.get("origin_domain"),
        origin_id=raw.get("origin_id"),
        origin_status=str(raw.get("origin_status", "legacy_unverified"))
        if form != LEGACY
        else "legacy_unverified",
    )
