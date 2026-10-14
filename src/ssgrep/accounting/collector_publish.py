"""What the read facade consumes: receipt/history files and snapshot coverage blocks.

Pure builders plus atomic writers, no clocks and no collection logic. Every document
is integers/strings only and must round-trip through ``service_records`` (receipts,
history) and ``service`` (snapshot coverage). Nothing here ever emits a filesystem
path, query, ref, transcript text, receipt content, or witness/action locators:
sources are only ever shown as ``collector_store.source_key`` hashes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting import storage
from ssgrep.accounting.collector_store import SourceRecord
from ssgrep.accounting.collector_types import RetentionCoverage
from ssgrep.accounting.contracts import SCHEMA_VERSION, Reuse, canonical_bytes, digest_of
from ssgrep.accounting.counters import LiveCounters, view
from ssgrep.accounting.receipts import ReceiptRecord, receipt_id_for
from ssgrep.accounting.reduce_types import Reduction
from ssgrep.accounting.replay import LedgerRows
from ssgrep.accounting.revisions import history
from ssgrep.accounting.service_records import MAX_RECEIPT_BYTES, RECEIPT_ID_RE
from ssgrep.accounting.units import UnitSpec
from ssgrep.accounting.valuation_inputs import Candidate, Methodology
from ssgrep.accounting.windows_estimates import quantity_dict

RECEIPT_DIR = "receipts"
HISTORY_DIR = "history"
HISTORY_FILE = "periods.json"
EXCLUSION_PREFIX = "exclusion."
LOOKUP_COUNTERS = (
    "recorded_lookups",
    "attributed_lookups",
    "candidate_receipts",
    "source_costs_available",
    "ambiguous_joins",
    "unavailable_sources",
)
_JOURNAL_GAPS = (
    ("incomplete_tails", "journal_incomplete_tail"),
    ("quarantined", "journal_quarantined"),
    ("conflicts", "journal_conflicts"),
)
_JOURNAL_KEYS = ("segments", "incomplete_tails", "quarantined", "conflicts")
_REUSE_KEYS = frozenset({"class", "relay", "report", "action_ok", "novel", "user_supplied"})
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# ---- receipts -----------------------------------------------------------------------------


def receipt_document(record: ReceiptRecord) -> dict[str, Any]:
    """The stored receipt: ``service_records.read_receipt`` shape plus explanatory extras."""
    receipt = record.receipt
    return {
        "schema": receipt.schema,
        "receipt_id": receipt.receipt_id,
        "revision": receipt.revision,
        "root": receipt.root,
        "task_turn": receipt.task_turn,
        "candidate_type": receipt.candidate_type,
        "atoms": list(receipt.atoms),
        "allocations": [[scenario, list(atoms)] for scenario, atoms in receipt.allocations],
        "exclusions": list(receipt.exclusions),
        "costs": [[name, quantity_dict(amount)] for name, amount in receipt.costs],
        "methodology": receipt.methodology,
        "evidence_available": receipt.evidence_available,
        "execution_ids": list(receipt.execution_ids),
        "unit": record.unit,
        "period": record.period,
        "missing_costs": [[name, reason] for name, reason in record.missing_costs],
        "atom_digests": [[atom, digest] for atom, digest in record.atom_digests],
        "credit_raw": [[scenario, raw] for scenario, raw in record.credit_raw],
        "identities": {
            "frequency": record.frequency_identity,
            "tokenizer": record.tokenizer_identity,
            "pricing": record.pricing_identity,
        },
    }


def _latest_receipts(rows: LedgerRows) -> dict[str, ReceiptRecord]:
    latest: dict[str, ReceiptRecord] = {}
    for record in rows.receipts:
        known = latest.get(record.receipt_id)
        if known is None or record.receipt.revision > known.receipt.revision:
            latest[record.receipt_id] = record
    return dict(sorted(latest.items()))


def reuse_block(candidate: Candidate) -> dict[str, Any]:
    """How reuse was established: the reducer's own judgments, enum plus booleans, nothing else."""
    return {
        "class": candidate.reuse.value,
        "relay": candidate.explicit_relay,
        "report": candidate.report_reference,
        "action_ok": candidate.action_outcome_ok,
        "novel": candidate.novel,
        "user_supplied": candidate.user_supplied,
    }


def reuse_blocks(
    candidates: Iterable[Candidate], methodology: Methodology, superseded: Collection[str] = ()
) -> dict[str, dict[str, Any]]:
    """``reuse`` block per receipt id from the ledger's persisted active candidates."""
    chosen: dict[str, Candidate] = {}
    for cand in candidates:
        if digest_of(cand) in superseded:
            continue
        rid = receipt_id_for(cand.task, cand.candidate_id, methodology)
        known = chosen.get(rid)
        if known is None or cand.accepted > known.accepted:
            chosen[rid] = cand
    return {rid: reuse_block(c) for rid, c in sorted(chosen.items())}


def _valid_reuse(raw: Any) -> dict[str, Any] | None:
    """An already-stored block, kept only when it has exactly the bounded shape."""
    if not isinstance(raw, dict) or set(raw) != _REUSE_KEYS:
        return None
    if raw["class"] not in {r.value for r in Reuse}:
        return None
    if any(not isinstance(raw[k], bool) for k in _REUSE_KEYS - {"class"}):
        return None
    return {k: raw[k] for k in sorted(raw)}


def _stored_reuse(directory: int, receipt_id: str) -> dict[str, Any] | None:
    """The block of the receipt file already on disk (candidates of frozen roots are gone)."""
    try:
        data = storage.read_file(directory, f"{receipt_id}.json", MAX_RECEIPT_BYTES)
        raw = json.loads(data) if data else None
    except (ValueError, OSError):
        return None
    return _valid_reuse(raw.get("reuse")) if isinstance(raw, dict) else None


def _receipt_files(
    rows: LedgerRows,
    reuse: Mapping[str, Mapping[str, Any]] | None = None,
    directory: int | None = None,
) -> tuple[dict[str, bytes], int]:
    """(file bytes per writable receipt id, number of latest receipts that cannot be written).

    A receipt without a fresh block keeps the one already on disk, so rewriting never erases it.
    """
    files: dict[str, bytes] = {}
    skipped = 0
    for receipt_id, record in _latest_receipts(rows).items():
        document = receipt_document(record)
        block = dict(reuse[receipt_id]) if reuse and receipt_id in reuse else None
        if block is None and directory is not None and RECEIPT_ID_RE.fullmatch(receipt_id):
            block = _stored_reuse(directory, receipt_id)
        if block is not None:
            document["reuse"] = block
        data = canonical_bytes(document) + b"\n"
        if RECEIPT_ID_RE.fullmatch(receipt_id) and len(data) <= MAX_RECEIPT_BYTES:
            files[receipt_id] = data
        else:
            skipped += 1
    return files, skipped


def unwritable_receipts(rows: LedgerRows) -> int:
    """Latest receipts ``write_receipts`` skips (id could be a path, or document too large)."""
    return _receipt_files(rows)[1]


def _write_if_changed(directory: int, name: str, data: bytes) -> bool:
    try:
        existing = storage.read_file(directory, name, len(data))
    except ValueError:  # larger than the new bytes, so certainly different
        existing = None
    if existing == data:
        return False
    storage.atomic_write(directory, name, data)
    return True


def write_receipts(
    profile: str, rows: LedgerRows, reuse: Mapping[str, Mapping[str, Any]] | None = None
) -> int:
    """Write ``receipts/<id>.json`` for each receipt's latest revision; never deletes."""
    if not _receipt_files(rows, reuse)[0]:
        return 0
    directory = storage.open_profile_dir(profile, RECEIPT_DIR)
    try:
        files, _ = _receipt_files(rows, reuse, directory)
        return sum(_write_if_changed(directory, f"{rid}.json", data) for rid, data in files.items())
    finally:
        os.close(directory)


def representative_receipt(rows: LedgerRows) -> str | None:
    """Deterministic receipt for ``savings explain``: biggest positive central credit first."""
    credit: list[tuple[int, str]] = []
    costed: list[str] = []
    for receipt_id, record in _latest_receipts(rows).items():
        if not RECEIPT_ID_RE.fullmatch(receipt_id):
            continue
        central = dict(record.credit_raw).get("central", 0)
        if central > 0:
            credit.append((-central, receipt_id))
        if record.receipt.costs:
            costed.append(receipt_id)
    if credit:
        return min(credit)[1]
    return min(costed) if costed else None


# ---- history ------------------------------------------------------------------------------


def history_document(rows: LedgerRows, unit: UnitSpec, today: str) -> list[dict[str, Any]]:
    """Every revision of every period; superseded ones say so, ``recomputable`` is explicit."""
    return [
        {
            "schema": SCHEMA_VERSION,
            "period": record.period,
            "revision": record.revision,
            "status": record.status,
            "supersedes": record.supersedes,
            "net": quantity_dict(record.net),
            "kind": record.kind,
            "recomputable": record.recomputable,
        }
        for record in history(rows.periods, rows.namespace, unit, today)
    ]


def write_history(profile: str, document: list[dict[str, Any]]) -> bool:
    """Atomically replace ``history/periods.json``; False when the bytes are identical."""
    directory = storage.open_profile_dir(profile, HISTORY_DIR)
    try:
        return _write_if_changed(directory, HISTORY_FILE, canonical_bytes(document) + b"\n")
    finally:
        os.close(directory)


# ---- snapshot blocks ----------------------------------------------------------------------


def _positive(counts: Mapping[str, int]) -> dict[str, int]:
    return {name: int(n) for name, n in counts.items() if int(n) > 0}


def _facts(record: SourceRecord) -> tuple[bool, int, int]:
    """(fully processed, lag bytes, processed bytes) of one source.

    Lag is the unread suffix ``max(size - offset, 0)`` (it includes a torn pending tail),
    except for a baseline-skipped complete source: its pre-existing history was
    deliberately not imported, so nothing of it counts as unread.
    """
    size = max(record.size, 0)
    settled = record.baseline == "skipped" and record.complete
    lag = 0 if settled else max(size - record.offset, 0)
    processed = size if settled else min(record.offset, size)
    done = (
        record.status == "ok" and record.complete and not record.more and record.pending_tail == 0
    )
    return done, lag, processed


def _runtime_blocks(records: Sequence[SourceRecord]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for record in records:
        _, lag, processed = _facts(record)
        block = out.setdefault(
            record.runtime,
            {"sources": 0, "processed_bytes": 0, "total_bytes": 0, "lag_bytes": 0, "missing": 0},
        )
        block["sources"] += 1
        block["processed_bytes"] += processed
        block["total_bytes"] += max(record.size, 0)
        block["lag_bytes"] += lag
        block["missing"] += record.status == "missing"
    return dict(sorted(out.items()))


def _watermark(record: SourceRecord) -> dict[str, Any]:
    _, lag, _ = _facts(record)
    return {
        "source": record.key,
        "runtime": record.runtime,
        "offset": record.offset,
        "size": max(record.size, 0),
        "lag_bytes": lag,
        "pending_tail_bytes": record.pending_tail,
        "status": record.status,
    }


def snapshot_coverage(
    *,
    reduction: Reduction,
    sources: Mapping[str, SourceRecord],
    journal: Mapping[str, int],
    read_gaps: Mapping[str, int],
    dropped: Mapping[str, int],
    stopped_by: str,
    retention: RetentionCoverage,
    index: str,
    lag_bytes: int,
) -> dict[str, Any]:
    """Top-level snapshot ``coverage``: watermarks, lag, gaps, and reducer denominators."""
    records = sorted(sources.values(), key=lambda r: r.key)
    facts = [_facts(r) for r in records]
    journal_block = {k: int(journal[k]) for k in _JOURNAL_KEYS if k in journal}
    gaps: dict[str, int] = {}
    for reason, count in _positive(read_gaps).items():
        gaps[reason] = gaps.get(reason, 0) + count
    for key, reason in _JOURNAL_GAPS:
        if journal_block.get(key, 0) > 0:
            gaps[reason] = gaps.get(reason, 0) + journal_block[key]
    for reason, count in _positive(dropped).items():
        gaps[f"dropped_{reason}"] = gaps.get(f"dropped_{reason}", 0) + count
    complete = (
        all(done for done, _, _ in facts)
        and stopped_by == "complete"
        and journal_block.get("incomplete_tails", 0) == 0
    )
    counters = {k: v for k, v in reduction.coverage.items() if k in LOOKUP_COUNTERS}
    recorded = counters.get("recorded_lookups")
    attributed = counters.get("attributed_lookups")
    partial = (
        not complete
        or counters.get("unavailable_sources", 0) > 0
        or counters.get("ambiguous_joins", 0) > 0
        or (recorded is not None and attributed is not None and attributed < recorded)
    )
    return {
        "complete": complete,
        "partial": partial,
        "capture_completeness": "unknown",
        "sources": {
            "total": len(records),
            "complete": sum(done for done, _, _ in facts),
            "lag_bytes": sum(lag for _, lag, _ in facts),
            "pending_tail_bytes": sum(r.pending_tail for r in records),
            "missing": sum(r.status == "missing" for r in records),
            "unreadable": sum(r.status == "unreadable" for r in records),
            "baseline_skipped_bytes": sum(
                r.skipped_bytes for r in records if r.baseline == "skipped"
            ),
            "by_runtime": _runtime_blocks(records),
        },
        "watermarks": [_watermark(r) for r in records],
        "gaps": dict(sorted(gaps.items())),
        "journal": journal_block,
        "stopped_by": stopped_by,
        "lag_bytes": lag_bytes,
        "resets": sum(r.resets for r in records),
        "retention": {
            "evidence_expired_before": retention.evidence_expired_before,
            "aggregates_expired_before": retention.aggregates_expired_before,
            "frozen_roots": retention.frozen_roots,
        },
        **{name: counters[name] for name in LOOKUP_COUNTERS if name in counters},
        "exclusions": dict(
            sorted(
                (name[len(EXCLUSION_PREFIX) :], n)
                for name, n in reduction.coverage.items()
                if name.startswith(EXCLUSION_PREFIX)
            )
        ),
        "index": index,
    }


def snapshot_observations(
    *, counters: LiveCounters | None, backfills: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    """Top-level snapshot ``observations``: live counters, backfill series, counter epoch."""
    out: dict[str, Any] = {"backfill": [dict(series) for series in backfills]}
    if counters is not None:
        out["counters"] = view(counters)
        out["epoch"] = {
            "counter_epoch": counters.epoch,
            "started_at": counters.epoch_started_at,
        }
    return out


def source_manifest_digest(sources: Mapping[str, SourceRecord]) -> str:
    """Digest of what each source looked like (hash keys only: no paths, no clocks)."""
    entries = sorted(
        [r.key, r.runtime, r.offset, max(r.size, 0), r.status, r.baseline] for r in sources.values()
    )
    return digest_of(entries)


def _instant(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).replace(microsecond=0)


def coverage_through(
    sources: Mapping[str, SourceRecord], journal_last_ts: str | None, fallback: str
) -> str:
    """Latest evidence instant seen (sources + journal), never later than ``fallback``."""
    seen = [_instant(r.last_ts) for r in sources.values()] + [_instant(journal_last_ts)]
    instants = [t for t in seen if t is not None]
    ceiling = _instant(fallback)
    if not instants or ceiling is None:
        return fallback
    return min(max(instants), ceiling).strftime(_UTC_FORMAT)
