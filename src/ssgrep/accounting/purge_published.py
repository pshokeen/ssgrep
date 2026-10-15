"""Date/scope deletion of genuine original publications, with no valuation decoding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, NoReturn

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.published_documents import day, read_document, references
from ssgrep.accounting.purge_model import PurgeError
from ssgrep.accounting.purge_select import Scoper, Scopes, Selection


@dataclass(frozen=True)
class RemovedRows:
    period_rows_removed: int = 0
    receipt_rows_removed: int = 0


@dataclass
class PublishedPlan:
    checkpoint: store.Checkpoint
    scopes: Scopes
    manifest: dict[str, Any]
    documents: dict[str, Any]
    updates: dict[str, tuple[str, Any]] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)
    rewrite: frozenset[str] = frozenset()
    late_sessions: frozenset = frozenset()
    gone_sessions: frozenset = frozenset()
    evidence_records: int = 0
    ledger_items: int = 0
    executions_removed: int = 0
    invocations_removed: int = 0
    sealed_removed: tuple[str, ...] = ()
    backfills_removed: int = 0
    derived_caches: tuple[str, ...] = ()
    namespaces: tuple[RemovedRows, ...] = ()
    remaining_records: int = 0
    history_remaining: int | None = None
    kept_roots: set[str] = field(default_factory=set)
    archive_metadata_records: int = 0
    selected_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    history_minimum_day: str | None = None

    @property
    def changed(self) -> bool:
        return bool(
            self.updates
            or self.dropped
            or self.evidence_records
            or self.documents != self.manifest["published_documents"]
        )


def _unresolved(message: str) -> NoReturn:
    raise PurgeError("archive_scope_unresolved", message + " Nothing was changed.")


def _root(record):
    task = record.get("task")
    receipt = record.get("receipt")
    return (
        (task.get("root") if isinstance(task, dict) else None)
        or record.get("root")
        or (receipt.get("root") if isinstance(receipt, dict) else None)
    )


def _scopes(profile, manifest, checkpoint, sel):
    from ssgrep.accounting.collector_state import ReductionDoc
    from ssgrep.accounting.purge_select import classify

    roots = tuple(tuple(row) for row in manifest.get("session_roots", []))
    cwds = tuple(tuple(row) for row in manifest.get("session_cwd", []))
    if sel.workspace is not None and manifest.get("root_mapping_status") != "available":
        _unresolved("Original publication has no verified workspace/root map.")
    doc = ReductionDoc(session_roots=roots, session_cwd=cwds)
    scopes = classify(profile, checkpoint, doc, (row[2] for row in roots), Scoper(sel.workspace))
    if sel.workspace is not None and (scopes.unknown or scopes.mixed):
        _unresolved("Original publication contains unknown or mixed workspace provenance.")
    return scopes


def _array_subset(text: str, keep: set[int]) -> str:
    """Retain original JSON row slices, including their numeric spelling and key order."""
    decoder = json.JSONDecoder()
    index, row, pieces = text.index("[") + 1, 0, []
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if text[index] == "]":
            break
        _, end = decoder.raw_decode(text, index)
        if row in keep:
            pieces.append(text[index:end])
        row += 1
        index = end
        while index < len(text) and text[index].isspace():
            index += 1
        if text[index] == ",":
            index += 1
    return "[" + ",".join(pieces) + "]\n"


def _plan_selected(
    profile: str, manifest: dict[str, Any], sel: Selection, *, atomic: bool = False
) -> PublishedPlan:
    from ssgrep.accounting.purge_state import _scan_evidence

    documents = dict(manifest["published_documents"])
    references(documents)
    checkpoint = store.Checkpoint(
        profile, segments=[store.SegmentRef.from_dict(row) for row in manifest.get("segments", [])]
    )
    scopes = _scopes(profile, manifest, checkpoint, sel)
    plan = PublishedPlan(checkpoint, scopes, manifest, documents)
    selected = scopes.selected
    known_roots = set(scopes.root_of.values())
    scoper = Scoper(sel.workspace)
    period_roots = documents.get("roots_by_period")
    if selected is not None:
        plan.kept_roots.update(known_roots - selected)

    def note_remaining(record):
        root = _root(record) or scopes.root_of.get(
            (record.get("runtime"), record.get("session_id"))
        )
        if isinstance(root, str):
            plan.kept_roots.add(root)

    def original_period_selected(period):
        if day(period) is None:
            raise ValueError("Original publication period is invalid.")
        if period >= sel.cutoff_day:
            return False
        if selected is None:
            return True
        roots = period_roots.get(period) if isinstance(period_roots, dict) else None
        if (
            not isinstance(roots, list)
            or not roots
            or any(root not in known_roots for root in roots)
        ):
            _unresolved("Original aggregate has no complete period/root membership proof.")
        hits = set(roots) & selected
        if hits and hits != set(roots):
            _unresolved("Original multi-workspace aggregate cannot be split without revaluation.")
        return bool(hits)

    def selected_record(record, *, period=None, scope_alias=False):
        stamp = period or record.get("day") or record.get("period")
        if stamp is None and isinstance(record.get("ts_utc"), str):
            stamp = record["ts_utc"][:10]
        if day(stamp) is None or stamp >= sel.cutoff_day:
            return False
        if selected is None:
            return True
        if scope_alias:
            _unresolved("Original scope-epoch records have no per-epoch workspace proof.")
        root = _root(record) or scopes.root_of.get(
            (record.get("runtime"), record.get("session_id"))
        )
        if root in known_roots:
            return root in selected
        cwd = record.get("cwd")
        if isinstance(cwd, str):
            return scoper.inside(cwd)
        _unresolved("Original dated record has no verified workspace provenance.")

    removed_periods = removed_receipts = 0
    history = documents.get("history_blob")
    if history is not None:
        envelope, rows = read_document(profile, history, "history")
        keep = {
            index for index, row in enumerate(rows) if not original_period_selected(row["period"])
        }
        plan.history_remaining = len(keep)
        plan.history_minimum_day = min((rows[index]["period"] for index in keep), default=None)
        removed_periods = len(rows) - len(keep)
        if removed_periods:
            text = _array_subset(envelope["original_utf8"], keep)
            plan.updates[history] = (
                "published-history",
                {
                    **envelope,
                    "original_utf8": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "source_sha256": envelope.get("source_sha256", envelope["sha256"]),
                    "filtered_before": sel.cutoff_day,
                },
            )
    receipts = dict(documents.get("receipt_blobs", {}))
    receipt_periods, receipt_roots = {}, {}
    for receipt_id, filename in list(receipts.items()):
        _, row = read_document(profile, filename, "receipt")
        if row.get("receipt_id") != receipt_id:
            raise ValueError("Original receipt identity does not match.")
        if selected_record(row):
            receipts.pop(receipt_id)
            plan.dropped.add(filename)
            removed_receipts += 1
        else:
            note_remaining(row)
            receipt_periods[receipt_id], receipt_roots[receipt_id] = row["period"], row["root"]
    documents["receipt_blobs"] = receipts
    if removed_receipts:
        documents.update(
            receipt_periods=receipt_periods,
            receipt_roots=receipt_roots,
            receipt_count=len(receipts),
            receipt_min_period=min(receipt_periods.values(), default=None),
        )
    if isinstance(period_roots, dict):
        documents["roots_by_period"] = {
            period: roots
            for period, roots in period_roots.items()
            if not original_period_selected(period)
        }
        for roots in documents["roots_by_period"].values():
            plan.kept_roots.update(roots)
        plan.archive_metadata_records += sum(len(roots) for roots in period_roots.values()) - sum(
            len(roots) for roots in documents["roots_by_period"].values()
        )

    from ssgrep.accounting.purge_published_inputs import prune_inputs

    reductions = prune_inputs(
        profile,
        plan,
        sel,
        atomic=atomic,
        selected_record=selected_record,
        note_remaining=note_remaining,
    )

    scan = _scan_evidence(profile, checkpoint, sel.cutoff_day, scopes)
    plan.rewrite, plan.evidence_records, plan.gone_sessions, plan.late_sessions = scan
    plan.kept_roots.update(
        scopes.root_of[key] for key in plan.late_sessions if key in scopes.root_of
    )
    if plan.changed:
        for filename, (name, raw) in reductions.items():
            updated = dict(raw)
            if "session_roots" not in raw:
                continue  # No trusted relation: preserve unselected minimum context, never guess.
            sessions = {
                (runtime, session)
                for runtime, session, root in raw.get("session_roots", [])
                if root in plan.kept_roots
            }
            for key in ("session_roots", "session_cwd", "session_parents", "session_last"):
                if key in raw:
                    kept = [row for row in raw[key] if (row[0], row[1]) in sessions]
                    plan.archive_metadata_records += len(raw[key]) - len(kept)
                    updated[key] = kept
            if "invocation_sessions" in raw:
                kept = [row for row in raw["invocation_sessions"] if (row[1], row[2]) in sessions]
                plan.archive_metadata_records += len(raw["invocation_sessions"]) - len(kept)
                updated["invocation_sessions"] = kept
            if updated != raw:
                plan.updates[filename] = (name, updated)
    if plan.changed:
        caches = {
            "partitions",
            "journal-cache",
            "root-evaluations",
            "frequency-maintenance",
            "frequency-assignments",
        }
        plan.derived_caches = tuple(
            name for name in manifest.get("blobs", {}) if name.split("-scope-", 1)[0] in caches
        )
        plan.dropped.update(manifest["blobs"][name] for name in plan.derived_caches)
    plan.namespaces = (RemovedRows(removed_periods, removed_receipts),)
    return plan


def plan_published(profile: str, manifest: dict[str, Any], sel: Selection) -> PublishedPlan:
    return _plan_selected(profile, manifest, sel)


def write_published(
    profile: str, name: str, plan: PublishedPlan, sel: Selection, blobs: dict[str, str]
):
    from ssgrep.accounting.purge_state import _write_segments

    if not plan.changed:
        return dict(plan.manifest.get("blobs", {}))
    memo = {
        old: store.write_blob(profile, prefix, document)
        for old, (prefix, document) in plan.updates.items()
        if old not in plan.dropped
    }
    retained = {
        key: memo.get(value, value)
        for key, value in plan.manifest.get("blobs", {}).items()
        if value not in plan.dropped
    }
    documents = {
        **plan.documents,
        "history_blob": memo.get(
            plan.documents.get("history_blob"), plan.documents.get("history_blob")
        ),
        "receipt_blobs": {
            key: memo.get(value, value) for key, value in plan.documents["receipt_blobs"].items()
        },
    }
    segments = (
        _write_segments(profile, plan, sel) if plan.rewrite else list(plan.checkpoint.segments)
    )
    if (
        not segments
        and not plan.remaining_records
        and not plan.history_remaining
        and not documents["receipt_blobs"]
        and not plan.kept_roots
    ):
        blobs.pop(name, None)
        return {}
    manifest = {
        **plan.manifest,
        "blobs": retained,
        "published_documents": documents,
        "segments": [ref.to_dict() for ref in segments],
    }
    kept_sessions = {
        (runtime, session)
        for runtime, session, root in manifest.get("session_roots", [])
        if root in plan.kept_roots
    }
    manifest["session_roots"] = [
        row for row in manifest.get("session_roots", []) if (row[0], row[1]) in kept_sessions
    ]
    manifest["session_cwd"] = [
        row for row in manifest.get("session_cwd", []) if (row[0], row[1]) in kept_sessions
    ]
    blobs[name] = store.write_blob(profile, name, manifest)
    return retained
