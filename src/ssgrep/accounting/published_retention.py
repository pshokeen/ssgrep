"""Selected-child original-date expiry; no global archive replay or valuation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.published_documents import day, references
from ssgrep.accounting.purge_model import PurgeError
from ssgrep.accounting.purge_select import Selection

_CACHES = {
    "partitions",
    "journal-cache",
    "root-evaluations",
    "frequency-maintenance",
    "frequency-assignments",
}


@dataclass(frozen=True)
class ChildResult:
    changed: bool
    expired_receipts: tuple[str, ...]
    manifest: dict | None


def _minimum(days):
    minimum, complete = None, True
    for value in days:
        stamp = day(value)
        if stamp is None:
            complete = False
        elif minimum is None or stamp < minimum:
            minimum = stamp
    return minimum, complete


def _record_day(record):
    stamp = record.get("day") or record.get("period")
    if stamp is None and isinstance(record.get("ts_utc"), str):
        stamp = record["ts_utc"][:10]
    return stamp


def _raw_minimum(role, raw):
    if role.startswith("ledger-"):
        state = raw["state"]
        records = chain(
            state.get("candidates", []),
            state.get("debits", []),
            (
                row
                for root in state.get("frozen", [])
                for row in chain(root.get("leaves", []), root.get("receipts", []))
            ),
        )
        return _minimum(_record_day(row) for row in records)
    if role == "reduction":
        return _minimum(
            chain(
                (
                    _record_day(row)
                    for row in chain(raw.get("executions", []), raw.get("invocations", []))
                ),
                (
                    stamp[:10] if isinstance(stamp, str) else None
                    for _, _, stamp in raw.get("session_last", [])
                ),
            )
        )
    if role == "counters":
        return _minimum(row[1] for row in raw.get("counted", []))
    if role == "sealed":
        return _minimum(raw)
    return None, False


def _index(source, minimum, complete, before_day):
    return {
        "schema": 1,
        "source": source,
        "minimum": minimum,
        "complete": complete,
        "clean_before": before_day,
    }


def stage_children(
    profile,
    checkpoint,
    *,
    name,
    before_day,
    manifest=None,
    history=False,
    receipt_ids=(),
    blob_names=(),
    segment_names=(),
):
    """Process <=32 indexed receipts OR one explicitly selected atomic child.

    The caller stats/deadline-gates work. Raw input/history bodies retain their
    existing MAX_BLOB/MAX_SNAPSHOT limits; one segment may be read twice only
    when filtering writes a replacement. No unselected child body is read.
    """
    from ssgrep.accounting.purge_published import _plan_selected

    if day(before_day) is None or not name.startswith("identity-archive-"):
        raise ValueError("Invalid original publication expiry selection.")
    atoms = int(history) + len(blob_names) + len(segment_names)
    if len(receipt_ids) > 32 or atoms > 1 or receipt_ids and atoms:
        raise ValueError("Original expiry requires <=32 receipts or one other child.")
    if not receipt_ids and not atoms:
        return ChildResult(False, (), manifest)
    manifest = (
        manifest if manifest is not None else store.read_blob(profile, checkpoint.blobs[name])
    )
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise PurgeError("unsupported_schema", "Original publication manifest is unsupported.")
    original = manifest["published_documents"]
    references(original)
    documents, blobs = dict(original), dict(manifest.get("blobs", {}))
    refs = list(manifest.get("segments", []))
    updates, dropped, summaries = {}, set(), dict(original.get("retention_child_records", {}))
    indexes = dict(original.get("retention_child_dates", {}))
    raw_plan = None
    changed = False
    expired_receipts = []
    if receipt_ids:
        periods, roots = original.get("receipt_periods"), original.get("receipt_roots")
        periods = periods if isinstance(periods, dict) else {}
        roots = roots if isinstance(roots, dict) else {}
        receipts = dict(original.get("receipt_blobs", {}))
        periods = {key: value for key, value in periods.items() if key in receipts}
        roots = {key: value for key, value in roots.items() if key in receipts}
        changed = periods != original.get("receipt_periods") or roots != original.get(
            "receipt_roots"
        )
        for receipt in receipt_ids:
            if receipt not in receipts:
                continue
            if day(periods.get(receipt)) is None or not isinstance(roots.get(receipt), str):
                from ssgrep.accounting.published_documents import read_document

                _, row = read_document(profile, receipts[receipt], "receipt")
                if row.get("receipt_id") != receipt:
                    raise ValueError("Original receipt identity does not match.")
                periods[receipt], roots[receipt] = row["period"], row["root"]
                changed = True
            if periods[receipt] < before_day:
                dropped.add(receipts.pop(receipt))
                expired_receipts.append(receipt)
                periods.pop(receipt, None)
                roots.pop(receipt, None)
                changed = True
        complete = len(periods) == len(receipts) and all(
            day(value) is not None for value in periods.values()
        )
        documents.update(
            receipt_blobs=receipts,
            receipt_periods=periods,
            receipt_roots=roots,
            receipt_count=len(receipts),
            receipt_min_period=min(periods.values(), default=None) if complete else None,
        )
    elif segment_names:
        selected = set(segment_names)
        after = []
        for raw in refs:
            ref = store.SegmentRef.from_dict(raw)
            if ref.name not in selected:
                after.append(raw)
                continue
            minimum, complete, expired = None, True, False
            for item in store.read_segment(profile, ref):
                stamp = day(item.ts_utc[:10]) if isinstance(item.ts_utc, str) else None
                if stamp is None:
                    complete = False
                elif stamp < before_day:
                    expired = True
                elif minimum is None or stamp < minimum:
                    minimum = stamp
            indexes.pop("segment:" + ref.name, None)
            if not expired:
                after.append(raw)
                indexes["segment:" + ref.name] = _index(ref.name, minimum, complete, before_day)
                continue
            kept_days = []

            def kept(ref=ref, kept_days=kept_days):
                for item in store.read_segment(profile, ref):
                    stamp = day(item.ts_utc[:10]) if isinstance(item.ts_utc, str) else None
                    if stamp is None or stamp >= before_day:
                        kept_days.append(stamp)
                        yield item

            replacements = store.write_segments(profile, kept(), origin=ref.origin)
            position = 0
            for replacement in replacements:
                count = replacement.records
                part_minimum, part_complete = _minimum(kept_days[position : position + count])
                indexes["segment:" + replacement.name] = _index(
                    replacement.name, part_minimum, part_complete, before_day
                )
                position += count
            after.extend(replacement.to_dict() for replacement in replacements)
            changed = True
        refs = after
    else:
        selected = {key: value for key, value in blobs.items() if key in blob_names}
        subset_documents = {
            **original,
            "history_blob": original.get("history_blob") if history else None,
            "receipt_blobs": {},
            "roots_by_period": original.get("roots_by_period") if history else None,
        }
        subset = {
            **manifest,
            "blobs": selected,
            "segments": [],
            "session_roots": [],
            "session_cwd": [],
            "published_documents": subset_documents,
        }
        plan = _plan_selected(
            profile, subset, Selection(profile, before_day + "T00:00:00Z", None), atomic=True
        )
        raw_plan = plan
        updates.update(plan.updates)
        dropped.update(plan.dropped)
        for key, value in selected.items():
            if key.split("-scope-", 1)[0] in _CACHES:
                dropped.add(value)  # disposable, never another workspace's canonical measurements
        if history:
            documents["roots_by_period"] = plan.documents.get("roots_by_period")
            documents["history_rows"] = plan.history_remaining
        else:
            for key in selected:
                role = key.split("-scope-", 1)[0]
                summaries[key] = (
                    plan.remaining_records
                    if role.startswith("ledger-") or role == "reduction"
                    else None
                )
        changed = bool(
            updates
            or dropped
            or plan.documents != subset_documents
            or history
            and documents.get("history_rows") != original.get("history_rows")
            or summaries != original.get("retention_child_records", {})
        )
    memo = {
        old: store.write_blob(profile, prefix, document)
        for old, (prefix, document) in updates.items()
        if old not in dropped
    }
    retained = {key: memo.get(value, value) for key, value in blobs.items() if value not in dropped}
    documents["history_blob"] = memo.get(
        documents.get("history_blob"), documents.get("history_blob")
    )
    documents["retention_child_records"] = summaries
    if raw_plan is not None:
        for key in blob_names:
            old = blobs.get(key)
            source = raw_plan.selected_inputs.get(old)
            if key in retained and source is not None:
                after = updates[old][1] if old in updates else source
                minimum, complete = _raw_minimum(key.split("-scope-", 1)[0], after)
                indexes["blob:" + key] = _index(retained[key], minimum, complete, before_day)
        if history and documents.get("history_blob"):
            periods = list((documents.get("roots_by_period") or {}).keys())
            if raw_plan.history_minimum_day is not None:
                periods.append(raw_plan.history_minimum_day)
            minimum, complete = _minimum(periods)
            indexes["history"] = _index(documents["history_blob"], minimum, complete, before_day)
    allowed = {"blob:" + key for key in retained} | {"segment:" + ref["name"] for ref in refs}
    if documents.get("history_blob"):
        allowed.add("history")
    documents["retention_child_dates"] = {
        key: value for key, value in indexes.items() if key in allowed
    }
    changed = changed or documents != original
    if not changed:
        return ChildResult(False, (), manifest)
    updated_manifest = {
        **manifest,
        "blobs": retained,
        "segments": refs,
        "published_documents": documents,
    }
    checkpoint.blobs = {**checkpoint.blobs, name: store.write_blob(profile, name, updated_manifest)}
    if checkpoint.identity_migration.get("status") == "complete":
        prior = checkpoint.identity_migration.get("superseded_blobs", {})
        pointers = {key: value for key, value in prior.items() if blobs.get(key) != value}
        for key, value in retained.items():
            pointers.setdefault(key, value)
        checkpoint.identity_migration = {
            **checkpoint.identity_migration,
            "superseded_blobs": pointers,
        }
    return ChildResult(True, tuple(expired_receipts), updated_manifest)
