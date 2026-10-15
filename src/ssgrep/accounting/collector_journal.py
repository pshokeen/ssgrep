"""Per-segment journal parse reuse under the collector's existing commit point.

Unchanged descriptors are never read. Append-only growth resumes at the last
complete offset (a pending tail is reparsed); replacement/truncation reparses only
that segment. The retained cache is scrubbed by the same committed privacy floor
as physical maintenance, and remembers proof-only pending cleanup after scrubbing.
Missing accepted segments require exact committed maintenance authority and all
retained measurements in surviving inputs; otherwise collection fails closed.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import ExecutionEvent, digest_of
from ssgrep.accounting.plain_codec import from_plain, to_plain

BLOB = "journal-cache"


def reuse_segment(result, proof, cached) -> int | None:
    """Return suffix offset, None for unchanged, or zero for a replacement."""
    from ssgrep.accounting.journal_reader import Damage, SegmentProof

    if cached is None:
        return 0
    try:
        before = from_plain(SegmentProof, cached["proof"])
        events = [from_plain(ExecutionEvent, row) for row in cached["events"]]
        damage = [from_plain(Damage, row) for row in cached["damage"]]
    except (KeyError, ValueError, TypeError) as error:
        raise CollectError("state_corrupt", "Cached journal segment is invalid.") from error
    equal = before == proof
    append = (before.device, before.inode) == (
        proof.device,
        proof.inode,
    ) and proof.size > before.size
    if not equal and not append:
        return 0
    incomplete = [row.offset for row in damage if row.kind == "incomplete_tail"]
    offset = min(incomplete, default=before.size)
    result.events.extend(events)
    result.segment_events[proof.name] = events
    result.damage.extend(
        damage if equal else [row for row in damage if row.kind != "incomplete_tail"]
    )
    if cached.get("retention_pending"):
        result.retention_pending.add(proof.name)
    return None if equal else offset


def _require_cached_segments(journal, previous: dict[str, dict], retention: dict | None) -> None:
    """Distinguish proven retention from loss of an accepted journal input."""
    from ssgrep.accounting.journal_reader import Damage, SegmentProof
    from ssgrep.accounting.journal_retention import retained_event

    missing = previous.keys() - {proof.name for proof in journal.proofs}
    if not missing:
        return
    available: set[str] | None = None
    for name in sorted(missing):
        cached = previous[name]
        try:
            proof = from_plain(SegmentProof, cached["proof"])
            events = [from_plain(ExecutionEvent, row) for row in cached["events"]]
            damage = [from_plain(Damage, row) for row in cached["damage"]]
            authorized = retention is not None and proof in [
                from_plain(SegmentProof, row) for row in retention["proofs"]
            ]
        except (KeyError, ValueError, TypeError) as error:
            raise CollectError("state_corrupt", "Cached journal segment is invalid.") from error
        if retention is None or not authorized or proof.name != name:
            raise CollectError(
                "state_corrupt",
                "A committed journal segment is missing without retention authority; "
                "restore the accepted journal before collecting.",
            )
        modified_day = (
            datetime.fromtimestamp(proof.modified_ns // 1_000_000_000, UTC).date().isoformat()
        )
        if damage and modified_day >= retention["horizon_day"]:
            raise CollectError(
                "state_corrupt", "A committed journal segment with a pending record is missing."
            )
        expected = [
            kept for event in events if (kept := retained_event(event, retention)) is not None
        ]
        if not expected:
            continue
        if available is None:
            available = {
                digest_of(kept)
                for rows in journal.segment_events.values()
                for event in rows
                if (kept := retained_event(event, retention)) is not None
            }
        if any(digest_of(event) not in available for event in expected):
            raise CollectError(
                "state_corrupt",
                "A committed journal segment is missing its retained measurements; "
                "restore the accepted journal or its retained replacement before collecting.",
            )


def read_cached_journal(profile: str, checkpoint: store.Checkpoint | None):
    from ssgrep.accounting.journal_reader import read_journal
    from ssgrep.accounting.journal_retention import expand_journal_retention

    raw = store.read_blob(profile, checkpoint.blobs.get(BLOB)) if checkpoint else None
    if raw is not None and (not isinstance(raw, dict) or raw.get("schema") != 1):
        raise CollectError("state_corrupt", "Journal cache manifest is invalid.")
    names = raw.get("segments", {}) if raw else {}
    previous = {}
    for name, filename in names.items():
        segment = store.read_blob(profile, filename)
        if not isinstance(segment, dict):
            raise CollectError("state_corrupt", "Cached journal segment is invalid.")
        previous[name] = segment
    retention = checkpoint.retention.get("journal") if checkpoint else None
    expanded = expand_journal_retention(profile, retention) if retention else None
    journal = read_journal(profile, previous=previous, retention=expanded)
    _require_cached_segments(journal, previous, expanded)
    journal.cache_blobs = {
        proof.name: names[proof.name] for proof in journal.proofs if proof.name in names
    }
    journal.cache_retention_tag = raw.get("retention_tag") if raw else None
    # Exact descriptor metadata detects same-size rewrites; journal writers only
    # append. Signature creation performs no native-content/retained-event reads.
    journal.signature = digest_of(
        {"proofs": [asdict(proof) for proof in journal.proofs], "retention": retention}
    )[:32]
    return journal


def save_journal_cache(
    profile: str,
    journal,
    blobs: dict[str, str],
    record: dict | None,
    *,
    retention_tag: str | None = None,
) -> None:
    from ssgrep.accounting.journal_retention import retained_event

    files = {}
    damage = {}
    for row in journal.damage:
        damage.setdefault(row.segment, []).append(to_plain(row))
    for proof in journal.proofs:
        name = proof.name
        if (
            retention_tag is not None
            and retention_tag == journal.cache_retention_tag
            and name not in journal.cache_dirty
            and name in journal.cache_blobs
        ):
            files[name] = journal.cache_blobs[name]
            continue
        original = journal.segment_events.get(name, ())
        retained = [
            retained_event(event, record) if record is not None else event for event in original
        ]
        scrubbed = any(old != new for old, new in zip(original, retained, strict=True))
        pending = name in journal.retention_pending or scrubbed
        if not scrubbed and name not in journal.cache_dirty and name in journal.cache_blobs:
            files[name] = journal.cache_blobs[name]
            continue
        files[name] = store.write_blob(
            profile,
            "journal-segment",
            {
                "proof": to_plain(proof),
                "events": [to_plain(row) for row in retained if row is not None],
                "damage": damage.get(name, []),
                "retention_pending": pending,
            },
        )
    blobs[BLOB] = store.write_blob(
        profile, BLOB, {"schema": 1, "segments": files, "retention_tag": retention_tag}
    )


def references(profile: str, filename: str) -> set[str]:
    raw = store.read_blob(profile, filename)
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != 1
        or not isinstance(raw.get("segments"), dict)
    ):
        raise CollectError("state_corrupt", "Journal cache manifest is invalid.")
    return set(raw["segments"].values())
