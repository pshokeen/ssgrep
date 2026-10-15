"""Explicit resumable compact-evidence indexing, retention and segment compaction.

The quantum is one existing bounded segment, never an unbounded corpus rewrite.
Logical retention floors commit independently of physical cleanup. Pending segment
names are durable backlog; a cycle with backlog cannot claim complete coverage.
"""

from __future__ import annotations

import os
from dataclasses import replace

from ssgrep.accounting import collector_store as store, storage
from ssgrep.accounting.collector_segment_index import IndexWriter, read_index, session_name
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import digest_of


def index_segment(profile, ref):
    directory = storage.open_profile_dir(profile, "collector", store.EVIDENCE_DIR, create=False)
    try:
        body = storage.read_file(directory, ref.name, ref.size + 1)
    except ValueError as error:
        raise CollectError(
            "state_corrupt", "Legacy retained segment is missing or damaged."
        ) from error
    finally:
        os.close(directory)
    if body is None or len(body) != ref.size:
        raise CollectError("state_corrupt", "Legacy retained segment is missing or damaged.")
    index = IndexWriter()
    offset = 0
    for line in body.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise CollectError("state_corrupt", "Legacy retained segment is incomplete.")
        index.add(store.evidence_from_line(line.rstrip(b"\n")), offset, line)
        offset += len(line)
    return replace(ref, index=index.write(profile))


def ensure_indexes(profile, checkpoint, *, clock, deadline, byte_budget, should_stop):
    """Run named one-time legacy index work before admitting new native evidence."""
    used = 0
    refs = []
    for ref in checkpoint.segments:
        if ref.index is None and used < byte_budget and not should_stop() and clock() < deadline:
            refs.append(index_segment(profile, ref))
            used += ref.size
        else:
            refs.append(ref)
    checkpoint.segments = refs
    pending = [ref.name for ref in refs if ref.index is None]
    if pending:
        checkpoint.retention["evidence_index_pending"] = pending
    else:
        checkpoint.retention.pop("evidence_index_pending", None)
    return used, bool(pending)


def compact_pair(refs):
    """A count target cannot make already-full segments require impossible work."""
    if len(refs) <= store.MAX_SEGMENTS:
        return None
    for origin in sorted({ref.origin for ref in refs}):
        candidates = sorted(
            (ref for ref in refs if ref.origin == origin), key=lambda ref: (ref.size, ref.name)
        )
        if (
            len(candidates) >= 2
            and candidates[0].size + candidates[1].size <= store.SEGMENT_TARGET_BYTES
        ):
            return candidates[0], candidates[1]
    return None


def maintain_evidence(
    profile, refs, *, expired_sessions, previous, clock, deadline, byte_budget, should_stop
):
    """Return immutable replacement refs + explicit remaining maintenance job."""
    job: dict[str, object] = dict(previous or {})
    expiry_digest = digest_of(sorted(expired_sessions))
    if job.get("expired_digest") != expiry_digest:
        job = {
            "expired_digest": expiry_digest,
            "pending": [ref.name for ref in refs] if expired_sessions else [],
        }
    pending = job.get("pending", [])
    if not isinstance(pending, list):
        raise CollectError("state_corrupt", "Collector evidence maintenance queue is invalid.")
    pending = list(pending)
    refs = list(refs)
    used = dropped = 0
    while pending and used < byte_budget and not should_stop() and clock() < deadline:
        name = pending.pop(0)
        ref = next((ref for ref in refs if ref.name == name), None)
        if ref is None:
            continue
        index = read_index(profile, ref)
        wanted = {session_name(key) for key in expired_sessions}
        if index is not None and not wanted.intersection(index["sessions"]):
            continue
        items = list(store.read_segment(profile, ref))
        kept = [item for item in items if (item.runtime, item.session_id) not in expired_sessions]
        replacements = store.write_segments(profile, kept, origin=ref.origin) if kept else []
        refs = [current for current in refs if current.name != name] + replacements
        used += ref.size
        dropped += len(items) - len(kept)
    # Compaction is incremental too. Preserve live/backfill segment origins.
    pair = compact_pair(refs)
    while (
        not pending
        and pair is not None
        and used < byte_budget
        and not should_stop()
        and clock() < deadline
    ):
        first, second = pair
        size = first.size + second.size
        if used and used + size > byte_budget:
            break
        replacements = store.write_segments(
            profile, store.iter_evidence(profile, (first, second)), origin=first.origin
        )
        if len(replacements) >= 2:
            # Both input segments are already full; no productive bounded merge.
            break
        refs = [ref for ref in refs if ref.name not in {first.name, second.name}] + replacements
        used += size
        pair = compact_pair(refs)
    job["pending"] = pending
    job["compaction_pending"] = compact_pair(refs) is not None
    return refs, job, used, dropped
