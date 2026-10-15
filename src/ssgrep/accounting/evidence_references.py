"""Explicit manifest reachability for crash-safe collector orphan collection."""

from __future__ import annotations

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CollectError

ARCHIVE_PREFIX = "identity-archive-"


def referenced_files(profile: str, checkpoint: store.Checkpoint) -> tuple[set[str], set[str]]:
    segments = {ref.name for ref in checkpoint.segments}
    refs = list(checkpoint.segments)
    blobs = set(checkpoint.blobs.values())
    pending = dict(checkpoint.identity_migration)
    from ssgrep.accounting.published_documents import references as publication_references

    capture = pending.get("published_capture")
    if capture is not None:
        if not isinstance(capture, str):
            raise CollectError(
                "state_corrupt", "Original publication capture reference is invalid."
            )
        blobs.add(capture)
        blobs.update(publication_references(store.read_blob(profile, capture)))
    blobs.update(str(name) for name in pending.get("superseded_blobs", {}).values())
    # A logical role is not a reference identity: active and superseded reductions
    # (and ledgers/maintenance inputs) may use the same key but different children.
    # Keep every ancestry entry; a role-keyed dict silently discarded active roots.
    nested = list(checkpoint.blobs.items())
    nested.extend(pending.get("superseded_blobs", {}).items())
    for name, filename in list(nested):
        if name.split("-scope-", 1)[0] == "frequency-maintenance":
            job = store.read_blob(profile, filename)
            if not isinstance(job, dict) or job.get("schema") != 1:
                raise CollectError("state_corrupt", "Frequency maintenance manifest is invalid.")
            for kind, child in job.get("sources", ()):
                blobs.add(child)
                nested.append((f"frequency-maintenance-ref:{kind}", child))
    for field in ("staged_segments", "compact_legacy_segments"):
        for raw in pending.get(field, []):
            ref = store.SegmentRef.from_dict(raw)
            segments.add(ref.name)
            refs.append(ref)
    manifests = [
        filename for name, filename in nested if name.rsplit(":", 1)[-1].startswith(ARCHIVE_PREFIX)
    ]
    seen = set()
    while manifests:
        filename = manifests.pop()
        if filename in seen:
            continue
        seen.add(filename)
        if len(seen) > 32:
            raise CollectError("state_corrupt", "Too many identity archive manifests.")
        raw = store.read_blob(profile, filename)
        if not isinstance(raw, dict) or raw.get("schema") != 1:
            raise CollectError("state_corrupt", "Identity archive manifest is invalid.")
        documents = raw.get("published_documents")
        if documents is not None:
            blobs.update(publication_references(documents))
        for segment in raw.get("segments", []):
            ref = store.SegmentRef.from_dict(segment)
            segments.add(ref.name)
            refs.append(ref)
        for name, child in raw.get("blobs", {}).items():
            blobs.add(str(child))
            nested.append((f"{filename}:{name}", str(child)))
            if name.startswith(ARCHIVE_PREFIX):
                manifests.append(str(child))
    from ssgrep.accounting.collector_assignment_store import references as assignment_references
    from ssgrep.accounting.collector_journal import references as journal_references
    from ssgrep.accounting.collector_ledger_store import references as ledger_references
    from ssgrep.accounting.collector_partitions import references as partition_references
    from ssgrep.accounting.collector_reduction_store import references as reduction_references
    from ssgrep.accounting.collector_root_cache import references as root_references
    from ssgrep.accounting.collector_segment_index import index_references

    blobs.update(index_references(profile, refs))
    for name, filename in nested:
        kind = name.rsplit(":", 1)[-1].split("-scope-", 1)[0]
        if kind == "partitions":
            blobs.update(partition_references(profile, filename))
        elif kind == "journal-cache":
            blobs.update(journal_references(profile, filename))
        elif kind == "root-evaluations":
            blobs.update(root_references(profile, filename))
        elif kind.startswith("ledger-"):
            blobs.update(ledger_references(store.read_blob(profile, filename)))
        elif kind == "reduction":
            blobs.update(reduction_references(store.read_blob(profile, filename)))
        elif kind == "frequency-assignments":
            blobs.update(assignment_references(store.read_blob(profile, filename)))
    return segments, blobs
