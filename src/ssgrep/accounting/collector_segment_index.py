"""Immutable compact-evidence indexes: session ranges and native-identity shards.

Only blob handles enter the checkpoint. Identity tables live in 256 independently
addressed shards, so admission never copies all retained IDs into a new checkpoint.
Each selected line is checked against its indexed SHA256 before decoding.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from ssgrep.accounting import collector_store as store, storage
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.identity_claims import target_key

Session = tuple[str, str]
NativeKey = tuple[str, str, str]


def session_name(key: Session) -> str:
    return store.dumps(list(key)).decode()


def identity_hash(key: NativeKey) -> str:
    return hashlib.sha256(store.dumps(list(key))).hexdigest()


@dataclass
class IndexWriter:
    sessions: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    identities: dict[str, dict[str, list]] = field(default_factory=lambda: defaultdict(dict))

    def add(self, item: RawEvidence, offset: int, line: bytes) -> None:
        location = [offset, len(line), hashlib.sha256(line).hexdigest()]
        self.sessions[session_name((item.runtime, item.session_id))].append(location)
        # Claims are fetched with their target, not as separate usage identities.
        key = target_key(item) or item.native_key
        digest = identity_hash(key)
        self.identities[digest[:2]].setdefault(digest, []).append(location)

    def write(self, profile: str) -> str:
        sessions = {
            key: store.write_blob(profile, "segment-session", rows)
            for key, rows in sorted(self.sessions.items())
        }
        identities = {
            key: store.write_blob(profile, "segment-identities", rows)
            for key, rows in sorted(self.identities.items())
        }
        return store.write_blob(
            profile,
            "segment-index",
            {
                "schema": 1,
                "sessions": sessions,
                "identities": identities,
            },
        )


def cached_blob(profile: str, filename: str | None, cache: dict | None):
    if cache is None or filename is None:
        return store.read_blob(profile, filename)
    if filename not in cache:
        cache[filename] = store.read_blob(profile, filename)
    return cache[filename]


def read_index(profile: str, ref: store.SegmentRef, cache: dict | None = None) -> dict | None:
    if ref.index is None:
        return None
    raw = cached_blob(profile, ref.index, cache)
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        raise CollectError("state_corrupt", "Retained evidence index is invalid.")
    if not isinstance(raw.get("sessions"), dict) or not isinstance(raw.get("identities"), dict):
        raise CollectError("state_corrupt", "Retained evidence index tables are invalid.")
    return raw


def iter_selected(
    profile: str,
    ref: store.SegmentRef,
    *,
    sessions: set[Session] | None = None,
    native_keys: set[NativeKey] | None = None,
    index_cache: dict | None = None,
) -> Iterator[RawEvidence]:
    index = read_index(profile, ref, index_cache)
    if index is None:
        # Legacy conversion is explicitly queued by the partition collector; this
        # compatibility read never hydrates native content and cannot claim freshness.
        for item in store.read_segment(profile, ref):
            if sessions is not None and (item.runtime, item.session_id) not in sessions:
                continue
            if native_keys is not None and (target_key(item) or item.native_key) not in native_keys:
                continue
            yield item
        return
    rows: dict[int, list] = {}
    if sessions is not None:
        for key in sorted(sessions):
            filename = index["sessions"].get(session_name(key))
            for row in cached_blob(profile, filename, index_cache) or ():
                rows[row[0]] = row
    elif native_keys is not None:
        shards: dict[str, set[str]] = defaultdict(set)
        for key in native_keys:
            digest = identity_hash(key)
            shards[digest[:2]].add(digest)
        for shard, digests in sorted(shards.items()):
            table = cached_blob(profile, index["identities"].get(shard), index_cache) or {}
            for digest in sorted(digests):
                for row in table.get(digest, ()):
                    rows[row[0]] = row
    else:
        yield from store.read_segment(profile, ref)
        return
    if not rows:
        return
    directory = storage.open_profile_dir(profile, "collector", store.EVIDENCE_DIR, create=False)
    try:
        fd = storage.open_regular(directory, ref.name)
    except OSError as error:
        raise CollectError("state_corrupt", "Indexed retained evidence is missing.") from error
    finally:
        os.close(directory)
    try:
        if os.fstat(fd).st_size != ref.size:
            raise CollectError("state_corrupt", "Indexed retained evidence size changed.")
        for offset, length, digest in (rows[key] for key in sorted(rows)):
            if offset < 0 or length <= 0 or offset + length > ref.size:
                raise CollectError("state_corrupt", "Retained evidence index range is invalid.")
            line = os.pread(fd, length, offset)
            if len(line) != length or hashlib.sha256(line).hexdigest() != digest:
                raise CollectError("state_corrupt", "Indexed retained evidence digest changed.")
            try:
                item = store.evidence_from_line(line.rstrip(b"\n"))
            except (ValueError, KeyError, TypeError) as error:
                raise CollectError(
                    "state_corrupt", "Indexed retained evidence is invalid."
                ) from error
            if sessions is not None and (item.runtime, item.session_id) not in sessions:
                raise CollectError("state_corrupt", "Retained evidence session index disagrees.")
            if native_keys is not None and (target_key(item) or item.native_key) not in native_keys:
                raise CollectError("state_corrupt", "Retained evidence identity index disagrees.")
            yield item
    finally:
        os.close(fd)


def index_references(profile: str, refs: Iterable[store.SegmentRef]) -> set[str]:
    keep: set[str] = set()
    for ref in refs:
        index = read_index(profile, ref)
        if index is not None and ref.index is not None:
            keep.add(ref.index)
            keep.update(index["sessions"].values())
            keep.update(index["identities"].values())
    return keep
