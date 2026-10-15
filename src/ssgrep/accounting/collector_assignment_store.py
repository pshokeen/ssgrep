"""Freeze per-lookup inputs without reserializing every ID on one new lookup."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.plain_codec import to_plain

FORMAT = "frequency-assignments-v2"


def expand_document(profile, raw):
    if raw.get("format") is None:
        return raw
    if raw.get("format") != FORMAT:
        raise CollectError("unsupported_schema", "Collector assignment format is unsupported.")
    rows = []
    for filename in raw["shards"].values():
        shard = store.read_blob(profile, filename)
        if not isinstance(shard, list):
            raise CollectError("state_corrupt", "Collector assignment shard is invalid.")
        rows.extend(shard)
    return {
        "schema": 1,
        "assignments": sorted(rows, key=lambda row: row["invocation_id"]),
        "unavailable_probe": raw.get("unavailable_probe"),
    }


def save_incremental(profile, name, state, blobs, *, previous=None):
    def groups(rows):
        result = defaultdict(list)
        for row in rows:
            shard = hashlib.sha256(row.invocation_id.encode()).hexdigest()[:2]
            result[shard].append(row)
        return result

    current = groups(state.frequency_assignments)
    old = groups(previous.frequency_assignments) if previous else {}
    before = store.read_blob(profile, blobs.get(name))
    files = before.get("shards", {}) if before and before.get("format") == FORMAT else {}
    shards = {}
    for shard, rows in sorted(current.items()):
        if shard in files and rows == old.get(shard):
            shards[shard] = files[shard]
        else:
            shards[shard] = store.write_blob(profile, "assignment-shard", to_plain(rows))
    blobs[name] = store.write_blob(
        profile,
        name,
        {
            "schema": 1,
            "format": FORMAT,
            "shards": shards,
            "unavailable_probe": to_plain(state.frequency_probe) if state.frequency_probe else None,
            "frequency_refs": sorted({row.view_id for row in state.frequency_assignments}),
        },
    )


def references(raw):
    return set(raw.get("shards", {}).values()) if raw.get("format") == FORMAT else set()
