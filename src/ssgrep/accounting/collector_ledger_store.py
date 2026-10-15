"""Per-root persistence for the existing LedgerState, with resumable legacy conversion.

The checkpoint references one immutable manifest. A legacy base remains readable
while at most 32 unchanged roots per cycle are materialized; changed roots always
get their own canonical blob. Explicit purge writes have no legacy base, so erased
metadata cannot remain reachable through a conversion overlay.
"""

from __future__ import annotations

import time
from collections import defaultdict

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.ledger_codec import ledger_state_from_dict, ledger_state_to_dict
from ssgrep.accounting.plain_codec import from_plain, to_plain
from ssgrep.accounting.replay import ArchivedRows, LedgerState

FORMAT = "root-ledger-v1"


def frequency_refs(value):
    """Actual frozen input IDs, not text tokens or methodology identities."""
    found = set()
    if isinstance(value, dict):
        for key, row in value.items():
            if (
                key in {"view_id", "frequency_view_id"}
                and isinstance(row, str)
                and row.startswith("freq-")
            ):
                found.add(row)
            found.update(frequency_refs(row))
    elif isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and value[0] in {"frequency", "frequency_view"}
            and isinstance(value[1], str)
            and value[1].startswith("freq-")
        ):
            found.add(value[1])
        for row in value:
            found.update(frequency_refs(row))
    return found


def root_states(state):
    grouped = defaultdict(lambda: [[], [], []])
    for row in state.candidates:
        grouped[row.task.root][0].append(row)
    for row in state.debits:
        grouped[row.task.root][1].append(row)
    for row in state.frozen:
        grouped[row.root][2].append(row)
    return {
        root: LedgerState(state.methodology, tuple(rows[0]), tuple(rows[1]), tuple(rows[2]))
        for root, rows in grouped.items()
    }


def expand_document(profile, raw, *, depth=0):
    if raw.get("format") is None:
        return raw
    if raw.get("format") != FORMAT:
        raise CollectError("unsupported_schema", "Collector ledger format is unsupported.")
    if depth > 1:
        raise CollectError("state_corrupt", "Collector ledger base chain is invalid.")
    base = raw.get("base_blob")
    document = (
        expand_document(profile, store.read_blob(profile, base), depth=depth + 1)
        if base
        else {
            "state": raw["state"],
            "superseded": {},
            "archives": {},
        }
    )
    roots = set(raw["roots"])
    state = document["state"]
    candidates = [row for row in state["candidates"] if row["task"]["root"] not in roots]
    debits = [row for row in state["debits"] if row["task"]["root"] not in roots]
    frozen = [row for row in state["frozen"] if row["root"] not in roots]
    superseded = dict(document.get("superseded", {}))
    for root, entry in sorted(raw["roots"].items()):
        if entry is None:
            continue
        child = store.read_blob(profile, entry["blob"])
        if not isinstance(child, dict):
            raise CollectError("state_corrupt", "Collector root document is invalid.")
        if child["state"]["methodology"] != raw["state"]["methodology"]:
            raise CollectError("state_corrupt", "Collector root method differs from its manifest.")
        if any(
            row["task"]["root"] != root
            for kind in ("candidates", "debits")
            for row in child["state"][kind]
        ):
            raise CollectError(
                "state_corrupt", "Collector root ownership differs from its manifest."
            )
        candidates.extend(child["state"]["candidates"])
        debits.extend(child["state"]["debits"])
        frozen.extend(child["state"]["frozen"])
        superseded.update(child.get("superseded", {}))
    archives = {
        name: store.read_blob(profile, filename)
        for name, filename in raw.get("archive_blobs", {}).items()
    }
    return {
        "state": {
            "methodology": raw["state"]["methodology"],
            "candidates": sorted(candidates, key=digest_of),
            "debits": sorted(debits, key=digest_of),
            "frozen": sorted(frozen, key=lambda row: row["root"]),
        },
        "superseded": superseded,
        "archives": archives,
        "collection_scope": raw.get("collection_scope", {}),
    }


def save_incremental(
    profile,
    name,
    state,
    blobs,
    *,
    previous=None,
    clock=time.monotonic,
    deadline=float("inf"),
    migration_limit=32,
    dirty_roots=None,
):
    before = store.read_blob(profile, blobs.get(name))
    current = root_states(state.ledger)
    old = root_states(previous.ledger) if previous and previous.ledger else {}
    if before and before.get("format") == FORMAT and previous is not None:
        roots = dict(before["roots"])
        base = before.get("base_blob")
        base_roots = set(before.get("base_roots", ()))
        old_archives = before.get("archive_blobs", {})
    else:
        roots, old_archives = {}, {}
        base = blobs.get(name) if before and previous is not None else None
        base_roots = set(old) if base else set()
    changed = {root for root in current.keys() | old.keys() if current.get(root) != old.get(root)}
    if previous is None:
        changed = set(current)
    if previous and state.superseded != previous.superseded:
        changed.update(dirty_roots if dirty_roots is not None else current.keys() | old.keys())
    unresolved = sorted(base_roots - roots.keys() - changed)
    scheduled = []
    for root in unresolved[:migration_limit]:
        if clock() >= deadline:
            break
        scheduled.append(root)
    for root in sorted(changed | set(scheduled)):
        local = current.get(root)
        if local is None:
            roots[root] = None
            continue
        identities = {digest_of(row) for row in (*local.candidates, *local.debits)}
        document = {
            "state": ledger_state_to_dict(local),
            "superseded": {
                key: value for key, value in state.superseded.items() if key in identities
            },
        }
        refs = sorted(frequency_refs(document))
        roots[root] = {
            "blob": store.write_blob(profile, "ledger-root", document),
            "frequency_refs": refs,
        }
    remaining = len(base_roots - roots.keys())
    if not remaining:
        base, base_roots = None, set()
        roots = {
            root: entry for root, entry in roots.items() if entry is not None and root in current
        }
    archives = {}
    archive_refs = set()
    for key, archive in sorted(state.archives.items()):
        if previous and previous.archives.get(key) == archive and key in old_archives:
            archives[key] = old_archives[key]
        else:
            archives[key] = store.write_blob(profile, "ledger-archive-rows", to_plain(archive))
        archive_refs.update(frequency_refs(to_plain(archive)))
    refs = archive_refs | {
        view for entry in roots.values() if entry for view in entry["frequency_refs"]
    }
    if base:
        if not isinstance(before, dict):
            raise CollectError("state_corrupt", "Collector ledger base document is invalid.")
        refs.update(
            before.get("frequency_refs", ())
            if before.get("format") == FORMAT
            else frequency_refs(before)
        )
    raw = {
        "format": FORMAT,
        "state": ledger_state_to_dict(LedgerState(state.ledger.methodology)),
        "roots": roots,
        "base_blob": base,
        "base_roots": sorted(base_roots),
        "collection_scope": state.collection_scope,
        "archive_blobs": archives,
        "frequency_refs": sorted(refs),
        "migration_pending_roots": remaining,
    }
    blobs[name] = store.write_blob(profile, name, raw)
    return remaining


def references(raw):
    if raw.get("format") != FORMAT:
        return set()
    return (
        ({raw["base_blob"]} if raw.get("base_blob") else set())
        | {entry["blob"] for entry in raw["roots"].values() if entry is not None}
        | set(raw.get("archive_blobs", {}).values())
    )


def write_document(profile, name, document):
    """Complete logical-document writer used by explicit privacy purge."""
    from ssgrep.accounting.collector_state import State

    state = State(
        ledger=ledger_state_from_dict(document["state"]),
        superseded=dict(document.get("superseded", {})),
        collection_scope=document.get("collection_scope", {}),
        archives={
            key: from_plain(ArchivedRows, row) for key, row in document.get("archives", {}).items()
        },
    )
    blobs = {}
    save_incremental(profile, name, state, blobs)
    return blobs[name]
