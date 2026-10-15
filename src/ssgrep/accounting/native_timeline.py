"""Canonical copy representatives and logical result ownership, without guessing roots."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import replace

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.identity_claims import CLAIM, claimed_owner, target_key
from ssgrep.accounting.readers.streaming import is_stream


def _offset(item: RawEvidence) -> int:
    try:
        return int(item.locator.split(":", 1)[1]) if item.locator.startswith("offset:") else -1
    except ValueError:
        return -1


def copy_order(item: RawEvidence) -> tuple:
    return (
        item.source_id,
        _offset(item),
        item.ts_utc or "",
        item.record_id,
        item.session_id,
        item.branch_id,
    )


def canonical_copies(best: dict, items: list[RawEvidence]) -> tuple[dict, dict]:
    """A persisted claimant can improve linkage or select the same owner as full replay."""
    links: dict[tuple, set[str]] = defaultdict(set)
    streams: dict[tuple, list[RawEvidence]] = defaultdict(list)
    for item in items:
        key = target_key(item) if item.kind == CLAIM else item.native_key
        if key is None or key not in best:
            continue
        link = item.payload.get("call_record_id")
        if isinstance(link, str) and item.payload.get("call_link_status") not in {
            "ambiguous",
            "missing_context",
        }:
            links[key].add(link)
        if is_stream(item):
            streams[key].append(item)
    stream_conflicts: dict[tuple, set[str]] = {}
    if streams:
        from ssgrep.accounting.readers.base import EvidenceLedger

        for key, updates in streams.items():
            ledger = EvidenceLedger()
            list(ledger.drain(sorted(updates, key=copy_order)))
            final = list(ledger.final_usage())
            if final:
                best[key] = final[0][0]
            if ledger.conflict_count:
                stream_conflicts[key] = {item.digest for item in updates}
    for claim in items:
        if claim.kind != CLAIM:
            continue
        key = target_key(claim)
        held = best.get(key)
        if (
            held is None
            or claim.payload.get("target_digest") != held.digest
            or claim.payload.get("origin_conflict") is True
        ):
            continue
        source, session, branch, cwd, parent = claimed_owner(claim)
        record_id = claim.payload.get("target_record_id")
        if not isinstance(record_id, str):
            continue
        payload = {**held.payload, "cwd": cwd}
        if parent is not None or "parent_session" in payload:
            payload["parent_session"] = parent
        candidate = replace(
            held,
            source_id=source,
            session_id=session,
            branch_id=branch,
            record_id=record_id,
            locator=claim.locator,
            ts_utc=claim.ts_utc,
            payload=payload,
        )
        if copy_order(candidate) < copy_order(held):
            best[key] = candidate
    for key, item in list(best.items()):
        if item.kind != "tool_result":
            continue
        proof = links.get(key, set())
        if len(proof) == 1:
            best[key] = replace(
                item,
                payload={
                    **item.payload,
                    "call_record_id": next(iter(proof)),
                    "call_link_status": "unique_native_copy",
                },
            )
        elif len(proof) > 1:
            best[key] = replace(
                item,
                payload={**item.payload, "call_record_id": None, "call_link_status": "ambiguous"},
            )
    return links, stream_conflicts


def logical_membership(best: dict) -> tuple[dict, dict]:
    """Results follow a uniquely proven call; RawEvidence containing fields stay unchanged."""
    membership = {key: (item.runtime, item.session_id) for key, item in best.items()}
    calls: dict[tuple, list[tuple]] = defaultdict(list)
    for key, item in best.items():
        if item.kind == "tool_call":
            calls[(item.runtime, item.record_id)].append(key)
    paired: dict[tuple, tuple] = {}
    for key, item in best.items():
        if item.kind != "tool_result":
            continue
        linked = item.payload.get("call_record_id")
        if not isinstance(linked, str):
            continue
        candidates = calls.get((item.runtime, linked), [])
        if item.origin_status not in {"confirmed_uuid", "provider_message_id"}:
            candidates = [c for c in candidates if c[1] == key[1]]
        if len(candidates) == 1:
            paired[key] = candidates[0]
            membership[key] = membership[candidates[0]]
    return membership, paired


def ordered(
    items: list[RawEvidence], observations: list[RawEvidence], paired: dict, order
) -> tuple[list[RawEvidence], set[str]]:
    """Stable source-offset dependencies plus proven call→result, not timestamps alone."""
    base = sorted(items, key=order)
    index = {item.native_key: n for n, item in enumerate(base)}
    edges: dict[tuple, set[tuple]] = defaultdict(set)
    incoming = {key: 0 for key in index}
    source_rows: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
    for item in observations:
        key = target_key(item) if item.kind == CLAIM else item.native_key
        offset = _offset(item)
        if key in index and offset >= 0:
            source_rows[item.source_id].append((offset, key))

    def edge(before, after):
        if before != after and before in index and after in index and after not in edges[before]:
            edges[before].add(after)
            incoming[after] += 1

    for rows in source_rows.values():
        sequence = sorted(set(rows))
        for (_, before), (_, after) in zip(sequence, sequence[1:], strict=False):
            edge(before, after)
    for result, call in paired.items():
        edge(call, result)
    ready = [(index[key], key) for key, n in incoming.items() if n == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        position, key = heapq.heappop(ready)
        result.append(base[position])
        for following in edges[key]:
            incoming[following] -= 1
            if incoming[following] == 0:
                heapq.heappush(ready, (index[following], following))
    remaining = {key for key, n in incoming.items() if n > 0}
    result.extend(base[index[key]] for key in sorted(remaining, key=index.__getitem__))
    return result, {best.native_identity for best in base if best.native_key in remaining}
