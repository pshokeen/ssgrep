"""native_identity_v2: explicit native domains, never output-text equivalence.

Domain                 Copy equivalence                  Non-equivalence / uncertainty
Claude record UUID     RFC UUID v4/v7 + semantic sub-ID   New UUID; local tool ID alone
Claude provider usage  API-shaped message ID + native UUID  Local/short message ID stays scoped
Pi / omp / Prime       native session + entry + sub-ID    Entry IDs are NOT globally unique
Missing session        source path + native entry         Cross-path copy unconfirmed
Missing entry          source path + byte offset          Rewrites/copies unconfirmed
Codex / OpenCode       existing session + native record   No broader domain asserted here

A same-domain same-ID changed digest is a conflict, not another event. Source,
branch, cwd and parent ownership remain on RawEvidence, not in the origin key.
UUID shape verifies a supported namespace, not authenticity or permission.
"""

from __future__ import annotations

# Native identity hash helpers.
import hashlib
import heapq
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import RFC_4122, UUID

from ssgrep.accounting.ledger.contracts import (
    MAX_RAW_RECORD_BYTES,
    PERSONAL_PROFILE,
    RawEvidence,
    digest_of,
    target_key,
)
from ssgrep.accounting.readers.base import evidence_digest, loads

IDENTITY_METHOD = "native_identity_v2"
CONTEXT_BYTES = 64 * 1024
MAX_PENDING_CALLS = 256


def native_uuid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = UUID(value)
    except ValueError:
        return None
    if parsed.variant != RFC_4122 or parsed.version not in {4, 7}:
        return None
    return str(parsed)


def with_origin(item: RawEvidence, domain: str, origin: str, status: str) -> RawEvidence:
    # These are reconstructed linkage/ownership, not native content variants.
    skip = {
        "call_record_id",
        "call_link_status",
        "occurrence",
        "originating_turn_record_id",
        "task_context_basis",
        "baseline_prior_context_unavailable",
    }
    if status in {"confirmed_uuid", "provider_message_id"}:
        skip.update({"parent_session", "cwd"})
    content = {k: v for k, v in item.payload.items() if k not in skip}
    return replace(
        item,
        origin_domain=domain,
        origin_id=origin,
        origin_status=status,
        digest=evidence_digest(item.kind, content),
    )


def remember_call(ctx: dict, call_id: str, record_id: str, entry_id: str) -> None:
    table = ctx.setdefault("pending_calls", {})
    pending = table.setdefault(call_id, [])
    pair = [record_id, entry_id]
    if pair not in pending:
        pending.append(pair)
    if sum(len(v) for v in table.values()) > MAX_PENDING_CALLS:
        # Losing context must not turn a formerly ambiguous link into a unique one.
        table.clear()
        ctx["call_context_incomplete"] = True


def link_result(ctx: dict, call_id: str, parent_id: object = None) -> tuple[str | None, str]:
    table = ctx.setdefault("pending_calls", {})
    pending = table.get(call_id, [])
    linked = [p for p in pending if p[1] == parent_id] if isinstance(parent_id, str) else []
    candidates = linked or pending
    if len(candidates) == 1 and (linked or not ctx.get("call_context_incomplete")):
        chosen = candidates[0]
        pending.remove(chosen)
        if not pending:
            table.pop(call_id, None)
        return chosen[0], "native_parent" if linked else "unique_pending"
    return None, "ambiguous" if len(candidates) > 1 else "missing_context"


MAX_ENTRIES = 256


def observe(obj: dict, ctx: dict[str, Any], locator: str) -> None:
    entry = obj.get("id")
    entry = entry if isinstance(entry, str) and entry else locator
    parent = obj.get("parentId")
    parent = parent if isinstance(parent, str) and parent else None
    message = obj.get("message")
    user = isinstance(message, dict) and message.get("role") == "user"
    table = ctx.setdefault("entry_turns", {})
    if user:
        turn = f"turn:{entry}"
        basis = "native_user_turn"
    elif parent is not None:
        turn = table.get(parent)
        basis = "native_parent_chain" if turn is not None else "native_parent_context_missing"
    else:
        turn = ctx.get("turn_id")
        basis = "bounded_source_order" if turn is not None else "native_turn_context_missing"
    table[entry] = turn
    if len(table) > MAX_ENTRIES:
        table.pop(next(iter(table)))
    ctx["turn_id"], ctx["turn_basis"] = turn, basis


def fields(ctx: dict[str, Any]) -> dict:
    return {
        "originating_turn_record_id": ctx.get("turn_id"),
        "task_context_basis": ctx.get("turn_basis", "native_turn_context_missing"),
        "baseline_prior_context_unavailable": bool(ctx.get("baseline_context")),
    }


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


CLAIM = "ownership_claim"


def owner(item: RawEvidence) -> tuple[str, str, str, str | None, str | None]:
    cwd, parent = item.payload.get("cwd"), item.payload.get("parent_session")
    return (
        item.source_id,
        item.session_id,
        item.branch_id,
        cwd if isinstance(cwd, str) else None,
        parent if isinstance(parent, str) else None,
    )


def claimed_owner(item: RawEvidence) -> tuple[str, str, str, str | None, str | None]:
    p = item.payload
    return (
        p.get("claimant_source_id", item.source_id),
        p.get("claimant_session_id", item.session_id),
        p.get("claimant_branch_id", item.branch_id),
        p.get("claimant_cwd"),
        p.get("claimant_parent_session"),
    )


class OwnershipClaims:
    def __init__(self) -> None:
        self.first: dict[bytes, str] = {}
        self.extra: dict[bytes, set[str]] = {}
        self.conflicts: dict[bytes, dict[str, RawEvidence]] = {}

    @staticmethod
    def signature(item: RawEvidence, *, claimed: bool = False) -> str:
        value = claimed_owner(item) if claimed else owner(item)
        linked = item.payload.get("call_record_id")
        if isinstance(linked, str) and item.payload.get("call_link_status") not in {
            "ambiguous",
            "missing_context",
        }:
            return digest_of([value, linked])
        return digest_of(value)

    def note(self, item: RawEvidence, key: bytes) -> None:
        if item.kind == CLAIM:
            target = target_key(item)
            if target is None:
                return
            key = hashlib.blake2b("\0".join(target).encode(), digest_size=16).digest()
            signature = self.signature(item, claimed=True)
            if item.payload.get("origin_conflict") is True:
                signature = digest_of([signature, item.payload.get("target_digest")])
                conflict = self.conflict_evidence(item)
                if conflict is not None:
                    self.conflicts.setdefault(key, {})[conflict.digest] = conflict
            self.extra.setdefault(key, set()).add(signature)
        else:
            self.first.setdefault(key, self.signature(item))

    def take(self, item: RawEvidence, key: bytes, *, conflict: bool = False) -> RawEvidence | None:
        signature = self.signature(item)
        if conflict:
            signature = digest_of([signature, item.digest])
        if self.first.get(key) == signature or signature in self.extra.get(key, ()):
            return None
        self.extra.setdefault(key, set()).add(signature)
        source, session, branch, cwd, parent = owner(item)
        payload = {
            "target_native_key": list(item.native_key),
            "original_kind": item.kind,
            "target_record_id": item.record_id,
            "target_digest": item.digest,
            "call_record_id": item.payload.get("call_record_id"),
            "call_link_status": item.payload.get("call_link_status"),
            "claimant_source_id": source,
            "claimant_session_id": session,
            "claimant_branch_id": branch,
            "claimant_cwd": cwd,
            "claimant_parent_session": parent,
        }
        if conflict:
            payload.update(target_digest=item.digest, origin_conflict=True)
        record_id = "ownership:" + digest_of(payload)
        return replace(
            item,
            kind=CLAIM,
            record_id=record_id,
            digest=digest_of(payload),
            payload=payload,
            origin_domain="native_ownership_claim_v1",
            origin_id=record_id,
            origin_status="metadata_claim",
        )

    @staticmethod
    def conflict_evidence(item: RawEvidence) -> RawEvidence | None:
        if item.kind != CLAIM or item.payload.get("origin_conflict") is not True:
            return None
        target, digest = target_key(item), item.payload.get("target_digest")
        if target is None or not isinstance(digest, str):
            return None
        return replace(
            item,
            runtime=target[0],
            session_id=target[1],
            record_id=target[2],
            origin_domain=target[1],
            origin_id=target[2],
            digest=digest,
            kind=item.payload.get("original_kind", "unknown"),
            payload={},
        )


def native_in_scope(record: Any, timelines: Any, scoper: Any) -> bool:
    """Unknown origin ownership is local-observable, never organization-authoritative."""
    personal = scoper.policy.profile_id == PERSONAL_PROFILE and not scoper.policy.roots
    if not personal and (
        record.identity in timelines.ownership_unknown
        or record.identity in timelines.order_unknown
        or record.ev.origin_status
        in {"missing_session_id", "missing_entry_id", "legacy_unverified"}
    ):
        return False
    claims = timelines.ownership_claims.get(record.identity, ())
    if claims and not personal:
        return all(scoper.in_scope(claim[3]) for claim in claims)
    cwd = record.ev.payload.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        cwd = timelines.sessions[(record.ev.runtime, record.ev.session_id)].cwd_at(record.seq)
    return scoper.in_scope(cwd)


MIGRATED_RUNTIMES = frozenset({"claude", "pi", "omp", "prime-agent"})


def _legacy_payload(item: RawEvidence, candidate: RawEvidence, source: Any, obj: dict) -> dict:
    p = candidate.payload
    if item.runtime == "claude":
        names = {
            "tool_call": {"call_id", "name", "arguments", "message_id", "cwd"},
            "tool_result": {"call_id", "output", "output_bytes", "is_error"},
            "user_turn": {"text", "cwd"},
            "usage": {"message_id", "model", "tokens", "streaming_updates"},
        }.get(item.kind, set(p))
        result = {k: v for k, v in p.items() if k in names}
        if source.role == "child":
            result["parent_session"] = p.get("parent_session")
        return result
    if item.kind == "tool_call":
        names = {"call_id", "name", "arguments", "entry_id", "cwd", "occurrence"}
    elif item.kind == "tool_result":
        names = {"occurrence", "call_id", "tool_name", "output", "output_bytes", "is_error"}
    elif item.kind == "user_turn":
        names = {"text"}
    else:
        names = set(p) - {
            "entry_id",
            "native_parent_id",
            "session_context",
            "parent_session",
            "originating_turn_record_id",
            "task_context_basis",
            "baseline_prior_context_unavailable",
        }
        if item.kind == "usage":
            names.discard("cwd")
    result = {k: v for k, v in p.items() if k in names}
    if item.kind in {"tool_call", "tool_result"}:
        suffix = item.record_id.rsplit("#", 1)
        result["occurrence"] = int(suffix[-1]) if len(suffix) == 2 and suffix[-1].isdigit() else 0
    if item.kind == "tool_call" and item.session_id == source.path.stem and not obj.get("cwd"):
        # The defective EOF baseline had no header/cwd context; verify that exact old digest.
        result["cwd"] = item.payload.get("cwd")
    return result


def _context(reader: Any, source: Any, fd: int, offset: int) -> tuple[dict, int]:
    """At most two metadata windows, using the already verified descriptor."""
    from ssgrep.accounting.readers.pi_family import HEADER_LINES, PiFamilyReader

    ctx: dict = {}
    first = os.pread(fd, min(offset, CONTEXT_BYTES), 0)
    consumed = len(first)
    if isinstance(reader, PiFamilyReader):
        for raw in first.splitlines(keepends=True)[:HEADER_LINES]:
            if not raw.endswith(b"\n"):
                continue
            try:
                obj = loads(raw)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "session":
                reader.decode(source, obj, ctx, "offset:0")
                break
    start = max(0, offset - CONTEXT_BYTES)
    tail = os.pread(fd, offset - start, start)
    consumed += len(tail)
    if start:
        boundary = tail.find(b"\n")
        start += boundary + 1 if boundary >= 0 else len(tail)
        tail = tail[boundary + 1 :] if boundary >= 0 else b""
        ctx["call_context_incomplete"] = True
    for raw in tail.splitlines(keepends=True):
        locator = f"offset:{start}"
        start += len(raw)
        if not raw.endswith(b"\n"):
            continue
        try:
            obj = loads(raw)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if isinstance(reader, PiFamilyReader) and obj.get("type") not in {"session", "message"}:
            observe(obj, ctx, locator)
        message = obj.get("message")
        if isinstance(message, dict):
            reader.decode(
                source,
                {**obj, "message": {k: v for k, v in message.items() if k != "usage"}},
                ctx,
                locator,
            )
    return ctx, consumed


@dataclass(frozen=True)
class RecordMigration:
    evidence: RawEvidence | None
    reason: str = ""
    bytes_read: int = 0


def migrate_record(item: RawEvidence, readers: dict, scoper: Any, budget: int) -> RecordMigration:
    from ssgrep.accounting.ledger.valuation_inputs import AtomStatus
    from ssgrep.accounting.readers.native_source import SourceReadError, open_source

    if item.origin_domain is not None or item.runtime not in MIGRATED_RUNTIMES:
        return RecordMigration(item)
    reader = readers.get(item.runtime)
    prefix = item.runtime + ":"
    if (
        reader is None
        or not item.source_id.startswith(prefix)
        or not item.locator.startswith("offset:")
    ):
        return RecordMigration(None, "source_identity_mismatch")
    try:
        offset = int(item.locator.split(":", 1)[1])
    except ValueError:
        return RecordMigration(None, "source_identity_mismatch")
    if offset < 0 or budget <= 2 * CONTEXT_BYTES:
        return RecordMigration(None, "source_read_bound")
    source = reader.source_for(Path(item.source_id[len(prefix) :]))
    try:
        descriptor = open_source(source.path, tuple(reader.approved_roots()))
        with os.fdopen(descriptor, "rb") as handle:
            context, context_bytes = _context(reader, source, handle.fileno(), offset)
            handle.seek(offset)
            raw = handle.readline(min(MAX_RAW_RECORD_BYTES, budget - 2 * CONTEXT_BYTES) + 1)
        consumed = len(raw) + context_bytes
        if not raw.endswith(b"\n") or len(raw) > MAX_RAW_RECORD_BYTES:
            return RecordMigration(None, "source_read_bound", consumed)
        obj = loads(raw)
        decoded = reader.decode(source, obj, context, item.locator) or []
        candidates = [
            c
            for c in decoded
            if c.kind == item.kind
            and all(
                item.payload.get(k) in (None, c.payload.get(k)) for k in ("call_id", "message_id")
            )
        ]
        matches = [
            c
            for c in candidates
            if evidence_digest(item.kind, _legacy_payload(item, c, source, obj)) == item.digest
        ]
        if len(matches) != 1:
            return RecordMigration(
                None, "source_digest_drift" if candidates else "source_identity_mismatch", consumed
            )
        candidate = matches[0]
        cwd = candidate.payload.get("cwd") or context.get("cwd")
        scope, reason = scoper.source_status(cwd)
        if scope is not AtomStatus.OK and item.kind not in {"session_header", "lineage"}:
            return RecordMigration(None, reason, consumed)
        return RecordMigration(candidate, bytes_read=consumed)
    except SourceReadError as error:
        return RecordMigration(None, error.status.value)
    except (OSError, ValueError, TypeError):
        return RecordMigration(None, "source_unreadable")


STREAM_KEY = "streaming_updates"
EQUAL, UP, DOWN, MIXED = "equal", "up", "down", "mixed"


def is_stream(evidence: RawEvidence) -> bool:
    return evidence.kind == "usage" and STREAM_KEY in evidence.payload


def compare(old: dict[str, int], new: dict[str, int]) -> str:
    """Relation of ``new`` to ``old`` over the union of counter names."""
    up = down = False
    for name in old.keys() | new.keys():
        before, after = old.get(name, 0), new.get(name, 0)
        up |= after > before
        down |= after < before
    if up and down:
        return MIXED
    return UP if up else DOWN if down else EQUAL


def same_context(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Everything except the counters and the update count must match exactly."""
    skip = {"tokens", STREAM_KEY}
    return {k: v for k, v in old.items() if k not in skip} == {
        k: v for k, v in new.items() if k not in skip
    }


class StreamState:
    """Per-identity resolution state inside an ``EvidenceLedger``."""

    def __init__(self, evidence: RawEvidence) -> None:
        self.best = evidence
        self.last: dict[str, dict[str, int]] = {}
        self.updates: dict[str, int] = {}
        self.note(evidence)

    def note(self, evidence: RawEvidence) -> None:
        source = evidence.source_id
        self.last[source] = evidence.payload["tokens"]
        self.updates[source] = self.updates.get(source, 0) + evidence.payload[STREAM_KEY]

    @property
    def total_updates(self) -> int:
        return max(self.updates.values())


def collapse(evidence: list[RawEvidence]) -> list[RawEvidence]:
    """Merge consecutive-by-identity streamed usage within one read cycle.

    A record joins the open run for its identity only while counters do not decrease and
    the context is identical; otherwise it starts a new run, which the ledger then reports
    as a conflict. The merged record keeps the greatest counters and the update count.
    """
    out: list[RawEvidence] = []
    runs: dict[tuple[str, str, str], int] = {}
    for item in evidence:
        if not is_stream(item):
            out.append(item)
            continue
        key = (item.runtime, item.session_id, item.record_id)
        index = runs.get(key)
        if index is not None:
            current = out[index]
            relation = compare(current.payload["tokens"], item.payload["tokens"])
            if same_context(current.payload, item.payload) and relation in {EQUAL, UP}:
                updates = current.payload[STREAM_KEY] + item.payload[STREAM_KEY]
                body = {**item.payload, STREAM_KEY: updates}
                digest = evidence_digest(item.kind, body)
                out[index] = replace(item, payload=body, digest=digest)
                continue
        runs[key] = len(out)
        out.append(item)
    return out
