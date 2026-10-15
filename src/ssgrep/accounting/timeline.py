"""Deduplicated, ordered per-session views of raw evidence (pure, no I/O).

Real transcripts reuse call ids across sessions, so every key here is
``(runtime, session, ...)``; records are ordered by native file order
(``source, offset``) and then timestamp, never by timestamp alone, so a wall-clock
discontinuity cannot reorder a session. Identical native copies collapse; one
identity with different content is a *conflict*: the smallest digest is kept
deterministically and the conflict is reported, never counted twice.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.identity_claims import CLAIM, claimed_owner, owner, target_key
from ssgrep.accounting.native_timeline import canonical_copies, logical_membership, ordered
from ssgrep.accounting.readers.streaming import is_stream

SessionKey = tuple[str, str]
TEXT_KINDS = frozenset({"user_turn", "tool_call", "tool_result"})
OFFSET_RE = re.compile(r"^offset:(\d+)$")


def key_str(key: SessionKey) -> str:
    return f"{key[0]}:{key[1]}"


def ts_ms(ts: str | None) -> int | None:
    """UTC epoch milliseconds of an RFC 3339 string; None when absent or invalid."""
    if not ts:
        return None
    try:
        moment = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return calendar.timegm(moment.utctimetuple()) * 1000 + moment.microsecond // 1000


def day_of(ms: int | None) -> str | None:
    """UTC ``YYYY-MM-DD`` of epoch milliseconds."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms // 1000, UTC).strftime("%Y-%m-%d")


def session_key(evidence: RawEvidence) -> SessionKey:
    return (evidence.runtime, evidence.session_id)


@dataclass(frozen=True)
class Record:
    """One evidence record at a position in its session."""

    ev: RawEvidence
    seq: int
    ts: int | None

    @property
    def identity(self) -> str:
        return self.ev.native_identity

    @property
    def kind(self) -> str:
        return self.ev.kind


@dataclass
class Timeline:
    key: SessionKey
    conflicted: frozenset[str]
    conflict_times: Mapping[str, tuple[int | None, ...]]
    calls_with_conflicted_results: frozenset[str]
    results_with_conflicted_calls: frozenset[str]
    records: list[Record] = field(default_factory=list)
    by_id: dict[str, Record] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)
    turns: list[Record] = field(default_factory=list)  # user_turn records in order
    cwds: list[str | None] = field(default_factory=list)  # effective cwd per record
    branch: str = "main"
    result_by_call: dict[str, Record] = field(default_factory=dict)
    call_by_result: dict[str, Record] = field(default_factory=dict)

    def result_for(self, call: Record) -> Record | None:
        """Unique proven result; explicit missing/ambiguous links never use suffixes."""
        return self.result_by_call.get(call.identity)

    def call_for(self, result: Record) -> Record | None:
        return self.call_by_result.get(result.identity)

    def cwd_at(self, seq: int) -> str | None:
        return self.cwds[seq] if 0 <= seq < len(self.cwds) else None

    def is_conflicted(self, record: Record) -> bool:
        return record.identity in self.conflicted

    def has_conflicted_result(self, call: Record) -> bool:
        return call.identity in self.calls_with_conflicted_results

    def has_conflicted_call(self, result: Record) -> bool:
        return result.identity in self.results_with_conflicted_calls


@dataclass
class Timelines:
    sessions: dict[SessionKey, Timeline]
    lineage: list[RawEvidence]  # kind == "lineage" (child launch/parent rows)
    duplicates: int = 0
    conflicts: int = 0
    conflicted: set[str] = field(default_factory=set)  # record identities with variants
    ownership_unknown: set[str] = field(default_factory=set)
    ownership_claims: dict[str, set[tuple[str, str, str, str | None, str | None]]] = field(
        default_factory=dict
    )
    order_unknown: set[str] = field(default_factory=set)


def _order(evidence: RawEvidence) -> tuple[str, int, int, str]:
    found = OFFSET_RE.match(evidence.locator)
    offset = int(found.group(1)) if found else -1
    return (evidence.source_id, offset, ts_ms(evidence.ts_utc) or 0, evidence.record_id)


def _linked_call_id(item: RawEvidence) -> str | None:
    payload = item.payload
    if "call_record_id" in payload:
        linked = payload["call_record_id"]
        return (
            linked
            if isinstance(linked, str)
            and payload.get("call_link_status") not in {"ambiguous", "missing_context"}
            else None
        )
    if item.runtime in {"pi", "omp", "prime-agent"}:
        return None
    return "call:" + item.record_id.split(":", 1)[-1]


def _pair(line: Timeline, conflicted: set[str]) -> None:
    candidates: dict[str, list[Record]] = {}
    for result in line.records:
        if result.kind != "tool_result" or result.identity in conflicted:
            continue
        linked = _linked_call_id(result.ev)
        if linked is None:
            continue
        call = line.by_id.get(linked)
        if call is not None and call.kind == "tool_call" and call.identity not in conflicted:
            candidates.setdefault(call.identity, []).append(result)
    for identity, results in candidates.items():
        if len(results) == 1:
            result = results[0]
            line.result_by_call[identity] = result
            linked = result.ev.payload.get("call_record_id")
            if linked is None:
                linked = "call:" + result.ev.record_id.split(":", 1)[-1]
            line.call_by_result[result.identity] = line.by_id[linked]


def _conflicted_links(best, links, conflicted) -> tuple[frozenset[str], frozenset[str]]:
    if not conflicted:
        return frozenset(), frozenset()
    calls: dict[tuple[str, str], list[RawEvidence]] = {}
    for item in best.values():
        if item.kind == "tool_call":
            calls.setdefault((item.runtime, item.record_id), []).append(item)
    blocked_calls, blocked_results = set(), set()
    for key, result in best.items():
        if result.kind != "tool_result":
            continue
        proven = set(links.get(key, ()))
        if linked := _linked_call_id(result):
            proven.add(linked)
        for linked in proven:
            for call in calls.get((result.runtime, linked), ()):
                if (
                    result.origin_status in {"confirmed_uuid", "provider_message_id"}
                    or call.native_key[1] == result.native_key[1]
                ):
                    if result.native_identity in conflicted:
                        blocked_calls.add(call.native_identity)
                    if call.native_identity in conflicted:
                        blocked_results.add(result.native_identity)
    return frozenset(blocked_calls), frozenset(blocked_results)


def build_timelines(evidence: Iterable[RawEvidence]) -> Timelines:
    """Dedupe by identity (conflicts keep the smallest digest) and order per session."""
    best: dict[tuple[str, str, str], RawEvidence] = {}
    seen: set[tuple[str, str, str, str]] = set()
    variants: dict[tuple[str, str, str], set[str]] = {}
    claims: dict[str, set[tuple[str, str, str, str | None, str | None]]] = {}
    observed = list(evidence)
    claim_conflicts: set[tuple[str, str, str]] = set()
    total = 0
    for item in observed:
        if item.kind == CLAIM:
            target = target_key(item)
            if target is not None:
                claims.setdefault(":".join(target), set()).add(claimed_owner(item))
                digest = item.payload.get("target_digest")
                if item.payload.get("origin_conflict") is True and isinstance(digest, str):
                    variants.setdefault(target, set()).add(digest)
                    claim_conflicts.add(target)
            continue
        total += 1
        seen.add(item.identity)
        claims.setdefault(item.native_identity, set()).add(owner(item))
        key = item.native_key
        variants.setdefault(key, set()).add(item.digest)
        held = best.get(key)
        # smallest (digest, order) wins: independent of input order, copies included
        if held is None or (item.digest, _order(item)) < (held.digest, _order(held)):
            best[key] = item
    links, stream_conflicts = canonical_copies(best, observed)
    for key, item in best.items():
        if is_stream(item) and key not in stream_conflicts and key not in claim_conflicts:
            variants[key] = {item.digest}
    membership, paired = logical_membership(best)
    duplicates = total - len(seen)
    conflicted = {":".join(key) for key, values in variants.items() if len(values) > 1} | {
        ":".join(key) for key in claim_conflicts
    }
    conflicts = len(conflicted)
    times: dict[str, set[int | None]] = {}
    for item in observed:
        target = target_key(item) if item.kind == CLAIM else item.native_key
        identity = ":".join(target) if target is not None else ""
        if identity in conflicted:
            times.setdefault(identity, set()).add(ts_ms(item.ts_utc))
    for key in claim_conflicts:
        if len(variants[key]) < 2:
            # The recorded conflict survives even when its other observation's
            # body/timestamp is unavailable. It cannot prove future-only context.
            times[":".join(key)].add(None)
    conflict_times = {
        identity: tuple(sorted(values, key=lambda value: (value is not None, value)))
        for identity, values in times.items()
    }
    blocked_calls, blocked_results = _conflicted_links(best, links, conflicted)
    conflict_ids = frozenset(conflicted)
    chosen = best
    lineage: list[RawEvidence] = []
    grouped: dict[SessionKey, list[RawEvidence]] = {}
    for native_key, item in chosen.items():
        if item.kind == "lineage":
            lineage.append(item)
        else:
            grouped.setdefault(membership[native_key], []).append(item)
    sessions: dict[SessionKey, Timeline] = {}
    order_unknown: set[str] = set()
    for key in sorted(grouped):
        line = Timeline(key, conflict_ids, conflict_times, blocked_calls, blocked_results)
        cwd: str | None = None
        sequence, missing_order = ordered(grouped[key], observed, paired, _order)
        order_unknown.update(missing_order)
        for seq, item in enumerate(sequence):
            stamp = ts_ms(item.ts_utc)
            variant_times = conflict_times.get(item.native_identity)
            if item.native_identity in missing_order or (
                variant_times is not None and (len(variant_times) != 1 or variant_times[0] is None)
            ):
                stamp = None
            record = Record(item, seq, stamp)
            for name in ("cwd", "directory"):
                value = item.payload.get(name)
                if isinstance(value, str) and value:
                    cwd = value
                    break
            line.cwds.append(cwd)
            if seq == 0:
                line.branch = item.branch_id
            line.records.append(record)
            line.by_id[item.record_id] = record
            line.sources.add(item.source_id)
            if item.kind == "user_turn":
                line.turns.append(record)
        _pair(line, conflicted)
        sessions[key] = line
    lineage.sort(key=lambda e: (e.runtime, e.session_id, e.record_id, e.digest))
    unknown = {
        identity for identity, owners in claims.items() if len({owner[1:] for owner in owners}) > 1
    }
    for result, call in paired.items():
        if ":".join(result) in unknown:
            unknown.add(":".join(call))
    return Timelines(
        sessions, lineage, duplicates, conflicts, conflicted, unknown, claims, order_unknown
    )


def record_text(record: Record) -> str:
    """The observable text of a record: prompt, call arguments, or result output."""
    payload = record.ev.payload
    if record.kind == "user_turn":
        text = payload.get("text")
    elif record.kind == "tool_result":
        text = payload.get("output")
    elif record.kind == "tool_call":
        return call_text(payload)
    else:
        return ""
    return text if isinstance(text, str) else ""


def _leaves(value: object, out: list[str], depth: int = 0) -> None:
    if depth > 8:
        return
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for key in sorted(value):
            _leaves(value[key], out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _leaves(item, out, depth + 1)


def call_text(payload: dict) -> str:
    """Model-written call text: a shell command verbatim, else its argument string leaves."""
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        for key in ("command", "cmd", "script"):
            value = arguments.get(key)
            if isinstance(value, str):
                return value
    parts: list[str] = []
    _leaves(arguments, parts)
    return "\n".join(parts)
