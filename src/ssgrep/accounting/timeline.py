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
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ssgrep.accounting.contracts import RawEvidence

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
        return f"{self.ev.runtime}:{self.ev.session_id}:{self.ev.record_id}"

    @property
    def kind(self) -> str:
        return self.ev.kind


@dataclass
class Timeline:
    key: SessionKey
    records: list[Record] = field(default_factory=list)
    by_id: dict[str, Record] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)
    turns: list[Record] = field(default_factory=list)  # user_turn records in order
    cwds: list[str | None] = field(default_factory=list)  # effective cwd per record
    branch: str = "main"

    def cwd_at(self, seq: int) -> str | None:
        return self.cwds[seq] if 0 <= seq < len(self.cwds) else None


@dataclass
class Timelines:
    sessions: dict[SessionKey, Timeline]
    lineage: list[RawEvidence]  # kind == "lineage" (child launch/parent rows)
    duplicates: int = 0
    conflicts: int = 0
    conflicted: set[str] = field(default_factory=set)  # record identities with variants


def _order(evidence: RawEvidence) -> tuple[str, int, int, str]:
    found = OFFSET_RE.match(evidence.locator)
    offset = int(found.group(1)) if found else -1
    return (evidence.source_id, offset, ts_ms(evidence.ts_utc) or 0, evidence.record_id)


def build_timelines(evidence: Iterable[RawEvidence]) -> Timelines:
    """Dedupe by identity (conflicts keep the smallest digest) and order per session."""
    best: dict[tuple[str, str, str], RawEvidence] = {}
    seen: set[tuple[str, str, str, str]] = set()
    variants: dict[tuple[str, str, str], set[str]] = {}
    total = 0
    for item in evidence:
        total += 1
        seen.add(item.identity)
        key = (item.runtime, item.session_id, item.record_id)
        variants.setdefault(key, set()).add(item.digest)
        held = best.get(key)
        # smallest (digest, order) wins: independent of input order, copies included
        if held is None or (item.digest, _order(item)) < (held.digest, _order(held)):
            best[key] = item
    duplicates = total - len(seen)
    conflicted = {f"{k[0]}:{k[1]}:{k[2]}" for k, v in variants.items() if len(v) > 1}
    conflicts = len(conflicted)
    chosen = best
    lineage: list[RawEvidence] = []
    grouped: dict[SessionKey, list[RawEvidence]] = {}
    for item in chosen.values():
        if item.kind == "lineage":
            lineage.append(item)
        else:
            grouped.setdefault(session_key(item), []).append(item)
    sessions: dict[SessionKey, Timeline] = {}
    for key in sorted(grouped):
        line = Timeline(key)
        cwd: str | None = None
        for seq, item in enumerate(sorted(grouped[key], key=_order)):
            record = Record(item, seq, ts_ms(item.ts_utc))
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
        sessions[key] = line
    lineage.sort(key=lambda e: (e.runtime, e.session_id, e.record_id, e.digest))
    return Timelines(sessions, lineage, duplicates, conflicts, conflicted)


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
