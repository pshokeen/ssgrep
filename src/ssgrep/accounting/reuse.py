"""Reuse evidence for one exposed candidate (task 6.2 and 6.5).

Looks for the candidate, as a whole token, in later ACTIONS of the same task: the
rest of the consumer's own turn, or -- for a child -- the launching parent's turn
but only through an explicit relay (a parent result that names this child and
carries the token). A sibling that merely mentions the same string never counts.
An action reference is an association, not causal learning; ``validated`` needs
explicit verification evidence and ``contradicted`` needs an interpreted failure.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from ssgrep.accounting.contracts import Reuse
from ssgrep.accounting.evidence import TokenKeys, is_verification, validates
from ssgrep.accounting.evidence_tokens import Token
from ssgrep.accounting.exposure import interpret
from ssgrep.accounting.lineage import Lineage
from ssgrep.accounting.timeline import (
    Record,
    SessionKey,
    Timeline,
    Timelines,
    call_text,
    record_text,
)

MAX_RELAY_DEPTH = 4


@dataclass(frozen=True)
class Finding:
    reuse: Reuse
    action: Record | None = None
    action_key: SessionKey | None = None
    outcome_ok: bool = False
    relay: bool = False
    outcome: str = "none"  # none | inspected | action | validated | contradicted


NONE = Finding(Reuse.NONE_OBSERVED)


class ActionIndex:
    """Where each match key occurs in one session's actions, lookups, and results."""

    def __init__(self, line: Timeline, lookup_calls: frozenset[str], tk: TokenKeys) -> None:
        self.line = line
        self.actions: dict[str, list[int]] = {}
        self.results: dict[str, list[int]] = {}
        for record in line.records:
            if (
                line.is_conflicted(record)
                or record.kind == "tool_call"
                and line.has_conflicted_result(record)
                or record.kind == "tool_result"
                and line.has_conflicted_call(record)
            ):
                continue
            if record.kind == "tool_call":
                if record.ev.record_id in lookup_calls:
                    continue  # a lookup is the exposure channel, not an action on it
                text, bucket = call_text(record.ev.payload), self.actions
            elif record.kind == "tool_result":
                text, bucket = record_text(record), self.results
            else:
                continue
            for key in tk.of_text(text):
                bucket.setdefault(key, []).append(record.seq)

    def first(self, bucket: dict[str, list[int]], key: str, start: int, end: int) -> int | None:
        seqs = bucket.get(key)
        if not seqs:
            return None
        at = bisect_right(seqs, start - 1)
        return seqs[at] if at < len(seqs) and seqs[at] < end else None

    def turn_end(self, after_seq: int) -> int:
        """Index (exclusive) where the task containing ``after_seq`` ends: the next user turn."""
        for turn in self.line.turns:
            if turn.seq > after_seq:
                return turn.seq
        return len(self.line.records)


def _outcome(line: Timeline, action: Record, end: int) -> tuple[bool, str]:
    """``(action_outcome_ok, 'action'|'validated'|'contradicted')`` for a reference."""
    result = line.result_for(action)
    if result is None:
        return False, "action"
    interp = interpret(result.ev.payload)
    if interp.is_error:
        return False, "contradicted"
    for record in line.records[action.seq : end]:
        if record.kind != "tool_call":
            continue
        if line.is_conflicted(record) or line.has_conflicted_result(record):
            continue
        command = call_text(record.ev.payload)
        if not is_verification(command):
            continue
        later = line.result_for(record)
        if later is not None and validates(command, interpret(later.ev.payload)):
            return True, "validated"
    return interp.action_ok, "action"


def _label(outcome: str) -> Reuse:
    return {
        "contradicted": Reuse.CONTRADICTED,
        "validated": Reuse.VALIDATED,
        "action": Reuse.ACTION_REFERENCE,
    }[outcome]


def _relay(
    token: Token,
    key: SessionKey,
    timelines: Timelines,
    lineage: Lineage,
    indexes: dict[SessionKey, ActionIndex],
) -> Finding | None:
    for _ in range(MAX_RELAY_DEPTH):
        link = lineage.launch.get(key)
        if link is None or key in lineage.broken:
            return None
        parent = timelines.sessions[link.parent]
        index = indexes[link.parent]
        names = lineage.identities(key)
        end = index.turn_end(link.call_seq)
        cursor = link.call_seq + 1
        while True:
            seq = index.first(index.results, token.key, cursor, end)
            if seq is None:
                break
            if any(name in record_text(parent.records[seq]) for name in names):
                at = index.first(index.actions, token.key, seq + 1, end)
                if at is not None:
                    action = parent.records[at]
                    ok, outcome = _outcome(parent, action, end)
                    return Finding(_label(outcome), action, link.parent, ok, True, outcome)
                break
            cursor = seq + 1
        key = link.parent
    return None


def find_reuse(
    token: Token,
    key: SessionKey,
    call_seq: int,
    start: int,
    timelines: Timelines,
    lineage: Lineage,
    indexes: dict[SessionKey, ActionIndex],
) -> Finding:
    """The strongest supported reuse of ``token`` after an exposure at ``start``."""
    index = indexes[key]
    line = timelines.sessions[key]
    end = len(line.records) if lineage.is_child(key) else index.turn_end(call_seq)
    at = index.first(index.actions, token.key, start, end)
    if at is not None:
        action = line.records[at]
        ok, outcome = _outcome(line, action, end)
        return Finding(_label(outcome), action, key, ok, False, outcome)
    relayed = _relay(token, key, timelines, lineage, indexes)
    if relayed is not None:
        return relayed
    return NONE
