"""Deterministic candidate extraction, prior-knowledge, and action references (6.1, 6.2, 6.5).

A candidate exists only from OBSERVED result content, is matched on whole-token
boundaries, and is dropped when the model or user already had it: in the request
itself, in an earlier user turn, or visible in earlier available branch/ancestor
context. Missing or compacted context does not drop a candidate; it lowers the
reduction's novelty confidence. An action that mentions a candidate later is an
*association*, never proof the model learned it from retrieval.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ssgrep.accounting.evidence_tokens import REF_RE, Token, scan_text, tokens_of
from ssgrep.accounting.exposure import Interpretation
from ssgrep.accounting.lineage import Lineage
from ssgrep.accounting.timeline import Record, SessionKey, Timeline, record_text

COMPACTION_RE = re.compile(
    r"^\s*(?:This session is being continued from a previous conversation|<summary>|\[compact"
    r"|The conversation history before this point was compacted|Summary of the conversation)",
    re.IGNORECASE,
)
VERIFY_RE = re.compile(
    r"\b(?:pytest|py\.test|unittest|tox|nox|jest|vitest|mocha|rspec|phpunit|mypy|tsc|ruff|flake8"
    r"|eslint|pyright|ty\s+check|cargo\s+(?:test|check|clippy)|go\s+(?:test|vet)|npm\s+(?:test|run\s+(?:test|lint|check))"
    r"|make\s+(?:test|check|lint)|poe\s+(?:test|lint|check|typecheck)|ctest)\b",
)
SUCCESS_RE = re.compile(
    r"\b\d+ passed\b(?!.*\b\d+ (?:failed|error))|\ball checks passed\b|\bOK\b\s*$"
    r"|\b0 (?:errors?|failures?)\b|\bBUILD SUCCESS",
    re.IGNORECASE | re.DOTALL,
)
FAILURE_RE = re.compile(
    r"\b[1-9]\d*\s+(?:failed|failures?|errors?)\b|\bFAILED\b|\bTraceback\b|\bBUILD FAILURE",
    re.IGNORECASE,
)
PURPOSE_RE = re.compile(r"\bSSGREP_PURPOSE=([a-z_]{1,32})\b")
PURPOSES = frozenset({"protocol_overhead", "task_work"})
MIN_PATH_TAIL = 2


def match_keys(token: Token) -> frozenset[str]:
    """Keys under which ``token`` is compared: itself plus whole-segment path tails."""
    keys = {token.key}
    if token.kind == "path":
        parts = [p for p in token.text.split("/") if p not in ("", "~", ".", "..")]
        for start in range(len(parts) - MIN_PATH_TAIL + 1):
            keys.add("path\0" + "/".join(parts[start:]))
    return frozenset(keys)


class TokenKeys:
    """Memoized match keys of record text; every index shares one scan per distinct text."""

    def __init__(self) -> None:
        self._cache: dict[bytes, frozenset[str]] = {}

    def of_text(self, text: str) -> frozenset[str]:
        if not text:
            return frozenset()
        digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
        keys = self._cache.get(digest)
        if keys is None:
            found: set[str] = set()
            for token in tokens_of(text, home=True):
                found |= match_keys(token)
            keys = self._cache[digest] = frozenset(found)
        return keys

    def of(self, record: Record) -> frozenset[str]:
        if record.kind not in ("user_turn", "tool_call", "tool_result"):
            return frozenset()
        return self.of_text(record_text(record))


@dataclass(frozen=True)
class Exposed:
    """A candidate token visible in one lookup result, with the cards that show it."""

    token: Token
    refs: tuple[str, ...]


@dataclass
class Extraction:
    exposed: list[Exposed] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)  # scan-level reason -> count


def extract(interp: Interpretation) -> Extraction:
    """Candidates from the text the model could actually see (never hidden/truncated text)."""
    out = Extraction()
    text_refs: tuple[str, ...] = ()
    if interp.cards:
        sources = [(card.ref, card.text) for card in interp.cards]
    else:
        sources = [("", interp.text)]
        found = {m.group(0) for m in REF_RE.finditer(interp.text) if ":ep:" in m.group(0)}
        text_refs = tuple(sorted(found)[:5])
    refs_by_token: dict[Token, set[str]] = {}
    for ref, text in sources:
        scan = scan_text(text)
        for token in scan.kept:
            refs_by_token.setdefault(token, set()).add(ref)
        for token, reason in scan.excluded.items():
            if token not in refs_by_token:
                out.excluded[reason] = out.excluded.get(reason, 0) + 1
    for token in sorted(refs_by_token):
        refs = tuple(sorted(r for r in refs_by_token[token] if r))
        out.exposed.append(Exposed(token, refs or text_refs))
    return out


class SessionIndex:
    """First positions at which tokens were available to the model in one session."""

    def __init__(
        self,
        line: Timeline,
        lookup_calls: frozenset[str],
        tk: TokenKeys,
    ) -> None:
        self.first: dict[str, dict[str, int]] = {}  # key -> class -> min seq
        self.compacted_at: int | None = None
        for record in line.records:
            if record.kind == "user_turn":
                klass = "user"
                if self.compacted_at is None and COMPACTION_RE.match(record_text(record)):
                    self.compacted_at = record.seq
            elif record.kind == "tool_call":
                klass = "query" if record.ev.record_id in lookup_calls else "context"
            elif record.kind == "tool_result":
                klass = "context"  # an earlier exposure is visible context for later lookups
            else:
                continue
            for key in tk.of(record):
                held = self.first.setdefault(key, {})
                if record.seq < held.get(klass, record.seq + 1):
                    held[klass] = record.seq

    def known_before(self, token: Token, upto: int) -> str | None:
        """Why ``token`` was already available before position ``upto``, else None.

        Only the token itself counts: knowing a shorter path says nothing about a longer one.
        """
        held = self.first.get(token.key, {})
        best = min(((seq, klass) for klass, seq in held.items() if seq < upto), default=None)
        if best is None:
            return None
        return {"user": "user_supplied", "query": "query_supplied"}.get(best[1], "already_visible")


@dataclass(frozen=True)
class Novelty:
    reason: str | None  # None: novel; else user_supplied | query_supplied | already_visible
    confidence: str  # high | reduced
    limits: tuple[str, ...]


def novelty(
    token: Token,
    key: SessionKey,
    seq: int,
    lineage: Lineage,
    index_for: Callable[[SessionKey], SessionIndex],
    timelines_sessions: dict[SessionKey, Timeline],
) -> Novelty:
    """Was ``token`` available in any earlier branch/ancestor context before ``seq``?"""
    chain, limits = lineage.context(key, seq)
    reasons = list(limits)
    found: str | None = None
    for session, upto in chain:
        index = index_for(session)
        if index.compacted_at is not None and index.compacted_at < upto:
            reasons.append("compaction_marker_in_context")
        line = timelines_sessions[session]
        first = line.records[0].kind if line.records else ""
        if first not in ("user_turn", "session_header"):
            reasons.append("session_start_not_observed")
        why = index.known_before(token, upto)
        if why and (found is None or why == "user_supplied"):
            found = why
    return Novelty(found, "reduced" if reasons else "high", tuple(sorted(set(reasons))))


def is_verification(command: str) -> bool:
    return bool(VERIFY_RE.search(command))


def validates(command: str, interp: Interpretation) -> bool:
    """Explicit validation evidence: a verification command whose result is positively good.

    A transport-level ``is_error: false`` alone never validates; the command must be a
    recognized verifier and the result must carry exit status 0 or a success marker.
    """
    if not is_verification(command) or interp.is_error:
        return False
    tail = interp.text[-4000:]
    if FAILURE_RE.search(tail):
        return False
    return interp.exit_code == 0 or bool(SUCCESS_RE.search(tail))


def purpose_of(command: str) -> str:
    """Protocol/task purpose from explicit invocation metadata only; otherwise unknown."""
    found = PURPOSE_RE.findall(command)
    return found[0] if len(set(found)) == 1 and found[0] in PURPOSES else "unknown"
