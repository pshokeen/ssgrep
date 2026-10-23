"""Candidate discovery, reuse evidence, and debits for the reducer."""

from __future__ import annotations

import hmac
import re
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ssgrep.accounting.collect.frequency import (
    FrequencyAssignments,
    FrequencyView,
    eligible_scenarios,
)
from ssgrep.accounting.collect.replay_evidence import (
    NORMALIZATION,
    Consumer,
    Resolution,
    Resolver,
    Token,
    TokenKeys,
    extract,
)
from ssgrep.accounting.ledger.contracts import Basis, Quantity, Reuse
from ssgrep.accounting.ledger.reduce_join import Interpretation, interpret
from ssgrep.accounting.ledger.reduce_records import Call, CandidateRecord, Exclusion, shown_refs
from ssgrep.accounting.ledger.units import (
    CHARS_DIV4_V1,
    CONTEXT_TOKENS,
    WORK_SECONDS,
    UnitSpec,
    context_estimate,
)
from ssgrep.accounting.ledger.valuation_inputs import (
    AtomStatus,
    Candidate,
    Debit,
    Eligibility,
    ValuationAtom,
)
from ssgrep.accounting.readers.pi_family import Lineage
from ssgrep.accounting.readers.timeline import (
    Record,
    SessionKey,
    Timeline,
    Timelines,
    call_text,
    day_of,
    key_str,
    record_text,
)

if TYPE_CHECKING:
    from ssgrep.accounting.views.filter import Scoper

CANDIDATE_KEY = b"ssgrep-candidate-v1"
CREDITED = frozenset({Reuse.ACTION_REFERENCE, Reuse.VALIDATED, Reuse.CONTRADICTED})
CHARS = UnitSpec(CONTEXT_TOKENS, CHARS_DIV4_V1)
WORK = UnitSpec(WORK_SECONDS, "capped_gap_v1")

STRENGTH = {
    Reuse.NONE_OBSERVED: 0,
    Reuse.INSPECTED: 1,
    Reuse.ACTION_REFERENCE: 2,
    Reuse.VALIDATED: 3,
    Reuse.CONTRADICTED: 4,
}


@dataclass
class Outputs:
    candidates: list[CandidateRecord] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    debits: list[Debit] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    reuse: dict[str, Reuse] = field(default_factory=dict)  # invocation id -> strongest reuse
    purpose: dict[str, str] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)


def candidate_id(token: Token, hmac_key: bytes | None) -> str:
    """Stable local fingerprint of a normalized candidate (keyed when a profile key exists)."""
    message = f"{NORMALIZATION}\0{token.key}".encode()
    return hmac.digest(hmac_key or CANDIDATE_KEY, message, "sha256").hex()[:32]


def exposure_day(timelines: Timelines, key: SessionKey, seq: int) -> str | None:
    """UTC day of the nearest timestamped record at or before ``seq`` (else after it)."""
    records = timelines.sessions[key].records
    for record in list(reversed(records[: seq + 1])) + list(records[seq + 1 :]):
        day = day_of(record.ts)
        if day:
            return day
    return None


def debit_for(call: Call, day: str) -> list[Debit]:
    """Known visible context plus unique linked, explicitly modeled lookup work."""
    interp = call.interp
    exposure = call.invocation_id
    if interp is None:
        context = Quantity.unavailable(CONTEXT_TOKENS, "downstream_exposure_unobserved")
        chars = None
    elif call.shape == "mixed" and not interp.cards:
        context = Quantity.unavailable(CONTEXT_TOKENS, "mixed_shell_output_unattributable")
        chars = None
    else:
        chars = interp.chars
        context = context_estimate(chars)
    partial = call.shape != "pure" or (interp is None)
    costs = [Debit(exposure, call.task, day, "exposure", CHARS, context, partial, exposure, chars)]
    for row in call.timing.rows:
        costs.append(
            Debit(
                f"{row.execution_id}:work",
                call.task,
                day,
                "exposure",
                WORK,
                row.work,
                not call.timing.complete,
                exposure,
                work_ns=row.work_ns,
                mapping_version=call.timing.mapping_version,
                assumptions=call.timing.assumptions,
            )
        )
    if not call.timing.complete:
        costs.append(
            Debit(
                f"{exposure}:work:unavailable",
                call.task,
                day,
                "exposure",
                WORK,
                Quantity.unavailable(WORK_SECONDS, call.timing.reason),
                True,
                exposure,
                mapping_version=call.timing.mapping_version,
                assumptions=call.timing.assumptions,
            )
        )
    return costs


@dataclass
class Builder:
    timelines: Timelines
    lineage: Lineage
    scoper: Scoper
    frequency: FrequencyView
    resolver: Resolver
    hmac_key: bytes | None
    lookup_ids: dict[SessionKey, frozenset[str]]
    frequencies: FrequencyAssignments | None = None
    tk: TokenKeys = field(default_factory=TokenKeys)
    out: Outputs = field(default_factory=Outputs)
    _sessions: dict[SessionKey, SessionIndex] = field(default_factory=dict)
    _actions: dict[SessionKey, ActionIndex] = field(default_factory=dict)

    def session_index(self, key: SessionKey) -> SessionIndex:
        found = self._sessions.get(key)
        if found is None:
            line = self.timelines.sessions[key]
            found = SessionIndex(line, self.lookup_ids.get(key, frozenset()), self.tk)
            self._sessions[key] = found
        return found

    def action_index(self, key: SessionKey) -> ActionIndex:
        found = self._actions.get(key)
        if found is None:
            line = self.timelines.sessions[key]
            found = ActionIndex(line, self.lookup_ids.get(key, frozenset()), self.tk)
            self._actions[key] = found
        return found

    class _Indexes(dict):
        def __init__(self, builder: Builder) -> None:
            super().__init__()
            self.builder = builder

        def __missing__(self, key: SessionKey) -> ActionIndex:
            return self.builder.action_index(key)

    def run(self, calls: list[Call], owned: dict[str, list[str]]) -> Outputs:
        indexes = self._Indexes(self)
        later_shows = _later_shows(calls)
        for call in calls:
            if not call.lookup or call.actor != "solver":
                continue
            day = exposure_day(self.timelines, call.key, call.call.seq)
            if day is None:
                self.out.counts["untimed_exposures"] += 1
                continue
            self.out.debits.extend(debit_for(call, day))
            self.out.purpose[call.invocation_id] = purpose_of(_command(call))
            interp = call.interp
            if interp is None or interp.is_error:
                continue
            if call.shape == "mixed" and not interp.cards:
                self.out.counts["exclusion.mixed_shell_unattributable"] += 1
                continue
            frequency = (
                self.frequencies.view_for(call.invocation_id)
                if self.frequencies
                else self.frequency
            )
            self._candidates(
                call,
                interp,
                day,
                owned.get(call.invocation_id, []),
                indexes,
                later_shows,
                frequency,
            )
        self._cycles()
        return self.out

    def _candidates(self, call, interp, day, executions, indexes, later_shows, frequency) -> None:
        result = call.result
        assert result is not None
        shown = later_shows.get(call.invocation_id, frozenset())
        best = Reuse.INSPECTED if shown & {c.ref for c in interp.cards} else Reuse.NONE_OBSERVED
        seen_rows = extract(interp)
        for reason, n in seen_rows.excluded.items():
            self.out.counts[f"exclusion.{reason}"] += n
        for exposed in seen_rows.exposed:
            token = exposed.token
            nov = novelty(
                token, call.key, call.call.seq + 1, self.lineage, self.session_index,
                self.timelines.sessions,
            )  # fmt: skip
            finding = find_reuse(
                token,
                call.key,
                call.call.seq,
                result.seq + 1,
                self.timelines,
                self.lineage,
                indexes,
            )
            rarity_evidence = frequency.rarity_evidence(token)
            rarity = rarity_evidence.value if not rarity_evidence.reason else None
            if finding.reuse not in CREDITED:
                if finding.reuse is Reuse.NONE_OBSERVED and shown & set(exposed.refs):
                    finding = Finding(Reuse.INSPECTED, outcome="inspected")
                self.out.counts["candidates_unreferenced"] += 1
                best = max(best, finding.reuse, key=STRENGTH.__getitem__)
                continue
            best = max(best, finding.reuse, key=STRENGTH.__getitem__)
            scenarios = eligible_scenarios(rarity)
            reasons: list[str] = []
            eligibility = Eligibility.SUPPORTED
            if (
                nov.reason
                or finding.reuse is Reuse.CONTRADICTED
                or rarity is not None
                and not scenarios
            ):
                eligibility = Eligibility.KNOWN_REJECTED
                reasons.append(
                    nov.reason
                    or (
                        "contradicted"
                        if finding.reuse is Reuse.CONTRADICTED
                        else "rarity_below_threshold"
                    )
                )
            else:
                if rarity is None:
                    reasons.append("rarity_unknown")
                    if rarity_evidence.reason:
                        reasons.append(rarity_evidence.reason)
                    if frequency.unavailable_reason:
                        reasons.append(frequency.unavailable_reason)
                if nov.confidence == "reduced":
                    reasons.extend(("prior_context_unavailable", *nov.limits))
                if reasons:
                    eligibility = Eligibility.UNAVAILABLE
            for reason in reasons:
                self._excluded(call, token, reason, finding)
            self._emit(
                call,
                token,
                exposed.refs,
                day,
                executions,
                finding,
                rarity,
                scenarios,
                nov,
                eligibility,
                tuple(sorted(set(reasons))),
                frequency,
                rarity_evidence,
            )
        self.out.reuse[call.invocation_id] = best

    def _emit(
        self,
        call,
        token,
        refs,
        day,
        executions,
        finding,
        rarity,
        scenarios,
        nov,
        eligibility,
        reasons,
        frequency,
        rarity_evidence,
    ) -> None:
        consumer = Consumer(call.task.root, call.key, call.call.seq, call.ts)
        res = (
            self.resolver.resolve(token, refs, consumer)
            if eligibility is Eligibility.SUPPORTED
            else Resolution()
        )
        action = finding.action
        action_locators = (f"{action.ev.source_id}@{action.ev.locator}",) if action else ()
        candidate = Candidate(
            candidate_id=candidate_id(token, self.hmac_key),
            candidate_type=token.kind,
            task=call.task,
            day=day,
            reuse=finding.reuse,
            novel=nov.reason is None,
            user_supplied=res.source_user_supplied,
            action_outcome_ok=finding.outcome_ok,
            immediate=res.immediate,
            discovery=res.discovery,
            episode=res.episode,
            explicit_relay=finding.relay,
            report_reference=False,
            exposure_ids=(call.invocation_id,),
            execution_ids=tuple(sorted(executions)),
            witness_locators=res.locators,
            action_locators=action_locators,
            eligibility=eligibility,
            eligibility_reasons=reasons,
            rarity_lower_bound_milli=(
                rarity_evidence.value
                if rarity is not None or rarity_evidence.kind == "lower_bound"
                else None
            ),
            provenance=(
                *(
                    self.frequencies.provenance(call.invocation_id)
                    if self.frequencies
                    else (("frequency", frequency.version_id),)
                ),
                ("normalization", NORMALIZATION),
                ("novelty", nov.confidence),
                ("rarity_evidence_kind", rarity_evidence.kind),
                ("rarity_df_upper_bound", str(rarity_evidence.upper_df or "")),
            ),
        )
        # Per-scenario eligibility is applied by ``valuation_inputs.rarity_proven``; atoms are
        # never pruned here, or the central/optimistic bands would lose the immediate tier.
        gated = candidate
        for reason, source in res.reasons:
            self.out.exclusions.append(
                Exclusion(candidate.candidate_id, call.invocation_id, reason, source)
            )
            self.out.counts[f"exclusion.{reason}"] += 1
        self.out.counts[f"eligibility.{eligibility.value}"] += 1
        self.out.candidates.append(
            CandidateRecord(
                gated,
                call.invocation_id,
                token.kind,
                nov.confidence,
                nov.limits,
                rarity,
                outcome=finding.outcome,
                rarity_evidence_kind=rarity_evidence.kind,
                frequency_view_id=(
                    dict(self.frequencies.provenance(call.invocation_id))["frequency"]
                    if self.frequencies
                    else frequency.version_id
                ),
                frequency_provenance=(
                    self.frequencies.provenance(call.invocation_id) if self.frequencies else ()
                ),
            )
        )
        self.out.edges.append(Edge(call.task.root, res.source_roots, res.witness_ts, call.ts))

    def _excluded(self, call: Call, token: Token, reason: str, finding: Finding) -> None:
        self.out.counts[f"exclusion.{reason}"] += 1
        if finding.reuse in CREDITED:
            self.out.exclusions.append(
                Exclusion(candidate_id(token, self.hmac_key), call.invocation_id, reason)
            )

    def _cycles(self) -> None:
        for index in sorted(find_cycles(self.out.edges)):
            record = self.out.candidates[index]
            cand = record.candidate
            marked = replace(
                cand,
                immediate=mark_cycle(cand.immediate),
                discovery=mark_cycle(cand.discovery),
                episode=mark_cycle(cand.episode),
            )
            self.out.candidates[index] = replace(record, candidate=marked)
            self.out.exclusions.append(
                Exclusion(cand.candidate_id, record.invocation_id, "provenance_cycle")
            )
            self.out.counts["exclusion.provenance_cycle"] += 1


def _command(call: Call) -> str:
    from ssgrep.accounting.readers.timeline import call_text

    return call_text(call.call.ev.payload)


def _later_shows(calls: list[Call]) -> dict[str, frozenset[str]]:
    """Per invocation: refs later opened by a ``show`` in the same session (inspection)."""
    by_session: dict[SessionKey, list[Call]] = {}
    for call in calls:
        by_session.setdefault(call.key, []).append(call)
    out: dict[str, frozenset[str]] = {}
    for items in by_session.values():
        items.sort(key=lambda c: c.call.seq)
        later: set[str] = set()
        for call in reversed(items):
            out[call.invocation_id] = frozenset(later)
            later |= set(shown_refs(call))
    return out


__all__ = ["CHARS", "WORK", "Builder", "Outputs", "key_str", "Interpretation", "Basis"]


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
        self.line = line
        self.conflicted_at: int | None = None
        self.conflict_since: int | None = None
        self.conflict_time_unknown = False
        for record in line.records:
            if line.is_conflicted(record):
                if self.conflicted_at is None:
                    self.conflicted_at = record.seq
                for stamp in line.conflict_times[record.identity]:
                    if stamp is None:
                        self.conflict_time_unknown = True
                    elif self.conflict_since is None or stamp < self.conflict_since:
                        self.conflict_since = stamp
                continue
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

    def uncertain_before(self, upto: int) -> bool:
        """Any conflicting body that cannot be proved outside this context window."""
        if self.conflicted_at is None:
            return False
        if self.conflicted_at < upto or self.conflict_time_unknown:
            return True
        boundary = self.line.records[min(upto, len(self.line.records)) - 1].ts if upto > 0 else None
        return boundary is None or self.conflict_since is None or self.conflict_since <= boundary


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
        if index.uncertain_before(upto):
            reasons.append("conflicted_prior_context")
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


@dataclass(frozen=True)
class Edge:
    consumer_root: str
    source_roots: tuple[str, ...]
    witness_ts: int | None
    exposure_ts: int | None


def find_cycles(edges: list[Edge]) -> set[int]:
    """Indexes of edges whose source content derives from the consumer root's own work.

    ``C`` reusing content ``w`` of root ``R`` is cyclic when ``R`` had itself absorbed
    content of ``C`` (transitively) before it produced ``w``.
    """
    by_consumer: dict[str, list[Edge]] = {}
    for edge in edges:
        by_consumer.setdefault(edge.consumer_root, []).append(edge)

    def absorbed(root: str, before: int | None, seen: frozenset[str]) -> set[str]:
        out: set[str] = set()
        if root in seen:
            return out
        for edge in by_consumer.get(root, ()):
            if before is not None and edge.exposure_ts is not None and edge.exposure_ts >= before:
                continue
            if before is not None and edge.exposure_ts is None:
                continue
            for source in edge.source_roots:
                if source == root:
                    continue
                out.add(source)
                out |= absorbed(source, edge.witness_ts, seen | {root})
        return out

    cyclic: set[int] = set()
    for index, edge in enumerate(edges):
        if edge.witness_ts is None:
            continue
        for source in edge.source_roots:
            if source != edge.consumer_root and edge.consumer_root in absorbed(
                source, edge.witness_ts, frozenset()
            ):
                cyclic.add(index)
    return cyclic


def mark_cycle(atoms: tuple[ValuationAtom, ...]) -> tuple[ValuationAtom, ...]:
    return tuple(
        replace(a, status=AtomStatus.CYCLE) if a.status is AtomStatus.OK else a for a in atoms
    )
