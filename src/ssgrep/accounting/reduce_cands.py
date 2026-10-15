"""Candidates, exposures, and debits from recognized lookups (tasks 6.1-6.5, 5.4).

Turns every in-scope solver lookup into (a) one exposure debit per invocation
-- repeated lookups cost every time -- and (b) candidate reuse records for facts
that were novel, rare enough, and later referenced. Rarity gating is applied
here, by shrinking a candidate's atom tiers: rarity >= 8 bits keeps all tiers,
>= 6 bits drops the immediate tier (central + optimistic only), >= 4 bits keeps the
episode tier only. Unavailable rarity/prior-context evidence is retained as
missing credit, not rejected zero. Thresholds are UNCALIBRATED.
"""

from __future__ import annotations

import hmac
from collections import Counter
from dataclasses import dataclass, field, replace

from ssgrep.accounting.contracts import Basis, Quantity, Reuse, Scenario
from ssgrep.accounting.evidence import (
    SessionIndex,
    TokenKeys,
    extract,
    novelty,
    purpose_of,
)
from ssgrep.accounting.evidence_tokens import NORMALIZATION, Token
from ssgrep.accounting.exposure import Interpretation
from ssgrep.accounting.frequency import FrequencyView, eligible_scenarios
from ssgrep.accounting.frequency_assignment import FrequencyAssignments
from ssgrep.accounting.lineage import Lineage
from ssgrep.accounting.reduce_calls import Call, shown_refs
from ssgrep.accounting.reduce_types import CandidateRecord, Exclusion
from ssgrep.accounting.reuse import ActionIndex, Finding, find_reuse
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.sources import Consumer, Edge, Resolution, Resolver, find_cycles, mark_cycle
from ssgrep.accounting.timeline import SessionKey, Timelines, day_of, key_str
from ssgrep.accounting.units import (
    CHARS_DIV4_V1,
    CONTEXT_TOKENS,
    WORK_SECONDS,
    UnitSpec,
    context_estimate,
)
from ssgrep.accounting.valuation_inputs import Candidate, Debit, Eligibility

CHARS = UnitSpec(CONTEXT_TOKENS, CHARS_DIV4_V1)
WORK = UnitSpec(WORK_SECONDS, "capped_gap_v1")
CANDIDATE_KEY = b"ssgrep-candidate-v1"
CREDITED = frozenset({Reuse.ACTION_REFERENCE, Reuse.VALIDATED, Reuse.CONTRADICTED})
STRENGTH = {
    Reuse.CONTRADICTED: 4,
    Reuse.VALIDATED: 3,
    Reuse.ACTION_REFERENCE: 2,
    Reuse.INSPECTED: 1,
    Reuse.NONE_OBSERVED: 0,
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


def _tiers(candidate: Candidate, scenarios: tuple[Scenario, ...]) -> Candidate:
    if Scenario.CONSERVATIVE in scenarios:
        return candidate
    if Scenario.CENTRAL in scenarios:
        return replace(candidate, immediate=())
    assert Scenario.OPTIMISTIC in scenarios
    return replace(candidate, immediate=(), discovery=())


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
                rarity_evidence.value if rarity_evidence.kind == "lower_bound" else None
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
        gated = _tiers(candidate, scenarios) if eligibility is Eligibility.SUPPORTED else candidate
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
    from ssgrep.accounting.timeline import call_text

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
