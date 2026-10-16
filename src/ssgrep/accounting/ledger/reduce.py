"""The combined deterministic reducer: raw evidence + journal -> :class:`Reduction`.

``reduce_profile`` is a pure function (no clocks, no I/O). Transcripts alone cannot
count loop executions and the journal alone cannot show reuse, so both are joined
(``reconcile``), tasks come from raw lineage (``lineage``), and exposure, reuse,
candidates and debits come from observed results only. Every output is sorted, so
shuffling the inputs yields an identical :attr:`Reduction.digest`.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.capture import fingerprint, journal_reader
from ssgrep.accounting.collect.frequency import FrequencyAssignments, FrequencyView, model_identity
from ssgrep.accounting.collect.replay_evidence import NORMALIZATION
from ssgrep.accounting.ledger.contracts import (
    THRESHOLD_MILLI,
    Attribution,
    ExecutionEvent,
    InvocationClass,
    RawEvidence,
)
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.ledger.reduce_cands import CHARS, WORK, Builder, Outputs
from ssgrep.accounting.ledger.reduce_join import (
    fold_events,
    join_index,
    reconcile,
)
from ssgrep.accounting.ledger.reduce_records import (
    Reduction,
    _executions,
    _invocations,
    attach_timing,
    collect_calls,
    summarize,
)
from ssgrep.accounting.ledger.valuation_inputs import (
    UNKNOWN_COST_STATUSES,
    AtomStatus,
    EvidenceBundle,
    Methodology,
)
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.privacy.policy import PolicyFile
from ssgrep.accounting.readers.base import EvidenceLedger, JsonlReader, ReadResult, Source
from ssgrep.accounting.readers.classify import CYCLE_BYTES
from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader, CodexReader
from ssgrep.accounting.readers.native_source import NativeSourceAdapter, NativeSourceReference
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.pi_family import OmpReader, PiReader, PrimeReader, build_lineage
from ssgrep.accounting.readers.timeline import build_timelines, key_str, ts_ms

if TYPE_CHECKING:
    from ssgrep.accounting.views.filter import Scoper


CONSUMED_KINDS = frozenset(
    {"tool_call", "tool_result", "user_turn", "session_header", "lineage", "ownership_claim"}
)


def methodology_for(frequency: FrequencyView) -> Methodology:
    """The production methodology namespace that bundles of this reduction belong to."""
    from ssgrep.accounting.views.lookups import LOOKUP_WORK_MAPPING

    return Methodology(
        label=f"production-{NORMALIZATION}-{LOOKUP_WORK_MAPPING}",
        units=(CHARS, WORK),
        thresholds=tuple((s.value, m // 1000) for s, m in THRESHOLD_MILLI.items()),
        frequency_view=model_identity(frequency),
        calibrated=False,
    )


def _bundles(outputs: Outputs) -> tuple[tuple[EvidenceBundle, ...], tuple[str, ...]]:
    candidates: dict[str, list] = {}
    debits: dict[str, list] = {}
    for record in outputs.candidates:
        candidates.setdefault(record.candidate.task.root, []).append(record.candidate)
    for debit in outputs.debits:
        debits.setdefault(debit.task.root, []).append(debit)
    roots = sorted(set(candidates) | set(debits))
    bundles = tuple(
        EvidenceBundle(
            tuple(
                sorted(
                    candidates.get(r, ()), key=lambda c: (c.task.turn or "", c.candidate_id, c.day)
                )
            ),
            tuple(sorted(debits.get(r, ()), key=lambda d: d.debit_id)),
        )
        for r in roots
    )
    return bundles, tuple(roots)


def _source_coverage(outputs: Outputs) -> tuple[int, int]:
    known = unavailable = 0
    for record in outputs.candidates:
        atoms = record.candidate.immediate + record.candidate.discovery + record.candidate.episode
        bad = any(a.status in UNKNOWN_COST_STATUSES or a.status is not AtomStatus.OK for a in atoms)
        if atoms and not bad:
            known += 1
        else:
            unavailable += 1
    return known, unavailable


def reduce_profile(
    profile: str,
    evidence: Iterable[RawEvidence],
    events: Iterable[ExecutionEvent],
    *,
    policy: ProfilePolicy,
    frequency: FrequencyView,
    fingerprint_key: bytes | None = None,
    verifier_sessions: frozenset[str] = frozenset(),
    known_atoms: Mapping[str, str] | None = None,
    ownership: PolicyFile | None = None,
    native_sources: NativeSourceAdapter | None = None,
    frequency_assignments: FrequencyAssignments | None = None,
    capture_span: tuple[int, int] | None = None,
    partitioned: bool = False,
) -> Reduction:
    """Reduce retained evidence and journal events to one deterministic :class:`Reduction`.

    ``verifier_sessions`` holds ``"runtime:session"`` ids (or roots) of verifier actors:
    their lookups stay in the execution denominators but never become solver exposure,
    debit or credit. ``known_atoms`` maps origin event identity to the digest recorded
    earlier, so source drift is detected and earns no credit.
    """
    from ssgrep.accounting.views.filter import Scoper
    from ssgrep.accounting.views.lookups import journal_timings

    timelines = build_timelines(item for item in evidence if item.kind in CONSUMED_KINDS)
    lineage = build_lineage(timelines)
    scoper = Scoper(policy, ownership)
    if native_sources is not None:
        native_sources.scoper = scoper
    callset = collect_calls(timelines, lineage, scoper, fingerprint_key, verifier_sessions)
    mine = [e for e in events if e.profile == profile]
    facts = fold_events(mine)
    requests = [r for call in callset.calls for r in call.requests]
    links = reconcile(facts, requests)
    joined = join_index(links)
    timings = journal_timings(mine)
    callset.calls = attach_timing(callset.calls, timings, joined)
    if frequency_assignments is None:
        from ssgrep.accounting.capture.fingerprint import key_id

        frequencies = FrequencyAssignments(frequency, fingerprint_key_id=key_id(fingerprint_key))
    else:
        frequencies = frequency_assignments
    bound_executions: set[str] = set()
    for call in callset.calls:
        if call.in_scope and call.lookup:
            owned = {e for request in call.requests for e in joined.owned.get(request.key, [])}
            frequencies.bind(call.invocation_id, owned)
            bound_executions.update(owned)
    for fact in facts:
        if (
            fact.operation in ("search", "show")
            and fact.execution_id not in bound_executions
            and scoper.in_scope(fact.cwd)
        ):
            frequencies.view_for(f"journal:{fact.execution_id}")
    calls_by_key = {r.key: call for call in callset.calls for r in call.requests}
    owned_by_invocation: dict[str, list[str]] = {}
    for call in callset.calls:
        owned_by_invocation[call.invocation_id] = sorted(
            {e for r in call.requests for e in joined.owned.get(r.key, [])}
        )
    lookup_ids: dict = {}
    for call in callset.calls:
        lookup_ids.setdefault(call.key, set()).add(call.call.ev.record_id)
    from ssgrep.accounting.collect.replay_evidence import Resolver

    resolver = Resolver(
        timelines, scoper, known_atoms or {}, lineage.root_of, native_sources=native_sources
    )
    builder = Builder(
        timelines, lineage, scoper, frequency, resolver, fingerprint_key,
        {k: frozenset(v) for k, v in lookup_ids.items()},
        frequencies=frequencies,
    )  # fmt: skip
    outputs = builder.run([c for c in callset.calls if c.in_scope], owned_by_invocation)
    executions, out_of_scope = _executions(
        facts, links, calls_by_key, joined.owned, scoper, outputs, timings
    )
    stamps = [t for t in (ts_ms(e.ts_utc) for e in mine) if t is not None]
    span = (
        capture_span if capture_span is not None else (min(stamps), max(stamps)) if stamps else None
    )
    invocations = _invocations(
        callset.calls, joined.owned, joined.ambiguous, links, span, outputs, executions, frequencies
    )
    bundles, bundle_roots = _bundles(outputs)
    coverage = _coverage(
        executions, invocations, callset.counts, outputs, lineage, timelines, out_of_scope, callset
    )
    sessions = tuple(sorted((k[0], k[1], key_str(lineage.root_of[k])) for k in timelines.sessions))
    parents = tuple(
        sorted(
            (k[0], k[1], lineage.parent_of[k][1] if k in lineage.parent_of else None)
            for k in timelines.sessions
        )
    )
    return Reduction(
        profile=profile,
        executions=tuple(executions),
        invocations=tuple(invocations),
        candidates=tuple(sorted(outputs.candidates, key=_candidate_order)),
        exclusions=tuple(
            sorted(
                set(outputs.exclusions),
                key=lambda x: (x.candidate_id, x.invocation_id, x.reason, x.detail),
            )
        ),
        bundles=bundles,
        bundle_roots=bundle_roots,
        session_roots=sessions,
        session_parents=parents,
        coverage=coverage,
        observations=summarize(executions),
        warnings=tuple(sorted(f"lineage:{i.kind}:{i.session}" for i in lineage.issues)),
        frequency_version=model_identity(frequency),
        frequency_assignments=frequencies.records(
            [
                *(call.invocation_id for call in callset.calls),
                *(f"journal:{fact.execution_id}" for fact in facts),
            ]
            if partitioned
            else None
        ),
    )


def _candidate_order(record) -> tuple:
    c = record.candidate
    return (c.task.root, c.task.turn or "", c.candidate_id, c.day, record.invocation_id)


def _coverage(
    executions, invocations, counts, outputs, lineage, timelines, out_of_scope, callset
) -> dict[str, int]:
    cov: Counter[str] = Counter()
    cov["recorded_executions"] = len(executions)
    for attribution in Attribution:
        cov[attribution.value] = sum(1 for e in executions if e.attribution is attribution)
    lookups = [e for e in executions if e.operation in ("search", "show")]
    cov["recorded_lookups"] = len(lookups)
    cov["attributed_lookups"] = sum(
        1 for e in lookups if e.attribution in (Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE)
    )
    cov["ambiguous_joins"] = cov[Attribution.AMBIGUOUS.value]
    cov["recognized_invocations"] = len(invocations)
    for cls in InvocationClass:
        cov[f"inv_{cls.value}"] = sum(1 for i in invocations if i.cls is cls)
    cov["inv_mention"] = counts["mention"]
    cov["inv_unclassified"] = counts["unclassified"]
    cov["classified_tool_calls"] = (
        counts["mention"] + counts["unclassified"] + counts["non_lookup"] + len(invocations)
    )
    cov["tasks_resolved"] = sum(1 for i in invocations if i.task is not None and i.task.resolved)
    cov["tasks_unknown"] = len(invocations) - cov["tasks_resolved"]
    cov["candidate_receipts"] = len(outputs.candidates)
    known, unavailable = _source_coverage(outputs)
    cov["source_costs_available"] = known
    cov["unavailable_sources"] = unavailable
    cov["executions_out_of_scope"] = out_of_scope
    cov["invocations_out_of_scope"] = callset.out_of_scope
    cov["evidence_duplicates"] = timelines.duplicates
    cov["evidence_conflicts"] = timelines.conflicts
    cov["sessions"] = len(timelines.sessions)
    for issue in lineage.issues:
        cov[f"lineage.{issue.kind}"] += 1
    for name, value in outputs.counts.items():
        cov[name] += value
    return dict(sorted(cov.items()))


@dataclass
class Loaded:
    """Everything one profile reduction needs, plus honest read coverage."""

    evidence: list[RawEvidence] = field(default_factory=list)
    events: list[ExecutionEvent] = field(default_factory=list)
    sources_read: dict[str, int] = field(default_factory=dict)
    gaps: dict[str, int] = field(default_factory=dict)
    unreadable_sources: int = 0
    pending_bytes: int = 0
    journal_damage: int = 0
    journal_segments: int = 0
    duplicates: int = 0
    conflicts: int = 0


def readers(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    return {
        "claude": ClaudeReader(env),
        "codex": CodexReader(env),
        "omp": OmpReader(env),
        "opencode": OpenCodeReader(env),
        "pi": PiReader(env),
        "prime-agent": PrimeReader(env),
    }


def _under(path: Path, bases: tuple[Path, ...]) -> bool:
    real = Path(os.path.realpath(path))
    return any(real == base or base in real.parents for base in bases)


def _read_source(reader: JsonlReader | OpenCodeReader, source: Source) -> Iterable[ReadResult]:
    state = None
    while True:
        budget = None if isinstance(reader, OpenCodeReader) else CYCLE_BYTES
        result = reader.read(source, state, budget)
        yield result
        state = result.state
        if not result.more:
            return


def load_evidence(
    env: Mapping[str, str] | None = None,
    *,
    only_under: Iterable[Path] | None = None,
    include: Callable[[Source], bool] | None = None,
    kinds: frozenset[str] = CONSUMED_KINDS,
) -> Loaded:
    """Raw evidence from every runtime store; identical native copies count once."""
    bases = tuple(Path(os.path.realpath(p)) for p in only_under) if only_under else None
    loaded = Loaded()
    ledger = EvidenceLedger()
    for name, reader in readers(env).items():
        count = 0
        for source in reader.discover():
            if bases is not None and not _under(source.path, bases):
                continue
            if include is not None and not include(source):
                continue
            count += 1
            try:
                for result in _read_source(reader, source):
                    for gap in result.gaps:
                        loaded.gaps[gap.reason] = loaded.gaps.get(gap.reason, 0) + 1
                    loaded.pending_bytes += result.pending_bytes
                    loaded.evidence.extend(
                        e for e in result.evidence if e.kind in kinds and ledger.admit(e)
                    )
            except OSError:
                loaded.unreadable_sources += 1
        loaded.sources_read[name] = count
    loaded.duplicates = ledger.duplicates
    loaded.conflicts = ledger.conflict_count
    return loaded


def load_journal(profile: str) -> tuple[list[ExecutionEvent], int, int]:
    """Accepted journal events, damaged-record count, and segment count (read-only)."""
    read = journal_reader.read_journal(profile, quarantine=False)
    return list(read.events), len(read.damage), read.segments


def native_source_adapter(
    profile_policy: ProfilePolicy,
    env: Mapping[str, str] | None = None,
    *,
    now_ms: int,
    references: Mapping[str, NativeSourceReference] | None = None,
    repository: Any = None,
    ownership: PolicyFile | None = None,
    scoper: Scoper | None = None,
) -> NativeSourceAdapter:
    """An on-demand source view with the full workspace ownership authority.

    Default Scoper reads current policy including explicit assignments; callers
    can supply the reducer's exact snapshot/instance. No source discovery scan.
    """
    from ssgrep.accounting.readers import classify as roots
    from ssgrep.accounting.views.filter import Scoper

    native_readers = readers(env)
    approved = tuple(
        root
        for reader in native_readers.values()
        if isinstance(reader, JsonlReader)
        for root in reader.roots()
    ) + (roots.opencode_db(env).parent,)
    return NativeSourceAdapter(
        references or {},
        native_readers,
        approved,
        scoper if scoper is not None else Scoper(profile_policy, ownership),
        now_ms,
        repository=repository,
    )


def reduce_stored_profile(
    profile: str,
    env: Mapping[str, str] | None = None,
    *,
    only_under: Iterable[Path] | None = None,
    frequency: FrequencyView | None = None,
    verifier_sessions: frozenset[str] = frozenset(),
    loaded: Loaded | None = None,
    native_sources: NativeSourceAdapter | None = None,
    source_now_ms: int | None = None,
    ownership: PolicyFile | None = None,
) -> tuple[Reduction, Loaded]:
    """Reduce one profile from the real stores; reads only, never writes."""
    data = loaded or load_evidence(env, only_under=only_under)
    if loaded is None:
        events, damage, segments = load_journal(profile)
        data.events, data.journal_damage, data.journal_segments = events, damage, segments
    if frequency is None:
        from ssgrep.accounting.collect.frequency_index import read_index_view

        view = read_index_view(profile)
    else:
        view = frequency
    profile_policy = policy.load_profile_policy(profile)
    if native_sources is None:
        from datetime import UTC, datetime

        now_ms = (
            source_now_ms
            if source_now_ms is not None
            else int(datetime.now(UTC).timestamp() * 1000)
        )
        native_sources = native_source_adapter(
            profile_policy, env, now_ms=now_ms, ownership=ownership
        )
    if ownership is None:
        ownership = native_sources.scoper.ownership
    key = fingerprint.load_key(profile, create=False)
    reduction = reduce_profile(
        profile,
        data.evidence,
        data.events,
        policy=profile_policy,
        ownership=ownership,
        frequency=view,
        fingerprint_key=key,
        verifier_sessions=verifier_sessions,
        native_sources=native_sources,
    )
    return reduction, data
