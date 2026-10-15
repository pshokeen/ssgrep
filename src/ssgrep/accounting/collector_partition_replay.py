"""Keep accepted canonical rows while admitting independent new supported lookups."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_ledger import candidate_identity
from ssgrep.accounting.collector_replay import preserve_unavailable
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import Attribution, InvocationClass
from ssgrep.accounting.plain_codec import from_plain
from ssgrep.accounting.reduce_types import Reduction, summarize
from ssgrep.accounting.valuation_inputs import AtomStatus, Eligibility, EvidenceBundle


def preserve(ledger, reduction, previous, unavailable, frequencies):
    from ssgrep.accounting.collector_steps import reduction_doc

    old_doc = reduction_doc(previous, {}) if previous else None
    relevant = {row.invocation_id for row in reduction.invocations}
    if previous:
        relevant.update(row.invocation_id for row in previous.invocations)
    relevant.update("journal:" + row.execution_id for row in reduction.executions)
    missing_frequencies = {
        ident: why for ident, why in frequencies.unavailable.items() if ident in relevant
    }
    reduction, desired = preserve_unavailable(
        ledger, reduction, old_doc, unavailable, missing_frequencies
    )
    blocked = set(missing_frequencies)
    actors = {runtime + ":" + session for runtime, session in unavailable}
    if previous:
        blocked.update(
            row.invocation_id
            for row in previous.invocations
            if (row.runtime, row.session_id) in unavailable
        )
    kept_candidates = [
        row
        for row in (previous.candidates if previous else ())
        if row.invocation_id in blocked
        or any(
            atom.actor in actors
            for atom in (*row.candidate.immediate, *row.candidate.discovery, *row.candidate.episode)
        )
    ]
    identities = {candidate_identity(row.candidate) for row in kept_candidates}
    candidates = [
        row for row in reduction.candidates if candidate_identity(row.candidate) not in identities
    ]
    candidates.extend(kept_candidates)
    old_invocations = {row.invocation_id for row in previous.invocations} if previous else set()
    missing_context = {
        row.invocation_id
        for row in reduction.invocations
        if (row.runtime, row.session_id) in unavailable and row.invocation_id not in old_invocations
    }
    updated_candidates = {}
    for index, row in enumerate(candidates):
        if row.invocation_id in missing_context:
            reason = "prior_context_replay_unavailable"
            candidate = replace(
                row.candidate,
                eligibility=Eligibility.UNAVAILABLE,
                eligibility_reasons=tuple(
                    sorted(set(row.candidate.eligibility_reasons) | {reason})
                ),
            )
            candidates[index] = replace(
                row,
                candidate=candidate,
                novelty_confidence="reduced",
                novelty_reasons=tuple(sorted(set(row.novelty_reasons) | {reason})),
            )
            updated_candidates[candidate_identity(candidate)] = candidate
    if updated_candidates:
        desired = {
            root: EvidenceBundle(
                tuple(
                    updated_candidates.get(candidate_identity(row), row)
                    for row in bundle.candidates
                ),
                bundle.debits,
            )
            for root, bundle in desired.items()
        }
    # Ledger accepted stamps are bookkeeping, not frozen reducer input content.
    pure = {
        root: EvidenceBundle(
            tuple(replace(row, accepted_day="") for row in bundle.candidates),
            tuple(replace(row, accepted_day="") for row in bundle.debits),
        )
        for root, bundle in desired.items()
    }
    coverage = dict(reduction.coverage)
    for name in (
        "sessions",
        "classified_tool_calls",
        "inv_mention",
        "inv_unclassified",
        "evidence_duplicates",
        "evidence_conflicts",
    ):
        coverage[name] = max(
            coverage.get(name, 0), previous.coverage.get(name, 0) if previous else 0
        )
    coverage["recorded_executions"] = len(reduction.executions)
    coverage["recorded_lookups"] = sum(
        row.operation in {"search", "show"} for row in reduction.executions
    )
    for attribution in Attribution:
        coverage[attribution.value] = sum(
            row.attribution is attribution for row in reduction.executions
        )
    coverage["attributed_lookups"] = sum(
        row.operation in {"search", "show"}
        and row.attribution in {Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE}
        for row in reduction.executions
    )
    coverage["ambiguous_joins"] = coverage[Attribution.AMBIGUOUS.value]
    coverage["recognized_invocations"] = len(reduction.invocations)
    for cls in InvocationClass:
        coverage["inv_" + cls.value] = sum(row.cls is cls for row in reduction.invocations)
    coverage["tasks_resolved"] = sum(
        row.task is not None and row.task.resolved for row in reduction.invocations
    )
    coverage["tasks_unknown"] = len(reduction.invocations) - coverage["tasks_resolved"]
    all_candidates = [row for bundle in pure.values() for row in bundle.candidates]
    coverage["candidate_receipts"] = len(all_candidates)
    available = 0
    for row in all_candidates:
        atoms = (*row.immediate, *row.discovery, *row.episode)
        available += bool(atoms) and all(atom.status is AtomStatus.OK for atom in atoms)
    coverage["source_costs_available"] = available
    coverage["unavailable_sources"] = len(all_candidates) - available
    return replace(
        reduction,
        candidates=tuple(
            sorted(
                candidates,
                key=lambda row: (
                    row.candidate.task.root,
                    row.candidate.task.turn or "",
                    row.candidate.candidate_id,
                    row.candidate.day,
                    row.invocation_id,
                ),
            )
        ),
        bundles=tuple(pure[root] for root in sorted(pure)),
        bundle_roots=tuple(sorted(pure)),
        coverage=dict(sorted(coverage.items())),
        observations=summarize(reduction.executions),
    ), desired


def combine(profile: str, parts: dict[str, dict], frequency_assignments) -> Reduction:
    reductions = [
        from_plain(Reduction, store.read_blob(profile, part["blob"]))
        for _, part in sorted(parts.items())
    ]
    coverage: Counter = Counter()
    bundles = {}
    for reduction in reductions:
        coverage.update(reduction.coverage)
        for root, bundle in zip(reduction.bundle_roots, reduction.bundles, strict=True):
            if root in bundles:
                # A late split can retain an old unavailable component. Never sum its
                # accepted rows with a replacement claiming the same root.
                raise CollectError("state_corrupt", "Collector partition roots overlap.")
            bundles[root] = bundle

    def collect(field):
        return [row for reduction in reductions for row in getattr(reduction, field)]

    executions = tuple(
        sorted(collect("executions"), key=lambda row: (row.ts_utc, row.execution_id))
    )
    invocations = tuple(
        sorted(
            collect("invocations"), key=lambda row: (row.runtime, row.session_id, row.invocation_id)
        )
    )
    candidates = tuple(
        sorted(
            collect("candidates"),
            key=lambda row: (
                row.candidate.task.root,
                row.candidate.task.turn or "",
                row.candidate.candidate_id,
                row.candidate.day,
                row.invocation_id,
            ),
        )
    )
    return Reduction(
        profile,
        executions=executions,
        invocations=invocations,
        candidates=candidates,
        exclusions=tuple(
            sorted(
                set(collect("exclusions")),
                key=lambda row: (row.candidate_id, row.invocation_id, row.reason, row.detail),
            )
        ),
        bundles=tuple(bundles[root] for root in sorted(bundles)),
        bundle_roots=tuple(sorted(bundles)),
        session_roots=tuple(sorted(collect("session_roots"))),
        session_parents=tuple(sorted(collect("session_parents"))),
        coverage=dict(sorted(coverage.items())),
        observations=summarize(executions),
        warnings=tuple(sorted(set(collect("warnings")))),
        frequency_version=reductions[0].frequency_version if reductions else "",
        frequency_assignments=frequency_assignments.records(),
    )
