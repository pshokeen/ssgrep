"""Preserve accepted measurements when an exact historical replay input disappears."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ssgrep.accounting.collector_ledger import VersionedLedger, candidate_identity
from ssgrep.accounting.collector_state import ReductionDoc
from ssgrep.accounting.reduce_types import Reduction
from ssgrep.accounting.valuation_inputs import EvidenceBundle


def preserve_unavailable(
    ledger: VersionedLedger,
    reduction: Reduction,
    previous: ReductionDoc | None,
    unavailable_sessions: Mapping[tuple[str, str], str],
    unavailable_frequencies: Mapping[str, str],
) -> tuple[Reduction, dict[str, EvidenceBundle]]:
    """Never turn a missing replay input into deletion/zero of an accepted lookup.

    Existing affected measurements stay frozen; independent new lookups in the
    same root still enter. Source actor identities identify accepted source atoms
    even when the consumer's own transcript remains readable. This is retention
    coverage, not permission to import missing historical usage or changed text.
    """
    desired = dict(zip(reduction.bundle_roots, reduction.bundles, strict=True))
    if not unavailable_sessions and not unavailable_frequencies:
        return reduction, desired
    old_invocations = previous.invocations if previous else ()
    blocked = set(unavailable_frequencies)
    blocked.update(
        row.invocation_id
        for row in old_invocations
        if (row.runtime, row.session_id) in unavailable_sessions
    )
    source_actors = {f"{runtime}:{session}" for runtime, session in unavailable_sessions}
    state = ledger.state()
    kept_candidates = [
        row
        for row in state.candidates
        if set(row.exposure_ids) & blocked
        or any(
            atom.actor in source_actors for atom in (*row.immediate, *row.discovery, *row.episode)
        )
    ]
    kept_candidates = [
        row for row in kept_candidates if ledger.superseded.get(_digest(row)) is None
    ]
    kept_debits = [
        row
        for row in state.debits
        if row.exposure_id in blocked and ledger.superseded.get(_digest(row)) is None
    ]
    candidate_keys = {candidate_identity(row) for row in kept_candidates}
    debit_keys = {row.debit_id for row in kept_debits}
    frozen_candidate_exposures = {
        exposure for row in kept_candidates for exposure in row.exposure_ids if exposure in blocked
    }
    roots = set(desired) | {row.task.root for row in (*kept_candidates, *kept_debits)}
    merged: dict[str, EvidenceBundle] = {}
    for root in roots:
        current = desired.get(root, EvidenceBundle())
        merged[root] = EvidenceBundle(
            tuple(
                row
                for row in current.candidates
                if candidate_identity(row) not in candidate_keys
                and not set(row.exposure_ids) & frozen_candidate_exposures
            )
            + tuple(row for row in kept_candidates if row.task.root == root),
            tuple(
                row
                for row in current.debits
                if row.debit_id not in debit_keys and row.exposure_id not in blocked
            )
            + tuple(row for row in kept_debits if row.task.root == root),
        )
    kept_invocations = [row for row in old_invocations if row.invocation_id in blocked]
    kept_executions = [
        row for row in (previous.executions if previous else ()) if row.invocation_id in blocked
    ]
    execution_ids = {row.execution_id for row in kept_executions}
    coverage = dict(reduction.coverage)
    coverage["replay_unavailable_sessions"] = len(unavailable_sessions)
    coverage["frequency_revision_unavailable"] = len(unavailable_frequencies)
    updated = replace(
        reduction,
        invocations=tuple(
            sorted(
                (
                    *[row for row in reduction.invocations if row.invocation_id not in blocked],
                    *kept_invocations,
                ),
                key=lambda row: row.invocation_id,
            )
        ),
        executions=tuple(
            sorted(
                (
                    *[row for row in reduction.executions if row.execution_id not in execution_ids],
                    *kept_executions,
                ),
                key=lambda row: (row.ts_utc, row.execution_id),
            )
        ),
        coverage=coverage,
        warnings=tuple(sorted(set(reduction.warnings) | {"historical_replay_unavailable"})),
    )
    return updated, merged


def _digest(row) -> str:
    from ssgrep.accounting.contracts import digest_of

    return digest_of(row)
