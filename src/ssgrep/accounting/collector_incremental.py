"""Small cycle integration helpers for partition queues and immutable state reuse."""

from __future__ import annotations

from dataclasses import replace

from ssgrep.accounting.collector_state import ReductionDoc
from ssgrep.accounting.contracts import Attribution, InvocationClass, digest_of


def source_proof(record):
    return (
        record.source_id,
        record.size,
        record.mtime_ns,
        record.status,
        record.resets,
        digest_of(
            [
                record.state.get("extra", {}).get("collector_source_signature"),
                record.state.get("extra", {}).get("collector_source_generation"),
            ]
        ),
    )


def changed_sources(previous, sources):
    return [record for key, record in sources.items() if previous.get(key) != source_proof(record)]


def pending_document(
    previous: ReductionDoc | None, current: ReductionDoc, partitions
) -> ReductionDoc:
    """A cache bootstrap/purge queue never discards accepted unvisited window rows."""
    if previous is None or (
        not partitions.pending and not partitions.pending_projections and not partitions.unavailable
    ):
        return current
    represented = {member for part in partitions.parts.values() for member in part["members"]}
    sessions = {tuple(name.split(":", 1)) for name in represented if not name.startswith("@")}
    execution_ids = {
        name[len("@journal:") :] for name in represented if name.startswith("@journal:")
    }
    executions = {
        row.execution_id: row
        for row in previous.executions
        if row.execution_id not in execution_ids and (row.runtime, row.session_id) not in sessions
    }
    executions.update((row.execution_id, row) for row in current.executions)
    invocations = {
        row.invocation_id: row
        for row in previous.invocations
        if (row.runtime, row.session_id) not in sessions
    }
    invocations.update((row.invocation_id, row) for row in current.invocations)

    def pairs(field):
        kept = {
            (row[0], row[1]): row
            for row in getattr(previous, field)
            if (row[0], row[1]) not in sessions
        }
        kept.update(((row[0], row[1]), row) for row in getattr(current, field))
        return tuple(kept[key] for key in sorted(kept))

    coverage = dict(current.coverage)
    coverage["recorded_executions"] = len(executions)
    coverage["recorded_lookups"] = sum(
        row.operation in {"search", "show"} for row in executions.values()
    )
    coverage["recognized_invocations"] = len(invocations)
    for attribution in Attribution:
        coverage[attribution.value] = sum(
            row.attribution is attribution for row in executions.values()
        )
    for cls in InvocationClass:
        coverage["inv_" + cls.value] = sum(row.cls is cls for row in invocations.values())
    coverage["ambiguous_joins"] = coverage[Attribution.AMBIGUOUS.value]
    coverage["attributed_lookups"] = sum(
        row.operation in {"search", "show"}
        and row.attribution in {Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE}
        for row in executions.values()
    )
    coverage["tasks_resolved"] = sum(
        row.task is not None and row.task.resolved for row in invocations.values()
    )
    coverage["tasks_unknown"] = len(invocations) - coverage["tasks_resolved"]
    return replace(
        current,
        executions=tuple(
            sorted(executions.values(), key=lambda row: (row.ts_utc, row.execution_id))
        ),
        invocations=tuple(
            sorted(
                invocations.values(),
                key=lambda row: (row.runtime, row.session_id, row.invocation_id),
            )
        ),
        coverage=coverage,
        session_roots=pairs("session_roots"),
        session_parents=pairs("session_parents"),
        session_cwd=pairs("session_cwd"),
        invocation_sessions=tuple(
            sorted((row.invocation_id, row.runtime, row.session_id) for row in invocations.values())
        ),
        warnings=tuple(sorted(set(current.warnings) | set(previous.warnings))),
    )


def partition_coverage(doc: ReductionDoc, state) -> ReductionDoc:
    coverage = dict(doc.coverage)
    coverage.update(
        reduction_pending_nodes=len(state.pending) + len(state.pending_projections),
        replay_unavailable_sessions=len(state.unavailable),
        reduction_cached_partitions=len(state.parts),
    )
    warnings = set(doc.warnings)
    if state.pending or state.pending_projections:
        warnings.add("reduction_backlog")
    else:
        warnings.discard("reduction_backlog")
    if state.unavailable:
        warnings.add("historical_replay_unavailable")
    else:
        warnings.discard("historical_replay_unavailable")
    return replace(doc, coverage=coverage, warnings=tuple(sorted(warnings)))
