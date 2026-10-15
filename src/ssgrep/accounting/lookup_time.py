"""Unique journal service latency and a separately versioned active-work proxy.

``journal_active_proxy_ns_v2`` maps each raw monotonic duration to modeled actor
occupancy, capped per operation at 300 seconds BEFORE any rounding. Raw ns
survive unique-operation sums; service milliseconds and whole work-seconds are
derived once. Source native-ms gaps retain their own observation precision.
This is not evidence of actual actor work, human waiting, critical-path savings
or a causal effect of retrieval. No later
transcript event, user prompt or provider timing is used to fill missing clocks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ssgrep.accounting.contracts import Basis, ExecutionEvent, Phase, Quantity, digest_of
from ssgrep.accounting.reconcile import Request
from ssgrep.accounting.timeunits import (
    CAPPED_GAP_V1,
    GAP_CAP_NS,
    MONOTONIC_V1,
    WORK_NS,
    measured_duration,
    ns_seconds_quantity,
    service_milliseconds,
)
from ssgrep.accounting.units import WORK_SECONDS

LOOKUP_WORK_MAPPING = "journal_active_proxy_ns_v2"
LOOKUP_WORK_ASSUMPTIONS = (
    "Each unique journal operation models one fully occupied active actor during service.",
    "Each modeled operation is capped at 300 seconds, matching capped_gap_v1 source work.",
    "Concurrent unique operations add actor work, not elapsed or human waiting time.",
    "Service occupancy is a proxy, not observed active work or causal time savings.",
    "Raw monotonic nanoseconds are capped per operation and summed before unit conversion.",
)


@dataclass(frozen=True)
class ExecutionTiming:
    """One unique execution: exact latency and its explicitly estimated debit."""

    execution_id: str
    measured_service_ns: Quantity
    work_ns: int | None

    @property
    def measured_service_ms(self) -> Quantity:
        return service_milliseconds(self.measured_service_ns)

    @property
    def work(self) -> Quantity:
        if self.work_ns is None:
            return Quantity.unavailable(WORK_SECONDS, self.measured_service_ns.reason)
        return ns_seconds_quantity(self.work_ns, CAPPED_GAP_V1, Basis.ESTIMATED)


@dataclass(frozen=True)
class LookupTiming:
    """Linked operation rows, plus coverage (partial timing never yields a total)."""

    rows: tuple[ExecutionTiming, ...] = ()
    complete: bool = False
    reason: str = "no_linked_journal_timing"
    mapping_version: str = LOOKUP_WORK_MAPPING
    assumptions: tuple[str, ...] = LOOKUP_WORK_ASSUMPTIONS

    @property
    def measured_service_ns(self) -> Quantity:
        if not self.complete:
            return Quantity.unavailable(WORK_NS, self.reason)
        for row in self.rows:
            if row.measured_service_ns.value is None:
                return Quantity.unavailable(WORK_NS, row.measured_service_ns.reason)
        return Quantity(
            sum(
                row.measured_service_ns.value
                for row in self.rows
                if row.measured_service_ns.value is not None
            ),
            WORK_NS,
            Basis.OBSERVED,
            MONOTONIC_V1,
        )

    @property
    def measured_service_ms(self) -> Quantity:
        return service_milliseconds(self.measured_service_ns)

    @property
    def work(self) -> Quantity:
        if not self.complete:
            return Quantity.unavailable(WORK_SECONDS, self.reason)
        if any(row.work_ns is None for row in self.rows):
            return Quantity.unavailable(WORK_SECONDS, "journal_monotonic_duration_missing")
        return ns_seconds_quantity(
            sum(row.work_ns for row in self.rows if row.work_ns is not None),
            CAPPED_GAP_V1,
            Basis.ESTIMATED,
        )


def _timing(execution_id: str, measured: Quantity) -> ExecutionTiming:
    if measured.value is None:
        return ExecutionTiming(execution_id, measured, None)
    return ExecutionTiming(execution_id, measured, min(measured.value, GAP_CAP_NS))


def journal_timings(events: Iterable[ExecutionEvent]) -> dict[str, ExecutionTiming]:
    """Deduplicate immutable phases; conflicting evidence never picks a duration.

    Capture's COMPLETED.mono_ns is an elapsed duration, not an absolute clock.
    Wall timestamps determine joins elsewhere; a wall discontinuity cannot change
    this independent monotonic measurement. A completed event suffices to retain
    its latency, even if the started/rendered phase is unavailable.
    """
    groups: dict[str, dict[Phase, dict[str, ExecutionEvent]]] = {}
    for event in events:
        phases = groups.setdefault(event.execution_id, {})
        phases.setdefault(event.phase, {})[digest_of(event)] = event
    out: dict[str, ExecutionTiming] = {}
    for execution_id, phases in sorted(groups.items()):
        done = phases.get(Phase.COMPLETED, {})
        reason = ""
        if any(len(variants) != 1 for variants in phases.values()):
            reason = "conflicting_journal_timing"
        elif not done:
            reason = "journal_monotonic_duration_missing"
        else:
            completed = next(iter(done.values()))
            identities = {
                (e.operation, e.transport, e.profile, e.fingerprint)
                for variants in phases.values()
                for e in variants.values()
            }
            if len(identities) != 1:
                reason = "conflicting_journal_timing"
            elif completed.mono_ns is None:
                reason = "journal_monotonic_duration_missing"
        if reason:
            measured = Quantity.unavailable(WORK_NS, reason)
        else:
            assert completed.mono_ns is not None
            measured = measured_duration(0, completed.mono_ns)
        out[execution_id] = _timing(execution_id, measured)
    return out


def link_call_timing(
    requests: Sequence[Request],
    owned: Mapping[tuple[str, int], Sequence[str]],
    ambiguous: Mapping[tuple[str, int], Sequence[str]],
    timings: Mapping[str, ExecutionTiming],
) -> LookupTiming:
    """Attach only uniquely owned journal rows; retain unknown invocation coverage.

    For a shell loop of unknown cardinality, observed rows remain supported but
    cannot prove that every operation was captured. Unknown/ambiguous requests
    therefore add an unavailable debit, rather than treating the known rows as a
    complete estimate. Notes do not contribute to lookup-work quantities.
    """
    execution_ids: set[str] = set()
    reasons: set[str] = set()
    lookups = [r for r in requests if r.operation in ("search", "show")]
    for request in lookups:
        ids = set(owned.get(request.key, ()))
        execution_ids.update(ids)
        if ambiguous.get(request.key):
            reasons.add("ambiguous_journal_timing")
        if request.capacity != 1:
            reasons.add("journal_cardinality_unbounded")
        elif len(ids) != 1:
            reasons.add("no_linked_journal_timing")
    rows = []
    for execution_id in sorted(execution_ids):
        row = timings.get(execution_id) or _timing(
            execution_id, Quantity.unavailable(WORK_NS, "journal_monotonic_duration_missing")
        )
        rows.append(row)
        if row.measured_service_ns.value is None:
            reasons.add(row.measured_service_ns.reason)
    if not lookups:
        reasons.add("no_lookup_request")
    complete = not reasons and bool(rows)
    return LookupTiming(tuple(rows), complete, "" if complete else sorted(reasons)[0])
