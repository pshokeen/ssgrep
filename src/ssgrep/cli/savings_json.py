"""JSON documents for ``savings``: integers and strings only, unknown is null + reason."""

from __future__ import annotations

from ssgrep.accounting.ledger.ledger_contracts import ScenarioEstimate
from ssgrep.accounting.privacy.storage import to_plain
from ssgrep.accounting.views.history import parse_estimate
from ssgrep.accounting.views.lookups import LookupsView
from ssgrep.accounting.views.service import STALE_AFTER_SECONDS
from ssgrep.accounting.views.types import (
    Condition,
    DeliveryLine,
    HistoryView,
    ReceiptView,
    ScenarioRow,
    SummaryView,
    quantity_json,
)
from ssgrep.cli.accounting_cli import envelope


def scenario_data(estimate: ScenarioEstimate) -> dict[str, object]:
    row = to_plain(estimate)
    for name in ("credit", "debit", "net"):
        row[name] = quantity_json(getattr(estimate, name))
    if estimate.bounds is not None:
        for name in (
            "credit_lower",
            "credit_upper",
            "debit_lower",
            "debit_upper",
            "net_lower",
            "net_upper",
        ):
            row["bounds"][name] = quantity_json(getattr(estimate.bounds, name))
    return row


def estimates_data(rows: tuple[ScenarioRow, ...]) -> dict[str, object]:
    out = {}
    for row in rows:
        estimates = {e.scenario: e for e in row.estimates}
        values = {}
        for (scenario, credit), (_, debit), (_, net) in zip(
            row.credit, row.debit, row.net, strict=True
        ):
            estimate = estimates.get(scenario) or parse_estimate(
                {"credit": to_plain(credit), "debit": to_plain(debit), "net": to_plain(net)},
                scenario,
                credit.unit,
            )
            values[scenario.value] = scenario_data(estimate)
        out[row.unit_key] = values
    return out


def condition_data(condition: Condition) -> dict[str, object]:
    action = condition.action
    return {
        "code": condition.code,
        "cause": condition.cause,
        "next_action": None if action is None else {"command": action.command, "why": action.why},
    }


def delivery_data(line: DeliveryLine) -> dict[str, object]:
    action = line.action
    return {
        "destination_id": line.destination_id,
        "signal": line.signal,
        "enabled": line.enabled,
        "state": line.state,
        "cause": line.cause,
        "pending_batches": line.pending_batches,
        "rejected_points": line.rejected_points,
        "unknown_batches": line.unknown_batches,
        "partial_batches": line.partial_batches,
        "lost_records": line.lost_records,
        "collected_generation": line.collected_generation,
        "attempted_generation": line.attempted_generation,
        "acked_generation": line.acked_generation,
        "last_acked_at": line.last_acked_at,
        "last_attempt_at": line.last_attempt_at,
        "next_retry_utc": line.next_retry_utc,
        "last_error_category": line.last_error_category,
        "blocked_reason": line.blocked_reason,
        "acked_age_seconds": line.acked_age_seconds,
        "attempt_age_seconds": line.attempt_age_seconds,
        "acknowledgment": "collector_acceptance_only",
        "next_action": None if action is None else {"command": action.command, "why": action.why},
    }


def summary_document(view: SummaryView) -> dict[str, object]:
    scope = view.scope
    data: dict[str, object] = {
        "profile": view.profile,
        "scope": {
            "window": None if (scope.since or scope.until) else scope.window,
            "since": scope.since,
            "until": scope.until,
            "interval": "utc_half_open",
            "project": scope.project,
            "runtime": scope.runtime,
            "session": scope.session,
            "task": scope.task,
            "by": scope.by,
            "include_children": scope.include_children,
        },
        "freshness": {
            "collected_at": view.collected_at,
            "age_seconds": view.age_seconds,
            "stale": view.stale,
            "stale_after_seconds": STALE_AFTER_SECONDS,
        },
        "paused": view.paused,
        "paused_at": view.paused_at,
        "collection_enabled": view.collection_enabled,
        "privacy": {
            "local_only": not view.exports,
            "exports_enabled": [{"destination": d, "signal": s} for d, s in view.exports],
        },
        "reason": view.reason,
        "measured": {name: quantity_json(q) for name, q in view.measured},
        "estimates": estimates_data(view.scenarios),
        "estimates_unavailable_reason": view.estimates_reason,
        "groups": None
        if view.scope.by is None
        else {
            "by": view.scope.by,
            "limit": view.limit,
            "total": view.group_total,
            "next_cursor": view.next_cursor,
            "rows": [
                {
                    "key": g.key,
                    "measured": {name: quantity_json(q) for name, q in g.measured},
                    "estimates": estimates_data(g.scenarios),
                    "receipts": g.receipts,
                }
                for g in view.groups
            ],
        },
        "conditions": [condition_data(c) for c in view.conditions],
        "delivery": [delivery_data(d) for d in view.delivery],
        "retention": {
            "evidence_expired_before": view.evidence_expired_before,
            "frozen_roots": view.frozen_roots,
        },
        "estimates_are_causal": False,
        "coverage": {
            **view.coverage,
            "partial": view.partial or view.replay_unavailable,
            "index": view.index,
        },
        "exclusions": [{"reason": r, "count": c} for r, c in view.exclusions],
        "representative_receipt": view.receipt_id,
    }
    provenance: dict[str, object] = {
        "snapshot_id": view.snapshot_id,
        "generation": view.generation,
        "methodology_id": view.methodology_id,
        "policy_version": view.policy_version,
        "coverage_through": view.coverage_through,
    }
    return envelope(
        view.state.value,
        data,
        warnings=view.warnings,
        next_actions=view.next_actions,
        provenance=provenance,
    )


def receipt_document(view: ReceiptView, profile: str) -> dict[str, object]:
    data: dict[str, object] = {
        "profile": profile,
        "receipt_id": view.receipt_id,
        "revision": view.revision,
        "root": view.root,
        "task_turn": view.task_turn,
        "candidate_type": view.candidate_type,
        "source_atoms": list(view.atoms),
        "allocations": {scenario: list(atoms) for scenario, atoms in view.allocations},
        "exclusions": list(view.exclusions),
        "costs": {name: quantity_json(q) for name, q in view.costs},
        "execution_ids": list(view.execution_ids),
        "exposure_ids": list(view.exposure_ids),
        "evidence_available": view.evidence_available,
        "content": view.content,
        "schema": view.schema,
        "unit": view.unit,
        "period": view.period,
        "missing_costs": [{"cost": n, "reason": r} for n, r in view.missing_costs],
        "credit_raw": {scenario: value for scenario, value in view.credit_raw},
        "raw_unit": view.raw_unit or None,
        "debit_raw": view.debit_raw,
        "debit_raw_min": view.debit_raw_min,
        "debit_raw_max": view.debit_raw_max,
        "identities": dict(view.identities),
        "atom_digests": [list(pair) for pair in view.atom_digests],
        "recomputation_available": view.evidence_available,
        "eligibility": view.eligibility,
        "eligibility_reasons": list(view.eligibility_reasons),
        "provenance": [list(p) for p in view.provenance],
        "scenarios": [scenario_data(s) for s in view.scenarios],
    }
    state = "available" if view.evidence_available else "evidence_expired"
    warnings = (view.note,) if view.note else ()
    return envelope(state, data, warnings=warnings, provenance={"methodology": view.methodology})


def history_document(view: HistoryView) -> dict[str, object]:
    data: dict[str, object] = {
        "profile": view.profile,
        "entries": [
            {
                "period": e.period,
                "revision": e.revision,
                "namespace": e.namespace,
                "revision_name": e.revision_name,
                "status": e.status,
                "current": e.current,
                "supersedes": e.supersedes,
                "net": quantity_json(e.net),
                "kind": e.kind,
                "recomputable": e.recomputable,
                "scenarios": [scenario_data(s) for s in e.scenarios],
            }
            for e in view.entries
        ],
        "current_total": quantity_json(view.current_total),
        "scenarios": [scenario_data(s) for s in view.scenarios],
        "next_cursor": view.next_cursor,
        "reason": view.reason,
    }
    failed = view.reason in {"history_unreadable", "history_unsupported_schema"}
    state = "unavailable" if failed else ("ok" if view.entries else "empty")
    return envelope(state, data, warnings=view.warnings, next_actions=view.next_actions)


def lookups_document(view: LookupsView) -> dict[str, object]:
    rows = []
    for row in view.rows:
        value = to_plain(row)
        value["measured_service_ms"] = quantity_json(row.measured_service_ms)
        value["known_costs"] = {name: quantity_json(q) for name, q in row.known_costs}
        value["scenarios"] = [scenario_data(e) for e in row.scenarios]
        value["causal_usefulness"] = None
        value["valuation_scope"] = "lookup_nonadditive"
        rows.append(value)
    data = {
        "profile": view.profile,
        "scope": to_plain(view.scope),
        "rows": rows,
        "total": view.total,
        "limit": view.limit,
        "next_cursor": view.next_cursor,
        "reason": view.reason,
        "coverage": view.coverage,
        "estimates_are_causal": False,
    }
    return envelope(
        "unavailable" if view.reason else "ok" if view.rows else "empty",
        data,
        warnings=view.warnings,
        provenance={"snapshot_id": view.snapshot_id, "generation": view.generation},
    )
