"""JSON documents for ``metrics export`` (integers/strings only; never a credential value)."""

from __future__ import annotations

from typing import Any

from ssgrep.accounting.export_consent import (
    DisableReport,
    EnablePlan,
    target_command,
)
from ssgrep.accounting.export_consent_preview import PreviewView
from ssgrep.accounting.service_types import NextAction, command_for
from ssgrep.cli.accounting_cli import envelope


def plan_data(plan: EnablePlan) -> dict[str, object]:
    """The privacy preview as data (the ``preview`` member of the enable document)."""
    return {
        "destination": plan.destination,
        "signal": plan.signal.value,
        "endpoint": plan.endpoint,
        "proxy_host": plan.proxy,
        "credential_ref": plan.credential_ref,
        "fields_version": plan.fields_version,
        "fields": list(plan.fields),
        "approved_roots": list(plan.roots),
        "roots_note": plan.roots_note,
        "policy_version": plan.policy_version,
        "policy_source": plan.policy_source,
        "managed": plan.managed,
        "retention": dict(plan.retention),
        "already_enabled": plan.already_enabled,
        "paused": plan.paused,
        "notices": list(plan.notices),
    }


def enable_document(plan: EnablePlan, changed: bool) -> dict[str, object]:
    profile, dest, sig = plan.profile, plan.destination, plan.signal
    actions = (
        NextAction(
            target_command(profile, dest, sig, "--check"),
            "Test connectivity and authentication with an empty request.",
        ),
        NextAction(
            target_command(profile, dest, sig, "--preview"),
            "Inspect the exact data that would be sent.",
        ),
    )
    return envelope(
        "enabled",
        {
            "profile": profile,
            "destination": dest,
            "signal": sig.value,
            "changed": changed,
            "preview": plan_data(plan),
        },
        warnings=("paused: nothing is sent until you resume",) if plan.paused else (),
        next_actions=actions,
        provenance={"policy_version": plan.policy_version, "policy_source": plan.policy_source},
    )


def disable_document(report: DisableReport, profile: str, policy_version: int) -> dict[str, object]:
    actions = (
        NextAction(
            target_command(profile, report.destination, report.signal, "--enable"),
            "Re-enable this destination/signal explicitly.",
        ),
    )
    return envelope(
        "disabled",
        {
            "profile": profile,
            "destination": report.destination,
            "signal": report.signal.value,
            "changed": report.was_enabled,
            "retained": {
                "pending_batches": report.pending_batches,
                "delivery_history": True,
                "spool": True,
                "registry": True,
                "tombstones": True,
            },
            "in_flight": report.in_flight,
            "in_flight_nonrecallable": report.in_flight_nonrecallable,
            "in_flight_note": "An in-flight request cannot be recalled.",
        },
        next_actions=actions,
        provenance={"policy_version": policy_version},
    )


def preview_document(view: PreviewView) -> dict[str, object]:
    batches: list[dict[str, Any]] = [
        {
            "batch_id": b.batch_id,
            "bytes": b.bytes,
            "items": b.items,
            "sha256": b.sha256,
            "payload": b.payload(),
        }
        for b in view.batches
    ]
    actions: tuple[NextAction, ...] = ()
    if view.state == "ok":
        actions = (
            NextAction(
                target_command(view.profile, view.destination, view.signal, "--enable"),
                "Enable sending for this destination/signal (asks for confirmation).",
            ),
        )
    elif view.state == "no_snapshot":
        actions = (NextAction(command_for(view.profile, "metrics collect"), "Collect a snapshot."),)
    return envelope(
        view.state,
        {
            "profile": view.profile,
            "destination": view.destination,
            "signal": view.signal.value,
            "state": view.state,
            "reason": view.reason,
            "policy_version": view.policy_version,
            "endpoint_host": view.endpoint_host,
            "fields_version": view.fields_version,
            "generation": view.generation,
            "snapshot_id": view.snapshot_id,
            "pseudonym_key_persisted": not view.ephemeral_key,
            "omitted": view.omitted,
            "notes": list(view.notes),
            "batch_count": len(view.batches),
            "bytes": view.total_bytes,
            "items": view.total_items,
            "batches": batches,
            "network": False,
            "state_changed": False,
        },
        next_actions=actions,
        provenance={"policy_version": view.policy_version, "snapshot_id": view.snapshot_id},
    )


def check_document(
    profile: str,
    destination: str,
    signal: str,
    state: str,
    ok: bool,
    status: int | None,
    category: str,
) -> dict[str, object]:
    return envelope(
        "ok" if ok else state,
        {
            "profile": profile,
            "destination": destination,
            "signal": signal,
            "ok": ok,
            "state": state,
            "http_status_category": category,
            "http_status": status,
            "claim": "connectivity and authentication only; no data was acknowledged",
        },
    )


def send_document(
    profile: str,
    cycle: dict[str, object],
    next_command: str | None,
    why: str,
    state: str,
) -> dict[str, object]:
    actions = (NextAction(next_command, why),) if next_command else ()
    return envelope(state, {"profile": profile, **cycle}, next_actions=actions)
