"""Per-destination delivery views for ``savings`` and ``metrics status`` (read-only).

Delivery documents are written only by the export module (``acks.update_delivery``); this
module reads them through ``acks.read_delivery`` (never creating anything) and names each
destination/signal state with its cause and a REAL next command. "Acknowledged" always
means collector acceptance, never backend durability or "synced". Collection freshness is a
separate fact: an offline or credential-blocked destination never changes local state.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from ssgrep.accounting import acks
from ssgrep.accounting.ledger_contracts import ProfilePolicy, Signal
from ssgrep.accounting.service_types import DeliveryLine, NextAction, command_for

_CREDENTIAL_BLOCKS = ("unauthorized", "forbidden", "credential_invalid")
_NETWORK = ("network", "network_error", "timeout")
_STATE_ORDER = (
    "credentials_expired",
    "blocked",
    "partial_export",
    "unknown",
    "offline",
    "unreadable",
    "pending",
    "waiting",
    "acknowledged",
    "disabled",
)


def _age(stamp: str | None, now: datetime) -> int | None:
    if stamp is None:
        return None
    # Delivery documents have already passed acks_model.as_time calendar validation.
    moment = datetime.fromisoformat(stamp)
    return max(int((now - moment.astimezone(UTC)).total_seconds()), 0)


def _act(profile: str, destination: str, signal: str, mode: str, why: str) -> NextAction:
    return NextAction(
        command_for(
            profile, f"metrics export --destination {destination} --signal {signal} {mode}"
        ),
        why,
    )


def _line(
    profile: str,
    destination: str,
    signal: str,
    enabled: bool,
    state: acks.Delivery | None,
    now: datetime,
) -> DeliveryLine:
    if state is None:
        if not enabled:
            return DeliveryLine(
                destination_id=destination,
                signal=signal,
                enabled=enabled,
                state="disabled",
                cause="Export is not enabled for this destination and signal; nothing is sent.",
                action=_act(
                    profile, destination, signal, "--preview", "Preview what would be sent."
                ),
            )
        return DeliveryLine(
            destination_id=destination,
            signal=signal,
            enabled=enabled,
            state="waiting",
            cause="Export is enabled but nothing has been queued or sent yet.",
            action=_act(profile, destination, signal, "--preview", "Preview the next payload."),
        )
    facts = acks.describe(state)
    kind = facts["state"]
    cause, label, action = "", str(kind), None
    if kind == "blocked" and state.blocked_reason in _CREDENTIAL_BLOCKS:
        label = "credentials_expired"
        cause = (
            f"The destination refused the credentials ({state.blocked_reason}); "
            f"{state.pending_batches} batch(es) wait. Local collection is unaffected."
        )
        action = _act(
            profile,
            destination,
            signal,
            "--check",
            "Rotate the credential file named by the policy's credential reference, then verify.",
        )
    elif kind == "blocked":
        cause = (
            f"Sending is blocked by policy ({state.blocked_reason}); "
            f"{state.pending_batches} batch(es) wait."
        )
        action = _act(profile, destination, signal, "--preview", "Review what is blocked.")
    elif kind == "partial":
        label = "partial_export"
        cause = (
            f"Not fully acknowledged: {state.rejected_points} point(s) rejected and "
            f"{state.partial_batches} batch(es) partly accepted."
        )
        action = _act(
            profile, destination, signal, "--preview", "Inspect what the collector partly rejected."
        )
    elif kind == "pending" and not state.pending_batches and state.in_flight is None:
        label = "waiting"
        cause = (
            "The latest collected generation is not fully acknowledged, "
            "but no batch is currently queued."
        )
        action = _act(
            profile, destination, signal, "--preview", "Inspect the next eligible payload."
        )
    elif kind in ("pending", "unknown") and (
        state.last_error_category in _NETWORK or (kind == "pending" and state.next_retry_utc)
    ):
        label = "offline"
        category = state.last_error_category or "unreachable"
        retry = f"; next retry {state.next_retry_utc}" if state.next_retry_utc else ""
        maybe = (
            f" {state.unknown_batches} of them may have reached it (outcome unknown; "
            "a resend carries the same bytes)."
            if state.unknown_batches
            else ""
        )
        cause = (
            f"Destination unreachable ({category}); {state.pending_batches} batch(es) pending"
            f"{retry}.{maybe} Not synced; local collection continues."
        )
        action = _act(profile, destination, signal, "--check", "Test connectivity now.")
    elif kind == "unknown":
        cause = f"{state.unknown_batches} batch(es) were sent but their outcome is unknown."
        action = _act(profile, destination, signal, "--check", "Verify connectivity.")
    elif kind == "pending":
        cause = f"{state.pending_batches} batch(es) pending delivery; not yet acknowledged."
        action = NextAction(command_for(profile, "metrics status"), "Watch delivery progress.")
    elif kind == "acknowledged":
        cause = (
            f"Collector accepted generation {state.last_acked_generation} "
            "(acceptance only, not a claim of backend durability)."
        )
    else:
        label = "waiting"
        cause = "Nothing queued yet."
        action = _act(profile, destination, signal, "--preview", "Preview the next payload.")
    if state.lost_records:
        cause += f" {state.lost_records} record(s) discarded without full acknowledgment."
        if state.loss_summary:
            latest = state.loss_summary[0]
            cause += f" Most recent loss: {latest.reason} ({latest.records} record(s))."
    if not enabled and label not in ("credentials_expired", "blocked"):
        cause += " (Export is currently disabled; history is kept.)"
    return DeliveryLine(
        destination_id=destination,
        signal=signal,
        enabled=enabled,
        state=label,
        cause=cause,
        action=action,
        pending_batches=state.pending_batches,
        rejected_points=state.rejected_points,
        unknown_batches=state.unknown_batches,
        partial_batches=state.partial_batches,
        lost_records=state.lost_records,
        collected_generation=state.last_collected_generation,
        attempted_generation=state.last_attempted_generation,
        acked_generation=state.last_acked_generation,
        last_acked_at=state.last_acked_at,
        last_attempt_at=state.last_attempt_at,
        next_retry_utc=state.next_retry_utc,
        last_error_category=state.last_error_category,
        blocked_reason=state.blocked_reason,
        acked_age_seconds=_age(state.last_acked_at, now),
        attempt_age_seconds=_age(state.last_attempt_at, now),
    )


def read_delivery_lines(
    profile: str, policy: ProfilePolicy, now: datetime
) -> tuple[DeliveryLine, ...]:
    """One line per policy destination and signal, in stable order."""
    enabled = set(policy.enabled_exports)
    lines: list[DeliveryLine] = []
    for destination in sorted(policy.destinations, key=lambda d: d.destination_id):
        for signal in destination.signals:
            is_on = (destination.destination_id, signal.value) in enabled
            try:
                state = acks.read_delivery(profile, destination.destination_id, Signal(signal))
            except (acks.DeliveryCorrupt, OSError, ValueError):
                lines.append(
                    DeliveryLine(
                        destination_id=destination.destination_id,
                        signal=signal.value,
                        enabled=is_on,
                        state="unreadable",
                        cause="The delivery state file is unreadable; nothing was changed.",
                        action=NextAction(
                            command_for(profile, "metrics status"), "Inspect profile state."
                        ),
                    )
                )
                continue
            line = _line(profile, destination.destination_id, signal.value, is_on, state, now)
            if (
                policy.paused
                and line.action is not None
                and line.action.command.endswith("--check")
            ):
                line = replace(
                    line,
                    cause=f"{line.cause} Sending is paused, so nothing is retried until resume.",
                    action=NextAction(
                        command_for(profile, "metrics resume"),
                        "Resume sending (a connectivity check exits 1 while paused).",
                    ),
                )
            lines.append(line)
    return tuple(lines)


def worst_first(lines: tuple[DeliveryLine, ...]) -> tuple[DeliveryLine, ...]:
    return tuple(
        sorted(
            lines,
            key=lambda line: _STATE_ORDER.index(line.state) if line.state in _STATE_ORDER else 99,
        )
    )
