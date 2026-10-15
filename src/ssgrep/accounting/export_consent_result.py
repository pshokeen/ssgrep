"""Framework-free delivery verdicts and correct recovery actions for authority holds."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ssgrep.accounting.ledger_contracts import Signal
from ssgrep.accounting.service_types import command_for

if TYPE_CHECKING:
    from ssgrep.accounting.delivery import CycleResult

FAILURE = 1


def target_command(profile: str, destination: str, signal: Signal, flag: str = "") -> str:
    tail = f" {flag}" if flag else ""
    return command_for(
        profile, f"metrics export --destination {destination} --signal {signal.value}{tail}"
    )


@dataclass(frozen=True)
class Verdict:
    ok: bool
    condition: str | None
    message: str
    next_command: str | None
    code: int


def classify_cycle(result: "CycleResult", profile: str) -> Verdict:
    """Stable conditions; migration/consent holds are not solved by profile resume."""
    here = target_command(profile, result.destination, result.signal)
    state, reason = result.state, result.reason
    if state == "sent":
        return Verdict(True, None, "Sent.", None, 0)
    if state == "idle":
        return Verdict(True, None, "Nothing new to send; delivery is up to date.", None, 0)
    if reason == "identity_migration_unavailable":
        return Verdict(
            False,
            reason,
            "Identity proof is unavailable; restore a supported checkpoint "
            "or use declared recovery before sending. Nothing was sent.",
            command_for(profile, "metrics status"),
            FAILURE,
        )
    if reason in ("identity_migration_required", "identity_migration_pending"):
        return Verdict(
            False,
            reason,
            "Current identity migration is incomplete; nothing was sent.",
            command_for(profile, "metrics collect"),
            FAILURE,
        )
    if reason in ("consent_reapproval_required", "consent_state_unavailable"):
        return Verdict(
            False,
            reason,
            "Review and explicitly approve the current export fields; nothing was sent.",
            target_command(profile, result.destination, result.signal, "--enable"),
            FAILURE,
        )
    if state == "paused":
        return Verdict(
            False,
            "export_paused",
            "Export is paused; nothing was sent.",
            command_for(profile, "metrics resume"),
            FAILURE,
        )
    if state == "not_enabled":
        return Verdict(
            False,
            "export_not_enabled",
            "Export is not enabled for this destination/signal; nothing was sent.",
            target_command(profile, result.destination, result.signal, "--enable"),
            FAILURE,
        )
    if state == "blocked":
        why = result.blocked_reason or reason or "blocked"
        action = (
            " Update the credential reference, then run --check."
            if why in ("unauthorized", "forbidden", "credential_invalid")
            else ""
        )
        return Verdict(
            False,
            "export_blocked",
            f"Sending is blocked ({why}).{action}",
            target_command(profile, result.destination, result.signal, "--check"),
            FAILURE,
        )
    if state == "waiting":
        return Verdict(
            False,
            "retry_pending",
            f"A retry is scheduled for {result.retry_scheduled_utc or 'later'}; nothing was sent.",
            here,
            FAILURE,
        )
    if state == "busy":
        return Verdict(
            False, "export_busy", "Another export is running for this target.", here, FAILURE
        )
    if state == "no_snapshot":
        return Verdict(
            False,
            "no_snapshot",
            "There is no snapshot to export yet.",
            command_for(profile, "metrics collect"),
            FAILURE,
        )
    if state == "revoked":
        return Verdict(
            False,
            reason or "export_revoked",
            "Policy no longer permits this export; nothing was sent.",
            command_for(profile, "metrics status"),
            FAILURE,
        )
    return Verdict(
        False, "send_failed", f"The send did not complete ({reason or state}).", here, FAILURE
    )
