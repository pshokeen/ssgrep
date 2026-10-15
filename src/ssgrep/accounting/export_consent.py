"""Framework-free consent logic for ``ssgrep metrics export`` (design D11, tasks 12.3-12.4).

Everything here is import-light and CLI-free: the command modules only parse flags and render.

* ``validate_modes`` runs FIRST and reads nothing: mutually exclusive modes, a required and
  well-formed destination, a known signal, a valid profile id.
* ``plan_enable`` is the read-only privacy preview of enabling one destination/signal. It never
  resolves a credential: only the reference name (``env:NAME`` / ``file:<basename>``) is shown.
* ``apply_enable`` persists explicit current field-version approval and enables only the
  ``(destination, signal)`` pair through narrowing-only ``policy.save_profile``. Enabling
  never sends and is no connectivity claim; disabling keeps spool, delivery history, registry and
  tombstones untouched. Neither resume nor any other path re-enables a signal.
* ``consent_state`` / ``require_target`` classify a target against the CURRENT policy.
* ``export_consent_preview.build_preview`` wraps the read-only candidate selection.
* ``classify_cycle`` turns a delivery ``CycleResult`` into a stable verdict for the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from ssgrep.accounting import acks, export_files, export_policy, export_spool, policy
from ssgrep.accounting.acks_model import DeliveryCorrupt
from ssgrep.accounting.export_consent_fields import approval_reason, approve
from ssgrep.accounting.export_consent_result import (
    FAILURE,
    Verdict as Verdict,
    classify_cycle as classify_cycle,
    target_command,
)
from ssgrep.accounting.export_model import FIELDS_VERSION
from ssgrep.accounting.export_spool_meta import SpoolCorrupt
from ssgrep.accounting.ledger_contracts import ProfilePolicy, Signal, valid_profile_id
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.retention import effective_retention
from ssgrep.accounting.scope import check_profile
from ssgrep.accounting.service_types import command_for

MODES = ("preview", "check", "enable", "disable")
SEND = "send"
USAGE = 2

METRICS_FIELDS = (
    "counters: captured executions and rendered bytes (cumulative)",
    "histograms: operation duration (cumulative, bounded buckets)",
    "estimate gauges: modeled credit, debit and net tokens/work seconds (may be negative or "
    "unknown; never a causal claim)",
    "separate known-subtotal, finite interval-bound and missing-evidence gauges",
    "measured monotonic service latency is separate from capped_gap_v1 modeled actor work",
    "coverage counts: lookup attribution, reuse evidence and source coverage",
    "resource: service name/version, pseudonymous installation id, policy version, counter epoch",
)
LOGS_FIELDS = (
    "pseudonymous per-task revisioned ledger records: record id, revision, record type, "
    "pseudonymous task id, UTC day, compatible unit/method, credit/debit/net per scenario, "
    "known subtotals, finite/unavailable bounds, sanitized missing-reason counts, coverage "
    "and methodology",
    "never exported: prompts, queries, paths, facts, titles, source refs, private receipt content",
)
NOT_A_CLAIM = (
    "Enabling sends nothing and is not a connectivity or delivery claim; use --check to test "
    "connectivity with an empty request."
)
ACCEPTANCE = (
    "Acknowledgment means the collector accepted the request; it does not mean durable "
    "end-to-end backend delivery."
)
DELETION_LIMITS = (
    "Remote deletion is limited: ssgrep cannot erase data a collector already stored, and "
    "previously acknowledged aggregate metrics may be NON-REVERSIBLE."
)
TOMBSTONE_NOTE = (
    "For logs, deletion requests (tombstones) for previously exported records can be requested "
    "only while policy permits them; the organization operates backend erasure."
)
PAUSE_NOTE = (
    "While the profile is paused nothing is sent, and resuming never re-enables a destination "
    "or signal you disabled."
)


@dataclass(frozen=True)
class ExportRequest:
    mode: str  # send | preview | check | enable | disable
    destination: str
    signal: Signal
    profile: str


def _usage(condition: str, message: str, command: str) -> PolicyError:
    return PolicyError(condition, message, command, exit_code=USAGE)


def validate_modes(
    *,
    destination: str | None,
    signal: str,
    profile: str,
    preview: bool = False,
    check: bool = False,
    enable: bool = False,
    disable: bool = False,
) -> ExportRequest:
    """Validate every argument before any file is read; raises ``PolicyError`` (exit 2)."""
    chosen = [n for n, on in zip(MODES, (preview, check, enable, disable), strict=True) if on]
    if len(chosen) > 1:
        raise _usage(
            "invalid_mode",
            f"--{' and --'.join(chosen)} cannot be combined: choose at most one of "
            "--preview, --check, --enable, --disable. Nothing was changed.",
            "ssgrep metrics export --help",
        )
    if not destination:
        raise _usage(
            "destination_required",
            "--destination <id> is required. Nothing was changed.",
            "ssgrep metrics status",
        )
    if not valid_profile_id(destination):
        raise _usage(
            "invalid_destination",
            "Destination ids are 1-32 characters of a-z, 0-9, '-' or '_'. Nothing was changed.",
            "ssgrep metrics status",
        )
    try:
        parsed = Signal(signal)
    except ValueError:
        raise _usage(
            "invalid_signal",
            "--signal must be metrics or logs. Nothing was changed.",
            "ssgrep metrics export --help",
        ) from None
    check_profile(profile)
    return ExportRequest(chosen[0] if chosen else SEND, destination, parsed, profile)


# ---- consent state ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsentState:
    state: str  # enabled | disabled | paused | <gate reason>
    enabled: bool
    paused: bool
    reason: str | None
    policy: ProfilePolicy


def consent_state(profile: str, destination: str, signal: Signal) -> ConsentState:
    """Where this destination/signal stands under the CURRENT policy (reads only)."""
    verdict = export_policy.gate(
        profile, destination, signal, need_enabled=False, need_unpaused=False
    )
    current = verdict.policy
    if not verdict.ok:
        reason = "policy_corrupt" if verdict.reason == "policy_unreadable" else verdict.reason
        return ConsentState(str(reason), False, False, verdict.detail, current)
    enabled = (destination, signal.value) in current.enabled_exports
    held = policy.policy_status(profile).sending_paused_reason
    paused = bool(current.paused or held)
    if enabled and (reason := approval_reason(profile, destination, signal)):
        return ConsentState(reason, False, paused, reason, current)
    return ConsentState(
        "paused" if paused else ("enabled" if enabled else "disabled"),
        enabled,
        paused,
        held,
        current,
    )


_USAGE_STATES = {
    "unknown_destination": "This destination is not in the profile policy.",
    "signal_not_permitted": "Policy does not permit this signal for the destination.",
    "logs_not_permitted": "Logs need separate explicit policy permission for this destination.",
    "endpoint_invalid": "The destination endpoint or proxy route is not supported.",
}


def require_target(profile: str, destination: str, signal: Signal) -> ConsentState:
    """The consent state, or ``PolicyError``: exit 2 for a target policy never allowed, exit 1
    when the policy itself is unreadable (sending stays paused; never a permissive fallback)."""
    state = consent_state(profile, destination, signal)
    if state.state in _USAGE_STATES:
        raise _usage(
            state.state,
            f"{_USAGE_STATES[state.state]} Nothing was changed.",
            command_for(profile, "metrics status"),
        )
    if state.state == "policy_corrupt":
        raise PolicyError(
            "policy_corrupt",
            "The accounting policy cannot be read safely, so export is paused and nothing "
            "was sent or changed.",
            "ssgrep metrics setup",
            exit_code=FAILURE,
        )
    return state


# ---- enable ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnablePlan:
    profile: str
    destination: str
    signal: Signal
    endpoint: str
    proxy: str | None
    credential_ref: str | None
    fields_version: str
    fields: tuple[str, ...]
    roots: tuple[str, ...]
    roots_note: str | None
    policy_version: int
    policy_source: str
    managed: bool
    retention: dict[str, int]
    already_enabled: bool
    paused: bool
    notices: tuple[str, ...]


def endpoint_display(url: str) -> str:
    """``scheme://host[:port]/path``; userinfo/query are refused by policy and never shown."""
    parts = urlsplit(url)
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname or ''}{port}{parts.path}"


def host_display(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.hostname or ''}{f':{parts.port}' if parts.port else ''}"


def credential_display(ref: str | None) -> str | None:
    """The reference NAME only: ``env:NAME`` or ``file:<basename>``; never a value or a path."""
    if ref is None:
        return None
    scheme, _, rest = ref.partition(":")
    if scheme == "file":
        return f"file:{rest.rstrip('/').rsplit('/', 1)[-1]}"
    return f"{scheme}:{rest}"


def plan_enable(profile: str, destination: str, signal: Signal) -> EnablePlan:
    """The privacy preview for enabling; read-only. Raises ``PolicyError`` on a refused target."""
    state = require_target(profile, destination, signal)
    current = state.policy
    dest = next(d for d in current.destinations if d.destination_id == destination)
    ceiling = policy.read_policy_file().ceilings.get(profile)
    effective = effective_retention(current, ceiling)
    roots_note = None
    if not current.roots:
        roots_note = (
            "all working directories on this machine (personal-local)"
            if profile == "personal-local"
            else "none approved"
        )
    notices = [NOT_A_CLAIM, ACCEPTANCE, DELETION_LIMITS]
    if signal is Signal.LOGS:
        notices.append(TOMBSTONE_NOTE)
    notices.append(PAUSE_NOTE)
    return EnablePlan(
        profile=profile,
        destination=destination,
        signal=signal,
        endpoint=endpoint_display(dest.endpoint),
        proxy=host_display(dest.proxy) if dest.proxy else None,
        credential_ref=credential_display(dest.credential_ref),
        fields_version=FIELDS_VERSION[signal.value],
        fields=METRICS_FIELDS if signal is Signal.METRICS else LOGS_FIELDS,
        roots=tuple(current.roots),
        roots_note=roots_note,
        policy_version=current.policy_version,
        policy_source=current.policy_source,
        managed=current.managed,
        retention={
            "evidence_days": effective.evidence_days,
            "receipt_days": effective.receipt_days,
            "spool_days": effective.spool_days,
            "spool_bytes": effective.spool_bytes,
        },
        already_enabled=state.enabled,
        paused=state.paused,
        notices=tuple(notices),
    )


def apply_enable(profile: str, destination: str, signal: Signal) -> bool:
    """Persist consent for exactly this pair under the CURRENT policy; True when it changed."""
    require_target(profile, destination, signal)
    current = policy.policy_status(profile).profile
    updated = export_policy.with_enabled(current, destination, signal, True)
    try:
        changed = approve(profile, destination, signal)
    except (OSError, ValueError, export_files.StateCorrupt):
        raise PolicyError(
            "consent_state_unavailable",
            "Field-version consent could not be stored safely; nothing was sent.",
            target_command(profile, destination, signal, "--enable"),
            exit_code=FAILURE,
        ) from None
    if updated != current:
        policy.save_profile(updated)
        changed = True
    return changed


# ---- disable ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DisableReport:
    destination: str
    signal: Signal
    was_enabled: bool
    pending_batches: int | None
    in_flight: str | None
    in_flight_nonrecallable: str | None


def apply_disable(profile: str, destination: str, signal: Signal) -> DisableReport:
    """Stop this signal only; spool, delivery history, registry and tombstones are untouched."""
    require_target(profile, destination, signal)
    current = policy.policy_status(profile).profile
    was = (destination, signal.value) in current.enabled_exports
    if was:
        policy.save_profile(export_policy.with_enabled(current, destination, signal, False))
    pending: int | None
    try:
        pending = export_spool.usage(profile, destination, signal).batches
    except (SpoolCorrupt, ValueError, OSError):
        pending = None
    in_flight = nonrecallable = None
    try:
        delivery = acks.read_delivery(profile, destination, signal)
    except (DeliveryCorrupt, ValueError):
        delivery = None
    if delivery is not None:
        in_flight = delivery.in_flight.batch_id if delivery.in_flight else None
        nonrecallable = delivery.in_flight_nonrecallable
    return DisableReport(destination, signal, was, pending, in_flight, nonrecallable)
