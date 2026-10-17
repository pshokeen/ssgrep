"""Framework-free consent logic for ``savings export`` (design D11, tasks 12.3-12.4).

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

import hashlib
import json
import os
import shlex
import sys as _sys
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.collect import store as collector_store
from ssgrep.accounting.collect.collector import CollectError
from ssgrep.accounting.export import acks, spool as export_spool
from ssgrep.accounting.export.acks import DeliveryCorrupt
from ssgrep.accounting.export.creds import (
    credential_display as _credential_display,
    endpoint_display as _endpoint_display,
    host_display as _host_display,
)
from ssgrep.accounting.export.ids import pseudonym
from ssgrep.accounting.export.otlp import FIELDS_VERSION as _FIELDS_VERSION
from ssgrep.accounting.export.spool import SpoolCorrupt
from ssgrep.accounting.ledger.contracts import digest_of
from ssgrep.accounting.ledger.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotEnvelope,
    valid_profile_id,
)
from ssgrep.accounting.privacy import policy, storage
from ssgrep.accounting.privacy.policy import (
    PolicyError,
    check_endpoint,
    check_proxy,
    valid_collection_scope,
)
from ssgrep.accounting.privacy.retention import effective_retention
from ssgrep.accounting.privacy.storage import loads
from ssgrep.accounting.views import snapshots
from ssgrep.accounting.views.types import check_profile, command_for

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


def not_a_claim(profile: str, destination: str, signal: Signal) -> str:
    return (
        "Enabling sends nothing and is not a connectivity or delivery claim; run "
        f"`{target_command(profile, destination, signal, '--check')}` to test connectivity "
        "with an empty request."
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
            "ssgrep savings export --help",
        )
    if not destination:
        raise _usage(
            "destination_required",
            "A destination is required. Nothing was changed.",
            "ssgrep savings export --help",
        )
    if not valid_profile_id(destination):
        raise _usage(
            "invalid_destination",
            "Destination ids are 1-32 characters of a-z, 0-9, '-' or '_'. Nothing was changed.",
            "ssgrep savings status",
        )
    try:
        parsed = Signal(signal)
    except ValueError:
        raise _usage(
            "invalid_signal",
            "A signal must be chosen by the export command. Nothing was changed.",
            "ssgrep savings export --help",
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
    verdict = _policy.gate(profile, destination, signal, need_enabled=False, need_unpaused=False)
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
            command_for(profile, "savings status"),
        )
    if state.state == "policy_corrupt":
        raise PolicyError(
            "policy_corrupt",
            "The accounting policy cannot be read safely, so export is paused and nothing "
            "was sent or changed.",
            "ssgrep savings setup",
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
    notices = [not_a_claim(profile, destination, signal), ACCEPTANCE, DELETION_LIMITS]
    if signal is Signal.LOGS:
        notices.append(TOMBSTONE_NOTE)
    notices.append(PAUSE_NOTE)
    return EnablePlan(
        profile=profile,
        destination=destination,
        signal=signal,
        endpoint=_endpoint_display(dest.endpoint),
        proxy=_host_display(dest.proxy) if dest.proxy else None,
        credential_ref=_credential_display(dest.credential_ref),
        fields_version=_FIELDS_VERSION[signal.value],
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
    updated = _policy.with_enabled(current, destination, signal, True)
    try:
        changed = approve(profile, destination, signal)
    except (OSError, ValueError, export_spool.StateCorrupt):
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
        policy.save_profile(_policy.with_enabled(current, destination, signal, False))
    pending: int | None
    try:
        pending = export_spool.live_count(profile, destination, signal)
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


"""Explicit per-target field-version approval, separate from the policy permission.

Only confirmed enable calls write this owner-only metadata. Missing, old or unsafe
approval cannot silently authorize newly added fields; preview/check never writes.
"""


_MAX_BYTES = 4096


def _name(signal: Signal) -> str:
    return f"consent-{signal.value}.json"


def _expected(profile: str, destination: str, signal: Signal) -> dict:
    if not valid_profile_id(profile) or not valid_profile_id(destination):
        raise ValueError("invalid_consent_target")
    return {
        "schema": 1,
        "profile": profile,
        "destination": destination,
        "signal": signal.value,
        "fields_version": _FIELDS_VERSION[signal.value],
    }


def _matches(document: object, expected: dict) -> bool:
    return (
        isinstance(document, dict) and type(document.get("schema")) is int and document == expected
    )


def approval_reason(profile: str, destination: str, signal: Signal) -> str | None:
    expected = _expected(profile, destination, signal)
    try:
        directory = storage.open_profile_dir(profile, "export", destination, create=False)
    except FileNotFoundError:
        return "consent_reapproval_required"
    except OSError:
        return "consent_state_unavailable"
    try:
        raw = storage.read_file(directory, _name(signal), _MAX_BYTES)
        document = loads(raw) if raw is not None else None
    except (OSError, ValueError, TypeError):
        return "consent_state_unavailable"
    finally:
        os.close(directory)
    return None if _matches(document, expected) else "consent_reapproval_required"


def approve(profile: str, destination: str, signal: Signal) -> bool:
    """Called ONLY after explicit current-field preview confirmation; idempotent."""
    document = _expected(profile, destination, signal)
    with export_spool.locked(
        profile, ("export", destination), f"consent-{signal.value}"
    ) as directory:
        if directory is None:
            raise ValueError("consent_state_unavailable")
        raw = storage.read_file(directory, _name(signal), _MAX_BYTES)
        previous = loads(raw) if raw is not None else None
        if _matches(previous, document):
            return False
        export_spool.write_json(directory, _name(signal), document)
    return True


"""Framework-free delivery verdicts and correct recovery actions for authority holds."""


if TYPE_CHECKING:
    from ssgrep.accounting.export.delivery import CycleResult

FAILURE = 1


def target_command(profile: str, destination: str, signal: Signal, flag: str = "") -> str:
    mode = {
        "--preview": "preview",
        "--check": "check",
        "--enable": "enable",
        "--disable": "disable",
    }.get(flag.split(" ", 1)[0] if flag else "", "send")
    parts = ["savings", "export", mode, shlex.quote(destination)]
    if signal is Signal.LOGS:
        parts.append("--logs")
    if "--yes" in flag.split():
        parts.append("--yes")
    return command_for(profile, " ".join(parts))


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
            command_for(profile, "savings status"),
            FAILURE,
        )
    if reason in ("identity_migration_required", "identity_migration_pending"):
        return Verdict(
            False,
            reason,
            "Current identity migration is incomplete; nothing was sent.",
            command_for(profile, "savings collect"),
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
            command_for(profile, "savings resume"),
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
            " It stays blocked until the credential value, endpoint or proxy changes; a passing "
            "check does not clear it. Change what the credential reference names, then send again."
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
            command_for(profile, "savings collect"),
            FAILURE,
        )
    if state == "revoked":
        return Verdict(
            False,
            reason or "export_revoked",
            "Policy no longer permits this export; nothing was sent.",
            command_for(profile, "savings status"),
            FAILURE,
        )
    return Verdict(
        False, "send_failed", f"The send did not complete ({reason or state}).", here, FAILURE
    )


"""Read-only ``--preview`` view of the exact sanitized candidate for one destination/signal.

Wraps ``projection.preview_candidate`` (which writes nothing: no key, no scope, no
registry/spool/delivery state) into plain data for the renderers. Request bytes are the
"""


@dataclass(frozen=True)
class BatchView:
    batch_id: str
    bytes: int
    items: int
    sha256: str
    body: bytes

    def payload(self) -> Any:
        return json.loads(self.body)


@dataclass(frozen=True)
class PreviewView:
    profile: str
    destination: str
    signal: Signal
    state: str
    reason: str | None
    policy_version: int
    endpoint_host: str | None
    fields_version: str
    ephemeral_key: bool
    generation: int | None
    snapshot_id: str | None
    omitted: int
    notes: tuple[str, ...]
    batches: tuple[BatchView, ...]

    @property
    def total_bytes(self) -> int:
        return sum(b.bytes for b in self.batches)

    @property
    def total_items(self) -> int:
        return sum(b.items for b in self.batches)


def _empty_snapshot_store(profile: str) -> bool:
    """Prove absence without mistaking an unreadable pointer or unsafe directory for setup."""
    try:
        directory = storage.open_profile_dir(profile, snapshots.SNAPSHOT_DIR, create=False)
    except FileNotFoundError:
        return True
    try:
        raw = storage.read_file(directory, snapshots.CURRENT_NAME, snapshots.MAX_SNAPSHOT_BYTES)
        if raw is not None:
            snapshots.decode(raw)  # unlike current_generation, corruption must not mean absence
            return False
        return not storage.list_names(directory)
    finally:
        os.close(directory)


def build_preview(profile: str, destination: str, signal: Signal) -> PreviewView:
    """The exact sanitized candidate, read-only. Raises ``PolicyError`` for refused targets."""
    authority = require_target(profile, destination, signal)
    from ssgrep.accounting.export.registry import RecordsUnavailable

    try:
        checkpoint = collector_store.load_checkpoint(profile)
    except CollectError as error:
        raise RecordsUnavailable(error.condition) from None
    migration = migration_of(checkpoint)
    if migration.reason:
        dest = next(d for d in authority.policy.destinations if d.destination_id == destination)
        if checkpoint is None and _empty_snapshot_store(profile):
            return PreviewView(
                profile,
                destination,
                signal,
                "no_snapshot",
                None,
                authority.policy.policy_version,
                _host_display(dest.endpoint),
                _FIELDS_VERSION[signal.value],
                False,
                None,
                None,
                0,
                (),
                (),
            )
        return PreviewView(
            profile,
            destination,
            signal,
            migration.reason,
            migration.reason,
            authority.policy.policy_version,
            _host_display(dest.endpoint),
            _FIELDS_VERSION[signal.value],
            False,
            None,
            None,
            0,
            ("current_projection_unavailable_until_identity_migration",),
            (),
        )
    from ssgrep.accounting.export.projection import preview_candidate

    result = preview_candidate(profile, destination, signal)
    dest = result.destination
    built = result.candidate
    batches = tuple(
        BatchView(b.batch_id, len(b.body), b.items, hashlib.sha256(b.body).hexdigest(), b.body)
        for b in (built.batches if built is not None else ())
    )
    reason = None
    if result.state == "policy_blocked":
        reason = _policy.gate(
            profile, destination, signal, need_enabled=False, need_unpaused=False
        ).reason
    return PreviewView(
        profile=profile,
        destination=destination,
        signal=signal,
        state=result.state,
        reason=reason,
        policy_version=result.policy.policy_version,
        endpoint_host=_host_display(dest.endpoint) if dest is not None else None,
        fields_version=_FIELDS_VERSION[signal.value],
        ephemeral_key=result.ephemeral_key,
        generation=built.generation if built is not None else None,
        snapshot_id=built.snapshot_id if built is not None else None,
        omitted=built.omitted if built is not None else 0,
        notes=tuple(built.notes) if built is not None else (),
        batches=batches,
    )


"""Read-only company preflight: native identity migration and explicit field approval."""


NATIVE_IDENTITY = "native_identity_v2"
AUTHORITY_HOLD_REASONS = frozenset(
    {
        "identity_migration_required",
        "identity_migration_pending",
        "identity_migration_unavailable",
        "consent_reapproval_required",
        "consent_state_unavailable",
    }
)


@dataclass(frozen=True)
class MigrationView:
    state: str  # current | legacy | pending | unavailable
    reason: str | None = None


def migration_of(checkpoint) -> MigrationView:
    if checkpoint is None:
        return MigrationView("legacy", "identity_migration_required")
    marker = checkpoint.identity_migration
    if not isinstance(marker, dict):
        return MigrationView("unavailable", "identity_migration_unavailable")
    if not marker and checkpoint.identity_method != NATIVE_IDENTITY:
        return MigrationView("legacy", "identity_migration_required")
    if marker.get("target") != NATIVE_IDENTITY:
        return MigrationView("unavailable", "identity_migration_unavailable")
    status = marker.get("status")
    if status == "complete":
        if checkpoint.identity_method != NATIVE_IDENTITY:
            return MigrationView("unavailable", "identity_migration_unavailable")
        return MigrationView("current")
    if status == "pending":
        return MigrationView("pending", "identity_migration_pending")
    return MigrationView("unavailable", "identity_migration_unavailable")


def migration_view(profile: str) -> MigrationView:
    try:
        return migration_of(collector_store.load_checkpoint(profile))
    except (CollectError, OSError, ValueError, TypeError):
        return MigrationView("unavailable", "identity_migration_unavailable")


def authority_reason(profile: str, destination: str, signal: Signal) -> str | None:
    migration = migration_view(profile)
    return migration.reason or approval_reason(profile, destination, signal)


"""Pre-dispatch revalidation: may THIS destination/signal be sent to, with THESE bytes, NOW?

Authorization is never inherited from the moment a batch was queued (design D11). Immediately
before every dispatch ``gate`` rechecks the profile's current policy (readable, not paused,
sending not held paused by an invalid managed update, destination allowlisted, signal
permitted, logs separately permitted, endpoint/proxy still valid, destination+signal
explicitly enabled) and ``scope_document`` rebuilds the identity of what the policy would
authorize today. A queued batch is sendable only when the scope document it was created under
equals today's; otherwise its bytes are invalid and must be reprojected or discarded.

Snapshots carry the digest of the actual collection authority (canonical workspace roots,
assignments and profile ownership). The first export cannot bless older or unknown authority:
the snapshot provenance must match today's scope. Scope-state corruption requires a collection
after recovery; it is never treated as an unobserved first send. Destination consent and
credential rotation do not change the collection authority.
Deletion requests (tombstones) are judged only by destination/endpoint/identity/logs permission:
narrowing roots must never block a request to delete data that was already exported.
"""


SCOPE_DIR = "export"
SCOPE_SCHEMA = 1
DATA_KEYS = ("collection_scope", "fields", "policy_version", "logs_permitted")
TOMBSTONE_KEYS = ("endpoint", "proxy", "fields", "logs_permitted", "identity")
BLOCKING_REASONS = frozenset({"policy_unreadable", "unknown_destination", "signal_not_permitted"})


@dataclass(frozen=True)
class Gate:
    """The verdict of one authorization check; ``reason`` is a stable machine code."""

    ok: bool
    reason: str | None
    detail: str
    policy: ProfilePolicy
    destination: DestinationPolicy | None

    @property
    def revoked(self) -> bool:
        """The destination/signal is no longer allowed by policy (not merely disabled/paused)."""
        return self.reason in {"unknown_destination", "signal_not_permitted", "logs_not_permitted"}


def _denied(
    reason: str, detail: str, current: ProfilePolicy, dest: DestinationPolicy | None
) -> Gate:
    return Gate(False, reason, detail, current, dest)


def gate(
    profile: str,
    destination_id: str,
    signal: Signal,
    *,
    need_enabled: bool = True,
    need_unpaused: bool = True,
) -> Gate:
    """Authorize one destination/signal against the CURRENT policy; reads only, never writes."""
    status = policy.policy_status(profile)
    current = status.profile
    if status.error is not None:
        return _denied("policy_unreadable", status.error.condition, current, None)
    dest = next((d for d in current.destinations if d.destination_id == destination_id), None)
    if dest is None:
        return _denied(
            "unknown_destination", "destination is not in the profile policy", current, None
        )
    if signal not in dest.signals:
        return _denied("signal_not_permitted", f"{signal.value} is not permitted", current, dest)
    if signal is Signal.LOGS and not dest.logs_permitted:
        return _denied("logs_not_permitted", "logs need explicit policy permission", current, dest)
    try:
        check_endpoint(dest.endpoint)
        if dest.proxy is not None:
            check_proxy(dest.proxy, endpoint=dest.endpoint)
    except PolicyError as error:
        return _denied("endpoint_invalid", str(error), current, dest)
    if need_unpaused:
        try:
            policy.require_not_paused(profile, "export")
        except PolicyError as error:
            return _denied(error.condition, str(error), current, dest)
    if need_enabled and (destination_id, signal.value) not in current.enabled_exports:
        return _denied("not_enabled", "this destination/signal was not enabled", current, dest)
    if need_enabled and (reason := authority_reason(profile, destination_id, signal)):
        return _denied(
            reason, "current identity and explicit field-version consent required", current, dest
        )
    return Gate(True, None, "", current, dest)


def scope_document(
    current: ProfilePolicy, dest: DestinationPolicy, signal: Signal, installation: str
) -> dict[str, Any]:
    """What a payload built today would be authorized under (plain, comparable JSON)."""
    return {
        "roots": sorted(current.roots),
        "collection_scope": policy.collection_scope(current),
        "endpoint": dest.endpoint,
        "proxy": dest.proxy,
        "fields": _FIELDS_VERSION[signal.value],
        "policy_version": current.policy_version,
        "logs_permitted": bool(dest.logs_permitted) if signal is Signal.LOGS else False,
        "identity": installation,
    }


def scope_changes(old: dict[str, Any], new: dict[str, Any]) -> tuple[str, ...]:
    """Names of scope keys whose value differs (values themselves are never reported)."""
    return tuple(sorted(k for k in {*old, *new} if old.get(k) != new.get(k)))


def credential_fingerprint(
    key: bytes, destination: str, dest: DestinationPolicy, secret: str | None
) -> str:
    """Keyed fingerprint of what a block depends on: credential value, endpoint, proxy."""
    return pseudonym(key, "credential", destination, dest.endpoint, dest.proxy or "", secret or "")


# ---- scope tracking ---------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeState:
    digest: str
    changed_at: str | None
    collection_scope: dict[str, Any] | None = None


def _scope_file(signal: Signal) -> str:
    return f"scope-{signal.value}.json"


def _dirs(destination: str) -> tuple[str, ...]:
    return (SCOPE_DIR, destination)


def _parse(raw: Any) -> ScopeState | None:
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema", "digest", "changed_at", "collection_scope"}
        or type(raw.get("schema")) is not int
        or raw["schema"] != SCOPE_SCHEMA
    ):
        return None
    digest, changed = raw.get("digest"), raw.get("changed_at")
    provenance = raw.get("collection_scope")
    if not valid_collection_scope(provenance):
        return None
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    if any(char not in "0123456789abcdef" for char in digest):
        return None
    if changed is not None:
        try:
            datetime.strptime(changed, "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError):
            return None
    return ScopeState(digest, changed, provenance)


def _read_scope_fd(dir_fd: int, signal: Signal) -> ScopeState | None:
    try:
        data = storage.read_file(dir_fd, _scope_file(signal), export_spool.MAX_STATE_BYTES)
        if data is None:
            return None
        return _parse(loads(data)) or ScopeState("", "")
    except (OSError, ValueError):
        return ScopeState("", "")


def read_scope(profile: str, destination: str, signal: Signal) -> ScopeState | None:
    """None only when absent; corrupt state is an invalid sentinel, never first-send authority."""
    try:
        dir_fd = storage.open_profile_dir(profile, *_dirs(destination), create=False)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return ScopeState("", "")
    try:
        return _read_scope_fd(dir_fd, signal)
    finally:
        os.close(dir_fd)


def next_scope(scope: dict[str, Any], known: ScopeState | None, now: str) -> ScopeState:
    """The same fail-closed transition for preview and durable observation."""
    digest = digest_of({k: scope.get(k) for k in DATA_KEYS})
    changed = None if known is None else (now if not known.digest else known.changed_at)
    return ScopeState(digest, changed, scope.get("collection_scope"))


def observe_scope(
    profile: str, destination: str, signal: Signal, scope: dict[str, Any], now: str
) -> ScopeState:
    """Persist collection authority; invalid prior state installs a recollection barrier."""
    with export_spool.locked(profile, _dirs(destination), _scope_file(signal)) as dir_fd:
        assert dir_fd is not None
        known = _read_scope_fd(dir_fd, signal)
        state = next_scope(scope, known, now)
        if state == known:
            return state
        export_spool.write_json(
            dir_fd,
            _scope_file(signal),
            {
                "schema": SCOPE_SCHEMA,
                "digest": state.digest,
                "changed_at": state.changed_at,
                "collection_scope": state.collection_scope,
            },
        )
        return state


def snapshot_eligible(envelope: SnapshotEnvelope, state: ScopeState, current_version: int) -> bool:
    """Actual collection provenance must match current authority, even on the first send."""
    provenance = envelope.coverage.get("collection_scope")
    if (
        envelope.policy_version != current_version
        or not valid_collection_scope(provenance)
        or not valid_collection_scope(state.collection_scope)
        or provenance != state.collection_scope
    ):
        return False
    return state.changed_at is None or envelope.collected_at > state.changed_at


def batch_current(batch_scope: Any, scope: dict[str, Any], kind: str = "data") -> bool:
    """Whether bytes queued under ``batch_scope`` may still be sent under today's ``scope``."""
    if not isinstance(batch_scope, dict):
        return False
    keys = TOMBSTONE_KEYS if kind == "tombstone" else tuple(scope)
    return all(batch_scope.get(k) == scope.get(k) for k in keys)


def with_enabled(
    current: ProfilePolicy, destination: str, signal: Signal, on: bool
) -> ProfilePolicy:
    """The profile with ``(destination, signal)`` added/removed from ``enabled_exports``."""
    pair = (destination, signal.value)
    kept = tuple(p for p in current.enabled_exports if p != pair)
    return replace(current, enabled_exports=tuple(sorted({*kept, *(({pair}) if on else set())})))


_policy = _sys.modules[__name__]
