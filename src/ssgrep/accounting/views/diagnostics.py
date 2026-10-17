"""Stable warning identities and bounded category summaries, without storage reads.

Repeated identical warnings are aliases, not extra records. Distinct native
subjects stay distinct even if they have the same diagnostic code. Messages
remain private unless the caller explicitly requests them.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

from ssgrep.accounting.export import acks
from ssgrep.accounting.export.consent import target_command
from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy, Signal, SnapshotState
from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.views.types import (
    Condition,
    DeliveryLine,
    NextAction,
    Scope,
    SummaryView,
    command_for,
    issue,
    redeem,
)

_SEVERE = frozenset(
    {
        "corrupt",
        "storage_corrupt",
        "snapshot_unreadable",
        "unsupported_schema",
        "blocked",
        "credentials_expired",
        "unreadable",
        "policy_invalid",
        "identity_conflict",
        "replay_unavailable_sessions",
        "frequency_revision_unavailable",
    }
)
_RANK = {"error": 0, "warning": 1, "info": 2}


def warning_code(message: str) -> str:
    parts = message.split(":", 2)
    if len(parts) > 1 and parts[0] == "lineage" and _CODE.fullmatch(parts[1]):
        return f"lineage.{parts[1]}"
    return parts[0] if _CODE.fullmatch(parts[0]) else "diagnostic_note"


def severity(code: str) -> str:
    if code in _SEVERE or any(word in code for word in ("corrupt", "conflict", "cyclic", "denied")):
        return "error"
    return "warning"


def warning_records(
    warnings: Iterable[str], *, include_text: bool = False
) -> tuple[tuple[DiagnosticRecord, ...], int]:
    records: dict[str, DiagnosticRecord] = {}
    duplicates = 0
    for message in warnings:
        code = warning_code(message)
        identity = hashlib.sha256(f"warning\0{code}\0{message}".encode()).hexdigest()
        if identity in records:
            duplicates += 1
            continue
        records[identity] = DiagnosticRecord(
            identity, code, severity(code), "snapshot_warning", message if include_text else None
        )
    return tuple(
        sorted(records.values(), key=lambda r: (_RANK[r.severity], r.code, r.record_id))
    ), duplicates


def grouped_warnings(warnings: Iterable[str]) -> tuple[DiagnosticGroup, ...]:
    unique = set(warnings)
    counts = Counter(warning_code(message) for message in unique)
    return tuple(
        DiagnosticGroup(code, severity(code), count)
        for code, count in sorted(
            counts.items(), key=lambda item: (_RANK[severity(item[0])], item[0])
        )
    )


@dataclass(frozen=True)
class DiagnosticRecord:
    record_id: str
    code: str
    severity: str
    source: str
    message: str | None = None


@dataclass(frozen=True)
class DiagnosticGroup:
    code: str
    severity: str
    count: int


@dataclass(frozen=True)
class DiagnosticPage:
    profile: str
    scope: Scope
    snapshot_id: str | None
    generation: int | None
    records: tuple[DiagnosticRecord, ...]
    total: int
    alias_duplicates: int
    next_cursor: str | None
    limit: int = 20
    code: str | None = None
    include_text: bool = False
    reason: str | None = None


_ID = re.compile(r"^[0-9a-f]{16,64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.]{0,79}$")


def diagnostics_command(
    profile: str, *, cursor: str | None = None, include_text: bool = False
) -> str:
    """Return the public command for the current status diagnostics page."""
    parts = ["savings", "status", "--details"]
    if cursor is not None:
        parts.extend(("--page", shlex.quote(cursor)))
    if include_text:
        parts.append("--include-text")
    return command_for(profile, " ".join(parts))


def read_diagnostics(
    profile: str = PERSONAL_PROFILE,
    scope: Scope | None = None,
    *,
    snapshot_id: str | None = None,
    generation: int | None = None,
    limit: int = 20,
    cursor: str | None = None,
    code: str | None = None,
    include_text: bool = False,
) -> DiagnosticPage:
    from ssgrep.accounting.privacy.policy import policy_status
    from ssgrep.accounting.views.service import read_snapshot, read_summary
    from ssgrep.accounting.views.types import build_scope, check_limit, check_profile

    check_profile(profile)
    check_limit(limit)
    scope = scope or Scope()
    # Reuse the existing selection contract, including interval and child validation.
    scope = build_scope(**{key: value for key, value in asdict(scope).items() if key != "window"})
    restart = diagnostics_command(profile)
    if (snapshot_id is not None and not _ID.fullmatch(snapshot_id)) or (
        generation is not None and (isinstance(generation, bool) or not 1 <= generation < 10**10)
    ):
        raise PolicyError(
            "invalid_snapshot", "Invalid diagnostic snapshot identity or generation.", restart, 2
        )
    if code is not None and not _CODE.fullmatch(code):
        raise PolicyError("invalid_filter", "--code must be a stable diagnostic code.", restart, 2)
    if cursor and (snapshot_id is None or generation is None):
        raise PolicyError(
            "invalid_cursor",
            "A diagnostic cursor requires --snapshot and --generation.",
            restart,
            2,
        )
    status = policy_status(profile)
    if status.error is not None:
        if cursor:
            raise PolicyError(
                "invalid_cursor",
                "Policy changed; this diagnostic page is unavailable. Start again.",
                restart,
                2,
            )
        records, duplicates = warning_records((str(status.error),), include_text=include_text)
        records = tuple(
            replace(record, source="operational_warning", severity="error")
            for record in records
            if code is None or record.code == code
        )
        return DiagnosticPage(
            profile,
            scope,
            None,
            None,
            records,
            len(records),
            duplicates,
            None,
            limit,
            code,
            include_text,
            status.error.condition,
        )
    current, _, why = read_snapshot(profile)
    envelope = current
    if snapshot_id is not None or generation is not None:
        if (
            current is None
            or (snapshot_id is not None and current.snapshot_id != snapshot_id)
            or (generation is not None and current.generation != generation)
        ):
            from ssgrep.accounting.views.snapshots import read_by_snapshot_id, read_generation

            try:
                if generation is not None:
                    envelope = read_generation(profile, generation)
                else:
                    assert snapshot_id is not None
                    envelope = read_by_snapshot_id(profile, snapshot_id)
            except (OSError, ValueError) as error:
                raise PolicyError(
                    "diagnostic_snapshot_unreadable",
                    "The selected diagnostic snapshot cannot be read; nothing changed.",
                    restart,
                    2,
                ) from error
        if envelope is None:
            raise PolicyError(
                "diagnostic_snapshot_expired",
                "The selected diagnostic snapshot expired; start a new view. "
                "Current data was not substituted.",
                restart,
                2,
            )
        if snapshot_id is not None and envelope.snapshot_id != snapshot_id:
            raise PolicyError(
                "invalid_snapshot", "Snapshot identity and generation do not match.", restart, 2
            )
    if envelope is None:
        summary = read_summary(profile, scope)
        return DiagnosticPage(
            profile,
            scope,
            None,
            None,
            (),
            0,
            0,
            None,
            limit,
            code,
            include_text,
            why or summary.reason,
        )
    warnings = envelope.warnings
    if not status.profile.collection_enabled:
        warnings = (*warnings, "measurement_off: showing retained history only")
    if not scope.is_default:
        if current is None or current.generation != envelope.generation:
            raise PolicyError(
                "diagnostic_selection_expired",
                "Filtered notes need this snapshot's committed ledger; "
                "start a new selection after collection.",
                restart,
                2,
            )
        summary = read_summary(profile, scope)
        if summary.snapshot_id != envelope.snapshot_id or summary.generation != envelope.generation:
            raise PolicyError(
                "diagnostic_selection_expired",
                "Collection advanced during selection; start again.",
                restart,
                2,
            )
        warnings = summary.warnings
    selected_warnings = tuple(
        message for message in warnings if code is None or warning_code(message) == code
    )
    records, duplicates = warning_records(selected_warnings, include_text=True)
    native = set(envelope.warnings)
    selected = tuple(
        replace(
            record,
            source="snapshot_warning"
            if record.message in native
            else ("operational_warning" if record.code == "measurement_off" else "selection_note"),
            message=record.message if include_text else None,
        )
        for record in records
    )
    binding = json.dumps(
        [
            profile,
            envelope.snapshot_id,
            envelope.generation,
            asdict(scope),
            code,
            limit,
            include_text,
            [r.record_id for r in selected],
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    binding = hashlib.sha256(binding.encode()).hexdigest()
    offset = redeem("diag", cursor, binding, restart)
    page = selected[offset : offset + limit]
    next_offset = offset + len(page)
    next_cursor = issue("diag", next_offset, binding) if next_offset < len(selected) else None
    return DiagnosticPage(
        profile,
        scope,
        envelope.snapshot_id,
        envelope.generation,
        page,
        len(selected),
        duplicates,
        next_cursor,
        limit,
        code,
        include_text,
    )


def diagnostics_document(view: DiagnosticPage) -> dict[str, object]:
    from ssgrep.utilities.textsafe import printable

    records = []
    for record in view.records:
        row = asdict(record)
        if record.message is None:
            row.pop("message")
        else:
            row["message"] = printable(record.message)
        records.append(row)
    return {
        "profile": view.profile,
        "scope": asdict(view.scope),
        "diagnostic_scope": "snapshot-wide; filters also select ledger notes",
        "snapshot_id": view.snapshot_id,
        "generation": view.generation,
        "records": records,
        "total": view.total,
        "alias_duplicates": view.alias_duplicates,
        "next_cursor": view.next_cursor,
        "limit": view.limit,
        "code": view.code,
        "include_text": view.include_text,
        "reason": view.reason,
    }


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
    return NextAction(target_command(profile, destination, Signal(signal), mode), why)


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
            "Rotate the credential the policy's credential reference names (a successful check "
            "alone does not unblock sending), verify it with this check, then send.",
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
        action = NextAction(command_for(profile, "savings status"), "Watch delivery progress.")
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
                            command_for(profile, "savings status"), "Inspect profile state."
                        ),
                    )
                )
                continue
            line = _line(profile, destination.destination_id, signal.value, is_on, state, now)
            if (
                policy.paused
                and line.action is not None
                and " savings export check " in f" {line.action.command} "
            ):
                line = replace(
                    line,
                    cause=f"{line.cause} Sending is paused, so nothing is retried until resume.",
                    action=NextAction(
                        command_for(profile, "savings resume"),
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


STALE_LIMIT_NOTE = "three collection intervals (180 s)"
_DELIVERY_PROBLEMS = (
    "credentials_expired",
    "blocked",
    "partial_export",
    "unknown",
    "offline",
    "unreadable",
)


def _state_action(profile: str, text: str, why: str) -> NextAction:
    return NextAction(command_for(profile, text), why)


def _age_text(seconds: int) -> str:
    """Format a stale snapshot's known age (strictly above the 180-second limit)."""
    if seconds < 7200:
        return f"{seconds // 60} min"
    if seconds < 172800:
        return f"{seconds // 3600} h"
    return f"{seconds // 86400} d"


def _partial_cause(view: SummaryView) -> str:
    cov = view.coverage
    known = []
    recorded, attributed = cov.get("recorded_lookups"), cov.get("attributed_lookups")
    if recorded is not None and attributed is not None:
        known.append(f"{attributed} of {recorded} recorded lookups are attributed")
    costs, receipts = cov.get("source_costs_available"), cov.get("candidate_receipts")
    if costs is not None and receipts is not None and receipts:
        known.append(f"source costs are known for {costs} of {receipts} candidate receipts")
    for name, label in (
        ("ambiguous_joins", "ambiguous joins"),
        ("unavailable_sources", "unavailable source records"),
    ):
        value = cov.get(name)
        if value:
            known.append(f"{value} {label}")
    detail = "; ".join(known) if known else "some evidence was not fully read"
    return f"Estimates cover only part of the evidence: {detail}."


def summary_conditions(
    view: SummaryView,
    delivery: tuple[DeliveryLine, ...],
    sending_paused_reason: str | None,
) -> tuple[Condition, ...]:
    p = view.profile
    out: list[Condition] = []
    state = view.state
    if state is SnapshotState.DISABLED:
        out.append(
            Condition(
                "disabled",
                "Measurement is off; nothing is collected or exported, "
                "and reading does not turn it on.",
                _state_action(p, "savings setup", "Preview what would be kept, then enable."),
            )
        )
    elif state is SnapshotState.NO_SNAPSHOT:
        out.append(
            Condition(
                "enabled_no_snapshot",
                "Capture is enabled but no collection has completed, so there are no numbers yet.",
                _state_action(
                    p, "savings", "Run the first summary; it refreshes a missing snapshot."
                ),
            )
        )
    elif state in (SnapshotState.CORRUPT, SnapshotState.UNSUPPORTED_SCHEMA):
        why = (view.reason or state.value).split(":")[0]
        out.append(
            Condition(
                state.value,
                f"The latest snapshot cannot be read ({why}), so no "
                "numbers are shown. Nothing was changed or rewritten.",
                _state_action(p, "savings status", "Inspect state without changing it."),
            )
        )
    if view.paused:
        since = view.paused_at or "an unknown time"
        out.append(
            Condition(
                "paused",
                f"Paused since {since}: capture, scheduled collection and sending are stopped; "
                "retained values stay readable and the paused interval is not imported.",
                _state_action(p, "savings resume", "Resume capture, collection and sending."),
            )
        )
    if view.stale and view.collected_at:
        assert view.age_seconds is not None
        out.append(
            Condition(
                "stale",
                f"Last good snapshot was collected {_age_text(view.age_seconds)} ago "
                f"({view.collected_at}), older than {STALE_LIMIT_NOTE}. "
                "These are the last good values, not new ones.",
                None if view.paused else _state_action(p, "savings", "Refresh the stale snapshot."),
            )
        )
    if view.no_reuse:
        reasons = ", ".join(f"{n} {r}" for r, n in view.exclusions) or "none recorded"
        out.append(
            Condition(
                "no_reuse",
                "No traceable reuse was found: usage is measured, so no savings are claimed. "
                f"Excluded: {reasons}.",
                _state_action(p, "savings status", "See coverage and what was read."),
            )
        )
    elif view.partial and view.has_values:
        out.append(
            Condition(
                "partial",
                _partial_cause(view),
                _state_action(p, f"savings explain {view.receipt_id}", "Inspect a real receipt.")
                if view.receipt_id
                else _state_action(p, "savings status", "See coverage detail."),
            )
        )
    if view.has_negative:
        out.append(
            Condition(
                "negative",
                "In a negative scenario, retrieval cost exceeded the estimated credit "
                "(minus sign = net cost, not a benefit).",
                _state_action(p, f"savings explain {view.receipt_id}", "See what drove the cost.")
                if view.receipt_id
                else _state_action(p, "savings status", "See coverage detail."),
            )
        )
    if view.index == "unavailable" and view.has_values:
        out.append(
            Condition(
                "index_unavailable",
                "The search index is unavailable: usage is still measured, but valuation of "
                "source content may be missing. Nothing is indexed automatically.",
                NextAction("ssgrep index", "Build the search index yourself if you want it."),
            )
        )
    expired = view.evidence_expired_before
    if expired or view.frozen_roots:
        boundary = f" before {expired}" if expired else ""
        out.append(
            Condition(
                "evidence_expired",
                f"Detailed evidence{boundary} is past retention ({view.frozen_roots or 0} "
                "conversation root(s) frozen): retained totals remain but cannot be recomputed.",
                _state_action(p, "savings history", "See which periods are not recomputable."),
            )
        )
    for line in delivery:
        if line.state in _DELIVERY_PROBLEMS:
            out.append(
                Condition(
                    line.state, f"{line.destination_id}/{line.signal}: {line.cause}", line.action
                )
            )
    if sending_paused_reason and not view.paused:
        out.append(
            Condition(
                "sending_paused",
                f"Sending is paused until the policy is corrected ({sending_paused_reason}).",
                NextAction(command_for(p, "savings setup <policy-file>"), "Correct the policy."),
            )
        )
    return tuple(out)


def merged_actions(
    primary: tuple[NextAction, ...], conditions: tuple[Condition, ...]
) -> tuple[NextAction, ...]:
    """Primary action first, then each condition's distinct command (placeholders skipped)."""
    seen: set[str] = set()
    out: list[NextAction] = []
    candidates = [*primary, *(c.action for c in conditions if c.action is not None)]
    for action in candidates:
        if action.command in seen or "<" in action.command:
            continue
        seen.add(action.command)
        out.append(action)
    return tuple(out)
