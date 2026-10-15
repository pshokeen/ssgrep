"""Complete bounded warning pages tied to an immutable snapshot and selection.

Reading never pins or creates files. Retained generation files provide continuity
across collection; expiry is explicit. Filter-derived notes require the matching
committed ledger, so an advanced ledger fails rather than inventing old notes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import asdict, replace

from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.diagnostic_types import DiagnosticPage
from ssgrep.accounting.diagnostics import warning_code, warning_records
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.service_cursor import issue, redeem
from ssgrep.accounting.service_types import Scope, command_for

_ID = re.compile(r"^[0-9a-f]{16,64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.]{0,79}$")


def diagnostics_command(
    profile: str,
    scope: Scope,
    snapshot_id: str | None,
    generation: int | None,
    *,
    limit: int = 20,
    cursor: str | None = None,
    code: str | None = None,
    include_text: bool = False,
) -> str:
    parts = ["savings", "diagnostics"]
    for flag, value in (
        ("--generation", generation),
        ("--snapshot", snapshot_id),
        ("--since", scope.since),
        ("--until", scope.until),
        ("--project", scope.project),
        ("--runtime", scope.runtime),
        ("--session", scope.session),
        ("--task", scope.task),
        ("--by", scope.by),
        ("--code", code),
        ("--cursor", cursor),
    ):
        if value is not None:
            parts.extend((flag, shlex.quote(str(value))))
    if scope.include_children:
        parts.append("--include-children")
    if include_text:
        parts.append("--include-text")
    if limit != 20:
        parts.extend(("--limit", str(limit)))
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
    from ssgrep.accounting.policy import policy_status
    from ssgrep.accounting.scope import build_scope, check_limit, check_profile
    from ssgrep.accounting.service import read_snapshot, read_summary

    check_profile(profile)
    check_limit(limit)
    scope = scope or Scope()
    # Reuse the existing selection contract, including interval and child validation.
    scope = build_scope(**{key: value for key, value in asdict(scope).items() if key != "window"})
    restart = command_for(profile, "savings diagnostics")
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
            from ssgrep.accounting.snapshots import read_by_snapshot_id, read_generation

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
