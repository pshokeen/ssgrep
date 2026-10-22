"""Private, read-only JSON reports of one committed accounting publication.

This is not an OTLP projection or an approval bypass. Original lookup and history
rows are explanations, not independent savings that can be added together.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import sys
from datetime import date
from typing import Any, cast

from ssgrep.accounting.collect import store
from ssgrep.accounting.ledger.contracts import digest_of
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.policy import PolicyError, check_profile_id
from ssgrep.accounting.views import service
from ssgrep.accounting.views.types import Scope
from ssgrep.cli.savings_json import history_document, lookups_document, summary_document

_PAGE_SIZE = 20


class UnsafeReportOutput(ValueError):
    """The requested output path is unsafe, distinct from malformed report data."""


def _failure(profile: str, condition: str, message: str) -> PolicyError:
    return PolicyError(condition, message, f"ssgrep savings status --profile {profile}", 1)


def _selection(since: str | None, until: str | None) -> Scope:
    for boundary in (since, until):
        if boundary is None:
            continue
        try:
            if not isinstance(boundary, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}", boundary
            ):
                raise ValueError
            date.fromisoformat(boundary)
        except ValueError:
            raise PolicyError(
                "invalid_interval", "Report dates must be valid YYYY-MM-DD UTC days."
            ) from None
    if until == "0001-01-01" or (since is not None and until is not None and since >= until):
        raise PolicyError("invalid_interval", "Start must precede the exclusive end UTC day.")
    start = f"{since}T00:00:00Z" if since is not None else None
    end = f"{until}T00:00:00Z" if until is not None else None
    return Scope(window="lifetime", since=start, until=end)


def _checkpoint(profile: str):
    try:
        checkpoint = store.load_checkpoint(profile)
    except (store.CollectError, OSError, ValueError):
        raise _failure(
            profile, "report_unavailable", "The accepted checkpoint is unavailable."
        ) from None
    if checkpoint is None:
        raise _failure(profile, "report_unavailable", "No accepted accounting checkpoint exists.")
    return checkpoint


def _history_digest(profile: str) -> str:
    try:
        raw = storage.load_json(
            profile, "history", "periods.json", limit=storage.MAX_SNAPSHOT_BYTES
        )
        if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
            raise ValueError
        return digest_of(raw)
    except (OSError, ValueError, storage.UnsafeStorageError):
        raise _failure(profile, "report_unavailable", "Retained history is unavailable.") from None


def build_report(profile: str, since: str | None = None, until: str | None = None) -> dict:
    """Read all retained pages, failing if their committed publication changes.

    Dates select UTC days with an exclusive upper bound. No date selection means
    the lifetime retained view, not the default seven-day interactive summary.
    """
    check_profile_id(profile)
    scope = _selection(since, until)
    snapshot, failure, _ = service.read_snapshot(profile)
    if snapshot is None:
        raise _failure(
            profile,
            "report_unavailable",
            "No readable accounting snapshot is available"
            + (f" ({failure.value})." if failure else "."),
        )
    checkpoint = _checkpoint(profile)
    if (
        checkpoint.generation != snapshot.generation
        or checkpoint.methodology != snapshot.methodology_id
    ):
        raise _failure(
            profile,
            "report_snapshot_changed",
            "Accounting publication is not committed consistently.",
        )
    summary = service.read_summary(profile, scope)
    if (summary.snapshot_id, summary.generation) != (snapshot.snapshot_id, snapshot.generation):
        raise _failure(
            profile, "report_snapshot_changed", "The accounting snapshot changed during the report."
        )
    if not summary.has_values or (not scope.is_default and not summary.filtered):
        raise _failure(
            profile, "report_unavailable", "The requested accounting summary is unavailable."
        )
    data = cast(dict[str, Any], summary_document(summary)["data"])
    lookup_data: dict[str, Any] | None = None
    lookup_rows = []
    cursor = None
    lookup_warnings = 0
    while True:
        try:
            page = service.read_lookups(
                profile,
                scope,
                snapshot_id=snapshot.snapshot_id,
                generation=snapshot.generation,
                limit=_PAGE_SIZE,
                cursor=cursor,
            )
        except PolicyError as error:
            if error.condition in {"invalid_cursor", "lookup_snapshot_expired"}:
                error = _failure(
                    profile,
                    "report_snapshot_changed",
                    "Accounting data changed during the report; retry this export.",
                )
            raise error from None
        if page.reason:
            raise _failure(
                profile, "report_unavailable", "Retained lookup evidence is unavailable."
            )
        encoded = cast(dict[str, Any], lookups_document(page)["data"])
        if lookup_data is None:
            lookup_data = encoded
            lookup_warnings = len(page.warnings)
        lookup_rows.extend(encoded["rows"])
        cursor = page.next_cursor
        if cursor is None:
            break
    history_digest = _history_digest(profile)
    first_history = None
    history_data: dict[str, Any] | None = None
    history_rows = []
    cursor = None
    while True:
        try:
            page = service.read_history(
                profile, scope.since, scope.until, limit=_PAGE_SIZE, cursor=cursor
            )
        except PolicyError as error:
            if error.condition == "invalid_cursor":
                error = _failure(
                    profile,
                    "report_snapshot_changed",
                    "Accounting data changed during the report; retry this export.",
                )
            raise error from None
        if page.reason in {"history_unreadable", "history_unsupported_schema"}:
            raise _failure(
                profile, "report_unavailable", "Retained history cannot be interpreted safely."
            )
        encoded = cast(dict[str, Any], history_document(page)["data"])
        if first_history is None:
            first_history, history_data = page, encoded
        history_rows.extend(encoded["entries"])
        cursor = page.next_cursor
        if cursor is None:
            break
    # Re-reading the first history page also rechecks its whole-history digest
    # cursor. For a single page, equality checks every retained projected row.
    final_history = service.read_history(profile, scope.since, scope.until, limit=_PAGE_SIZE)
    final_snapshot, _, _ = service.read_snapshot(profile)
    final_checkpoint = _checkpoint(profile)
    if (
        final_history != first_history
        or _history_digest(profile) != history_digest
        or final_snapshot != snapshot
        or final_checkpoint.generation != checkpoint.generation
        or final_checkpoint.methodology != checkpoint.methodology
        or final_checkpoint.blobs != checkpoint.blobs
    ):
        raise _failure(
            profile,
            "report_snapshot_changed",
            "Accounting data changed during the report; try again.",
        )
    assert lookup_data is not None and history_data is not None
    lookup_data.update(rows=lookup_rows, next_cursor=None)
    lookup_data.pop("limit")
    history_data.update(entries=history_rows, next_cursor=None)
    return {
        "schema_version": 1,
        "metadata": {
            "profile": profile,
            "generation": snapshot.generation,
            "snapshot_id": snapshot.snapshot_id,
            "collected_at": snapshot.collected_at,
            "methodology_id": snapshot.methodology_id,
            "selection": {
                "window": None if scope.since or scope.until else "lifetime",
                "since": scope.since,
                "until": scope.until,
                "interval": "utc_half_open",
            },
            "retention": data["retention"],
            "availability": {
                "state": summary.state.value,
                "estimates_reason": summary.estimates_reason,
                "history_reason": history_data["reason"],
                "summary_warning_count": len(summary.warnings),
                "lookup_warning_count": lookup_warnings,
            },
            "caveats": [
                "Only retained committed data is included; missing evidence is not zero.",
                "Untimed lookups remain in lifetime results but are excluded by date filters.",
                "Summary coverage and retention describe collection, "
                "not proof of completeness in the selected interval.",
                "This private local report does not grant permission for remote export.",
            ],
            "semantics": {
                "lookup_values": "nonadditive",
                "original_receipt_values": "nonadditive",
                "history_totals": "current_revisions_only",
                "estimates_are_causal": False,
                "billing_savings": None,
            },
        },
        "usage": data["measured"],
        "estimates": data["estimates"],
        "lookups": lookup_data,
        "history": history_data,
        "coverage": {
            **data["coverage"],
            "partial": bool(
                data["coverage"]["partial"]
                or lookup_warnings
                or history_data["reason"] not in (None, "no_closed_periods")
            ),
        },
    }


def _output_directory(output: str) -> tuple[int, str]:
    if (
        not isinstance(output, str)
        or not output
        or len(os.fsencode(output)) > 4096
        or any(ord(char) < 32 for char in output)
        or output.endswith("/")
        or ".." in output.split("/")
    ):
        raise UnsafeReportOutput("Unsafe report output path.")
    parts = os.path.abspath(output).split("/")[1:]
    if not parts or any(not part or len(os.fsencode(part)) > 255 for part in parts):
        raise UnsafeReportOutput("Unsafe report output path.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open("/", flags)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        return directory, parts[-1]
    except BaseException:
        os.close(directory)
        raise


def _check_target(directory: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise UnsafeReportOutput("Report output must be an owned regular file, never a symlink.")


def write_report(document: dict, output: str) -> None:
    """Write one UTF-8 JSON document, privately and atomically, or only to stdout.

    Serialization completes before opening any output, and parent directories
    are never created. Every directory descriptor is closed on every exit path.
    """
    text = (
        json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    )
    payload = text.encode("utf-8")
    if output == "-":
        sys.stdout.write(text)
        return
    directory, name = _output_directory(output)
    temporary = None
    try:
        _check_target(directory, name)
        candidate = f".ssgrep-report-{secrets.token_hex(12)}.tmp"
        descriptor = os.open(
            candidate,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        temporary = candidate
        try:
            os.fchmod(descriptor, 0o600)
            storage.write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _check_target(directory, name)
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        temporary = None
        os.fsync(directory)
    finally:
        try:
            if temporary is not None:
                os.unlink(temporary, dir_fd=directory)
        finally:
            os.close(directory)
