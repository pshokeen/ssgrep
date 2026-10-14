"""Read-only response builders for the loopback endpoints (no collect, index, send, or create).

Every function here only OPENS EXISTING owner-safe files (``open_profile_dir(create=False)``)
and returns bytes plus headers. Snapshot bodies are the exact verified bytes of one immutable
generation file, so a read can never mix generations. The ETag is the snapshot digest;
freshness (state, age, pause) travels in ``X-Ssgrep-*`` headers so the body stays exactly the
digested content and a 304 cannot hide a state change.

Delivery state is read from ``delivery/<destination>/<signal>.json`` (written only by the export
module) through a positive allowlist: unknown keys, wrong types, and free text are dropped, so no
remote error body, credential, or path can reach a response.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ssgrep.accounting import metrics, snapshots, storage
from ssgrep.accounting.export_spool_meta import valid_batch_id
from ssgrep.accounting.ledger_contracts import Signal, SnapshotEnvelope, valid_profile_id
from ssgrep.accounting.metrics_render import CONTENT_TYPE, render_openmetrics

DELIVERY_DIR = "delivery"
MAX_DELIVERY_BYTES = 1024 * 1024
MAX_LOSS_ROWS = 16
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_CATEGORY = re.compile(r"[a-z][a-z0-9_]{0,47}")
_SNAPSHOT_ID = re.compile(r"[0-9a-f]{16,64}")
_INT_FIELDS = (
    "last_collected_generation",
    "last_attempted_generation",
    "last_acked_generation",
    "pending_batches",
    "rejected_points",
    "unknown_batches",
    "partial_batches",
    "quarantined_batches",
    "lost_records",
)
_CATEGORY_FIELDS = ("blocked_reason", "last_error_category", "warning_category")
_UTC_FIELDS = ("last_attempt_at", "last_acked_at", "next_retry_utc")
_BOOL_FIELDS = ("identity_unavailable", "enabled")
JSON_TYPE = "application/json; charset=utf-8"


@dataclass(frozen=True)
class Response:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


def json_response(status: int, document: Any, extra: tuple[tuple[str, str], ...] = ()) -> Response:
    body = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    return Response(status, (("Content-Type", JSON_TYPE), *extra), body)


def error_response(status: int, code: str, extra: tuple[tuple[str, str], ...] = ()) -> Response:
    """Fixed machine-readable error; ``code`` is always a constant, never request text."""
    return json_response(status, {"error": code}, extra)


# ---- snapshot reads --------------------------------------------------------------------------


@dataclass(frozen=True)
class Current:
    envelope: SnapshotEnvelope
    raw: bytes


def load_current(profile: str) -> tuple[Current | None, str | None]:
    """(snapshot, None) | (None, None) when none ever published | (None, failure code)."""
    try:
        directory = storage.open_profile_dir(profile, snapshots.SNAPSHOT_DIR, create=False)
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, "snapshot_unreadable"
    try:
        raw = storage.read_file(directory, snapshots.CURRENT_NAME, snapshots.MAX_SNAPSHOT_BYTES)
    except (OSError, ValueError):
        return None, "snapshot_unreadable"
    finally:
        os.close(directory)
    if raw is None:
        return None, None
    try:
        return Current(snapshots.decode(raw), raw), None
    except ValueError:
        return None, "snapshot_unreadable"


def _etag(snapshot_id: str) -> str:
    return f'"{snapshot_id}"'


def etag_matches(header: str | None, snapshot_id: str) -> bool:
    """RFC 9110 weak comparison of an ``If-None-Match`` list against the snapshot ETag."""
    if header is None:
        return False
    for item in header.split(","):
        candidate = item.strip()
        if candidate == "*" or candidate.removeprefix("W/") == _etag(snapshot_id):
            return True
    return False


def freshness(
    envelope: SnapshotEnvelope, now: datetime, stale_after: int, paused: bool
) -> tuple[tuple[str, str], ...]:
    try:
        collected = datetime.strptime(envelope.collected_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=UTC
        )
    except ValueError:
        age, stale = None, True
    else:
        elapsed = now - collected
        age, stale = max(int(elapsed.total_seconds()), 0), elapsed > timedelta(seconds=stale_after)
    return (
        ("X-Ssgrep-State", "stale" if stale else "fresh"),
        ("X-Ssgrep-Paused", "true" if paused else "false"),
        ("X-Ssgrep-Age-Seconds", "unknown" if age is None else str(age)),
        ("X-Ssgrep-Stale-After-Seconds", str(stale_after)),
        ("X-Ssgrep-Generation", str(envelope.generation)),
        ("X-Ssgrep-Collected-At", envelope.collected_at),
    )


def snapshot_response(
    current: Current,
    if_none_match: str | None,
    state_headers: tuple[tuple[str, str], ...],
    view: str = "current",
) -> Response:
    headers = (
        ("ETag", _etag(current.envelope.snapshot_id)),
        ("Cache-Control", "no-cache"),
        ("X-Ssgrep-View", view),
        *state_headers,
    )
    if etag_matches(if_none_match, current.envelope.snapshot_id):
        return Response(304, headers, b"")
    return Response(200, (("Content-Type", JSON_TYPE), *headers), current.raw)


# ---- delivery state ---------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        return None
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return value


def _loss_rows(raw: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw[:MAX_LOSS_ROWS] if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        reason = item.get("reason")
        row: dict[str, Any] = {
            "reason": reason if isinstance(reason, str) and _CATEGORY.fullmatch(reason) else "other"
        }
        for name in ("batches", "records"):
            row[name] = item[name] if _is_int(item.get(name)) else None
        row["at"] = _timestamp(item.get("at"))
        rows.append(row)
    return rows


def _nonrecallable(data: dict[str, Any]) -> bool | None:
    """Expose a validated pending-send fact, never its private batch identifier."""
    if "in_flight_nonrecallable" not in data:
        return None
    value = data["in_flight_nonrecallable"]
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return True if valid_batch_id(value) else None


def sanitize_delivery(raw: object) -> dict[str, Any]:
    """Allowlisted view of one delivery file; absent/invalid fields are ``None`` (unknown)."""
    data = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {}
    for name in _INT_FIELDS:
        out[name] = data[name] if _is_int(data.get(name)) else None
    for name in _CATEGORY_FIELDS:
        value = data.get(name)
        out[name] = value if isinstance(value, str) and _CATEGORY.fullmatch(value) else None
    for name in _UTC_FIELDS:
        out[name] = _timestamp(data.get(name))
    for name in _BOOL_FIELDS:
        out[name] = data[name] if isinstance(data.get(name), bool) else None
    out["in_flight_nonrecallable"] = _nonrecallable(data)
    acked = data.get("last_acked_snapshot_id")
    out["last_acked_snapshot_id"] = (
        acked if isinstance(acked, str) and _SNAPSHOT_ID.fullmatch(acked) else None
    )
    out["loss_summary"] = _loss_rows(data.get("loss_summary"))
    return out


def _list_names(directory: int) -> list[str]:
    try:
        return storage.list_names(directory)
    except OSError:
        return []


def read_delivery_file(profile: str, destination: str, signal: str) -> tuple[str, Any]:
    """('present', parsed) | ('absent', None) | ('unreadable', None); never creates anything."""
    try:
        directory = storage.open_profile_dir(profile, DELIVERY_DIR, destination, create=False)
    except FileNotFoundError:
        return "absent", None
    except (OSError, ValueError):
        return "unreadable", None
    try:
        raw = storage.read_file(directory, f"{signal}.json", MAX_DELIVERY_BYTES)
        return ("absent", None) if raw is None else ("present", json.loads(raw))
    except (OSError, ValueError, RecursionError):
        return "unreadable", None
    finally:
        os.close(directory)


def destination_ids(profile: str, configured: tuple[tuple[str, tuple[str, ...]], ...]) -> list[str]:
    found = {destination for destination, _ in configured}
    try:
        root = storage.open_profile_dir(profile, DELIVERY_DIR, create=False)
    except (OSError, ValueError):
        root = None
    if root is not None:
        try:
            found.update(n for n in _list_names(root) if valid_profile_id(n))
        finally:
            os.close(root)
    return sorted(found)


def delivery_rows(
    profile: str, configured: tuple[tuple[str, tuple[str, ...]], ...]
) -> list[dict[str, Any]]:
    """One row per destination x signal (configured or having state), bounded and sanitized."""
    signals = {destination: set(sigs) for destination, sigs in configured}
    rows: list[dict[str, Any]] = []
    for destination in destination_ids(profile, configured):
        for signal in Signal:
            status, raw = read_delivery_file(profile, destination, signal.value)
            if status == "absent" and signal.value not in signals.get(destination, ()):
                continue
            state = {"present": "reported", "absent": "no_delivery_state"}.get(status, status)
            rows.append(
                {
                    "destination_id": destination,
                    "signal": signal.value,
                    "state": state,
                    **sanitize_delivery(raw),
                }
            )
    return rows


def export_status_response(rows: list[dict[str, Any]], profile: str) -> Response:
    destinations = [
        {
            "destination_id": r["destination_id"],
            "signal": r["signal"],
            "state": r["state"],
            "enabled": r["enabled"],
            "last_attempt": {
                "at": r["last_attempt_at"],
                "generation": r["last_attempted_generation"],
            },
            "last_full_ack": {
                "at": r["last_acked_at"],
                "generation": r["last_acked_generation"],
                "snapshot_id": r["last_acked_snapshot_id"],
            },
            "last_collected_generation": r["last_collected_generation"],
            "pending_batches": r["pending_batches"],
            "rejected_points": r["rejected_points"],
            "unknown_batches": r["unknown_batches"],
            "partial_batches": r["partial_batches"],
            "quarantined_batches": r["quarantined_batches"],
            "lost_records": r["lost_records"],
            "loss_summary": r["loss_summary"],
            "blocked_reason": r["blocked_reason"],
            "last_error_category": r["last_error_category"],
            "warning_category": r["warning_category"],
            "next_retry_utc": r["next_retry_utc"],
            "in_flight_nonrecallable": r["in_flight_nonrecallable"],
            "identity_unavailable": r["identity_unavailable"],
        }
        for r in rows
    ]
    document = {
        "profile": profile,
        "acknowledgment": "collector_accepted_backend_durability_not_claimed",
        "destinations": destinations,
    }
    return json_response(200, document, (("Cache-Control", "no-store"),))


# ---- acknowledged view ------------------------------------------------------------------------


def acknowledged_current(
    profile: str, destination: str, signal: str
) -> tuple[Current | None, Response | None]:
    """The exact retained snapshot last fully acknowledged; never the current one."""
    status, raw = read_delivery_file(profile, destination, signal)
    if status == "unreadable":
        return None, error_response(500, "delivery_state_unreadable")
    state = sanitize_delivery(raw)
    snapshot_id = state["last_acked_snapshot_id"]
    if status == "absent" or snapshot_id is None:
        return None, error_response(404, "not_acknowledged")
    try:
        envelope = snapshots.read_by_snapshot_id(profile, snapshot_id)
    except (OSError, ValueError):
        return None, error_response(500, "acknowledged_snapshot_unreadable")
    if envelope is None:
        return None, error_response(410, "acknowledged_snapshot_expired")
    return Current(envelope, snapshots.encode(envelope)), None


# ---- metrics ----------------------------------------------------------------------------------


def _epoch(text: str | None) -> int | None:
    if text is None:
        return None
    return int(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp())


def export_health(rows: list[dict[str, Any]]) -> list[metrics.ExportHealth]:
    health = []
    for r in rows:
        lost = tuple(
            (row["reason"], row["records"])
            for row in r["loss_summary"]
            if row["records"] is not None
        )
        health.append(
            metrics.ExportHealth(
                destination=r["destination_id"],
                signal=r["signal"],
                last_full_ack_epoch=_epoch(r["last_acked_at"]),
                pending_batches=r["pending_batches"],
                rejected_points=r["rejected_points"],
                lost=lost,
            )
        )
    return health


def metrics_response(current: Current | None, rows: list[dict[str, Any]]) -> Response:
    try:
        samples = metrics.all_samples(
            None if current is None else current.envelope, export_health(rows)
        )
    except metrics.MetricsValidationError as error:
        return error_response(500, error.condition)
    return Response(
        200,
        (("Content-Type", CONTENT_TYPE), ("Cache-Control", "no-store")),
        render_openmetrics(samples),
    )
