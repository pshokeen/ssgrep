"""Closed grammar for optional subprocess measurements on observer result events.

An independently observed native result or a process-start/signal failure need not have
returned subprocess measurements. Absence never synthesizes capture facts or grants trust.
When present, capture is a flat, bounded schema; it cannot carry text, paths or extra data.
Raw pipe-byte counts deliberately remain distinct from rendered result/blob byte counts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MAX_CAPTURE_LIMIT_BYTES = 1_000_000  # The actual observed SQL/ssgrep shim stream ceiling.
_MAX_COUNT = 2**63 - 1
_FIELDS = frozenset(
    {
        "returncode",
        "pid",
        "timed_out",
        "cancelled",
        "capture_status",
        "capture_complete",
        "termination_cause",
        "pipes_drained",
        "observed_over_limit_streams",
        "output_limit",
        "output_limit_streams",
        "stdout_bytes",
        "stderr_bytes",
        "stdout_limit_bytes",
        "stderr_limit_bytes",
        "stdout_truncated",
        "stderr_truncated",
    }
)
_FLAGS = (
    "timed_out",
    "cancelled",
    "capture_complete",
    "pipes_drained",
    "output_limit",
    "stdout_truncated",
    "stderr_truncated",
)
_STATUSES = frozenset(
    {
        "completed",
        "output_limit",
        "timeout",
        "cancelled",
        "interrupted",
        "invalid_output_encoding",
        "capture_error",
    }
)
_STREAMS = ("stdout", "stderr")


def _integer(value: Any, low: int = 0, high: int = _MAX_COUNT) -> bool:
    return type(value) is int and low <= value <= high


def _streams(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= len(_STREAMS)
        and all(type(item) is str and item in _STREAMS for item in value)
        and value == [name for name in _STREAMS if name in value]
    )


def valid_capture(value: Any, result: Mapping[str, Any]) -> bool:
    """Validate measurements without treating solver-provided metadata as authority."""
    if not isinstance(value, dict):
        return False
    keys = set(value)
    if not _FIELDS <= keys or keys - _FIELDS - {"capture_observation"}:
        return False
    status = value["capture_status"]
    if type(status) is not str or status not in _STATUSES:
        return False
    if any(type(value[key]) is not bool for key in _FLAGS):
        return False
    if not _integer(value["pid"], 1, 2**31 - 1) or not _integer(
        value["returncode"], -(2**31), 2**31 - 1
    ):
        return False
    for name in _STREAMS:
        if (
            not _integer(value[name + "_bytes"])
            or not _integer(value[name + "_limit_bytes"], 0, MAX_CAPTURE_LIMIT_BYTES)
            or value[name + "_truncated"] != (value[name + "_bytes"] > value[name + "_limit_bytes"])
        ):
            return False
    observed = value["observed_over_limit_streams"]
    triggered = value["output_limit_streams"]
    if not _streams(observed) or not _streams(triggered):
        return False
    if observed != [name for name in _STREAMS if value[name + "_truncated"]]:
        return False
    if (
        value["timed_out"] != (status == "timeout")
        or value["cancelled"] != (status == "cancelled")
        or value["output_limit"] != (status == "output_limit")
        or value["capture_complete"] != (status == "completed")
        or value["termination_cause"] != (None if status == "completed" else status)
    ):
        return False
    if status == "output_limit":
        if not triggered or any(name not in observed for name in triggered):
            return False
    elif triggered:
        return False
    if "capture_observation" in value:
        phase = value["capture_observation"]
        if phase not in ("final", "termination_snapshot"):
            return False
        if phase == "termination_snapshot" and (status != "output_limit" or value["pipes_drained"]):
            return False
    code = result.get("exit_code")
    if code is not None and code != value["returncode"]:
        return False
    if status == "completed":
        return (
            value["pipes_drained"]
            and not observed
            and value["returncode"] >= 0
            and result.get("state") == "verified"
            and code is not None
        )
    return (
        result.get("state") == "interrupted"
        and result.get("reason") == status
        and (code is not None or status == "timeout")
        and result.get("stdout_chars") == 0
        and result.get("stdout_bytes") == 0
        and result.get("blob") is None
    )
