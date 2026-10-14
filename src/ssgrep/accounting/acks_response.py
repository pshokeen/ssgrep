"""Classify an HTTP 200 OTLP/HTTP export response (design D10); never retains receiver text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ssgrep.accounting.acks_model import INT64_MAX, is_int
from ssgrep.accounting.ledger_contracts import Signal

DECIMAL_RE = re.compile(r"0|[1-9][0-9]{0,18}")
REJECTED_FIELD = {Signal.METRICS: "rejectedDataPoints", Signal.LOGS: "rejectedLogRecords"}


@dataclass(frozen=True)
class AckResult:
    outcome: str  # 'full' | 'full_with_warning' | 'partial' | 'unparseable'
    rejected: int = 0
    warning_category: str | None = None
    identity_unavailable: bool = False


class _FloatLiteral(str):
    """Marks a JSON float so it is never turned into a Python float (and never accepted)."""


_UNPARSEABLE = AckResult("unparseable")


def _rejected_count(value: object) -> int | None:
    if is_int(value) and 0 <= value <= INT64_MAX:
        return value
    if (
        isinstance(value, str)
        and not isinstance(value, _FloatLiteral)
        and DECIMAL_RE.fullmatch(value)
    ):
        return int(value) if int(value) <= INT64_MAX else None
    return None


def parse_response(signal: Signal, body: bytes | None) -> AckResult:
    """Classify an HTTP 200 OTLP response body; ``errorMessage`` text is never retained."""
    if not body:
        return AckResult("full")
    try:
        document = json.loads(
            body.decode("utf-8"), parse_float=_FloatLiteral, parse_constant=_FloatLiteral
        )
    except (ValueError, RecursionError):
        return _UNPARSEABLE
    if not isinstance(document, dict):
        return _UNPARSEABLE
    if "partialSuccess" not in document:
        return AckResult("full")
    partial = document["partialSuccess"]
    if not isinstance(partial, dict):
        return _UNPARSEABLE
    if not partial or partial == {"errorMessage": ""}:
        return AckResult("full")  # proto3 JSON default values carry no information
    rejected = _rejected_count(partial.get(REJECTED_FIELD[Signal(signal)], 0))
    if rejected is None:
        return _UNPARSEABLE
    if rejected == 0:
        return AckResult("full_with_warning", warning_category="receiver_warning")
    return AckResult("partial", rejected=rejected, identity_unavailable=True)
