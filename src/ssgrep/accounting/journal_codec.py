"""Bounded canonical serialization and strict parsing of journal events.

One event is one newline-terminated canonical JSON object of at most
``MAX_EVENT_BYTES`` (including the newline). When optional reference metadata
would exceed the bound it is dropped from the end and the event carries
``refs_omitted=true``; identity and outcome fields are never shortened. A
strict prefix of a canonical JSON object is never itself valid JSON, so a
short write can never be mistaken for a complete record.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from ssgrep.accounting.contracts import (
    MAX_EVENT_BYTES,
    SCHEMA_VERSION,
    Basis,
    ExecutionEvent,
    Operation,
    Outcome,
    Phase,
    Quantity,
    Transport,
    canonical_bytes,
)

_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_TEXT_FIELDS = ("profile", "ts_utc", "fingerprint", "version")
_OPTIONAL_TEXT = ("cwd", "index_generation", "correlation_id")
_OPTIONAL_INT = ("mono_ns", "limit", "token_budget", "omitted_count")


class EventTooLarge(ValueError):
    """Identity and outcome alone exceed the event bound (never truncated)."""


class MalformedEvent(ValueError):
    """A record that must not be admitted as complete evidence."""


def _line(event: ExecutionEvent) -> bytes:
    return canonical_bytes(event) + b"\n"


def serialize_event(event: ExecutionEvent) -> bytes:
    """The bounded journal line for ``event``; excess references are marked omitted."""
    data = _line(event)
    if len(data) <= MAX_EVENT_BYTES:
        return data
    low, high = 0, max(len(event.refs), len(event.ranks_scores))
    best: bytes | None = None
    while low < high:  # largest prefix that fits (the empty prefix may not)
        mid = (low + high + 1) // 2
        trial = _line(_trimmed(event, mid))
        if len(trial) <= MAX_EVENT_BYTES:
            low, best = mid, trial
        else:
            high = mid - 1
    if best is None:
        best = _line(_trimmed(event, 0))
    if len(best) > MAX_EVENT_BYTES:
        raise EventTooLarge(f"event identity and outcome need {len(best)} bytes")
    return best


def _trimmed(event: ExecutionEvent, keep: int) -> ExecutionEvent:
    return replace(
        event,
        refs=event.refs[:keep],
        ranks_scores=event.ranks_scores[:keep],
        refs_omitted=True,
    )


def _expect(condition: bool, reason: str) -> None:
    if not condition:
        raise MalformedEvent(reason)


def _quantity(raw: Any) -> Quantity | None:
    if raw is None:
        return None
    _expect(isinstance(raw, dict), "rendered_bytes is not an object")
    value = raw.get("value")
    _expect(value is None or (isinstance(value, int) and not isinstance(value, bool)), "bad value")
    try:
        return Quantity(
            value,
            str(raw["unit"]),
            Basis(raw["basis"]),
            str(raw.get("method", "")),
            str(raw.get("reason", "")),
        )
    except (KeyError, ValueError) as error:
        raise MalformedEvent(f"invalid quantity: {error}") from error


def _int_pairs(raw: Any) -> tuple[tuple[int, int], ...]:
    _expect(isinstance(raw, list), "ranks_scores is not a list")
    pairs: list[tuple[int, int]] = []
    for item in raw:
        ok = isinstance(item, list) and len(item) == 2
        _expect(ok and all(type(n) is int for n in item), "ranks_scores holds a non-integer pair")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def parse_event(raw: bytes) -> ExecutionEvent:
    """Strictly decode one journal line (without its newline) or raise ``MalformedEvent``."""
    try:
        document = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise MalformedEvent("not valid JSON (partial or corrupted record)") from error
    _expect(isinstance(document, dict), "record is not a JSON object")
    _expect(document.get("schema") == SCHEMA_VERSION, "unsupported schema version")
    execution_id = document.get("execution_id")
    _expect(isinstance(execution_id, str) and bool(_ID_RE.match(execution_id)), "bad execution_id")
    for name in _TEXT_FIELDS:
        _expect(isinstance(document.get(name), str), f"{name} must be text")
    for name in _OPTIONAL_TEXT:
        _expect(document.get(name) is None or isinstance(document[name], str), f"bad {name}")
    for name in _OPTIONAL_INT:
        value = document.get(name)
        _expect(value is None or type(value) is int, f"{name} must be an integer")
    for name in ("truncated",):
        _expect(document.get(name) is None or type(document[name]) is bool, f"bad {name}")
    _expect(type(document.get("refs_omitted")) is bool, "refs_omitted must be boolean")
    refs = document.get("refs")
    _expect(isinstance(refs, list) and all(isinstance(r, str) for r in refs), "refs not text")
    try:
        phase = Phase(document["phase"])
        operation = Operation(document["operation"])
        transport = Transport(document["transport"])
        outcome = None if document.get("outcome") is None else Outcome(document["outcome"])
    except (KeyError, ValueError) as error:
        raise MalformedEvent(f"unknown enum value: {error}") from error
    _expect(phase is not Phase.COMPLETED or outcome is not None, "completed needs an outcome")
    return ExecutionEvent(
        execution_id=execution_id,
        phase=phase,
        operation=operation,
        transport=transport,
        profile=document["profile"],
        ts_utc=document["ts_utc"],
        cwd=document.get("cwd"),
        fingerprint=document["fingerprint"],
        version=document["version"],
        outcome=outcome,
        mono_ns=document.get("mono_ns"),
        refs=tuple(refs),
        ranks_scores=_int_pairs(document.get("ranks_scores")),
        limit=document.get("limit"),
        token_budget=document.get("token_budget"),
        omitted_count=document.get("omitted_count"),
        truncated=document.get("truncated"),
        rendered_bytes=_quantity(document.get("rendered_bytes")),
        index_generation=document.get("index_generation"),
        refs_omitted=document["refs_omitted"],
        correlation_id=document.get("correlation_id"),
    )
