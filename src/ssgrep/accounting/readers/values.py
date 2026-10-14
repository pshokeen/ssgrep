"""Value normalization shared by every raw reader.

Raw transcripts contain floats (costs, timestamps) but accounting data may not,
so transcripts are decoded with ``parse_float=Decimal`` and normalized here:
integral numbers become ints, other decimals become exact strings, and costs
become integer nano-USD.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

from ssgrep.accounting.contracts import RawEvidence, digest_of

NANO = Decimal(10) ** 9


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def loads(data: bytes | bytearray | str) -> Any:
    """Decode JSON without ever creating floats."""
    return json.loads(data, parse_float=Decimal, parse_constant=_reject_constant)


def clean(value: Any) -> Any:
    """Make a decoded value canonical-safe: no floats, no Decimals, plain containers."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else format(value, "f")
    if isinstance(value, float):
        return clean(Decimal(repr(value)))
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clean(item) for item in value]
    return value


def obj(value: Any) -> dict[str, Any]:
    """``value`` when it is a JSON object, else an empty dict (untrusted shapes never raise)."""
    return value if isinstance(value, dict) else {}


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    return None


def nano_usd(value: Any) -> int | None:
    """USD amount to integer nano-USD; None when the amount is absent/not numeric."""
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        return None
    return int((Decimal(value) * NANO).to_integral_value(ROUND_HALF_EVEN))


def to_utc(value: Any) -> str | None:
    """RFC 3339 UTC with milliseconds from an ISO string or epoch seconds/milliseconds."""
    moment: datetime | None = None
    if isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
    elif isinstance(value, int | Decimal) and not isinstance(value, bool):
        seconds = Decimal(value)
        seconds = seconds / 1000 if seconds > Decimal(10) ** 11 else seconds
        try:
            moment = datetime.fromtimestamp(float(seconds), UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if moment is None:
        return None
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def text_of(content: Any) -> str:
    """Concatenate the text of a string or a list of ``{type:text,text}`` blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def pick_ints(source: Any, names: dict[str, str]) -> dict[str, int]:
    """Map native counter names to normalized names, keeping only integer values."""
    if not isinstance(source, dict):
        return {}
    picked = {}
    for native, normalized in names.items():
        number = as_int(source.get(native))
        if number is not None:
            picked[normalized] = number
    return picked


def evidence_digest(kind: str, payload: dict[str, Any]) -> str:
    """Digest of native content; the streaming update count is bookkeeping, not content."""
    content = {k: v for k, v in payload.items() if k != "streaming_updates"}
    return digest_of({"kind": kind, "payload": content})


def make_evidence(
    *,
    runtime: str,
    source_id: str,
    branch_id: str,
    session_id: str,
    record_id: str,
    locator: str,
    kind: str,
    ts: str | None,
    payload: dict[str, Any],
) -> RawEvidence:
    """Build evidence whose digest covers native content only.

    Source path, offset, and timestamp are excluded: a message that streams across
    several records (each with its own timestamp) is still one native event.
    """
    body = clean(payload)
    return RawEvidence(
        runtime=runtime,
        source_id=source_id,
        branch_id=branch_id,
        session_id=session_id,
        record_id=record_id,
        locator=locator,
        digest=evidence_digest(kind, body),
        ts_utc=ts,
        kind=kind,
        payload=body,
    )
