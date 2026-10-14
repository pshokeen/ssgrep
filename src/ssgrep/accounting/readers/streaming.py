"""Streamed usage: repeated records of one native message with growing counters.

Claude writes one record per streamed content block, all with the same
``message.id`` and a ``usage`` that grows (``output_tokens`` 2, 2, 344). That is
one message whose final usage is the greatest value per counter, not several
events and not a conflict. Only a counter that *goes down* in stream order (or
moves in opposite directions across variants) is a true conflict.

Counters are compared per field with an absent field treated as 0 *for ordering
only*; the resolved usage is the later, greater record, never a synthesized sum.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.readers.values import evidence_digest

STREAM_KEY = "streaming_updates"
EQUAL, UP, DOWN, MIXED = "equal", "up", "down", "mixed"


def is_stream(evidence: RawEvidence) -> bool:
    return evidence.kind == "usage" and STREAM_KEY in evidence.payload


def compare(old: dict[str, int], new: dict[str, int]) -> str:
    """Relation of ``new`` to ``old`` over the union of counter names."""
    up = down = False
    for name in old.keys() | new.keys():
        before, after = old.get(name, 0), new.get(name, 0)
        up |= after > before
        down |= after < before
    if up and down:
        return MIXED
    return UP if up else DOWN if down else EQUAL


def same_context(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Everything except the counters and the update count must match exactly."""
    skip = {"tokens", STREAM_KEY}
    return {k: v for k, v in old.items() if k not in skip} == {
        k: v for k, v in new.items() if k not in skip
    }


class StreamState:
    """Per-identity resolution state inside an ``EvidenceLedger``."""

    def __init__(self, evidence: RawEvidence) -> None:
        self.best = evidence
        self.last: dict[str, dict[str, int]] = {}
        self.updates: dict[str, int] = {}
        self.note(evidence)

    def note(self, evidence: RawEvidence) -> None:
        source = evidence.source_id
        self.last[source] = evidence.payload["tokens"]
        self.updates[source] = self.updates.get(source, 0) + evidence.payload[STREAM_KEY]

    @property
    def total_updates(self) -> int:
        return max(self.updates.values())


def collapse(evidence: list[RawEvidence]) -> list[RawEvidence]:
    """Merge consecutive-by-identity streamed usage within one read cycle.

    A record joins the open run for its identity only while counters do not decrease and
    the context is identical; otherwise it starts a new run, which the ledger then reports
    as a conflict. The merged record keeps the greatest counters and the update count.
    """
    out: list[RawEvidence] = []
    runs: dict[tuple[str, str, str], int] = {}
    for item in evidence:
        if not is_stream(item):
            out.append(item)
            continue
        key = (item.runtime, item.session_id, item.record_id)
        index = runs.get(key)
        if index is not None:
            current = out[index]
            relation = compare(current.payload["tokens"], item.payload["tokens"])
            if same_context(current.payload, item.payload) and relation in {EQUAL, UP}:
                updates = current.payload[STREAM_KEY] + item.payload[STREAM_KEY]
                body = {**item.payload, STREAM_KEY: updates}
                digest = evidence_digest(item.kind, body)
                out[index] = replace(item, payload=body, digest=digest)
                continue
        runs[key] = len(out)
        out.append(item)
    return out
