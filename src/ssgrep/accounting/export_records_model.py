"""Types, bounds, and stable errors shared by the ledger-log projection and its registry.

Everything here is plain frozen data. ``Candidate`` is the sanitized, revision-less content of
one (task, period, unit) record; ``Planned`` is a complete receiver-valid body for one
revision; ``Registry`` mirrors the local ``records.json`` (what was queued, unresolved, or
accepted per record). Errors carry only a stable ``condition`` code, never raw text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

SCHEMA = 1
MAX_RECORDS = 200_000
MAX_QUEUED = 64  # unsent batches holding one record at once
MAX_REGISTRY_BYTES = 32 * 1024 * 1024
INT64_MAX = 2**63 - 1
CONTRIBUTION, ADJUSTMENT, DELETION = "contribution", "adjustment", "deletion"
RECORD_TYPES = (CONTRIBUTION, ADJUSTMENT, DELETION)
SCENARIOS = ("conservative", "central", "optimistic")
BASES = ("observed", "estimated", "unavailable")

HEX16 = re.compile(r"[0-9a-f]{16}")
HEX32 = re.compile(r"[0-9a-f]{32}")
HEX64 = re.compile(r"[0-9a-f]{64}")
METHODOLOGY = re.compile(r"[0-9a-f]{16,64}")
BATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
UNIT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
# Closed vocabularies: a unit's method is one of the shipped versioned methods and its tokenizer
# is exactly ``units.TokenizerPin.identity``; anything else is omitted (counted), never copied.
METHOD_NAME = re.compile(
    r"chars_div4_v1|capped_gap_v1|transcript_intervals_v1|monotonic_v1|provider_usage_v1"
)
TOKENIZER = re.compile(r"[a-z0-9_.\-]{1,64}==[0-9A-Za-z.+\-]{1,32}:sha256:[0-9a-f]{64}")
PERIOD = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
CURRENCY = re.compile(r"[A-Z]{3}")

UnitRef = tuple[str, str, str | None, str | None]  # (unit, method, tokenizer, currency)


class RecordsUnavailable(Exception):
    """The committed collector ledger cannot be read; ``condition`` is a stable code."""

    def __init__(self, condition: str) -> None:
        super().__init__(condition)
        self.condition = condition


class RegistryCorrupt(Exception):
    """``records.json`` is unsafe, oversized, or invalid; nothing was rewritten."""

    def __init__(self, condition: str) -> None:
        super().__init__(condition)
        self.condition = condition


@dataclass(frozen=True)
class Candidate:
    """Sanitized content of one record before it is given a revision."""

    record_id: str
    task_id: str
    root_key: str  # local-only (never in a body): 16 hex of sha256(root)
    period: str
    unit: UnitRef
    methodology: str
    core: dict[str, Any]  # body without revision / supersedes_revision / record_type
    content_digest: str


@dataclass(frozen=True)
class Planned:
    """A complete, receiver-valid body for one revision of one record."""

    record_id: str
    revision: int
    supersedes: int | None
    record_type: str
    body: dict[str, Any]
    body_digest: str
    period: str
    task: str
    root_key: str
    content_digest: str


@dataclass(frozen=True)
class State:
    """One known version of a record: its revision, body digest, content digest, and type."""

    revision: int
    digest: str
    content: str
    type: str
    batch: str | None = None  # only for queued versions


@dataclass(frozen=True)
class Entry:
    """Everything remembered about one record id (no ledger access is needed to delete it)."""

    task: str
    root_key: str
    period: str
    unit: UnitRef
    methodology: str
    acked: State | None = None
    unknown: State | None = None
    queued: tuple[State, ...] = ()  # ascending revision; one per spooled, unsent batch

    @property
    def latest(self) -> State | None:
        """Highest known revision (queued > unknown > acked on a tie)."""
        ranked = [(s.revision, 0, s) for s in (self.acked,) if s]
        ranked += [(s.revision, 1, s) for s in (self.unknown,) if s]
        ranked += [(s.revision, 2, s) for s in self.queued]
        return max(ranked, key=lambda item: item[:2])[2] if ranked else None


@dataclass(frozen=True)
class Registry:
    entries: Mapping[str, Entry] = field(default_factory=dict)
