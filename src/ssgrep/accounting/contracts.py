"""Frozen evidence contracts shared by capture, readers, and the reducer.

Design rules enforced here (see design.md D3-D6):

* Unknown is never zero: every measurement is a :class:`Quantity` whose
  ``value`` is ``None`` when ``basis`` is ``unavailable``.
* Canonical serialization is deterministic -- sorted keys, integer units, no
  clocks -- so full replay and incremental reduction produce identical bytes.
* Observations (what happened) and estimates (modeled benefit) are different
  types and are never summed together.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = 1
MAX_EVENT_BYTES = 64 * 1024
MAX_RAW_RECORD_BYTES = 8 * 1024 * 1024
PERSONAL_PROFILE = "personal-local"
DEFAULT_LIMIT = 20
DEFAULT_PORT = 9469


class Operation(StrEnum):
    SEARCH = "search"
    SHOW = "show"
    NOTE = "note"


class Transport(StrEnum):
    CLI = "cli"
    MCP = "mcp"
    API = "api"


class Phase(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    RENDERED = "rendered"


class Outcome(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class Attribution(StrEnum):
    EXPLICIT = "explicit"
    INFERRED_UNIQUE = "inferred_unique"
    AMBIGUOUS = "ambiguous"
    UNATTRIBUTED = "unattributed"


class Exposure(StrEnum):
    OBSERVED = "observed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class Reuse(StrEnum):
    NONE_OBSERVED = "none_observed"
    INSPECTED = "inspected"
    ACTION_REFERENCE = "action_reference"
    VALIDATED = "validated"
    CONTRADICTED = "contradicted"


class InvocationClass(StrEnum):
    MATCHED = "matched"
    TRANSCRIPT_ONLY = "transcript_only"
    JOURNAL_GAP = "journal_gap"
    MENTION = "mention"
    UNCLASSIFIED = "unclassified"


class Scenario(StrEnum):
    CONSERVATIVE = "conservative"
    CENTRAL = "central"
    OPTIMISTIC = "optimistic"


class Basis(StrEnum):
    OBSERVED = "observed"
    REPORTED = "reported"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


RUNTIMES = ("claude", "codex", "opencode", "pi", "omp", "prime-agent")


@dataclass(frozen=True)
class Quantity:
    """One measurement with its provenance; ``value`` is None iff unavailable."""

    value: int | None
    unit: str
    basis: Basis
    method: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if (self.value is None) != (self.basis is Basis.UNAVAILABLE):
            raise ValueError("value is None exactly when basis is unavailable")
        if self.basis is Basis.UNAVAILABLE and not self.reason:
            raise ValueError("an unavailable quantity must state a reason")

    @classmethod
    def unavailable(cls, unit: str, reason: str) -> Quantity:
        return cls(None, unit, Basis.UNAVAILABLE, reason=reason)

    @classmethod
    def observed(cls, value: int, unit: str, method: str = "") -> Quantity:
        return cls(value, unit, Basis.OBSERVED, method)


@dataclass(frozen=True)
class ExecutionEvent:
    """One immutable journal phase of one real search/show/note execution."""

    execution_id: str
    phase: Phase
    operation: Operation
    transport: Transport
    profile: str
    ts_utc: str
    cwd: str | None
    fingerprint: str
    version: str
    outcome: Outcome | None = None
    mono_ns: int | None = None
    refs: tuple[str, ...] = ()
    ranks_scores: tuple[tuple[int, int], ...] = ()
    limit: int | None = None
    token_budget: int | None = None
    omitted_count: int | None = None
    truncated: bool | None = None
    rendered_bytes: Quantity | None = None
    index_generation: str | None = None
    refs_omitted: bool = False
    correlation_id: str | None = None
    schema: int = SCHEMA_VERSION
    fingerprint_key_id: str | None = None
    scope_basis: str = "legacy"


@dataclass(frozen=True)
class RawEvidence:
    """A verifiable native record read from a runtime store."""

    runtime: str
    source_id: str
    branch_id: str
    session_id: str
    record_id: str
    locator: str
    digest: str
    ts_utc: str | None
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    schema: int = SCHEMA_VERSION
    origin_domain: str | None = None
    origin_id: str | None = None
    origin_status: str = "legacy_session_scoped"

    @property
    def native_key(self) -> tuple[str, str, str]:
        """Origin identity; containing file, ownership and collection mode are separate."""
        return (
            self.runtime,
            self.origin_domain or self.session_id,
            self.origin_id or self.record_id,
        )

    @property
    def native_identity(self) -> str:
        return ":".join(self.native_key)

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """One content variant of a native event, never a text-only dedup key."""
        return (*self.native_key, self.digest)


@dataclass(frozen=True)
class TaskId:
    """Root conversation + originating user turn + branch; never a session id."""

    root: str
    turn: str | None
    branch: str = "main"

    @property
    def resolved(self) -> bool:
        return self.turn is not None


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, frozenset):
        return sorted(_plain(v) for v in value)
    if isinstance(value, float):
        raise TypeError("floats are not allowed in canonical accounting data")
    return value


def canonical_bytes(value: Any) -> bytes:
    """Deterministic UTF-8 JSON: sorted keys, no whitespace, no floats."""
    text = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode()


def digest_of(value: Any) -> str:
    """Content digest of the canonical form (hex sha256)."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
