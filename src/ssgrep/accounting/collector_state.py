"""The collector's derived state, persisted as content-addressed blobs named by the checkpoint.

Everything a *clock-only* cycle needs to republish without re-reading sources or re-reducing:
the ledger of the CURRENT methodology (frozen roots plus live candidates/debits and the
superseded-version map), live counters, sealed UTC days, and the last reduction's executions
and coverage. A methodology change is a new namespace: its ledger blob is separate, and the
old namespace's blob is kept untouched (never summed with the new one).

Each blob is written before the checkpoint that names it, so a crash can never leave the
checkpoint pointing at a blob that is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_store import Checkpoint
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.counters import LiveCounters
from ssgrep.accounting.ledger_codec import ledger_state_from_dict, ledger_state_to_dict
from ssgrep.accounting.plain_codec import from_plain, to_plain
from ssgrep.accounting.reduce_types import ExecutionRecord
from ssgrep.accounting.replay import LedgerState
from ssgrep.accounting.windows import SealedDay

COUNTERS, SEALED, REDUCTION = "counters", "sealed", "reduction"
LEDGER_PREFIX = "ledger-"
NAMESPACE_CHARS = 12


def ledger_blob_name(methodology_hash: str) -> str:
    return f"{LEDGER_PREFIX}{methodology_hash[:NAMESPACE_CHARS]}"


@dataclass
class ReductionDoc:
    """What a clock-only cycle reuses of the last reduction (plain data, no sources)."""

    executions: tuple[ExecutionRecord, ...] = ()
    coverage: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    digest: str = ""
    session_roots: tuple[tuple[str, str, str], ...] = ()  # (runtime, session_id, root)
    session_last: tuple[tuple[str, str, str], ...] = ()  # (runtime, session_id, last evidence ts)
    # additive (readers tolerate absence): lineage parent and working directory per session
    session_parents: tuple[tuple[str, str, str | None], ...] = ()  # (runtime, session_id, parent)
    session_cwd: tuple[tuple[str, str, str], ...] = ()  # (runtime, session_id, cwd)
    # (invocation_id, runtime, session_id): what Candidate.exposure_ids / Debit.exposure_id name
    invocation_sessions: tuple[tuple[str, str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "executions": [to_plain(e) for e in self.executions],
            "coverage": dict(sorted(self.coverage.items())),
            "warnings": list(self.warnings),
            "digest": self.digest,
            "session_roots": [list(t) for t in self.session_roots],
            "session_last": [list(t) for t in self.session_last],
            "session_parents": [list(t) for t in self.session_parents],
            "session_cwd": [list(t) for t in self.session_cwd],
            "invocation_sessions": [list(t) for t in self.invocation_sessions],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ReductionDoc:
        return cls(
            tuple(from_plain(ExecutionRecord, e) for e in raw["executions"]),
            {str(k): int(v) for k, v in raw["coverage"].items()},
            tuple(str(w) for w in raw.get("warnings", [])),
            str(raw.get("digest", "")),
            tuple((str(a), str(b), str(c)) for a, b, c in raw.get("session_roots", [])),
            tuple((str(a), str(b), str(c)) for a, b, c in raw.get("session_last", [])),
            tuple(
                (str(a), str(b), None if c is None else str(c))
                for a, b, c in raw.get("session_parents", [])
            ),
            tuple((str(a), str(b), str(c)) for a, b, c in raw.get("session_cwd", [])),
            tuple((str(a), str(b), str(c)) for a, b, c in raw.get("invocation_sessions", [])),
        )


@dataclass
class State:
    """Loaded derived state; ``None`` fields mean the blob does not exist yet."""

    ledger: LedgerState | None = None
    superseded: dict[str, str] = field(default_factory=dict)
    counters: LiveCounters | None = None
    sealed: dict[str, SealedDay] = field(default_factory=dict)
    reduction: ReductionDoc | None = None
    collection_scope: dict[str, Any] = field(default_factory=dict)


def _decode(name: str, build: Any, raw: Any) -> Any:
    try:
        return build(raw)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise CollectError("state_corrupt", f"Collector state blob {name!r} is invalid.") from error


def decode_ledger(raw: Any) -> tuple[LedgerState, dict[str, str]]:
    """Validate a referenced ledger before its provenance can trigger a scope rotation."""
    try:
        state = ledger_state_from_dict(raw["state"])
        return state, {str(k): str(v) for k, v in raw.get("superseded", {}).items()}
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise CollectError("state_corrupt", "Collector ledger state is invalid.") from error


def load_state(profile: str, checkpoint: Checkpoint, methodology_hash: str) -> State:
    """Every blob the checkpoint names (corrupt or missing committed blob => state_corrupt)."""
    blobs = checkpoint.blobs

    def blob(name: str) -> Any | None:
        return store.read_blob(profile, blobs.get(name))

    state = State()
    name = ledger_blob_name(methodology_hash)
    raw = blob(name)
    if raw is not None:
        state.ledger, state.superseded = decode_ledger(raw)
        state.collection_scope = raw.get("collection_scope", {})
    raw = blob(COUNTERS)
    if raw is not None:
        state.counters = _decode(COUNTERS, LiveCounters.from_dict, raw)
    raw = blob(SEALED)
    if raw is not None:
        state.sealed = _decode(
            SEALED, lambda r: {str(k): SealedDay.from_dict(v) for k, v in r.items()}, raw
        )
    raw = blob(REDUCTION)
    if raw is not None:
        state.reduction = _decode(REDUCTION, ReductionDoc.from_dict, raw)
    return state


def save_state(profile: str, state: State, methodology_hash: str, blobs: dict[str, str]) -> None:
    """Write every blob (not yet committed) and record their names in ``blobs``."""
    if state.ledger is not None:
        document = {
            "state": ledger_state_to_dict(state.ledger),
            "superseded": dict(sorted(state.superseded.items())),
            "collection_scope": state.collection_scope,
        }
        blobs[ledger_blob_name(methodology_hash)] = store.write_blob(
            profile, ledger_blob_name(methodology_hash), document
        )
    if state.counters is not None:
        blobs[COUNTERS] = store.write_blob(profile, COUNTERS, state.counters.to_dict())
    sealed = {day: value.to_dict() for day, value in state.sealed.items()}
    blobs[SEALED] = store.write_blob(profile, SEALED, sealed)
    if state.reduction is not None:
        blobs[REDUCTION] = store.write_blob(profile, REDUCTION, state.reduction.to_dict())
