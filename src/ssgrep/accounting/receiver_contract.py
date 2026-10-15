"""Receiver-side deduplication, conflict, and tombstone contract (pure reference).

This module is the executable statement of what an organization's receiver must
do with ssgrep ledger-log record bodies *after* it has authenticated the
sender. It is a pure, deterministic function over an immutable value: no I/O,
no clocks, no globals, no client database, and no hosted receiver. An
organization applies the same rules inside its own backend; the replay of
``docs/examples/receiver-conformance.json`` through :func:`apply` is the
conformance fixture.

Vanilla OTLP log acceptance alone is NOT exactly-once aggregation: the OTLP
protocol only acknowledges that a request was accepted, so retries, reordering
and replays would be summed as separate rows. Exactly-once *current* totals need
the revision-aware keyed upsert implemented here.

Contract (key = authenticated ``(tenant, device, record_id)``; tenant and device
come from the enrolled credential, never from the body, and bodies that claim
identity fields are invalid):

* first sight of a key -> ``APPLIED_NEW``; greater revision -> ``APPLIED_UPSERT``;
* same revision and same digest -> ``DUPLICATE_NOOP`` (state unchanged);
* same revision, different digest -> ``QUARANTINED`` (held row kept, never
  double counted, only the conflicting *digest* is recorded);
* lower revision than the current row -> ``REJECTED_OLD_REVISION``;
* ``deletion`` with a revision above the current row removes it and keeps a
  tombstone forever (``APPLIED_TOMBSTONE``), also for a key never seen; a
  deletion not above the current row is ``REJECTED_OLD_REVISION`` and never
  removes the row;
* any body at or below the tombstone revision -> ``REJECTED_TOMBSTONED``; a
  body above it is an intentional re-creation (``APPLIED_NEW``, tombstone kept);
* anything failing strict schema validation -> ``REJECTED_INVALID``, no change.

``current_totals`` sums only current, non-deleted rows of ONE authenticated tenant,
methodology and unit; mixed-tenant state requires an explicit tenant selector.
Integers only, negatives preserved, unknown values counted separately and never
treated as zero. A deletion below the current row cannot delete it
(``REJECTED_OLD_REVISION``) but its tombstone is still recorded, so any
permutation of distinct-revision bodies for a key converges to the same current
row and tombstone set.

``apply`` copies the keyed maps per call so states stay immutable values; it is
a reference/conformance implementation, not a high-volume store.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from ssgrep.accounting.contracts import canonical_bytes, digest_of
from ssgrep.accounting.receiver_values import (
    MAX_ENTRIES as MAX_ENTRIES,
    MAX_INT as MAX_INT,
    MAX_KEY_CHARS as MAX_KEY_CHARS,
    MAX_STRING,
    _fail,
    _flat,
    _int,
    _Invalid,
    _mapping,
    _scenarios,
    _text,
    _unit,
)

MAX_BODY_BYTES = 64 * 1024
LOG_LIMIT = 1000
SCENARIOS = ("conservative", "central", "optimistic")
RECORD_TYPES = ("contribution", "adjustment", "deletion")
UnitKey = tuple[str, str, str | None, str | None]

_TOP_KEYS = frozenset(
    (
        "schema",
        "record_id",
        "revision",
        "supersedes_revision",
        "record_type",
        "task_id",
        "period",
        "unit",
        "scenarios",
        "known_costs",
        "coverage",
        "methodology",
    )
)
_AMOUNT_KEYS = frozenset(("credit", "debit", "net"))
_PERIOD = re.compile(r"\d{4}-\d{2}-\d{2}")


class Outcome(StrEnum):
    APPLIED_NEW = "APPLIED_NEW"
    APPLIED_UPSERT = "APPLIED_UPSERT"
    DUPLICATE_NOOP = "DUPLICATE_NOOP"
    QUARANTINED = "QUARANTINED"
    REJECTED_OLD_REVISION = "REJECTED_OLD_REVISION"
    REJECTED_TOMBSTONED = "REJECTED_TOMBSTONED"
    APPLIED_TOMBSTONE = "APPLIED_TOMBSTONE"
    REJECTED_INVALID = "REJECTED_INVALID"


@dataclass(frozen=True)
class Key:
    """Authenticated receiver identity; never derived from the body."""

    tenant: str
    device: str
    record_id: str


@dataclass(frozen=True)
class Applied:
    """One stored current row; ``digest`` is ``digest_of(body)``."""

    revision: int
    digest: str
    record_type: str
    body: Mapping[str, Any]


@dataclass(frozen=True)
class Quarantined:
    """A same-revision conflict; holds digests only, never the conflicting body."""

    key: Key
    revision: int
    digest: str
    held_digest: str
    reason: str


@dataclass(frozen=True)
class Rejection:
    key: Key
    revision: int
    digest: str
    reason: str


def _frozen_map(value: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return value if isinstance(value, MappingProxyType) else MappingProxyType(dict(value))


def _empty() -> Mapping[Any, Any]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ReceiverState:
    """Immutable receiver value; every ``apply`` returns a new state.

    ``tombstones`` holds the highest deleted revision per key (retained forever
    against replay) and ``tombstone_digests`` the digest of that deletion so an
    identical retry is a no-op and a conflicting one is quarantined.
    ``quarantine`` and ``rejected`` keep the newest ``LOG_LIMIT`` entries.
    """

    current: Mapping[Key, Applied] = field(default_factory=_empty)
    tombstones: Mapping[Key, int] = field(default_factory=_empty)
    tombstone_digests: Mapping[Key, str] = field(default_factory=_empty)
    quarantine: tuple[Quarantined, ...] = ()
    rejected: tuple[Rejection, ...] = ()

    def __post_init__(self) -> None:
        for name in ("current", "tombstones", "tombstone_digests"):
            object.__setattr__(self, name, _frozen_map(getattr(self, name)))
        object.__setattr__(self, "quarantine", tuple(self.quarantine))
        object.__setattr__(self, "rejected", tuple(self.rejected))


def _period(raw: Any) -> str:
    text = _text(raw)
    if not _PERIOD.fullmatch(text):
        raise _fail()
    try:
        date.fromisoformat(text)
    except ValueError:
        raise _fail() from None
    return text


def _normalize(body: Any, key: Key) -> dict[str, Any]:
    """Strictly validate ``body`` and return a plain deep copy (never the input)."""
    b = _mapping(body, _TOP_KEYS)
    if _int(b["schema"]) != 1 or _text(b["record_id"]) != key.record_id:
        raise _fail()
    revision = _int(b["revision"], 1)
    kind = b["record_type"]
    if kind not in RECORD_TYPES:
        raise _fail()
    sup = b["supersedes_revision"]
    if revision == 1:
        if sup is not None:
            raise _fail()
    elif sup is None or not 1 <= _int(sup) < revision:
        raise _fail()
    if (kind == "contribution") != (revision == 1) and kind != "deletion":
        raise _fail()
    scenarios = _scenarios(b["scenarios"])
    unit = _unit(b["unit"])
    if any(
        row[part]["unit"] != unit["unit"] for row in scenarios.values() for part in _AMOUNT_KEYS
    ):
        raise _fail()
    known = _flat(b["known_costs"], allow_bool=False)
    coverage = _flat(b["coverage"], allow_bool=True)
    if kind == "deletion":
        if scenarios or known or coverage:
            raise _fail()
    elif not scenarios:
        raise _fail()
    out = {
        "schema": 1,
        "record_id": key.record_id,
        "revision": revision,
        "supersedes_revision": sup,
        "record_type": kind,
        "task_id": _text(b["task_id"]),
        "period": _period(b["period"]),
        "unit": unit,
        "scenarios": scenarios,
        "known_costs": known,
        "coverage": coverage,
        "methodology": _text(b["methodology"]),
    }
    if len(canonical_bytes(out)) > MAX_BODY_BYTES:
        raise _fail()
    return out


def _check_key(key: Any) -> None:
    if not isinstance(key, Key):
        raise _fail()
    for part in (key.tenant, key.device, key.record_id):
        _text(part, limit=MAX_STRING)


def _tail(items: tuple[Any, ...], entry: Any) -> tuple[Any, ...]:
    if entry in items:
        return items
    return (*items, entry)[-LOG_LIMIT:]


def _evolve(state: ReceiverState, **changes: Any) -> ReceiverState:
    return ReceiverState(
        current=changes.get("current", state.current),
        tombstones=changes.get("tombstones", state.tombstones),
        tombstone_digests=changes.get("tombstone_digests", state.tombstone_digests),
        quarantine=changes.get("quarantine", state.quarantine),
        rejected=changes.get("rejected", state.rejected),
    )


def _reject(
    state: ReceiverState, key: Key, revision: int, digest: str, reason: str, outcome: Outcome
) -> tuple[ReceiverState, Outcome]:
    rejected = _tail(state.rejected, Rejection(key, revision, digest, reason))
    return _evolve(state, rejected=rejected), outcome


def _quarantine(
    state: ReceiverState, key: Key, revision: int, digest: str, held: str
) -> tuple[ReceiverState, Outcome]:
    entry = Quarantined(key, revision, digest, held, "same_revision_different_digest")
    return _evolve(state, quarantine=_tail(state.quarantine, entry)), Outcome.QUARANTINED


def apply(state: ReceiverState, key: Key, body: Mapping[str, Any]) -> tuple[ReceiverState, Outcome]:
    """Apply one authenticated body; returns the new state and what happened."""
    try:
        _check_key(key)
        norm = _normalize(body, key)
    except _Invalid:
        return state, Outcome.REJECTED_INVALID
    revision = norm["revision"]
    digest = digest_of(norm)
    deletion = norm["record_type"] == "deletion"
    row = state.current.get(key)
    tomb = state.tombstones.get(key)
    if tomb is not None and revision <= tomb:
        if deletion and revision == tomb:
            held = state.tombstone_digests[key]
            if held == digest:
                return state, Outcome.DUPLICATE_NOOP
            return _quarantine(state, key, revision, digest, held)
        return _reject(state, key, revision, digest, "tombstoned", Outcome.REJECTED_TOMBSTONED)
    if row is not None and revision <= row.revision:
        if not deletion and revision == row.revision:
            if row.digest == digest:
                return state, Outcome.DUPLICATE_NOOP
            return _quarantine(state, key, revision, digest, row.digest)
        rejected, outcome = _reject(
            state, key, revision, digest, "old_revision", Outcome.REJECTED_OLD_REVISION
        )
        if deletion and revision < row.revision:
            # Cannot delete a newer row, but remember the deletion so the final
            # tombstone set does not depend on arrival order.
            rejected = _evolve(
                rejected,
                tombstones={**state.tombstones, key: revision},
                tombstone_digests={**state.tombstone_digests, key: digest},
            )
        return rejected, outcome
    if deletion:
        current = {k: v for k, v in state.current.items() if k != key}
        return (
            _evolve(
                state,
                current=current,
                tombstones={**state.tombstones, key: revision},
                tombstone_digests={**state.tombstone_digests, key: digest},
            ),
            Outcome.APPLIED_TOMBSTONE,
        )
    applied = Applied(revision, digest, norm["record_type"], norm)
    outcome = Outcome.APPLIED_NEW if row is None else Outcome.APPLIED_UPSERT
    return _evolve(state, current={**state.current, key: applied}), outcome


def _check_unit_key(unit_key: Any) -> UnitKey:
    if (
        not isinstance(unit_key, tuple)
        or len(unit_key) != 4
        or not all(v is None or isinstance(v, str) for v in unit_key)
        or not all(isinstance(v, str) for v in unit_key[:2])
    ):
        raise ValueError("unit_key must be (unit, method, tokenizer|None, currency|None)")
    return unit_key


def current_totals(
    state: ReceiverState, *, methodology: str, unit_key: UnitKey, tenant: str | None = None
) -> dict[str, Any]:
    """Sum current non-deleted rows of exactly one tenant, methodology and unit.

    Per scenario and part (credit/debit/net) the result has ``total`` (int; None
    when every contributing value is unknown), ``known`` and ``unknown`` counts.
    A row whose value is None, or that lacks the scenario, counts as unknown,
    never as 0. Multiple tenants require an authenticated ``tenant`` selector;
    foreign rows are excluded before strict methodology and unit validation.
    A selected current row of another methodology or unit raises ValueError.
    """
    wanted = _check_unit_key(unit_key)
    if tenant is not None and (
        not isinstance(tenant, str) or not tenant or len(tenant) > MAX_STRING
    ):
        raise ValueError("invalid authenticated tenant")
    if tenant is None and len({key.tenant for key in state.current}) > 1:
        raise ValueError("current rows mix tenants; select authenticated tenant")
    chosen = [row for key, row in state.current.items() if tenant is None or key.tenant == tenant]
    sums = {(s, p): [0, 0, 0] for s in SCENARIOS for p in sorted(_AMOUNT_KEYS)}
    for row in chosen:
        body = row.body
        unit = body["unit"]
        have = (unit["unit"], unit["method"], unit["tokenizer"], unit["currency"])
        if body["methodology"] != methodology or have != wanted:
            raise ValueError("current rows mix methodologies or units; select one first")
        for (scenario, part), acc in sums.items():
            entry = body["scenarios"].get(scenario)
            value = None if entry is None else entry[part]["value"]
            if value is None:
                acc[2] += 1
            else:
                acc[0] += value
                acc[1] += 1
    scenarios: dict[str, Any] = {s: {} for s in SCENARIOS}
    for (scenario, part), (total, known, unknown) in sums.items():
        scenarios[scenario][part] = {
            "total": None if known == 0 and unknown > 0 else total,
            "known": known,
            "unknown": unknown,
        }
    return {
        "methodology": methodology,
        "unit": {
            "unit": wanted[0],
            "method": wanted[1],
            "tokenizer": wanted[2],
            "currency": wanted[3],
        },
        "rows": len(chosen),
        "scenarios": scenarios,
    }
