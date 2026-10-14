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

``current_totals`` sums only current, non-deleted rows of ONE methodology and
unit; integers only, negatives preserved, unknown values counted separately
and never treated as zero. A deletion below the current row cannot delete it
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

from ssgrep.accounting.contracts import Basis, canonical_bytes, digest_of

MAX_BODY_BYTES = 64 * 1024
MAX_STRING = 1024
MAX_ENTRIES = 64
MAX_KEY_CHARS = 64
MAX_INT = 2**63 - 1
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
_UNIT_KEYS = frozenset(("unit", "method", "tokenizer", "currency"))
_QUANTITY_KEYS = frozenset(("value", "unit", "basis", "method", "reason"))
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


class _Invalid(Exception):
    pass


def _fail() -> _Invalid:
    return _Invalid()


def _int(value: Any, low: int = -MAX_INT - 1) -> int:
    if type(value) is not int or not low <= value <= MAX_INT:
        raise _fail()
    return value


def _text(value: Any, *, empty: bool = False, limit: int = MAX_STRING) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value):
        raise _fail()
    return value


def _mapping(value: Any, keys: frozenset[str] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise _fail()
    if keys is not None and set(value) != keys:
        raise _fail()
    return value


def _quantity(raw: Any) -> dict[str, Any]:
    q = _mapping(raw, _QUANTITY_KEYS)
    value = q["value"]
    if value is not None:
        _int(value)
    try:
        basis = Basis(q["basis"])
    except ValueError:
        raise _fail() from None
    reason = _text(q["reason"], empty=True)
    if (value is None) != (basis is Basis.UNAVAILABLE) or (value is None and not reason):
        raise _fail()
    return {
        "value": value,
        "unit": _text(q["unit"]),
        "basis": basis.value,
        "method": _text(q["method"], empty=True),
        "reason": reason,
    }


def _scenarios(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, amounts in _mapping(raw).items():
        if name not in SCENARIOS:
            raise _fail()
        a = _mapping(amounts, _AMOUNT_KEYS)
        out[name] = {part: _quantity(a[part]) for part in sorted(_AMOUNT_KEYS)}
    return out


def _flat(raw: Any, *, allow_bool: bool) -> dict[str, Any]:
    m = _mapping(raw)
    if len(m) > MAX_ENTRIES:
        raise _fail()
    out: dict[str, Any] = {}
    for k, v in m.items():
        _text(k, limit=MAX_KEY_CHARS)
        if isinstance(v, str):
            _text(v, empty=True)
        elif type(v) is bool:
            if not allow_bool:
                raise _fail()
        else:
            _int(v)
        out[k] = v
    return out


def _unit(raw: Any) -> dict[str, Any]:
    u = _mapping(raw, _UNIT_KEYS)
    out: dict[str, Any] = {"unit": _text(u["unit"]), "method": _text(u["method"])}
    for name in ("tokenizer", "currency"):
        out[name] = None if u[name] is None else _text(u[name])
    return out


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
        "unit": _unit(b["unit"]),
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


def current_totals(state: ReceiverState, *, methodology: str, unit_key: UnitKey) -> dict[str, Any]:
    """Sum current non-deleted rows of exactly one methodology and unit.

    Per scenario and part (credit/debit/net) the result has ``total`` (int; None
    when every contributing value is unknown), ``known`` and ``unknown`` counts.
    A row whose value is None, or that lacks the scenario, counts as unknown,
    never as 0. A current row of another methodology or unit raises ValueError.
    """
    wanted = _check_unit_key(unit_key)
    sums = {(s, p): [0, 0, 0] for s in SCENARIOS for p in sorted(_AMOUNT_KEYS)}
    for row in state.current.values():
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
        "rows": len(state.current),
        "scenarios": scenarios,
    }
