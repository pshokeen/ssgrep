"""Local record identity, revision planning, and durable registry state."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from ssgrep.accounting.export import spool as export_spool
from ssgrep.accounting.export.ids import EXPORT_DIR
from ssgrep.accounting.ledger.contracts import digest_of
from ssgrep.accounting.ledger.ledger_contracts import Signal, valid_profile_id
from ssgrep.accounting.privacy import storage

"""Types, bounds, and stable errors shared by the ledger-log projection and its registry.

Everything here is plain frozen data. ``Candidate`` is the sanitized, revision-less content of
one (task, period, unit) record; ``Planned`` is a complete receiver-valid body for one
revision; ``Registry`` mirrors the local ``records.json`` (what was queued, unresolved, or
accepted per record). Errors carry only a stable ``condition`` code, never raw text.
"""


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


"""Pure revision planning for ledger-log records (no I/O, no clocks).

``plan_revisions`` turns the sanitized candidates of the current ledger into the NEXT revision
of every record that changed since the registry last knew it:

* nothing known                      -> revision 1 ``contribution`` (supersedes ``None``);
* latest known content == candidate  -> nothing (idempotent: no duplicate additive credit);
* otherwise                          -> revision ``latest + 1`` ``adjustment`` superseding it.

"Latest known" is the highest revision over the acked, unknown and queued versions, so an
unresolved or unsent revision r still counts: a changed body then becomes r+1 (it supersedes r
at the receiver) instead of a second additive revision r. A record whose latest version is a
``deletion`` is not re-created from identical content (a deletion keeps the content it
removed); changed content re-creates it as an ``adjustment`` at ``deletion revision + 1``,
because the receiver only accepts ``contribution`` at revision 1 and treats any body above the
tombstone as an intentional re-creation.

An explicitly selected current namespace retires records from older namespaces
with higher-revision deletions, including queued/unacknowledged old methods. No
task/root alias is guessed; archived identities cannot become additional usage.
"""


def _planned(cand: Candidate, revision: int, supersedes: int | None, record_type: str) -> Planned:
    body = {
        **cand.core,
        "revision": revision,
        "supersedes_revision": supersedes,
        "record_type": record_type,
    }
    return Planned(
        cand.record_id,
        revision,
        supersedes,
        record_type,
        body,
        digest_of(body),
        cand.period,
        cand.task_id,
        cand.root_key,
        cand.content_digest,
    )


def plan_revisions(
    registry: Registry,
    candidates: Iterable[Candidate],
    *,
    retracted: frozenset[str] = frozenset(),
    current_namespace: str | None = None,
) -> list[Planned]:
    """The bodies that must be sent so the receiver converges on ``candidates``.

    ``retracted`` record ids are never planned (a purge/tombstone is pending for them).
    Output is ordered by ``(period, record_id)`` and independent of input order or duplicates.
    """
    if current_namespace is not None and not METHODOLOGY.fullmatch(current_namespace):
        raise ValueError("invalid_current_namespace")
    by_id: dict[str, Candidate] = {}
    for candidate in candidates:
        held = by_id.setdefault(candidate.record_id, candidate)
        if held != candidate:
            raise ValueError("conflicting_export_candidate_identity")
    planned: list[Planned] = []
    for cand in sorted(by_id.values(), key=lambda c: (c.period, c.record_id)):
        if cand.record_id in retracted:
            continue
        entry = registry.entries.get(cand.record_id)
        latest = entry.latest if entry is not None else None
        if latest is None:
            planned.append(_planned(cand, 1, None, CONTRIBUTION))
        elif latest.content != cand.content_digest:
            planned.append(_planned(cand, latest.revision + 1, latest.revision, ADJUSTMENT))
    if current_namespace is not None:
        for record_id, entry in sorted(registry.entries.items()):
            if record_id not in by_id and entry.methodology != current_namespace:
                retired = _deletion(record_id, entry)
                if retired is not None:
                    planned.append(retired)
    return sorted(planned, key=lambda item: (item.period, item.record_id))


def _deletion(record_id: str, entry: Entry) -> Planned | None:
    latest = entry.latest
    if latest is None or latest.type == DELETION:
        return None
    unit, method, tokenizer, currency = entry.unit
    body = {
        "schema": SCHEMA,
        "record_id": record_id,
        "revision": latest.revision + 1,
        "supersedes_revision": latest.revision,
        "record_type": DELETION,
        "task_id": entry.task,
        "period": entry.period,
        "unit": {"unit": unit, "method": method, "tokenizer": tokenizer, "currency": currency},
        "scenarios": {},
        "known_costs": {},
        "coverage": {},
        "methodology": entry.methodology,
    }
    return Planned(
        record_id,
        latest.revision + 1,
        latest.revision,
        DELETION,
        body,
        digest_of(body),
        entry.period,
        entry.task,
        entry.root_key,
        latest.content,
    )


def build_deletions(
    registry: Registry, record_ids: Iterable[str], *, key: bytes, destination: str
) -> list[Planned]:
    """Deletion bodies (revision ``latest + 1``) for every named record not already deleted.

    Everything a deletion carries is read from the registry (task pseudonym, period, unit,
    methodology), so it needs neither the ledger nor the original conversation; ``key`` and
    ``destination`` are accepted for call-site symmetry with the projection and are not
    needed because no identity is recomputed. Unknown ids and records whose latest version is
    already a deletion produce nothing.
    """
    del key, destination
    out: list[Planned] = []
    for record_id in sorted(set(record_ids)):
        entry = registry.entries.get(record_id)
        planned = _deletion(record_id, entry) if entry is not None else None
        if planned is not None:
            out.append(planned)
    return out


"""Pure registry state transitions (queued -> acked / unknown, or discarded).

Every function takes a ``Registry`` and returns a new one; ``export_registry`` runs them under
the registry flock. The invariants they keep:

* ``acked`` only ever advances: an older acknowledgement never replaces a newer one;
* ``unknown`` (sent, acceptance unresolved) never regresses either and disappears once an
  acknowledgement at or above its revision exists;
* a record with no remaining state at all is forgotten, so planning starts again at revision 1
  and a discarded-unsent batch re-plans the SAME revision numbers it held.
"""


def _unit_ref(body: Mapping[str, object]) -> UnitRef:
    unit = body["unit"]
    assert isinstance(unit, Mapping)
    return (unit["unit"], unit["method"], unit["tokenizer"], unit["currency"])


def _tidy(entry: Entry) -> Entry | None:
    """Drop unknown states already covered by an acknowledgement; None when nothing is left."""
    unknown = entry.unknown
    if unknown is not None and entry.acked is not None and unknown.revision <= entry.acked.revision:
        entry = replace(entry, unknown=None)
    if entry.acked is None and entry.unknown is None and not entry.queued:
        return None
    return entry


def _rebuild(registry: Registry, changed: Mapping[str, Entry | None]) -> Registry:
    entries = dict(registry.entries)
    for record_id, entry in changed.items():
        tidy = None if entry is None else _tidy(entry)
        if tidy is None:
            entries.pop(record_id, None)
        else:
            entries[record_id] = tidy
    return Registry(entries)


def queue(registry: Registry, planned: Iterable[Planned], batch: str) -> Registry:
    """Remember ``planned`` as spooled in ``batch`` (a same-revision entry is re-pointed)."""
    changed: dict[str, Entry | None] = {}
    for item in planned:
        entry = changed.get(item.record_id) or registry.entries.get(item.record_id)
        state = State(item.revision, item.body_digest, item.content_digest, item.record_type, batch)
        queued = tuple(s for s in (entry.queued if entry else ()) if s.revision != item.revision)
        queued = tuple(sorted((*queued, state), key=lambda s: s.revision))
        if entry is None:
            entry = Entry(
                item.task,
                item.root_key,
                item.period,
                _unit_ref(item.body),
                str(item.body["methodology"]),
            )
        changed[item.record_id] = replace(
            entry,
            unit=_unit_ref(item.body),
            methodology=str(item.body["methodology"]),
            queued=queued,
        )
    return _rebuild(registry, changed)


def _resolve(registry: Registry, batch: str, settle: Callable[[Entry, State], Entry]) -> Registry:
    changed: dict[str, Entry | None] = {}
    for record_id, entry in registry.entries.items():
        mine = [s for s in entry.queued if s.batch == batch]
        if not mine:
            continue
        rest = replace(entry, queued=tuple(s for s in entry.queued if s.batch != batch))
        changed[record_id] = settle(rest, max(mine, key=lambda s: s.revision))
    return _rebuild(registry, changed)


def _plain(state: State) -> State:
    return replace(state, batch=None)


def _ack(entry: Entry, state: State) -> Entry:
    held = entry.acked
    return (
        entry
        if held is not None and held.revision > state.revision
        else replace(entry, acked=_plain(state))
    )


def _unresolved(entry: Entry, state: State) -> Entry:
    acked, held = entry.acked, entry.unknown
    if acked is not None and acked.revision >= state.revision:
        return entry
    if held is not None and held.revision > state.revision:
        return entry
    return replace(entry, unknown=_plain(state))


def _discard(entry: Entry, state: State) -> Entry:
    return entry


def ack(registry: Registry, batch: str) -> Registry:
    """Queued states of ``batch`` were fully accepted: they become ``acked`` (never regress)."""
    return _resolve(registry, batch, _ack)


def unresolve(registry: Registry, batch: str) -> Registry:
    """Queued states of ``batch`` were sent but not resolved (timeout, partial, quarantine)."""
    return _resolve(registry, batch, _unresolved)


def discard(registry: Registry, batch: str) -> Registry:
    """``batch`` was dropped unsent: its queued states vanish (re-planning reuses revisions)."""
    return _resolve(registry, batch, _discard)


def ids_for_roots(registry: Registry, root_keys: frozenset[str]) -> list[str]:
    return sorted(i for i, e in registry.entries.items() if e.root_key in root_keys)


def ids_before(registry: Registry, day: str, root_keys: frozenset[str] | None) -> list[str]:
    return sorted(
        i
        for i, e in registry.entries.items()
        if e.period < day and (root_keys is None or e.root_key in root_keys)
    )


"""``accounting/<profile>/export/<destination>/records.json``: the local ledger-log registry.

Written only through ``update_registry`` (flock-serialised read-modify-write, atomic
replacement, like ``acks.update_delivery``); read through ``load_registry``, which is
read-only and never creates anything. The document is strict, float-free, and bounded
(``MAX_RECORDS`` entries, ``MAX_REGISTRY_BYTES`` bytes): anything else raises
``RegistryCorrupt`` and NOTHING is rewritten, so a damaged registry is never "repaired" into
an empty one that would re-send history as new contributions.

Document::

    {"schema": 1, "records": {record_id: {"task", "root_key", "period", "methodology",
        "unit": {"unit", "method", "tokenizer", "currency"},
        "acked": S | null, "unknown": S | null, "queued": [S + {"batch"}, ...]}}}
    S = {"revision": int, "digest": body digest, "content": content digest, "type": ...}

The registry holds only pseudonyms, hashes, the period day and the unit label: enough to plan
revisions and to delete records without the ledger, and nothing a conversation could be
recovered from (``root_key`` is an unkeyed 16-hex hash used only locally by purge).
"""


NAME = "records.json"
LOCK = ".records.lock"
LOCK_OPEN_ATTEMPTS = 16
_STATE_KEYS = frozenset(("revision", "digest", "content", "type"))
_ENTRY_KEYS = frozenset(
    ("task", "root_key", "period", "methodology", "unit", "acked", "unknown", "queued")
)
_UNIT_KEYS = frozenset(("unit", "method", "tokenizer", "currency"))


def _bad(condition: str) -> RegistryCorrupt:
    return RegistryCorrupt(condition)


def _no_float(_: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def _no_constant(_: str) -> Any:
    raise ValueError("non-finite numbers are not allowed")


def _text(value: Any, pattern: Any) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise _bad("registry_invalid")
    return value


def _opt_text(value: Any, pattern: Any) -> str | None:
    return None if value is None else _text(value, pattern)


def _mapping(value: Any, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise _bad("registry_invalid")
    return value


def _state(raw: Any, *, queued: bool) -> State:
    data = _mapping(raw, _STATE_KEYS | {"batch"} if queued else _STATE_KEYS)
    revision = data["revision"]
    if type(revision) is not int or not 1 <= revision <= INT64_MAX:
        raise _bad("registry_invalid")
    if data["type"] not in RECORD_TYPES:
        raise _bad("registry_invalid")
    batch = _text(data["batch"], BATCH_ID) if queued else None
    return State(
        revision, _text(data["digest"], HEX64), _text(data["content"], HEX64), data["type"], batch
    )


def _entry(raw: Any) -> Entry:
    data = _mapping(raw, _ENTRY_KEYS)
    if not isinstance(data["queued"], list) or len(data["queued"]) > MAX_QUEUED:
        raise _bad("registry_invalid")
    period = _text(data["period"], PERIOD)
    try:
        date.fromisoformat(period)
    except ValueError:
        raise _bad("registry_invalid") from None
    unit = _mapping(data["unit"], _UNIT_KEYS)
    queued = tuple(_state(item, queued=True) for item in data["queued"])
    if [s.revision for s in queued] != sorted({s.revision for s in queued}):
        raise _bad("registry_invalid")
    return Entry(
        _text(data["task"], HEX32),
        _text(data["root_key"], HEX16),
        period,
        (
            _text(unit["unit"], UNIT_NAME),
            _text(unit["method"], METHOD_NAME),
            _opt_text(unit["tokenizer"], TOKENIZER),
            _opt_text(unit["currency"], CURRENCY),
        ),
        _text(data["methodology"], METHODOLOGY),
        None if data["acked"] is None else _state(data["acked"], queued=False),
        None if data["unknown"] is None else _state(data["unknown"], queued=False),
        queued,
    )


def decode(data: bytes) -> Registry:
    """Strictly parse a registry document (``RegistryCorrupt`` on anything unexpected)."""
    if len(data) > MAX_REGISTRY_BYTES:
        raise _bad("registry_oversize")
    try:
        raw = json.loads(data, parse_float=_no_float, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        raise _bad("registry_invalid") from None
    if not isinstance(raw, dict) or set(raw) != {"schema", "records"}:
        raise _bad("registry_invalid")
    if type(raw["schema"]) is not int or raw["schema"] != SCHEMA:
        raise _bad("registry_schema")
    records = raw["records"]
    if not isinstance(records, dict) or len(records) > MAX_RECORDS:
        raise _bad("registry_oversize" if isinstance(records, dict) else "registry_invalid")
    return Registry({_text(rid, HEX32): _entry(value) for rid, value in records.items()})


def _state_dict(state: State | None, *, queued: bool = False) -> dict[str, Any] | None:
    if state is None:
        return None
    out: dict[str, Any] = {
        "revision": state.revision,
        "digest": state.digest,
        "content": state.content,
        "type": state.type,
    }
    if queued:
        out["batch"] = state.batch
    return out


def encode(registry: Registry) -> bytes:
    """Canonical bytes of ``registry``; refuses what ``decode`` would refuse."""
    if len(registry.entries) > MAX_RECORDS:
        raise _bad("registry_oversize")
    records = {}
    for rid, e in sorted(registry.entries.items()):
        if len(e.queued) > MAX_QUEUED:
            raise _bad("registry_oversize")
        unit, method, tokenizer, currency = e.unit
        records[rid] = {
            "task": e.task,
            "root_key": e.root_key,
            "period": e.period,
            "methodology": e.methodology,
            "unit": {"unit": unit, "method": method, "tokenizer": tokenizer, "currency": currency},
            "acked": _state_dict(e.acked),
            "unknown": _state_dict(e.unknown),
            "queued": [_state_dict(s, queued=True) for s in e.queued],
        }
    data = json.dumps(
        {"schema": SCHEMA, "records": records}, sort_keys=True, separators=(",", ":")
    ).encode()
    if len(data) > MAX_REGISTRY_BYTES:
        raise _bad("registry_oversize")
    return data


def _open_dir(profile: str, destination: str, *, create: bool) -> int | None:
    if not valid_profile_id(profile) or not valid_profile_id(destination):
        raise _bad("registry_unsafe")
    try:
        return storage.open_profile_dir(profile, EXPORT_DIR, destination, create=create)
    except FileNotFoundError:
        return None
    except OSError:
        raise _bad("registry_unsafe") from None


def _read(dir_fd: int) -> Registry:
    try:
        data = storage.read_file(dir_fd, NAME, MAX_REGISTRY_BYTES)
    except ValueError:
        raise _bad("registry_oversize") from None
    except OSError:
        raise _bad("registry_unsafe") from None
    return Registry() if data is None else decode(data)


def load_registry(profile: str, destination: str) -> Registry:
    """The stored registry; empty when absent. Read-only: never creates a file or directory."""
    dir_fd = _open_dir(profile, destination, create=False)
    if dir_fd is None:
        return Registry()
    try:
        return _read(dir_fd)
    finally:
        os.close(dir_fd)


def _open_lock(dir_fd: int) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    for _ in range(LOCK_OPEN_ATTEMPTS - 1):
        with contextlib.suppress(FileNotFoundError):  # a concurrent first creation can race
            return os.open(LOCK, flags, 0o600, dir_fd=dir_fd)
    return os.open(LOCK, flags, 0o600, dir_fd=dir_fd)


def update_registry(
    profile: str,
    destination: str,
    change: Callable[[Registry], Registry],
    *,
    create: bool = True,
) -> Registry:
    """Lock, read, apply ``change``, validate by encoding, and atomically write.

    With ``create`` false an absent registry is left absent (a transition on nothing is a
    no-op that creates no file). The file is rewritten only when the content changed.
    """
    dir_fd = _open_dir(profile, destination, create=create)
    if dir_fd is None:
        return Registry()
    try:
        try:
            lock = _open_lock(dir_fd)
        except OSError:
            raise _bad("registry_unsafe") from None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                before = storage.read_file(dir_fd, NAME, MAX_REGISTRY_BYTES)
            except ValueError:
                raise _bad("registry_oversize") from None
            except OSError:
                raise _bad("registry_unsafe") from None
            current = Registry() if before is None else decode(before)
            updated = change(current)
            encoded = encode(updated)
            if encoded != before and not (before is None and not updated.entries):
                storage.atomic_write(dir_fd, NAME, encoded)
            return updated
        finally:
            os.close(lock)  # closing releases the flock
    finally:
        os.close(dir_fd)


"""Ledger-log registry transitions the delivery cycle needs beyond ``export_records``.

``export_records`` settles one queued batch at a time; the cycle also needs

* ``queue_batches``: register EVERY batch of one enqueue in a single atomic registry write, so a
  crash leaves either none or all of them registered (never a half-registered snapshot that a
  rebuild would plan differently and split into conflicting batch ids);
* ``promote_unknown``: a batch that was sent (timeout, restart in flight) and later accepted turns
  its ``unknown`` record states into ``acked``;
* ``require_queued_spool``: refuse lost/corrupt registered bytes rather than rebuild a smaller
  generation from surviving metas and incorrectly acknowledge the missing records.

A batch whose bytes are later discarded keeps its ``unknown`` states on purpose: the receiver may
hold that revision, so the planner must supersede it (revision + 1) rather than reuse the number.

Records are matched by (record id, revision) read from the spooled body, so no ledger access and
no extra state is needed. Only pseudonymous ids and integers are handled here.
"""


Pair = tuple[str, int]


def body_pairs(body: bytes) -> list[Pair]:
    """(record_id, revision) of every log record in a serialized OTLP logs request."""
    found: list[Pair] = []
    try:
        document = json.loads(body)
        for resource in document.get("resourceLogs", []):
            for scope in resource.get("scopeLogs", []):
                for record in scope.get("logRecords", []):
                    inner = json.loads(record["body"]["stringValue"])
                    found.append((inner["record_id"], int(inner["revision"])))
    except (ValueError, KeyError, TypeError, AttributeError):
        return []
    return found


def queue_batches(
    profile: str, destination: str, groups: Sequence[tuple[str, Sequence[Planned]]]
) -> None:
    """Register ``(batch_id, planned...)`` groups as queued, atomically, in order."""

    def change(registry: Registry) -> Registry:
        for batch_id, planned in groups:
            registry = queue(registry, planned, batch_id)
        return registry

    update_registry(profile, destination, change)


def require_queued_spool(profile: str, destination: str) -> None:
    """Every queued registry reference must still have its original readable body and meta.

    A missing meta is not proof that the batch was never attempted. Preserve the registry and
    acknowledgment truth until the pair is restored or explicit retention/purge settles it.
    """
    registry = load_registry(profile, destination)
    queued = {state.batch for entry in registry.entries.values() for state in entry.queued}
    for batch_id in queued:
        assert batch_id is not None  # strict registry decoding requires a batch for queued state
        try:
            export_spool.read_body(profile, destination, Signal.LOGS, batch_id)
        except FileNotFoundError:
            raise export_spool.SpoolCorrupt("registry references a missing spooled batch") from None


def promote_unknown(profile: str, destination: str, pairs: Iterable[Pair]) -> None:
    """``unknown`` states at exactly these revisions were accepted: they become ``acked``."""
    wanted = list(pairs)

    def change(registry: Registry) -> Registry:
        entries = dict(registry.entries)
        for record_id, revision in wanted:
            entry = entries.get(record_id)
            if entry is None or entry.unknown is None or entry.unknown.revision != revision:
                continue
            held = entry.acked
            acked = held if held is not None and held.revision > revision else entry.unknown
            entries[record_id] = replace(entry, acked=acked, unknown=None)
        return Registry(entries)

    update_registry(profile, destination, change, create=False)


def _batch(batch_id: str) -> str:
    if not BATCH_ID.fullmatch(batch_id):
        raise ValueError("batch ids are bounded identifiers")
    return batch_id


def mark_queued(
    profile: str, destination: str, planned: Iterable[Planned], batch_id: str
) -> Registry:
    """Record ``planned`` as spooled (not yet sent) in ``batch_id``."""
    items = list(planned)
    batch = _batch(batch_id)
    return update_registry(profile, destination, lambda reg: queue(reg, items, batch))


def mark_acked(profile: str, destination: str, batch_id: str) -> Registry:
    """The collector fully accepted the batch: its queued states become ``acked``."""
    batch = _batch(batch_id)
    return update_registry(profile, destination, lambda r: ack(r, batch), create=False)


def mark_unknown(profile: str, destination: str, batch_id: str) -> Registry:
    """The batch was sent but its acceptance is unresolved (timeout, restart in flight)."""
    batch = _batch(batch_id)
    return update_registry(profile, destination, lambda r: unresolve(r, batch), create=False)


def mark_partial(profile: str, destination: str, batch_id: str) -> Registry:
    """The batch was partially accepted; item identities are unavailable, so all are unknown."""
    return mark_unknown(profile, destination, batch_id)


def mark_quarantined(profile: str, destination: str, batch_id: str) -> Registry:
    """A quarantined batch is treated like an unresolved one."""
    return mark_unknown(profile, destination, batch_id)


def reset_queued(profile: str, destination: str, batch_id: str) -> Registry:
    """The batch was discarded unsent: its queued states are removed."""
    batch = _batch(batch_id)
    return update_registry(profile, destination, lambda r: discard(r, batch), create=False)


def records_for_roots(profile: str, destination: str, root_keys: Iterable[str]) -> list[str]:
    """Record ids exported for conversations whose ``root_key`` is in ``root_keys``."""
    return ids_for_roots(load_registry(profile, destination), frozenset(root_keys))


def records_before(
    profile: str, destination: str, day: str, root_keys: Iterable[str] | None = None
) -> list[str]:
    """Record ids with a period before ``day`` (optionally only for the given roots)."""
    wanted = None if root_keys is None else frozenset(root_keys)
    return ids_before(load_registry(profile, destination), day, wanted)
