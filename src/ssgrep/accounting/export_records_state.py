"""Pure registry state transitions (queued -> acked / unknown, or discarded).

Every function takes a ``Registry`` and returns a new one; ``export_registry`` runs them under
the registry flock. The invariants they keep:

* ``acked`` only ever advances: an older acknowledgement never replaces a newer one;
* ``unknown`` (sent, acceptance unresolved) never regresses either and disappears once an
  acknowledgement at or above its revision exists;
* a record with no remaining state at all is forgotten, so planning starts again at revision 1
  and a discarded-unsent batch re-plans the SAME revision numbers it held.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

from ssgrep.accounting.export_records_model import Entry, Planned, Registry, State, UnitRef


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
