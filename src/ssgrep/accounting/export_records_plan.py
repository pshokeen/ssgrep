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

from __future__ import annotations

from collections.abc import Iterable

from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.export_records_model import (
    ADJUSTMENT,
    CONTRIBUTION,
    DELETION,
    METHODOLOGY,
    SCHEMA,
    Candidate,
    Entry,
    Planned,
    Registry,
)


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
