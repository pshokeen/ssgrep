"""Source-atom resolution from raw evidence with verified identity (task 6.4).

A ref returned by search/show is only a *locator*: it picks the source session.
Atoms are the retained raw records of that session, identified by their origin
event identity (``runtime:session:record``) plus the record's content digest. An atom
is credited only when it verifies:

* its digest matches the digest recorded earlier for the same atom id (drift -> no
  credit, receipt note) and no conflicting variant of the record exists;
* it is not the consumer's own later work, nor content that postdates the exposure;
* it lies inside the profile's approved workspaces (else missing coverage);
* it does not close a provenance cycle (``find_cycles``).

Nothing is guessed from episode ordinals or prompt-prefix text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from ssgrep.accounting.contracts import RUNTIMES
from ssgrep.accounting.evidence import TokenKeys
from ssgrep.accounting.evidence_tokens import Token
from ssgrep.accounting.ledger_contracts import SourceAtom
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.timeline import Record, SessionKey, Timeline, Timelines, record_text
from ssgrep.accounting.valuation_inputs import AtomStatus, ValuationAtom

REF_RE = re.compile(r"^(.+):ep:\d+$")
MAX_ATOMS = 600
MAX_REFS = 5


@dataclass(frozen=True)
class Consumer:
    """Where the consuming exposure happened (for self-later-work checks)."""

    root: str
    session: SessionKey
    seq: int
    ts: int | None


@dataclass(frozen=True)
class Resolution:
    """Atoms per tier for one candidate, plus why anything was not creditable."""

    immediate: tuple[ValuationAtom, ...] = ()
    discovery: tuple[ValuationAtom, ...] = ()
    episode: tuple[ValuationAtom, ...] = ()
    reasons: tuple[tuple[str, str], ...] = ()  # (reason, source)
    source_roots: tuple[str, ...] = ()
    witness_ts: int | None = None
    source_user_supplied: bool = False
    locators: tuple[str, ...] = ()


def resolve_ref(ref: str, sessions: Mapping[SessionKey, object]) -> SessionKey | None:
    """The session a ref locates; None when it cannot be resolved from retained evidence."""
    found = REF_RE.match(ref)
    if not found:
        return None
    base = found.group(1)
    for runtime in RUNTIMES:
        if base.startswith(runtime + ":"):
            key = (runtime, base[len(runtime) + 1 :])
            if key in sessions:
                return key
    key = ("claude", base)
    if key in sessions:
        return key
    matches = sorted(k for k in sessions if k[1] == base)
    return matches[0] if len(matches) == 1 else None


def atom_identity(record: Record) -> tuple[str, str, str]:
    origin = record.identity
    return origin, origin, record.ev.digest


class _SourceIndex:
    """Earliest occurrence of each candidate key in one source session, and its first result."""

    def __init__(self, line: Timeline, tk: TokenKeys) -> None:
        self.first: dict[str, tuple[int, str]] = {}
        self.result: dict[str, int] = {}
        for record in line.records:
            for key in tk.of(record):
                if key not in self.first:
                    self.first[key] = (record.seq, record.kind)
                if record.kind == "tool_result" and key not in self.result:
                    self.result[key] = record.seq

    def witness(self, token: Token) -> tuple[int, str, int | None] | None:
        """``(first seq, first kind, first result seq)`` of the whole token, else None."""
        first = self.first.get(token.key)
        return None if first is None else (first[0], first[1], self.result.get(token.key))


@dataclass
class Resolver:
    timelines: Timelines
    scoper: Scoper
    known_atoms: Mapping[str, str] = field(default_factory=dict)  # origin event -> digest
    root_of: Mapping[SessionKey, SessionKey] = field(default_factory=dict)
    tk: TokenKeys = field(default_factory=TokenKeys)
    _indexes: dict[SessionKey, _SourceIndex] = field(default_factory=dict)

    def _index(self, key: SessionKey) -> _SourceIndex:
        found = self._indexes.get(key)
        if found is None:
            found = self._indexes[key] = _SourceIndex(self.timelines.sessions[key], self.tk)
        return found

    def _status(self, line: Timeline, record: Record, consumer: Consumer) -> tuple[AtomStatus, str]:
        scope, why = self.scoper.source_status(line.cwd_at(record.seq))
        if scope is not AtomStatus.OK:
            return scope, why
        _, origin, digest = atom_identity(record)
        if origin in self.timelines.conflicted:
            return AtomStatus.DIGEST_CONFLICT, "conflicting_source_variants"
        same_root = self.root_of.get(line.key, line.key)
        if f"{same_root[0]}:{same_root[1]}" == consumer.root:
            same = line.key == consumer.session
            comparable = consumer.ts is not None and record.ts is not None
            later = record.seq >= consumer.seq if same else not comparable
            if later or (comparable and record.ts >= consumer.ts):
                return AtomStatus.SELF_LATER_WORK, "consumer_own_later_work"
        elif consumer.ts is not None and record.ts is not None and record.ts > consumer.ts:
            return AtomStatus.UNAVAILABLE, "source_postdates_exposure"
        return AtomStatus.OK, ""

    def _atom(
        self, line: Timeline, record: Record, consumer: Consumer, reasons: set
    ) -> ValuationAtom | None:
        text = record_text(record)
        if not text:
            return None
        _, origin, digest = atom_identity(record)
        atom_id = origin
        status, why = self._status(line, record, consumer)
        drift = self._drift(origin, digest)
        if status is AtomStatus.OK and drift:
            status, why = AtomStatus.DIGEST_CONFLICT, "source_digest_drift"
        if status is not AtomStatus.OK:
            reasons.add((why, line.key[0] + ":" + line.key[1]))
        following = line.records[record.seq + 1] if record.seq + 1 < len(line.records) else None
        t_start = record.ts // 1000 if record.ts is not None else None
        t_end = None
        if t_start is not None:
            end_ms = following.ts if following is not None and following.ts is not None else None
            t_end = max(t_start, end_ms // 1000) if end_ms is not None else t_start
        source = SourceAtom(atom_id, origin, digest, len(text))
        return ValuationAtom(source, 0, len(text), t_start, t_end, None, status)

    def _drift(self, origin: str, digest: str) -> bool:
        """The digest recorded for this origin event earlier differs from what resolves now."""
        held = self.known_atoms.get(origin)
        return held is not None and held != digest

    def resolve(self, token: Token, refs: tuple[str, ...], consumer: Consumer) -> Resolution:
        reasons: set[tuple[str, str]] = set()
        immediate: dict[str, ValuationAtom] = {}
        discovery: dict[str, ValuationAtom] = {}
        episode: dict[str, ValuationAtom] = {}
        roots: set[str] = set()
        locators: set[str] = set()
        witness_ts: int | None = None
        user_supplied = False
        for ref in sorted(set(refs))[:MAX_REFS]:
            key = resolve_ref(ref, self.timelines.sessions)
            if key is None:
                reasons.add(("source_unresolved", "ref"))
                continue
            line = self.timelines.sessions[key]
            hit = self._index(key).witness(token)
            if hit is None:
                reasons.add(("witness_not_found", key[0] + ":" + key[1]))
                continue
            first_seq, kind, seq_or_none = hit
            label = key[0] + ":" + key[1]
            if seq_or_none is None:
                reasons.add(("source_not_observed", label))
                continue
            seq = seq_or_none
            if kind == "user_turn" and first_seq < seq:
                user_supplied = True
                reasons.add(("source_user_supplied", label))
                continue
            if kind == "tool_call" and first_seq < seq:
                reasons.add(("source_known_before_witness", label))
                continue
            witness = line.records[seq]
            root_key = self.root_of.get(key, key)
            roots.add(f"{root_key[0]}:{root_key[1]}")
            locators.add(f"{witness.ev.source_id}@{witness.ev.locator}")
            if witness.ts is not None:
                witness_ts = witness.ts if witness_ts is None else min(witness_ts, witness.ts)
            turn = max((t.seq for t in line.turns if t.seq <= seq), default=0)
            later_turns = [t.seq for t in line.turns if t.seq > seq]
            end = (later_turns[0] if later_turns else len(line.records)) - 1
            call = line.by_id.get("call:" + witness.ev.record_id.split(":", 1)[1])
            invalid = self._atom(line, witness, consumer, set())
            if invalid is not None and invalid.status is not AtomStatus.OK:
                # the witness itself is not creditable, so nothing around it is either
                self._put(immediate, line, witness, consumer, reasons)
                status = invalid.status
            else:
                status = AtomStatus.OK
            for record in ([call] if call is not None else []) + [witness]:
                self._put(immediate, line, record, consumer, reasons, status)
            for record in line.records[turn : seq + 1][:MAX_ATOMS]:
                self._put(discovery, line, record, consumer, reasons, status)
            for record in line.records[turn : end + 1][:MAX_ATOMS]:
                self._put(episode, line, record, consumer, reasons, status)
        if not (immediate or discovery or episode) and not reasons:
            reasons.add(("source_unresolved", "ref"))
        placeholder = ()
        if not (immediate or discovery or episode) and reasons and not user_supplied:
            placeholder = (self._placeholder(token, sorted(reasons)[0][0]),)
        order = lambda items: tuple(items[k] for k in sorted(items))  # noqa: E731
        return Resolution(
            order(immediate) or placeholder,
            order(discovery),
            order(episode),
            tuple(sorted(reasons)),
            tuple(sorted(roots)),
            witness_ts,
            user_supplied,
            tuple(sorted(locators)),
        )

    def _put(self, bucket, line, record, consumer, reasons, forced=AtomStatus.OK) -> None:
        atom = self._atom(line, record, consumer, reasons)
        if atom is not None:
            if forced is not AtomStatus.OK and atom.status is AtomStatus.OK:
                atom = replace(atom, status=forced)
            bucket[atom.atom_id] = atom

    @staticmethod
    def _placeholder(token: Token, reason: str) -> ValuationAtom:
        identity = f"unresolved:{reason}:{token.kind}"
        atom = SourceAtom(identity, identity, "", 0)
        return ValuationAtom(atom, 0, 0, None, None, None, AtomStatus.UNAVAILABLE)


@dataclass(frozen=True)
class Edge:
    consumer_root: str
    source_roots: tuple[str, ...]
    witness_ts: int | None
    exposure_ts: int | None


def find_cycles(edges: list[Edge]) -> set[int]:
    """Indexes of edges whose source content derives from the consumer root's own work.

    ``C`` reusing content ``w`` of root ``R`` is cyclic when ``R`` had itself absorbed
    content of ``C`` (transitively) before it produced ``w``.
    """
    by_consumer: dict[str, list[Edge]] = {}
    for edge in edges:
        by_consumer.setdefault(edge.consumer_root, []).append(edge)

    def absorbed(root: str, before: int | None, seen: frozenset[str]) -> set[str]:
        out: set[str] = set()
        if root in seen:
            return out
        for edge in by_consumer.get(root, ()):
            if before is not None and edge.exposure_ts is not None and edge.exposure_ts >= before:
                continue
            if before is not None and edge.exposure_ts is None:
                continue
            for source in edge.source_roots:
                if source == root:
                    continue
                out.add(source)
                out |= absorbed(source, edge.witness_ts, seen | {root})
        return out

    cyclic: set[int] = set()
    for index, edge in enumerate(edges):
        if edge.witness_ts is None:
            continue
        for source in edge.source_roots:
            if source != edge.consumer_root and edge.consumer_root in absorbed(
                source, edge.witness_ts, frozenset()
            ):
                cyclic.add(index)
    return cyclic


def mark_cycle(atoms: tuple[ValuationAtom, ...]) -> tuple[ValuationAtom, ...]:
    return tuple(
        replace(a, status=AtomStatus.CYCLE) if a.status is AtomStatus.OK else a for a in atoms
    )
