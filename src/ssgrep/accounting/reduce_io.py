"""Read-only adapters that feed :func:`reduce_profile` from the real stores.

Loads raw evidence through the six bounded readers (never the retrieval adapters),
the accepted journal events of one profile, the profile's policy and the keyed
request-fingerprint key, then reduces. Nothing is written: no checkpoints, no
quarantine, no key creation, and transcript content is never executed or fetched.
A caller that wants to bound the work passes ``only_under`` (directories or files
of one root conversation) or an ``include`` predicate over sources.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ssgrep.accounting import fingerprint, journal_reader, policy
from ssgrep.accounting.contracts import ExecutionEvent, RawEvidence
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.native_source import NativeSourceAdapter, NativeSourceReference
from ssgrep.accounting.policy_rules import PolicyFile
from ssgrep.accounting.readers.base import EvidenceLedger, JsonlReader, ReadResult, Source
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader
from ssgrep.accounting.reduce import CONSUMED_KINDS, reduce_profile
from ssgrep.accounting.reduce_types import Reduction
from ssgrep.accounting.scope_events import Scoper

CYCLE_BYTES = 32 * 1024 * 1024


@dataclass
class Loaded:
    """Everything one profile reduction needs, plus honest read coverage."""

    evidence: list[RawEvidence] = field(default_factory=list)
    events: list[ExecutionEvent] = field(default_factory=list)
    sources_read: dict[str, int] = field(default_factory=dict)
    gaps: dict[str, int] = field(default_factory=dict)
    unreadable_sources: int = 0
    pending_bytes: int = 0
    journal_damage: int = 0
    journal_segments: int = 0
    duplicates: int = 0
    conflicts: int = 0


def readers(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    return {
        "claude": ClaudeReader(env),
        "codex": CodexReader(env),
        "omp": OmpReader(env),
        "opencode": OpenCodeReader(env),
        "pi": PiReader(env),
        "prime-agent": PrimeReader(env),
    }


def _under(path: Path, bases: tuple[Path, ...]) -> bool:
    real = Path(os.path.realpath(path))
    return any(real == base or base in real.parents for base in bases)


def _read_source(reader: JsonlReader | OpenCodeReader, source: Source) -> Iterable[ReadResult]:
    state = None
    while True:
        budget = None if isinstance(reader, OpenCodeReader) else CYCLE_BYTES
        result = reader.read(source, state, budget)
        yield result
        state = result.state
        if not result.more:
            return


def load_evidence(
    env: Mapping[str, str] | None = None,
    *,
    only_under: Iterable[Path] | None = None,
    include: Callable[[Source], bool] | None = None,
    kinds: frozenset[str] = CONSUMED_KINDS,
) -> Loaded:
    """Raw evidence from every runtime store; identical native copies count once."""
    bases = tuple(Path(os.path.realpath(p)) for p in only_under) if only_under else None
    loaded = Loaded()
    ledger = EvidenceLedger()
    for name, reader in readers(env).items():
        count = 0
        for source in reader.discover():
            if bases is not None and not _under(source.path, bases):
                continue
            if include is not None and not include(source):
                continue
            count += 1
            try:
                for result in _read_source(reader, source):
                    for gap in result.gaps:
                        loaded.gaps[gap.reason] = loaded.gaps.get(gap.reason, 0) + 1
                    loaded.pending_bytes += result.pending_bytes
                    loaded.evidence.extend(
                        e for e in result.evidence if e.kind in kinds and ledger.admit(e)
                    )
            except OSError:
                loaded.unreadable_sources += 1
        loaded.sources_read[name] = count
    loaded.duplicates = ledger.duplicates
    loaded.conflicts = ledger.conflict_count
    return loaded


def load_journal(profile: str) -> tuple[list[ExecutionEvent], int, int]:
    """Accepted journal events, damaged-record count, and segment count (read-only)."""
    read = journal_reader.read_journal(profile, quarantine=False)
    return list(read.events), len(read.damage), read.segments


def native_source_adapter(
    profile_policy: ProfilePolicy,
    env: Mapping[str, str] | None = None,
    *,
    now_ms: int,
    references: Mapping[str, NativeSourceReference] | None = None,
    repository: Any = None,
    ownership: PolicyFile | None = None,
    scoper: Scoper | None = None,
) -> NativeSourceAdapter:
    """An on-demand source view with the full workspace ownership authority.

    Default Scoper reads current policy including explicit assignments; callers
    can supply the reducer's exact snapshot/instance. No source discovery scan.
    """
    from ssgrep.accounting.readers import roots

    native_readers = readers(env)
    approved = tuple(
        root
        for reader in native_readers.values()
        if isinstance(reader, JsonlReader)
        for root in reader.roots()
    ) + (roots.opencode_db(env).parent,)
    return NativeSourceAdapter(
        references or {},
        native_readers,
        approved,
        scoper if scoper is not None else Scoper(profile_policy, ownership),
        now_ms,
        repository=repository,
    )


def reduce_stored_profile(
    profile: str,
    env: Mapping[str, str] | None = None,
    *,
    only_under: Iterable[Path] | None = None,
    frequency: FrequencyView | None = None,
    verifier_sessions: frozenset[str] = frozenset(),
    loaded: Loaded | None = None,
    native_sources: NativeSourceAdapter | None = None,
    source_now_ms: int | None = None,
    ownership: PolicyFile | None = None,
) -> tuple[Reduction, Loaded]:
    """Reduce one profile from the real stores; reads only, never writes."""
    data = loaded or load_evidence(env, only_under=only_under)
    if loaded is None:
        events, damage, segments = load_journal(profile)
        data.events, data.journal_damage, data.journal_segments = events, damage, segments
    if frequency is None:
        from ssgrep.accounting.index_frequency import read_index_view

        view = read_index_view(profile)
    else:
        view = frequency
    profile_policy = policy.load_profile_policy(profile)
    if native_sources is None:
        from datetime import UTC, datetime

        now_ms = (
            source_now_ms
            if source_now_ms is not None
            else int(datetime.now(UTC).timestamp() * 1000)
        )
        native_sources = native_source_adapter(
            profile_policy, env, now_ms=now_ms, ownership=ownership
        )
    if ownership is None:
        ownership = native_sources.scoper.ownership
    key = fingerprint.load_key(profile, create=False)
    reduction = reduce_profile(
        profile,
        data.evidence,
        data.events,
        policy=profile_policy,
        ownership=ownership,
        frequency=view,
        fingerprint_key=key,
        verifier_sessions=verifier_sessions,
        native_sources=native_sources,
    )
    return reduction, data
