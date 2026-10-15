"""Affected-closure collection with immutable per-component canonical reductions.

This is derived state in the existing collector blob store, not a second database.
The durable queue contains session/component work, never copies of every event ID
in the checkpoint. Global replay/policy changes are named maintenance. A component
is the current atomic reducer unit; budgets are checked before every component.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_inputs import Retained
from ssgrep.accounting.collector_ledger import CANDIDATE_IDENTITY_METHOD
from ssgrep.accounting.collector_partition_graph import (
    components,
    journal_node,
    key,
    metadata,
    node,
)
from ssgrep.accounting.collector_partition_replay import combine
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.identity_claims import target_key
from ssgrep.accounting.plain_codec import from_plain, to_plain
from ssgrep.accounting.reduce_types import Reduction
from ssgrep.accounting.timeline import ts_ms
from ssgrep.accounting.valuation_inputs import EvidenceBundle

BLOB = "partitions"


@dataclass
class PartitionState:
    metas: dict[str, str] = field(default_factory=dict)
    parts: dict[str, dict] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    journal_digests: dict[str, str] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    maintenance: str | None = None
    methodology: str = ""
    frequency_refs: list[str] = field(default_factory=list)
    reference_index_complete: bool = True

    @property
    def pending_projections(self) -> tuple[str, ...]:
        """Current cached components not yet accepted under lookup-level identity."""
        return tuple(
            ident
            for ident, part in sorted(self.parts.items())
            if part.get("candidate_identity_method") != CANDIDATE_IDENTITY_METHOD
        )

    def to_dict(self):
        return {
            "schema": 1,
            "metas": self.metas,
            "parts": self.parts,
            "pending": self.pending,
            "journal_digests": self.journal_digests,
            "unavailable": self.unavailable,
            "maintenance": self.maintenance,
            "methodology": self.methodology,
            "frequency_refs": self.frequency_refs,
            "reference_index_complete": self.reference_index_complete,
        }


def load_partitions(profile: str, checkpoint: store.Checkpoint) -> PartitionState:
    raw = store.read_blob(profile, checkpoint.blobs.get(BLOB))
    if raw is None:
        return PartitionState()
    try:
        if not isinstance(raw, dict) or type(raw.get("schema")) is not int or raw["schema"] != 1:
            raise ValueError("unsupported partition schema")
        raw.pop("schema")
        state = PartitionState(**raw)
        if not isinstance(state.pending, list) or not isinstance(state.parts, dict):
            raise ValueError("invalid partition queue")
        return state
    except (ValueError, KeyError, TypeError) as error:
        raise CollectError("state_corrupt", "Collector partition state is invalid.") from error


def save_partitions(profile: str, state: PartitionState, blobs: dict[str, str]) -> None:
    blobs[BLOB] = store.write_blob(profile, BLOB, state.to_dict())


def references(profile: str, filename: str) -> set[str]:
    raw = store.read_blob(profile, filename)
    if not isinstance(raw, dict) or type(raw.get("schema")) is not int or raw["schema"] != 1:
        raise CollectError("state_corrupt", "Collector partition manifest is invalid.")
    return set(raw["metas"].values()) | {part["blob"] for part in raw["parts"].values()}


@dataclass
class PartitionOutcome:
    state: PartitionState
    reduction: Reduction | None
    desired: dict[str, EvidenceBundle]
    processed: int = 0
    stopped_by: str | None = None


def reduce_partitions(
    profile: str,
    checkpoint: store.Checkpoint,
    *,
    reducer,
    new_items,
    events,
    policy,
    frequency,
    fingerprint_key,
    ownership,
    ledger,
    frequency_assignments,
    native_sources=None,
    env=None,
    previous_doc=None,
    force: str | None = None,
    changed_sources=(),
    expired_sessions=frozenset(),
    clock,
    deadline: float,
    should_stop,
) -> PartitionOutcome:
    state = load_partitions(profile, checkpoint)
    state.unavailable = {
        name: reason
        for name, reason in state.unavailable.items()
        if key(name) not in expired_sessions
    }
    method = reducer.methodology_for(frequency).hash
    if state.methodology and state.methodology != method:
        state = PartitionState(maintenance="methodology_replay")
        force = "methodology_replay"
    original = {ident: dict(part) for ident, part in state.parts.items()}
    metas = {}
    for name, filename in state.metas.items():
        meta = store.read_blob(profile, filename)
        if not isinstance(meta, dict):
            raise CollectError("state_corrupt", "Collector partition metadata is invalid.")
        metas[name] = meta
    fresh: dict[str, list] = defaultdict(list)
    seeds = set(state.pending)
    for item in new_items:
        name = node((item.runtime, item.session_id))
        fresh[name].append(item)
        seeds.add(name)
        target = target_key(item)
        if target is not None:
            for owner in store.iter_evidence(profile, checkpoint.segments, native_keys={target}):
                seeds.add(node((owner.runtime, owner.session_id)))
    # Detect journal changes by phase content, not only ID/phase presence.
    journal: dict[str, list] = defaultdict(list)
    for event in events:
        journal[event.execution_id].append(event)
    signatures = {ident: digest_of(sorted(rows, key=digest_of)) for ident, rows in journal.items()}
    seeds.update(
        journal_node(ident)
        for ident in signatures.keys() | state.journal_digests.keys()
        if signatures.get(ident) != state.journal_digests.get(ident)
    )
    state.journal_digests = signatures
    for source in changed_sources:
        for name, meta in metas.items():
            if source.source_id in meta["sources"]:
                seeds.add(name)
        for part in state.parts.values():
            if source.source_id in part.get("witnesses", ()):
                seeds.update(part["members"])
        extra = source.state.get("extra", {})
        sid = extra.get("session_id") or extra.get("session")
        if not sid:
            sid = source.source_id.rsplit("/", 1)[-1].removesuffix(".jsonl")
        if isinstance(sid, str):
            seeds.update(
                (node((source.runtime, sid)), "@ref:" + sid, "@ref:" + source.runtime + ":" + sid)
            )
    needs_bootstrap = not state.methodology
    if force or needs_bootstrap:
        state.maintenance = force or "partition_bootstrap"
        # Compact index tables enumerate sessions without opening retained native text.
        from ssgrep.accounting.collector_segment_index import read_index

        for ref in checkpoint.segments:
            index = read_index(profile, ref)
            if index is None:
                # Named, one-segment bounded legacy index maintenance; no fake fresh.
                for item in store.read_segment(profile, ref):
                    seeds.add(node((item.runtime, item.session_id)))
            else:
                seeds.update(node(tuple(store.loads(name.encode()))) for name in index["sessions"])
        seeds.update(metas)
        seeds.update(journal_node(ident) for ident in journal)
    desired = {}
    processed = 0
    for ident in state.pending_projections:
        part = state.parts[ident]
        if force or seeds.intersection(part["members"]):
            continue  # Changed components must accept their current normal reduction.
        if should_stop() or clock() >= deadline:
            continue
        cached = from_plain(Reduction, store.read_blob(profile, part["blob"]))
        desired.update(zip(cached.bundle_roots, cached.bundles, strict=True))
        part["candidate_identity_method"] = CANDIDATE_IDENTITY_METHOD
        processed += 1
    if not seeds:
        return PartitionOutcome(state, None, desired, processed)
    items_by_session = {}
    blocked: dict[str, str] = {}
    # Refresh only changed session classifiers. This prerequisite is itself resumable:
    # unvisited seeds remain durable, and accepted component rows stay unchanged.
    deferred = set()
    for name in sorted(seeds):
        if name.startswith("@"):
            continue
        if name not in fresh and name in metas and not force and not needs_bootstrap:
            continue
        if should_stop() or clock() >= deadline:
            deferred.add(name)
            continue
        retained = Retained(profile, checkpoint.segments, env=env, sessions={key(name)})
        items = ([] if key(name) in expired_sessions else retained.items()) + fresh.get(name, [])
        unavailable = {} if key(name) in expired_sessions else retained.unavailable
        if unavailable:
            state.unavailable.update((node(k), why) for k, why in unavailable.items())
            blocked.update((node(k), why) for k, why in unavailable.items())
        else:
            state.unavailable.pop(name, None)
        if not items:
            continue
        items_by_session[name] = items
        updated = metadata(
            items, policy=policy, ownership=ownership, fingerprint_key=fingerprint_key
        )
        if unavailable and name in metas:
            prior = metas[name]
            for field in ("sources", "parents", "paths", "refs", "origins", "claims"):
                updated[field] = sorted(set(updated[field]) | set(prior[field]))
            updated["lineages"] = [
                list(pair)
                for pair in sorted(
                    {tuple(pair) for pair in (*updated["lineages"], *prior["lineages"])}
                )
            ]
            requests = {digest_of(row): row for row in (*prior["requests"], *updated["requests"])}
            updated["requests"] = [requests[ident] for ident in sorted(requests)]
        metas[name] = updated
        state.metas[name] = store.write_blob(profile, "partition-meta", metas[name])
    groups, membership = components(metas, events, original)
    # A disappeared dependency must invalidate both sides of its old component.
    affected = set(seeds)
    for part in original.values():
        if affected.intersection(part["members"]):
            affected.update(part["members"])
    wanted = {membership[name] for name in affected if name in membership}
    stamps = [stamp for event in events if (stamp := ts_ms(event.ts_utc)) is not None]
    span = (min(stamps), max(stamps)) if stamps else None
    state.pending = sorted(deferred)
    for ident in sorted(wanted):
        members = groups[ident]
        if members.intersection(deferred) or should_stop() or clock() >= deadline:
            state.pending.extend(sorted(members))
            continue
        sessions = {key(name) for name in members if name in metas}
        evidence = []
        unavailable = {name: blocked[name] for name in members if name in blocked}
        for session in sorted(sessions):
            name = node(session)
            if name in items_by_session:
                evidence.extend(items_by_session[name])
            else:
                retained = Retained(profile, checkpoint.segments, env=env, sessions={session})
                if session in expired_sessions:
                    unavailable[name] = "evidence_expired"
                    continue  # Closed/frozen history is intentionally non-recomputable.
                else:
                    evidence.extend(retained.items())
                    unavailable.update((node(k), why) for k, why in retained.unavailable.items())
                    state.unavailable.update(
                        (node(k), why) for k, why in retained.unavailable.items()
                    )
                evidence.extend(fresh.get(name, ()))
        if unavailable:
            # Missing context is explicit; accepted measurements are merged below,
            # while independent new supported usage still enters.
            state.unavailable.update(
                (name, why) for name, why in unavailable.items() if why != "evidence_expired"
            )
        selected = [
            event
            for execution_id, rows in journal.items()
            if journal_node(execution_id) in members
            for event in rows
        ]
        reduction = reducer.reduce_profile(
            profile,
            evidence,
            selected,
            policy=policy,
            frequency=frequency,
            fingerprint_key=fingerprint_key,
            ownership=ownership,
            native_sources=native_sources,
            frequency_assignments=frequency_assignments,
            capture_span=span,
            partitioned=True,
        )
        previous_parts = {
            old: part for old, part in state.parts.items() if members.intersection(part["members"])
        }
        current_desired = dict(zip(reduction.bundle_roots, reduction.bundles, strict=True))
        if unavailable or frequency_assignments.unavailable:
            from ssgrep.accounting.collector_partition_replay import preserve

            previous = (
                combine(profile, previous_parts, frequency_assignments) if previous_parts else None
            )
            if previous is None and previous_doc is not None:
                previous = Reduction(
                    profile,
                    executions=tuple(
                        row
                        for row in previous_doc.executions
                        if journal_node(row.execution_id) in members
                        or node((row.runtime or "", row.session_id or "")) in members
                    ),
                    invocations=tuple(
                        row
                        for row in previous_doc.invocations
                        if node((row.runtime, row.session_id)) in members
                    ),
                    coverage=previous_doc.coverage,
                    session_roots=tuple(
                        row
                        for row in previous_doc.session_roots
                        if node((row[0], row[1])) in members
                    ),
                    session_parents=tuple(
                        row
                        for row in previous_doc.session_parents
                        if node((row[0], row[1])) in members
                    ),
                )
            reduction, current_desired = preserve(
                ledger,
                reduction,
                previous,
                {key(name): why for name, why in unavailable.items()},
                frequency_assignments,
            )
        removed = [
            old for old, part in state.parts.items() if members.intersection(part["members"])
        ]
        for old in removed:
            desired.update((root, EvidenceBundle((), ())) for root in state.parts[old]["roots"])
            del state.parts[old]
        desired.update(current_desired)
        state.parts[ident] = {
            "candidate_identity_method": CANDIDATE_IDENTITY_METHOD,
            "members": sorted(members),
            "roots": list(reduction.bundle_roots),
            "blob": store.write_blob(profile, "partition-rows", to_plain(reduction)),
            "frequency_refs": sorted({row.view_id for row in reduction.frequency_assignments}),
            "witnesses": sorted(
                {source for name in members if name in metas for source in metas[name]["sources"]}
                | {
                    reference.runtime + ":" + str(reference.path)
                    for name in members
                    if name in metas
                    for ref in metas[name]["refs"]
                    if native_sources is not None
                    and (reference := native_sources.observed_references.get(ref)) is not None
                }
            ),
        }
        processed += 1
    state.pending = sorted(set(state.pending))
    if not state.pending:
        state.maintenance = None
    state.reference_index_complete = all("frequency_refs" in part for part in state.parts.values())
    state.frequency_refs = sorted(
        {view for part in state.parts.values() for view in part.get("frequency_refs", ())}
    )
    state.methodology = method
    reduction = combine(profile, state.parts, frequency_assignments) if processed else None
    return PartitionOutcome(
        state,
        reduction,
        desired,
        processed,
        "reduction_budget" if state.pending or state.pending_projections else None,
    )
