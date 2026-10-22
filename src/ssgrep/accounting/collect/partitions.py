"""Affected-closure collection with immutable per-component canonical reductions.

This is derived state in the existing collector blob store, not a second database.
The durable queue contains session/component work, never copies of every event ID
in the checkpoint. Global replay/policy changes are named maintenance. A component
is the current atomic reducer unit; budgets are checked before every component.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, replace

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.read import REF_RE, Retained, resolve_ref
from ssgrep.accounting.collect.replay import preserve_unavailable
from ssgrep.accounting.collect.store import CollectError
from ssgrep.accounting.ledger.contracts import Attribution, InvocationClass, digest_of, target_key
from ssgrep.accounting.ledger.reduce_join import Request, fold_events, interpret, reconcile
from ssgrep.accounting.ledger.reduce_records import Reduction, summarize
from ssgrep.accounting.ledger.valuation_inputs import AtomStatus, Eligibility, EvidenceBundle
from ssgrep.accounting.privacy.storage import from_plain, to_plain
from ssgrep.accounting.readers.timeline import ts_ms


@dataclass
class PartitionOutcome:
    state: store.PartitionState
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
    from ssgrep.accounting.collect.cycle import read_index
    from ssgrep.accounting.collect.cycle_publish import combine
    from ssgrep.accounting.collect.state import CANDIDATE_IDENTITY_METHOD

    state = store.load_partitions(profile, checkpoint)

    state.unavailable = {
        name: reason
        for name, reason in state.unavailable.items()
        if key(name) not in expired_sessions
    }
    method = reducer.methodology_for(frequency).hash
    if state.methodology and state.methodology != method:
        state = store.PartitionState(maintenance="methodology_replay")
        force = force or "methodology_replay"
    # A replacement is a durable request, not permission to restart every cycle.
    if force == state.operation:
        force = None
    elif force:
        state.operation = force
    state.blocked_reason = None
    # Detect journal changes by phase content, not only ID/phase presence.
    journal: dict[str, list] = defaultdict(list)
    for event in events:
        journal[event.execution_id].append(event)
    signatures = {ident: digest_of(sorted(rows, key=digest_of)) for ident, rows in journal.items()}
    original = {ident: dict(part) for ident, part in state.parts.items()}
    cached_groups = (
        state.pending_groups
        if not (
            force
            or new_items
            or changed_sources
            or state.metadata_pending
            or signatures != state.journal_digests
        )
        else {}
    )

    def read_meta(name):
        meta = store.read_blob(profile, state.metas[name])
        if not isinstance(meta, dict):
            raise CollectError("state_corrupt", "Collector partition metadata is invalid.")
        return meta

    metas = {name: read_meta(name) for name in state.metas} if changed_sources else {}
    fresh: dict[str, list] = defaultdict(list)
    seeds = set(state.pending)
    dirty = set(state.metadata_pending)
    for item in new_items:
        name = node((item.runtime, item.session_id))
        fresh[name].append(item)
        seeds.add(name)
        dirty.add(name)
        target = target_key(item)
        if target is not None:
            for owner in store.iter_evidence(profile, checkpoint.segments, native_keys={target}):
                seeds.add(node((owner.runtime, owner.session_id)))
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
                dirty.add(name)
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
        state.maintenance = force.split(":", 1)[0] if force else "partition_bootstrap"
        # Compact index tables enumerate sessions without opening retained native text.

        for ref in checkpoint.segments:
            index = read_index(profile, ref)
            if index is None:
                # Named, one-segment bounded legacy index maintenance; no fake fresh.
                for item in store.read_segment(profile, ref):
                    seeds.add(node((item.runtime, item.session_id)))
            else:
                seeds.update(node(tuple(store.loads(name.encode()))) for name in index["sessions"])
        seeds.update(state.metas)
        seeds.update(journal_node(ident) for ident in journal)
        if needs_bootstrap or not (force or "").startswith("frequency_"):
            dirty.update(name for name in seeds if not name.startswith("@"))
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
        if state.pending_projections and not processed and not should_stop():
            state.blocked_reason = "reduction_blocked"
        return PartitionOutcome(state, None, desired, processed, state.blocked_reason)
    items_by_session = {}
    # Refresh only changed session classifiers. This prerequisite is itself resumable:
    # unvisited seeds remain durable, and accepted component rows stay unchanged.
    deferred = set()
    metadata_processed = 0
    for name in sorted(seeds):
        if name.startswith("@") or cached_groups:
            continue
        if name not in dirty and name in state.metas:
            continue
        if should_stop() or clock() >= deadline:
            deferred.add(name)
            continue
        if name in state.metas and name not in metas:
            metas[name] = read_meta(name)
        retained = Retained(profile, checkpoint.segments, env=env, sessions={key(name)})
        metadata_processed += 1
        items = ([] if key(name) in expired_sessions else retained.items()) + fresh.get(name, [])
        unavailable = {} if key(name) in expired_sessions else retained.unavailable
        if unavailable:
            state.unavailable.update((node(k), why) for k, why in unavailable.items())
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
    if deferred:
        state.pending = sorted(seeds)
        state.metadata_pending = sorted(deferred)
        state.pending_groups = {}
        state.methodology = method
        if not (metadata_processed or processed or should_stop()):
            state.blocked_reason = "reduction_blocked"
        return PartitionOutcome(
            state, None, desired, processed, state.blocked_reason or "reduction_budget"
        )
    if cached_groups:
        groups = {ident: set(members) for ident, members in cached_groups.items()}
        membership = {name: ident for ident, members in groups.items() for name in members}
    else:
        metas.update((name, read_meta(name)) for name in state.metas if name not in metas)
        groups, membership = components(metas, events, original)
    # A disappeared dependency must invalidate both sides of its old component.
    affected = set(seeds)
    for part in original.values():
        if affected.intersection(part["members"]):
            affected.update(part["members"])
    wanted = {membership[name] for name in affected if name in membership}
    stamps = [stamp for event in events if (stamp := ts_ms(event.ts_utc)) is not None]
    span = (min(stamps), max(stamps)) if stamps else None
    state.pending = []
    for ident in sorted(wanted):
        members = groups[ident]
        if should_stop() or clock() >= deadline:
            state.pending.extend(sorted(members))
            continue
        sessions = {key(name) for name in members if name in state.metas}
        for name in members:
            if name in state.metas and name not in metas:
                metas[name] = read_meta(name)
        evidence = []
        unavailable = {
            name: state.unavailable[name] for name in members if name in state.unavailable
        }
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
                    if not retained.unavailable:
                        state.unavailable.pop(name, None)
                        unavailable.pop(name, None)
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
            from ssgrep.accounting.collect.partitions import preserve

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
    state.metadata_pending = []
    pending = set(state.pending)
    state.pending_groups = {
        ident: sorted(groups[ident]) for ident in wanted if pending.intersection(groups[ident])
    }
    if (
        (state.pending or state.pending_projections)
        and not processed
        and not metadata_processed
        and not (state.pending_groups and not cached_groups)
        and not should_stop()
    ):
        state.blocked_reason = "reduction_blocked"
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
        state.blocked_reason
        or ("reduction_budget" if state.pending or state.pending_projections else None),
    )


"""Compact session metadata and conservative reduction dependency components.

Edges express actual parent/context, source witnesses, native ownership and journal
competition. A source dependency is bidirectional for *processing*, not accounting:
a changed source must invalidate its consumers and a consumer needs the source context.
Missing parents/refs stay as graph nodes, so late arrival invalidates earlier results.
"""


def node(key: tuple[str, str]) -> str:
    # Session IDs may contain ':', so split once only.
    return key[0] + ":" + key[1]


def key(name: str) -> tuple[str, str]:
    runtime, session = name.split(":", 1)
    return runtime, session


def journal_node(execution_id: str) -> str:
    return "@journal:" + execution_id


def metadata(items, *, policy, ownership, fingerprint_key) -> dict:
    """One hydrated session; reuse the reducer's classifier instead of guessing joins."""
    from ssgrep.accounting.ledger.reduce_records import collect_calls
    from ssgrep.accounting.readers.pi_family import build_lineage
    from ssgrep.accounting.readers.timeline import build_timelines
    from ssgrep.accounting.views.filter import Scoper

    timelines = build_timelines(items)
    calls = collect_calls(
        timelines, build_lineage(timelines), Scoper(policy, ownership), fingerprint_key, frozenset()
    )
    refs = set()
    parents = set()
    paths = set()
    origins = set()
    claims = set()
    lineages = set()
    sources = set()
    for item in items:
        sources.add(item.source_id)
        p = item.payload
        context = p.get("session_context")
        context = context if isinstance(context, dict) else {}
        for payload in (p, context):
            for payload_field in ("parent_session", "parent_session_id", "parent_id"):
                value = payload.get(payload_field)
                if isinstance(value, str) and value:
                    parents.add(node((item.runtime, value)))
            value = payload.get("parent_session_path")
            if isinstance(value, str) and value:
                paths.add(os.path.normpath(value))
        if item.kind == "lineage":
            child, parent = p.get("session_file"), p.get("parent_session_id")
            if isinstance(child, str) and isinstance(parent, str):
                lineages.add((os.path.normpath(child), node((item.runtime, parent))))
        if item.kind == "tool_result":
            refs.update(card.ref for card in interpret(p).cards)
        target = target_key(item)
        if target is not None:
            claims.add(digest_of(target))
        else:
            origins.add(digest_of(item.native_key))
    return {
        "sources": sorted(sources),
        "parents": sorted(parents),
        "paths": sorted(paths),
        "refs": sorted(refs),
        "origins": sorted(origins),
        "claims": sorted(claims),
        "lineages": [list(pair) for pair in sorted(lineages)],
        "requests": [to_plain(r) for c in calls.calls for r in c.requests],
    }


def components(
    metas: dict[str, dict], events, old_parts
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Exact join candidates plus conservative ancestry/witness/ownership closure."""
    links: dict[str, set[str]] = defaultdict(set)

    def edge(a, b):
        links[a].add(b)
        links[b].add(a)

    sessions = {key(name): True for name in metas}
    by_source: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_origin: dict[str, set[str]] = defaultdict(set)
    requests = []
    request_session = {}
    for name, meta in metas.items():
        links[name]
        runtime = key(name)[0]
        for source in meta["sources"]:
            by_source[(runtime, os.path.normpath(source.split(":", 1)[-1]))].append(name)
        for origin in (*meta["origins"], *meta["claims"]):
            by_origin[origin].add(name)
        for raw in meta["requests"]:
            request = from_plain(Request, raw)
            requests.append(request)
            request_session[request.key] = name
    for names in by_origin.values():
        names = sorted(names)
        for name in names[1:]:
            edge(names[0], name)
    for name, meta in metas.items():
        runtime = key(name)[0]
        for parent in meta["parents"]:
            edge(name, parent)
        for path in meta["paths"]:
            for parent in by_source.get((runtime, path), ()):
                edge(name, parent)
            # The path placeholder survives missing ancestors and late source arrival.
            placeholder = "@path:" + runtime + ":" + path
            edge(name, placeholder)
            for parent in by_source.get((runtime, path), ()):
                edge(placeholder, parent)
        for child_path, parent in meta["lineages"]:
            edge(name, parent)
            placeholder = "@path:" + runtime + ":" + child_path
            edge(name, placeholder)
            for child in by_source.get((runtime, child_path), ()):
                edge(placeholder, child)
        for ref in meta["refs"]:
            resolved = resolve_ref(ref, sessions)
            if resolved is not None:
                edge(name, node(resolved))
            found = REF_RE.match(ref)
            if found:
                placeholder = "@ref:" + found.group(1)
                edge(name, placeholder)
                for session in metas:
                    if session == found.group(1) or key(session)[1] == found.group(1):
                        edge(placeholder, session)
    facts = fold_events(list(events))
    joined = reconcile(facts, requests)
    for fact in facts:
        name = journal_node(fact.execution_id)
        links[name]
        link = joined[fact.execution_id]
        for request in (*link.requests, *link.candidates):
            if request in request_session:
                edge(name, request_session[request])
    # Keep a previously accepted component as one replacement transaction. This
    # conservative edge prevents a split's completed half from deleting the cached
    # other half when its context is unavailable or its budget has not yet arrived.
    # It does not alter reducer lineage/joins; it only widens the processing unit.
    for part in old_parts.values():
        members = part["members"]
        for member in members[1:]:
            edge(members[0], member)
    groups = {}
    membership = {}
    for seed in sorted(links):
        if seed in membership:
            continue
        members = set()
        pending = [seed]
        while pending:
            name = pending.pop()
            if name in members:
                continue
            members.add(name)
            pending.extend(links[name] - members)
        ident = digest_of(sorted(members))[:24]
        groups[ident] = members
        membership.update((name, ident) for name in members)
    return groups, membership


"""Keep accepted canonical rows while admitting independent new supported lookups."""


def preserve(ledger, reduction, previous, unavailable, frequencies):
    from ssgrep.accounting.collect.state import candidate_identity
    from ssgrep.accounting.collect.steps import reduction_doc

    old_doc = reduction_doc(previous, {}) if previous else None
    relevant = {row.invocation_id for row in reduction.invocations}
    if previous:
        relevant.update(row.invocation_id for row in previous.invocations)
    relevant.update("journal:" + row.execution_id for row in reduction.executions)
    missing_frequencies = {
        ident: why for ident, why in frequencies.unavailable.items() if ident in relevant
    }
    reduction, desired = preserve_unavailable(
        ledger, reduction, old_doc, unavailable, missing_frequencies
    )
    blocked = set(missing_frequencies)
    actors = {runtime + ":" + session for runtime, session in unavailable}
    if previous:
        blocked.update(
            row.invocation_id
            for row in previous.invocations
            if (row.runtime, row.session_id) in unavailable
        )
    kept_candidates = [
        row
        for row in (previous.candidates if previous else ())
        if row.invocation_id in blocked
        or any(
            atom.actor in actors
            for atom in (*row.candidate.immediate, *row.candidate.discovery, *row.candidate.episode)
        )
    ]
    identities = {candidate_identity(row.candidate) for row in kept_candidates}
    candidates = [
        row for row in reduction.candidates if candidate_identity(row.candidate) not in identities
    ]
    candidates.extend(kept_candidates)
    old_invocations = {row.invocation_id for row in previous.invocations} if previous else set()
    missing_context = {
        row.invocation_id
        for row in reduction.invocations
        if (row.runtime, row.session_id) in unavailable and row.invocation_id not in old_invocations
    }
    updated_candidates = {}
    for index, row in enumerate(candidates):
        if row.invocation_id in missing_context:
            reason = "prior_context_replay_unavailable"
            candidate = replace(
                row.candidate,
                eligibility=Eligibility.UNAVAILABLE,
                eligibility_reasons=tuple(
                    sorted(set(row.candidate.eligibility_reasons) | {reason})
                ),
            )
            candidates[index] = replace(
                row,
                candidate=candidate,
                novelty_confidence="reduced",
                novelty_reasons=tuple(sorted(set(row.novelty_reasons) | {reason})),
            )
            updated_candidates[candidate_identity(candidate)] = candidate
    if updated_candidates:
        desired = {
            root: EvidenceBundle(
                tuple(
                    updated_candidates.get(candidate_identity(row), row)
                    for row in bundle.candidates
                ),
                bundle.debits,
            )
            for root, bundle in desired.items()
        }
    # Ledger accepted stamps are bookkeeping, not frozen reducer input content.
    pure = {
        root: EvidenceBundle(
            tuple(replace(row, accepted_day="") for row in bundle.candidates),
            tuple(replace(row, accepted_day="") for row in bundle.debits),
        )
        for root, bundle in desired.items()
    }
    coverage = dict(reduction.coverage)
    for name in (
        "sessions",
        "classified_tool_calls",
        "inv_mention",
        "inv_unclassified",
        "evidence_duplicates",
        "evidence_conflicts",
    ):
        coverage[name] = max(
            coverage.get(name, 0), previous.coverage.get(name, 0) if previous else 0
        )
    coverage["recorded_executions"] = len(reduction.executions)
    coverage["recorded_lookups"] = sum(
        row.operation in {"search", "show"} for row in reduction.executions
    )
    for attribution in Attribution:
        coverage[attribution.value] = sum(
            row.attribution is attribution for row in reduction.executions
        )
    coverage["attributed_lookups"] = sum(
        row.operation in {"search", "show"}
        and row.attribution in {Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE}
        for row in reduction.executions
    )
    coverage["ambiguous_joins"] = coverage[Attribution.AMBIGUOUS.value]
    coverage["recognized_invocations"] = len(reduction.invocations)
    for cls in InvocationClass:
        coverage["inv_" + cls.value] = sum(row.cls is cls for row in reduction.invocations)
    coverage["tasks_resolved"] = sum(
        row.task is not None and row.task.resolved for row in reduction.invocations
    )
    coverage["tasks_unknown"] = len(reduction.invocations) - coverage["tasks_resolved"]
    all_candidates = [row for bundle in pure.values() for row in bundle.candidates]
    coverage["candidate_receipts"] = len(all_candidates)
    available = 0
    for row in all_candidates:
        atoms = (*row.immediate, *row.discovery, *row.episode)
        available += bool(atoms) and all(atom.status is AtomStatus.OK for atom in atoms)
    coverage["source_costs_available"] = available
    coverage["unavailable_sources"] = len(all_candidates) - available
    return replace(
        reduction,
        candidates=tuple(
            sorted(
                candidates,
                key=lambda row: (
                    row.candidate.task.root,
                    row.candidate.task.turn or "",
                    row.candidate.candidate_id,
                    row.candidate.day,
                    row.invocation_id,
                ),
            )
        ),
        bundles=tuple(pure[root] for root in sorted(pure)),
        bundle_roots=tuple(sorted(pure)),
        coverage=dict(sorted(coverage.items())),
        observations=summarize(reduction.executions),
    ), desired


REDUCTION_FORMAT = "root-reduction-v1"
ROWS = (
    "executions",
    "invocations",
    "session_roots",
    "session_parents",
    "session_cwd",
    "invocation_sessions",
)


def _groups(document):
    sessions = {(row[0], row[1]): row[2] for row in document.get("session_roots", ())}
    grouped = defaultdict(lambda: {field_name: [] for field_name in ROWS})
    for field_name in ROWS:
        for row in document.get(field_name, ()):
            if field_name in {"executions", "invocations"}:
                runtime, session = row.get("runtime"), row.get("session_id")
                root = sessions.get((runtime, session))
                if root is None and runtime and session:
                    root = runtime + ":" + session
                if root is None:
                    root = (
                        "@journal:" + hashlib.sha256(row["execution_id"].encode()).hexdigest()[:2]
                    )
            elif field_name == "invocation_sessions":
                root = sessions.get((row[1], row[2]), row[1] + ":" + row[2])
            else:
                root = sessions.get((row[0], row[1]), row[0] + ":" + row[1])
            grouped[root][field_name].append(row)
    return dict(grouped)


def expand_reduction_document(profile, raw):
    if raw.get("format") is None:
        return raw
    if raw.get("format") != REDUCTION_FORMAT:
        raise CollectError("unsupported_schema", "Collector native-row format is unsupported.")
    result = dict(raw["summary"])
    result.update((field_name, []) for field_name in ROWS)
    for filename in raw["roots"].values():
        child = store.read_blob(profile, filename)
        if not isinstance(child, dict):
            raise CollectError("state_corrupt", "Collector native-row document is invalid.")
        for field_name in ROWS:
            result[field_name].extend(child[field_name])
    result["executions"].sort(key=lambda row: (row["ts_utc"], row["execution_id"]))
    result["invocations"].sort(
        key=lambda row: (row["runtime"], row["session_id"], row["invocation_id"])
    )
    for field_name in ROWS[2:]:
        result[field_name].sort()
    return result


def write_reduction_document(profile, name, document, *, previous=None, prior_manifest=None):
    """Write a complete logical document; optional prior typed rows reuse unchanged roots."""
    groups = _groups(document)
    old = _groups(previous) if previous is not None else {}
    roots = {}
    files = (
        prior_manifest.get("roots", {})
        if prior_manifest and prior_manifest.get("format") == REDUCTION_FORMAT
        else {}
    )
    for root, rows in sorted(groups.items()):
        if root in files and rows == old.get(root):
            roots[root] = files[root]
        else:
            roots[root] = store.write_blob(profile, "reduction-root", rows)
    summary = {field: value for field, value in document.items() if field not in ROWS}
    return store.write_blob(
        profile,
        name,
        {
            "format": REDUCTION_FORMAT,
            "summary": summary,
            "roots": roots,
            "frequency_refs": sorted(
                {
                    row["frequency_view_id"]
                    for row in document.get("invocations", ())
                    if row.get("frequency_view_id")
                }
            ),
        },
    )


def save_reduction_incremental(profile, name, document, blobs, *, previous=None):
    """Reuse dataclass identity before serialization, not after serializing every root."""

    def groups(doc):
        found = defaultdict(lambda: {field_name: [] for field_name in ROWS})
        mapping = {(runtime, sid): root for runtime, sid, root in doc.session_roots}
        for field_name in ROWS:
            for row in getattr(doc, field_name):
                if field_name in {"executions", "invocations"}:
                    root = mapping.get((row.runtime, row.session_id))
                    if root is None and row.runtime and row.session_id:
                        root = row.runtime + ":" + row.session_id
                    if root is None:
                        root = (
                            "@journal:" + hashlib.sha256(row.execution_id.encode()).hexdigest()[:2]
                        )
                elif field_name == "invocation_sessions":
                    root = mapping.get((row[1], row[2]), row[1] + ":" + row[2])
                else:
                    root = mapping.get((row[0], row[1]), row[0] + ":" + row[1])
                found[root][field_name].append(row)
        return found

    current = groups(document)
    old = groups(previous) if previous is not None else {}
    manifest = store.read_blob(profile, blobs.get(name))
    files = (
        manifest.get("roots", {}) if manifest and manifest.get("format") == REDUCTION_FORMAT else {}
    )
    roots = {}
    for root, rows in sorted(current.items()):
        if root in files and rows == old.get(root):
            roots[root] = files[root]
        else:
            roots[root] = store.write_blob(profile, "reduction-root", to_plain(rows))
    summary = {
        field: getattr(document, field)
        for field in ("coverage", "warnings", "digest", "session_last", "lookup_schema")
    }
    blobs[name] = store.write_blob(
        profile,
        name,
        {
            "format": REDUCTION_FORMAT,
            "summary": to_plain(summary),
            "roots": roots,
            "frequency_refs": sorted(
                {row.frequency_view_id for row in document.invocations if row.frequency_view_id}
            ),
        },
    )


def reduction_references(raw):
    return set(raw.get("roots", {}).values()) if raw.get("format") == REDUCTION_FORMAT else set()
