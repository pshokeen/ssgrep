"""Compact session metadata and conservative reduction dependency components.

Edges express actual parent/context, source witnesses, native ownership and journal
competition. A source dependency is bidirectional for *processing*, not accounting:
a changed source must invalidate its consumers and a consumer needs the source context.
Missing parents/refs stay as graph nodes, so late arrival invalidates earlier results.
"""

from __future__ import annotations

import os
from collections import defaultdict

from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.exposure import interpret
from ssgrep.accounting.identity_claims import target_key
from ssgrep.accounting.plain_codec import from_plain, to_plain
from ssgrep.accounting.reconcile import Request, fold_events, reconcile
from ssgrep.accounting.sources import REF_RE, resolve_ref


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
    from ssgrep.accounting.lineage import build_lineage
    from ssgrep.accounting.reduce_calls import collect_calls
    from ssgrep.accounting.scope_events import Scoper
    from ssgrep.accounting.timeline import build_timelines

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
            for field in ("parent_session", "parent_session_id", "parent_id"):
                value = payload.get(field)
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
