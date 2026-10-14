"""Root conversation / originating user turn / branch task identity (task 5.3).

A task is ``(root conversation, originating user-turn, branch)`` -- never an
inactivity bucket or a session id. Parents are resolved from raw evidence only
(omp/pi ``parentSession`` paths matched to retained sources, prime lineage rows,
claude child paths, codex thread spawns, OpenCode ``parent_id``). A child inherits
the task of the parent turn that LAUNCHED it, even when the parent has since
received a newer prompt. Launch linkage is explicit: the earliest parent tool call
or result that names the child's own identity. Without it the child keeps the
supported root with ``task_unknown`` (turn ``None``); a missing or cyclic parent is
surfaced and never fabricates a task or a second credit.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.timeline import Record, SessionKey, Timeline, Timelines, key_str, record_text

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{3,}")
GENERIC_NAMES = frozenset(
    {"main", "agent", "task", "child", "subagent", "explore", "general", "plan", "root", "none"}
)
LAUNCH_SKEW_MS = 2000


@dataclass(frozen=True)
class Launch:
    """The parent position that launched a child (``call_seq`` in ``parent``)."""

    parent: SessionKey
    call_seq: int
    via: str  # the child identity that the launching record names


@dataclass(frozen=True)
class Issue:
    kind: str  # missing_parent | cyclic_parent | unlinked_launch | unresolved_lineage_row
    session: str


@dataclass
class Lineage:
    parent_of: dict[SessionKey, SessionKey] = field(default_factory=dict)
    declared: dict[SessionKey, str] = field(default_factory=dict)  # declared parent ref
    root_of: dict[SessionKey, SessionKey] = field(default_factory=dict)
    launch: dict[SessionKey, Launch] = field(default_factory=dict)
    broken: set[SessionKey] = field(default_factory=set)  # missing or cyclic ancestry
    names: dict[SessionKey, set[str]] = field(default_factory=dict)  # extra launch identities
    issues: list[Issue] = field(default_factory=list)
    timelines: Timelines | None = None
    _tasks: dict[SessionKey, TaskId] = field(default_factory=dict)

    def root(self, key: SessionKey) -> str:
        return key_str(self.root_of[key])

    def identities(self, key: SessionKey) -> list[str]:
        """Names under which a parent's records refer to child ``key``."""
        assert self.timelines is not None
        return _identity_words(key, self.timelines.sessions[key], self.names.get(key, set()))

    def is_child(self, key: SessionKey) -> bool:
        return key in self.declared

    def task_of(self, key: SessionKey, seq: int) -> TaskId:
        """The task of the record at ``seq`` of ``key`` (see module docstring)."""
        assert self.timelines is not None
        root = self.root(key)
        if key in self.declared:
            cached = self._tasks.get(key)
            if cached is None:
                link = self.launch.get(key)
                if key in self.broken or link is None:
                    cached = TaskId(root, None, "main")
                else:
                    cached = self.task_of(link.parent, link.call_seq)
                self._tasks[key] = cached
            return cached
        line = self.timelines.sessions[key]
        turn: Record | None = None
        for candidate in line.turns:
            if candidate.seq < seq:
                turn = candidate
            else:
                break
        return TaskId(root, turn.ev.record_id if turn else None, "main")

    def context(self, key: SessionKey, seq: int) -> tuple[list[tuple[SessionKey, int]], list[str]]:
        """Earlier available context: ``[(session, upto_seq), ...]`` plus limitation reasons."""
        chain: list[tuple[SessionKey, int]] = []
        reasons: list[str] = []
        seen: set[SessionKey] = set()
        current: SessionKey | None = key
        upto = seq
        while current is not None and current not in seen:
            seen.add(current)
            chain.append((current, upto))
            if current not in self.declared:
                break
            link = self.launch.get(current)
            if current in self.broken:
                reasons.append("missing_or_cyclic_ancestor_context")
                break
            if link is None:
                reasons.append("launch_context_unknown")
                break
            current, upto = link.parent, link.call_seq + 1  # the launching call text counts
        return chain, reasons


def _norm(path: str) -> str:
    return os.path.normpath(path)


def _declared_parent(line: Timeline) -> tuple[str | None, str | None, str | None]:
    """``(session_id, path, opaque)`` of the declared parent from one session's evidence."""
    for record in line.records:
        payload = record.ev.payload
        if record.kind == "session_header":
            pid = payload.get("parent_session_id") or payload.get("parent_id")
            path = payload.get("parent_session_path")
            if pid or path:
                return pid, path, None
        parent = payload.get("parent_session")
        if isinstance(parent, str) and parent:
            return parent, None, None
    return None, None, None


def _source_index(timelines: Timelines) -> dict[tuple[str, str], list[SessionKey]]:
    found: dict[tuple[str, str], list[SessionKey]] = {}
    for key, line in timelines.sessions.items():
        for source in line.sources:
            found.setdefault((key[0], _norm(source.split(":", 1)[1])), []).append(key)
    return found


def _resolve_parents(timelines: Timelines, out: Lineage) -> None:
    by_source = _source_index(timelines)
    for key, line in timelines.sessions.items():
        pid, path, _ = _declared_parent(line)
        if pid is None and path is None:
            continue
        out.declared[key] = path or pid or ""
        parent: SessionKey | None = None
        if path:
            hits = by_source.get((key[0], _norm(path)), [])
            parent = hits[0] if len(hits) == 1 else None
        elif pid:
            candidate = (key[0], pid)
            parent = candidate if candidate in timelines.sessions else None
        if parent is not None and parent != key:
            out.parent_of[key] = parent
    for row in timelines.lineage:  # prime ``rlm_subagent`` rows: child file -> parent id
        payload = row.payload
        child_file, parent_id = payload.get("session_file"), payload.get("parent_session_id")
        if not (isinstance(child_file, str) and isinstance(parent_id, str)):
            continue
        hits = by_source.get((row.runtime, _norm(child_file)), [])
        parent_key = (row.runtime, parent_id)
        if len(hits) == 1 and parent_key in timelines.sessions and hits[0] != parent_key:
            child = hits[0]
            out.declared.setdefault(child, parent_id)
            out.parent_of.setdefault(child, parent_key)
            child_id = payload.get("child_id")
            if isinstance(child_id, str):
                out.names.setdefault(child, set()).add(child_id)
        else:
            out.issues.append(Issue("unresolved_lineage_row", f"{row.runtime}:{parent_id}"))


def _roots(timelines: Timelines, out: Lineage) -> None:
    for key in sorted(timelines.sessions):
        trail: list[SessionKey] = []
        position: dict[SessionKey, int] = {}
        current = key
        broken = False
        while True:
            if current in out.root_of:
                root = out.root_of[current]
                broken = current in out.broken
                break
            if current in position:  # cycle: all members share the smallest member as root
                members = trail[position[current] :]
                root = min(members)
                for member in members:
                    out.issues.append(Issue("cyclic_parent", key_str(member)))
                    out.broken.add(member)
                broken = True
                break
            position[current] = len(trail)
            trail.append(current)
            parent = out.parent_of.get(current)
            if parent is None:
                if current in out.declared:
                    out.issues.append(Issue("missing_parent", key_str(current)))
                    out.broken.add(current)
                    broken = True
                root = current
                break
            current = parent
        for member in trail:
            out.root_of.setdefault(member, root)
            if broken and member in out.declared:
                out.broken.add(member)
    out.issues = sorted(set(out.issues), key=lambda i: (i.kind, i.session))


def _identity_words(key: SessionKey, line: Timeline, extra: set[str]) -> list[str]:
    names: set[str] = set(extra)
    session = key[1]
    names.add(session.rsplit("/", 1)[-1])
    names.add(session)
    for source in line.sources:
        stem = os.path.basename(source.split(":", 1)[1])
        stem = stem[: -len(".jsonl")] if stem.endswith(".jsonl") else stem
        names.add(stem[len("agent-") :] if stem.startswith("agent-") else stem)
        if "_" in stem:
            names.add(stem.rsplit("_", 1)[-1])
    return sorted(n for n in names if len(n) >= 4 and n.lower() not in GENERIC_NAMES)


def _first_words(line: Timeline) -> dict[str, int]:
    """Earliest call/result sequence at which each word appears (one pass per parent)."""
    first: dict[str, int] = {}
    for record in line.records:
        if record.kind not in ("tool_call", "tool_result"):
            continue
        for word in {w.rstrip(".-") for w in WORD.findall(record_text(record))}:
            first.setdefault(word, record.seq)
    return first


def _link_launches(timelines: Timelines, out: Lineage) -> None:
    index: dict[SessionKey, dict[str, int]] = {}
    for child, parent in sorted(out.parent_of.items()):
        if child in out.broken:
            continue
        line, pline = timelines.sessions[child], timelines.sessions[parent]
        words = index.get(parent)
        if words is None:
            words = index[parent] = _first_words(pline)
        best: tuple[int, str] | None = None
        for name in _identity_words(child, line, out.names.get(child, set())):
            seq = words.get(name)
            if seq is not None and (best is None or seq < best[0]):
                best = (seq, name)
        if best is None:
            out.issues.append(Issue("unlinked_launch", key_str(child)))
            continue
        record = pline.records[best[0]]
        if record.kind == "tool_result":  # the launching call is the one this result answers
            call = pline.by_id.get("call:" + record.ev.record_id.split(":", 1)[1])
            if call is not None:
                record = call
        first_ts = next((r.ts for r in line.records if r.ts is not None), None)
        if record.ts is not None and first_ts is not None and first_ts + LAUNCH_SKEW_MS < record.ts:
            out.issues.append(Issue("unlinked_launch", key_str(child)))  # child predates launch
            continue
        out.launch[child] = Launch(parent, record.seq, best[1])


def build_lineage(timelines: Timelines) -> Lineage:
    """Resolve roots, parents, and launch links; surfaces every defect in ``issues``."""
    out = Lineage(timelines=timelines)
    _resolve_parents(timelines, out)
    _roots(timelines, out)
    _link_launches(timelines, out)
    return out
