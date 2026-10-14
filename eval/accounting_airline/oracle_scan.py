"""Evidence scan for the lab accounting oracle: invocations, exposures, novelty and reuse.

Reads only the frozen observer evidence (observed dispatches + native session records). No
``ssgrep`` accounting code is imported; every rule is spelled out here and in ``oracle_rules``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from . import oracle_rules as rules
from .observer import Frozen, NativeRecord, NativeSession, ObservedCall
from .oracle_rules import Parsed

OPS = ("search", "show", "note")
LOOKUPS = ("search", "show")
CONSUMERS = ("solver-a", "solver-b")
ACTORS = ("solver-a", "solver-b", "verifier", "control")
STATES = ("verified", "attempted", "interrupted", "unknown")
UNOBSERVED = "downstream_exposure_unobserved"
UNATTRIBUTABLE = "mixed_shell_output_unattributable"


def _pre_api_usage(call: ObservedCall) -> bool:
    """A verified CLI parameter error rejected before the shared service API."""
    if call.state != "verified":
        return False
    for line in call.stdout.strip().splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            document = json.loads(line)
        except ValueError:
            return False
        error = document.get("error") if isinstance(document, dict) else None
        return isinstance(error, dict) and error.get("type") in ("BadParameter", "UsageError")
    return False


def operation(call: ObservedCall) -> str | None:
    """``search``/``show``/``note`` for an observed ssgrep dispatch, else None."""
    if call.tool == "ssgrep" and len(call.argv) > 1 and call.argv[1] in OPS:
        return call.argv[1]
    return None


@dataclass
class Invocation:
    """One native tool call hosting >=1 observed solver ssgrep executions."""

    session: str
    actor: str
    index: int  # tool_call record index in the session
    call_id: str
    task: str
    dispatches: list[ObservedCall] = field(default_factory=list)
    result: int | None = None  # tool_result record index
    visible: str = ""
    parsed: Parsed | None = None
    outcome: str = "ok"  # ok | empty | error
    chars: int | None = None
    reason: str | None = None  # why the debit is unknown
    partial: bool = False
    truncated: bool = False

    @property
    def ops(self) -> tuple[str, ...]:
        return tuple(op for op in (operation(d) for d in self.dispatches) if op)

    @property
    def is_lookup(self) -> bool:
        return any(op in LOOKUPS for op in self.ops)

    @property
    def is_execution(self) -> bool:
        """Whether any hosted dispatch entered the shared API."""
        return any(not _pre_api_usage(dispatch) for dispatch in self.dispatches)

    @property
    def query(self) -> str:
        return " ".join(arg for d in self.dispatches for arg in d.argv[2:])

    @property
    def has_cards(self) -> bool:
        """A visible, successful, parsable lookup result that may expose candidates."""
        return (
            self.is_lookup
            and self.chars is not None
            and self.outcome == "ok"
            and self.parsed is not None
            and self.parsed.status != "none"
        )


@dataclass
class Scan:
    sessions: Mapping[str, tuple[str, str | None, NativeSession]]
    records: dict[str, tuple[NativeRecord, ...]]
    alias: dict[str, str]  # native session_id -> frozen key
    ordinals: dict[str, list[int]]  # per session: record index -> user_turn ordinal
    invocations: list[Invocation]
    unlinked: list[ObservedCall]
    observed: dict[tuple[str, str], list[ObservedCall]]
    hosts: set[tuple[str, int]]
    counts: dict[str, int]
    atom_chars: dict[str, int]

    def task_id(self, session: str, index: int) -> str:
        ordinal = self.ordinals[session][index]
        return f"{session}#{ordinal}" if ordinal >= 0 else f"{session}#before_first_turn"

    def tasks(self) -> list[str]:
        out = []
        for key, (actor, _label, _native) in self.sessions.items():
            if actor in CONSUMERS:
                kinds = [r.kind for r in self.records[key]]
                out.extend(f"{key}#{n}" for n in range(kinds.count("user_turn")))
        return out

    # ---- novelty / reuse ------------------------------------------------------------
    def novelty(self, inv: Invocation, token: str) -> str | None:
        if rules.has_token(inv.query, token):
            return "query_mention"
        for record in self.records[inv.session][: inv.result]:
            if record.kind in rules.CONTENT_KINDS and rules.has_token(record.text, token):
                return "prior_known"
        return None

    def action_failed(self, session: str, actor: str, index: int) -> bool:
        records = self.records[session]
        call = records[index]
        for record in records[index + 1 :]:
            if record.kind == "tool_result" and record.call_id == call.call_id:
                if record.is_error or rules.is_error_text(record.text):
                    return True
                break
        seen = self.observed.get((actor, call.call_id or ""), ())
        return any(c.exit_code not in (0, None) for c in seen)

    def reuse(self, inv: Invocation, token: str, refs: list[str | None]) -> tuple[str, str | None]:
        """(reuse state, native call id of the deciding later call)."""
        records = self.records[inv.session]
        assert inv.result is not None
        for j in range(inv.result + 1, len(records)):
            record = records[j]
            if record.kind != "tool_call" or not rules.has_token(record.text, token):
                continue
            if (inv.session, j) in self.hosts:  # a lookup/note quoting the token is no action
                continue
            failed = self.action_failed(inv.session, inv.actor, j)
            return ("contradicted" if failed else "action_reference"), record.call_id
        wanted = {ref for ref in refs if ref}
        for other in self.invocations:
            if other.session != inv.session or other.index <= inv.result:
                continue
            for dispatch in other.dispatches:
                if operation(dispatch) == "show" and wanted & set(dispatch.argv[2:]):
                    return "inspected", other.call_id
        child = self.sessions[inv.session][2]
        for parent_key, (parent_actor, _label, parent) in self.sessions.items():
            if (
                parent_key == inv.session
                or child.parent_session is None
                or os.path.abspath(child.parent_session)
                not in {os.path.abspath(path) for path in (parent.path, *parent.source_aliases)}
            ):
                continue
            parent_records = self.records[parent_key]
            for j, record in enumerate(parent_records):
                if (
                    record.kind != "tool_result"
                    or child.session_id not in record.text
                    or not rules.has_token(record.text, token)
                ):
                    continue
                for action_index in range(j + 1, len(parent_records)):
                    action = parent_records[action_index]
                    if action.kind != "tool_call" or not rules.has_token(action.text, token):
                        continue
                    failed = self.action_failed(parent_key, parent_actor, action_index)
                    return ("contradicted" if failed else "action_reference", action.call_id)
        return "none_observed", None


def _find_native(
    sessions: Mapping[str, tuple[str, str | None, NativeSession]],
    records: dict[str, tuple[NativeRecord, ...]],
    call: ObservedCall,
) -> tuple[str, int] | None:
    """The unique native tool_call of the dispatching actor that ``call`` is linked to."""
    nid = getattr(call, "native_call_id", None)
    if not nid:
        return None
    hits = [
        (key, i)
        for key, (actor, _label, _native) in sessions.items()
        if actor == call.actor
        for i, r in enumerate(records[key])
        if r.kind == "tool_call" and r.call_id == nid
    ]
    if len(hits) > 1 and call.task is not None:
        hits = [h for h in hits if sessions[h[0]][1] == call.task] or hits
    return hits[0] if len(hits) == 1 else None


def _derive(inv: Invocation, scan: Scan) -> None:
    """Visible text, outcome and debit of one lookup invocation (what the model saw only)."""
    records = scan.records[inv.session]
    inv.result = next(
        (
            j
            for j in range(inv.index + 1, len(records))
            if records[j].kind == "tool_result" and records[j].call_id == inv.call_id
        ),
        None,
    )
    if not inv.is_lookup:
        return
    if inv.result is None:
        inv.reason = UNOBSERVED
        return
    result = records[inv.result]
    inv.visible = result.text
    mixed = rules.is_mixed(records[inv.index].text, len(inv.dispatches))
    hint = next((a for d in inv.dispatches for a in d.argv[2:] if rules.REF_RE.fullmatch(a)), None)
    parsed = rules.parse_visible(inv.visible, strict=mixed, hint_ref=hint)
    if not mixed:
        try:
            json.loads(inv.visible.strip())
        except ValueError:
            if parsed.status == "json" and inv.visible.count('{"ok"') > 1:
                refs = tuple(
                    dict.fromkeys(found.group(0) for found in rules.REF_RE.finditer(inv.visible))
                )
                parsed = Parsed(
                    "fallback", (rules.Card(None, inv.visible),), None, False, refs, inv.visible
                )
    inv.parsed = parsed
    codes = [d.exit_code for d in inv.dispatches if d.exit_code is not None]
    if any(c not in (0, 3) for c in codes) or result.is_error or parsed.ok is False:
        inv.outcome = "error"
    elif parsed.empty or (codes and all(c == 3 for c in codes)):
        inv.outcome = "empty"
    if mixed and parsed.status == "none":
        inv.reason, inv.partial = UNATTRIBUTABLE, True
        return
    inv.chars = len(inv.visible)
    inv.truncated = sum(d.stdout_chars for d in inv.dispatches) > inv.chars


def scan_frozen(frozen: Frozen) -> Scan:
    sessions = dict(frozen.sessions)
    records = {key: tuple(entry[2].records) for key, entry in sessions.items()}
    alias = {entry[2].session_id: key for key, entry in sessions.items()}
    ordinals: dict[str, list[int]] = {}
    for key, recs in records.items():
        seen, row = -1, []
        for record in recs:
            seen += record.kind == "user_turn"
            row.append(seen)
        ordinals[key] = row
    counts = {f"{a}.{op}": 0 for a in ACTORS for op in OPS}
    counts.update({f"{a}_{op}": 0 for a in ("verifier", "control") for op in OPS})
    counts.update({"executions": 0, **{f"executions_{s}": 0 for s in STATES}})
    counts.update({"journaled_search": 0, "journaled_show": 0, "journaled_note": 0})
    atom_chars = {f"{k}:{r.record_id}": len(r.text) for k, recs in records.items() for r in recs}
    scan = Scan(sessions, records, alias, ordinals, [], [], {}, set(), counts, atom_chars)
    hosts: dict[tuple[str, int], Invocation] = {}
    for call in sorted(frozen.calls, key=lambda c: c.dispatch_seq):
        nid = getattr(call, "native_call_id", None)
        if nid:
            scan.observed.setdefault((call.actor, nid), []).append(call)
        op = operation(call)
        if op is None:
            continue
        counts[f"{call.actor}.{op}"] += 1
        if call.actor in ("verifier", "control"):
            counts[f"{call.actor}_{op}"] += 1
        if call.state == "verified" and not _pre_api_usage(call):
            counts[f"journaled_{op}"] += 1
        if call.actor not in CONSUMERS:
            continue
        is_usage = _pre_api_usage(call)
        if not is_usage:
            counts["executions"] += 1
            counts[f"executions_{call.state if call.state in STATES else 'unknown'}"] += 1
        found = _find_native(sessions, records, call)
        if found is None:
            if not is_usage:
                scan.unlinked.append(call)
            continue
        if found not in hosts:
            key, index = found
            call_id = records[key][index].call_id or ""
            hosts[found] = Invocation(key, call.actor, index, call_id, scan.task_id(key, index))
        hosts[found].dispatches.append(call)
    scan.hosts = set(hosts)
    scan.invocations = [hosts[k] for k in sorted(hosts)]
    for inv in scan.invocations:
        _derive(inv, scan)
    return scan
