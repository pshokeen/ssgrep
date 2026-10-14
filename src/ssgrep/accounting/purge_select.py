"""What a purge selects: the validated cutoff/workspace and the conversations inside it.

``--before`` is an RFC3339 instant or a day. Ledgers, executions and evidence are dated by UTC
day, so the cutoff is the UTC DAY of that instant: data dated strictly before ``cutoff_day`` is
in scope (``--before 2026-10-10T15:00:00Z`` and ``--before 2026-10-10`` both keep 2026-10-10
itself). Journal segments use the same boundary: only a segment whose every event is dated
before ``cutoff_day`` AND whose file was last modified before that day's midnight UTC is quiet
enough to delete.

``--workspace`` is an approved workspace ROOT PATH, canonicalized with symlinks resolved. A
conversation (root) is selected when it has retained evidence with a known working directory
and EVERY known directory lies inside the workspace (component-wise, never by string prefix).
Conversations with no known directory are ``unknown`` scope and conversations that also worked
outside the workspace are ``mixed``; both are retained and counted, never guessed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_state import ReductionDoc
from ssgrep.accounting.collector_store import Checkpoint
from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.policy_rules import PolicyError, canonical_root, contains
from ssgrep.accounting.purge_model import PurgeError
from ssgrep.accounting.replay import LedgerState
from ssgrep.accounting.scope import check_profile, parse_instant

USAGE = "ssgrep metrics purge --before <date> [--workspace <path>] [--dry-run]"


@dataclass(frozen=True)
class Selection:
    profile: str
    before: str  # normalized UTC instant
    workspace: str | None  # canonical root path, None = the whole profile

    @property
    def cutoff_day(self) -> str:
        return self.before[:10]

    @property
    def cutoff_midnight(self) -> str:
        return f"{self.cutoff_day}T00:00:00Z"


def _usage(error: PolicyError, condition: str | None = None) -> PurgeError:
    return PurgeError(condition or error.condition, str(error), exit_code=2, command=USAGE)


def parse_selection(
    *, profile: str, before: str | None, workspace: str | None, policy: ProfilePolicy
) -> Selection:
    """Validate every argument; nothing has been read or written when this raises."""
    try:
        check_profile(profile)
    except PolicyError as error:
        raise _usage(error) from None
    if before is None or not before.strip():
        raise PurgeError(
            "invalid_interval",
            "--before is required: say which data to delete (a day or an RFC3339 instant). "
            "Nothing was changed.",
            exit_code=2,
            command=USAGE,
        )
    try:
        instant = parse_instant(before.strip(), "--before")
    except PolicyError as error:
        raise _usage(error) from None
    return Selection(profile, instant, _workspace(profile, workspace, policy))


def _workspace(profile: str, raw: str | None, policy: ProfilePolicy) -> str | None:
    if raw is None:
        return None
    try:
        chosen = canonical_root(raw)
    except PolicyError:
        raise PurgeError(
            "invalid_workspace",
            "--workspace must be an absolute path (or ~/...) other than '/'. Nothing was changed.",
            exit_code=2,
            command=USAGE,
        ) from None
    if not policy.roots:
        if profile == PERSONAL_PROFILE:
            return chosen
    else:
        for root in policy.roots:
            try:
                if contains(canonical_root(root), chosen):
                    return chosen
            except PolicyError:
                continue
    raise PurgeError(
        "workspace_not_in_scope",
        f"That workspace is not inside an approved root of profile {profile}. Nothing was changed.",
        exit_code=2,
        command="ssgrep metrics status",
    )


class Scoper:
    """Whether a working directory lies inside the selected workspace (realpath cached)."""

    def __init__(self, workspace: str | None) -> None:
        self.workspace = workspace
        self._cache: dict[str, bool] = {}

    def inside(self, cwd: str | None) -> bool:
        if self.workspace is None:
            return True
        if not isinstance(cwd, str) or not cwd or "\0" in cwd or not os.path.isabs(cwd):
            return False
        known = self._cache.get(cwd)
        if known is None:
            known = self._cache[cwd] = contains(self.workspace, os.path.realpath(cwd))
        return known


def evidence_cwd(payload: dict) -> str | None:
    for name in ("cwd", "directory"):
        value = payload.get(name)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass(frozen=True)
class Scopes:
    """Which conversation roots are selected; ``selected`` None means every root."""

    selected: frozenset[str] | None
    root_of: dict[tuple[str, str], str] = field(default_factory=dict)
    unknown: int = 0
    mixed: int = 0

    def root(self, runtime: str | None, session: str | None) -> str | None:
        if runtime is None or session is None:
            return None
        return self.root_of.get((runtime, session), f"{runtime}:{session}")

    def has_root(self, root: str | None) -> bool:
        return self.selected is None or (root is not None and root in self.selected)

    def has_session(self, runtime: str | None, session: str | None) -> bool:
        return self.has_root(self.root(runtime, session))


def ledger_roots(states: Iterable[LedgerState]) -> set[str]:
    """Every conversation root the ledgers know (live evidence and frozen)."""
    roots: set[str] = set()
    for state in states:
        roots.update(c.task.root for c in state.candidates)
        roots.update(d.task.root for d in state.debits)
        roots.update(f.root for f in state.frozen)
    return roots


def classify(
    profile: str,
    checkpoint: Checkpoint | None,
    doc: ReductionDoc | None,
    known_roots: Iterable[str],
    scoper: Scoper,
) -> Scopes:
    """Select conversation roots by the working directories of their retained evidence."""
    root_of = {(rt, sid): root for rt, sid, root in (doc.session_roots if doc else ())}
    if scoper.workspace is None:
        return Scopes(None, root_of)
    cwds: dict[str, set[str | None]] = {}
    if checkpoint is not None:
        for item in store.iter_evidence(profile, checkpoint.segments):
            root = root_of.get((item.runtime, item.session_id), f"{item.runtime}:{item.session_id}")
            cwds.setdefault(root, set()).add(evidence_cwd(item.payload))
    universe = set(cwds) | set(known_roots) | set(root_of.values())
    selected: set[str] = set()
    unknown = mixed = 0
    for root in sorted(universe):
        known = {c for c in cwds.get(root, ()) if c is not None}
        if not known:
            unknown += 1
        elif all(scoper.inside(c) for c in known):
            selected.add(root)
        elif any(scoper.inside(c) for c in known):
            mixed += 1
    return Scopes(frozenset(selected), root_of, unknown, mixed)
