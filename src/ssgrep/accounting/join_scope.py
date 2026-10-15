"""Project trusted native joins into local scope without changing raw journals.

Workspace and actor fields are local evidence, not remote identity attestation.
Unknown journals never acquire organization scope from a selector or server cwd.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.contracts import Attribution, ExecutionEvent

if TYPE_CHECKING:
    from ssgrep.accounting.reconcile import ExecFact, Link
    from ssgrep.accounting.reduce_calls import Call
    from ssgrep.accounting.reduce_types import ExecutionRecord
    from ssgrep.accounting.scope_events import Scoper

TRUSTED_BASES = frozenset({"process_cwd", "trusted_invocation", "native_call_metadata"})
JOINED = frozenset({Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE})


def execution_scope(
    fact: ExecFact, link: Link, call: Call | None, scoper: Scoper
) -> tuple[str | None, str, bool]:
    """One scope decision shared by execution rows and derived counter evidence."""
    if fact.cwd and (
        fact.scope_basis in TRUSTED_BASES
        or fact.scope_basis == "legacy"
        and fact.transport == "cli"
    ):
        basis = "process_cwd" if fact.scope_basis == "legacy" else fact.scope_basis
        return fact.cwd, basis, scoper.in_scope(fact.cwd)
    if (
        link.attribution in JOINED
        and len(link.requests) == 1
        and call is not None
        and call.in_scope
    ):
        request = next((item for item in call.requests if item.key == link.requests[0]), None)
        if (
            request is not None
            and request.cwd
            and request.cwd_trusted
            and scoper.in_scope(request.cwd)
        ):
            cwd = os.path.realpath(request.cwd)
            return cwd, "native_call_metadata", scoper.in_scope(cwd)
    return None, "unknown_scope", scoper.in_scope(None)


def invocation_provenance(
    call: Call, owned: list[str], ambiguous: list[str], links: Mapping[str, Link]
) -> dict[str, Any]:
    reasons = {
        reason for execution_id in owned + ambiguous for reason in links[execution_id].reasons
    }
    trusted = (
        bool(call.cwd) and bool(call.requests) and all(item.cwd_trusted for item in call.requests)
    )
    if not trusted:
        reasons.add("unknown_scope" if not call.cwd else "untrusted_cwd")
    key_ids = {item.fingerprint_key_id for item in call.requests}
    return {
        "scope_basis": "native_call_metadata" if trusted else "unknown_scope",
        "join_rejection_reasons": tuple(sorted(reasons)),
        "actor_basis": "configured_verifier" if call.actor == "verifier" else "native_lineage",
        "fingerprint_key_id": next(iter(key_ids)) if len(key_ids) == 1 else None,
    }


def effective_scoped_journal(
    raw: Iterable[ExecutionEvent], executions: Iterable[ExecutionRecord], scoper: Scoper
) -> list[ExecutionEvent]:
    """Counter inputs authorized for this profile, including safe native corrections.

    Derived copies preserve execution/phase identity; nothing writes them back to
    the journal. A trusted unique native consumer may resolve missing capture scope,
    but policy ownership must still agree. Unjoined unknown personal observations
    remain local with cwd=None; organization counters never include those events.
    """
    rows = {row.execution_id: row for row in executions}
    result = []
    for event in raw:
        if event.profile != scoper.policy.profile_id:
            continue
        known = event.scope_basis in TRUSTED_BASES or (
            event.scope_basis == "legacy" and str(event.transport) == "cli"
        )
        if known and event.cwd:
            if scoper.in_scope(event.cwd):
                result.append(event)
            continue
        row = rows.get(event.execution_id)
        if (
            row is not None
            and row.in_scope
            and row.scope_basis == "native_call_metadata"
            and row.attribution in JOINED
            and row.cwd
            and scoper.in_scope(row.cwd)
        ):
            result.append(replace(event, cwd=row.cwd, scope_basis="native_call_metadata"))
        elif scoper.in_scope(None):
            result.append(replace(event, cwd=None, scope_basis="unknown_scope"))
    return result
