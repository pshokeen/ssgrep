"""Pure join guards; rejected reasons contain no caller paths or request text."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssgrep.accounting.reconcile import ExecFact, Request


def contradictions(execution: ExecFact, request: Request) -> tuple[str, ...]:
    """Hard evidence mismatches cannot be overridden by correlation or timing."""
    reasons = []
    if execution.conflicted:
        reasons.append("journal_conflict")
    if execution.operation != request.operation:
        reasons.append("operation_mismatch")
    if execution.transport in ("cli", "mcp") and execution.transport != request.transport:
        reasons.append("transport_mismatch")
    if (
        execution.correlation_id
        and request.correlation
        and execution.correlation_id != request.correlation
    ):
        reasons.append("correlation_mismatch")
    if (
        execution.fingerprint_key_id is not None
        and request.fingerprint_key_id is not None
        and execution.fingerprint_key_id != request.fingerprint_key_id
    ):
        reasons.append("key_mismatch")
    elif request.fingerprint is not None and request.fingerprint != execution.fingerprint:
        reasons.append(
            "query_mismatch"
            if execution.fingerprint_key_id is not None and request.fingerprint_key_id is not None
            else "fingerprint_mismatch"
        )
    if not request.cwd:
        reasons.append("unknown_scope")
    elif not request.cwd_trusted:
        reasons.append("untrusted_cwd")
    elif not os.path.isabs(request.cwd) or "\0" in request.cwd:
        reasons.append("untrusted_cwd")
    if execution.cwd and request.cwd and request.cwd_trusted and "untrusted_cwd" not in reasons:
        if execution.scope_basis == "unknown_scope":
            reasons.append("untrusted_cwd")
        elif execution.cwd != os.path.realpath(request.cwd):
            reasons.append("cwd_mismatch")
    shown = request.result_refs
    if shown is not None and execution.refs and not set(execution.refs) <= shown:
        if execution.refs_complete or execution.operation == "show":
            reasons.append("result_mismatch")
    return tuple(reasons)


def rejection(execution: ExecFact, request: Request, *, explicit: bool = False) -> tuple[str, ...]:
    """A supported native workspace and compatible query/result are required.

    Unknown journal scope can join trusted native call metadata, but only the
    reducer's same-profile policy ownership check can authorize that workspace.
    Explicit correlation can replace a missing clock, never contradictory facts.
    """
    reasons = contradictions(execution, request)
    if reasons or explicit:
        return reasons
    if (
        request.window is None
        or request.window[1] < request.window[0]
        or execution.ts_ms is None
        or not execution.timing_known
    ):
        return ("clock_uncertainty",)
    if not request.window[0] <= execution.ts_ms <= request.window[1]:
        return ("outside_window",)
    shown = request.result_refs
    refs_ok = shown is not None and bool(execution.refs)
    if request.fingerprint is not None or (refs_ok and execution.refs_complete):
        return ()
    return ("insufficient_evidence",)
