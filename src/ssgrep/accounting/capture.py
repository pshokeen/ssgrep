"""Policy-aware, fail-open capture of search/show/note executions.

Hot-path contract (design D3): importing this module is cheap (stdlib and the
frozen contracts only). Policy and key cache hits still validate pinned,
owner-protected directory/file authority; disabled capture never reads transcript
content or creates accounting state. Enabled capture adds keyed fingerprints and
validated segment writes, with no ``fsync``, index/ledger scan, or network I/O.
Every public function swallows its own failures (``Exception``) and never alters the operation's
return value or exception; a capture problem can only mean "less evidence".

Authorization binds one configured root path and device/inode before reading
policy. Key creation, process identity and every append receive that same private
snapshot; neither root metadata nor request arguments are added to exports.
Configuration/root replacement drops in-flight phases without poisoning another
root's profile. Unsafe-profile latches remain attached to their original inode.

Phases: ``begin`` writes ``started`` at service entry; ``finish_*``/``fail``
write ``completed`` with the outcome category (never exception text);
transports that actually know the output shape write ``rendered`` through
``rendered``. ``rendered`` means "this layer produced the output", not that the
consumer received or saw it. Final byte counts are recorded only when observed
by the layer that owns serialization; otherwise ``Quantity.unavailable``.
"""

from __future__ import annotations

import contextvars
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting import capture_policy, datadir
from ssgrep.accounting.contracts import (
    ExecutionEvent,
    Operation,
    Outcome,
    Phase,
    Quantity,
    Transport,
)

CORRELATION_ENV = "SSGREP_CORRELATION_ID"
_CORRELATION_RE = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")
_MAX_REFS = 200
_transport: contextvars.ContextVar[Transport] = contextvars.ContextVar(
    "ssgrep_capture_transport", default=Transport.API
)
_caller: contextvars.ContextVar[tuple[str | None, str | None]] = contextvars.ContextVar(
    "ssgrep_capture_caller", default=(None, None)
)
_last: contextvars.ContextVar[Execution | None] = contextvars.ContextVar(
    "ssgrep_capture_last", default=None
)
_unsafe: set[tuple[datadir.SafeRootSnapshot, str]] = set()
_version_cache: list[str] = []


@dataclass(frozen=True, slots=True)
class Execution:
    """Identity and fixed request context of one captured execution."""

    execution_id: str
    profile: str
    operation: Operation
    transport: Transport
    cwd: str | None
    fingerprint: str
    limit: int | None
    token_budget: int | None
    correlation_id: str | None
    start_ns: int
    fingerprint_key_id: str | None
    scope_basis: str
    root: datadir.SafeRootSnapshot


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version() -> str:
    if not _version_cache:
        try:
            from importlib.metadata import version

            _version_cache.append(version("ssgrep"))
        except Exception:
            _version_cache.append("unknown")
    return _version_cache[0]


def _correlation() -> str | None:
    value = os.environ.get(CORRELATION_ENV)
    return value if value and _CORRELATION_RE.match(value) else None


def _failed(
    profile: str,
    error: Exception,
    *,
    root: datadir.SafeRootSnapshot,
) -> None:
    """Latch unsafe authority only for this root inode/profile, including on return."""
    from ssgrep.accounting.storage import UnsafeStorageError

    if isinstance(error, UnsafeStorageError):
        _unsafe.add((root, profile))


def _emit(execution: Execution, phase: Phase, **fields: Any) -> None:
    elapsed = time.monotonic_ns() - execution.start_ns if phase is not Phase.STARTED else None
    try:
        from ssgrep.accounting import journal

        journal.append(
            execution.profile,
            ExecutionEvent(
                execution_id=execution.execution_id,
                phase=phase,
                operation=execution.operation,
                transport=execution.transport,
                profile=execution.profile,
                ts_utc=_now(),
                cwd=execution.cwd,
                fingerprint=execution.fingerprint,
                version=_version(),
                mono_ns=elapsed,
                limit=execution.limit,
                token_budget=execution.token_budget,
                correlation_id=execution.correlation_id,
                fingerprint_key_id=execution.fingerprint_key_id,
                scope_basis=execution.scope_basis,
                **fields,
            ),
            root=execution.root,
        )
    except Exception as error:
        _failed(execution.profile, error, root=execution.root)


def arm(
    transport: Transport, *, caller_workspace: str | None = None, profile: str | None = None
) -> None:
    """Arm one invocation from the local transport host, not client arguments.

    ``caller_workspace`` is local host attestation, not cryptographic proof.
    Tool arguments, MCP client roots and server cwd are not caller authority.
    """
    _transport.set(transport)
    _caller.set((caller_workspace, profile))
    _last.set(None)


def take() -> Execution | None:
    """The execution started by the api call just made on this context (once)."""
    execution = _last.get()
    _last.set(None)
    return execution


def begin(
    operation: Operation,
    *,
    text: str | None = None,
    where: str | None = None,
    limit: int | None = None,
    token_budget: int | None = None,
    cwd: str | None = None,
) -> Execution | None:
    """Write ``started`` when policy allows capture for this event; else return None.

    Consumes the transport declared by :func:`arm` (one ``arm`` serves one ``begin``).
    """
    transport = _transport.get()
    caller_workspace, selected_profile = _caller.get()
    _caller.set((None, None))
    _transport.set(Transport.API)
    _last.set(None)
    profile: str | None = None
    root: datadir.SafeRootSnapshot | None = None
    try:
        if transport is Transport.MCP:
            scope = caller_workspace
            scope_basis = "trusted_invocation" if scope is not None else "unknown_scope"
        else:
            scope = caller_workspace if caller_workspace is not None else cwd
            scope_basis = "trusted_invocation" if scope is not None else "process_cwd"
            scope = scope if scope is not None else os.getcwd()
        if scope is not None:
            if not os.path.isabs(scope) or "\0" in scope:
                return None
            scope = os.path.realpath(scope)
        root = datadir.snapshot()
        if root is None:
            return None
        profile = capture_policy.capture_profile(scope, profile=selected_profile, root=root)
        if profile is None or (root, profile) in _unsafe:
            return None
        datadir.require_current(root)
        from ssgrep.accounting import fingerprint, journal

        key = fingerprint.load_key(profile, create=True, root=root)
        execution = Execution(
            execution_id=journal.next_execution_id(root=root),
            profile=profile,
            operation=operation,
            transport=transport,
            cwd=scope,
            fingerprint=fingerprint.request_fingerprint(key, operation, query=text, where=where),
            limit=limit,
            token_budget=token_budget,
            correlation_id=_correlation() if transport is not Transport.MCP else None,
            start_ns=time.monotonic_ns(),
            fingerprint_key_id=fingerprint.key_id(key),
            scope_basis=scope_basis,
            root=root,
        )
    except Exception as error:
        if profile is not None and root is not None:
            _failed(profile, error, root=root)
        return None
    _emit(execution, Phase.STARTED)
    _last.set(execution)
    return execution


def complete(
    execution: Execution | None,
    outcome: Outcome,
    *,
    refs: tuple[str, ...] = (),
    ranks_scores: tuple[tuple[int, int], ...] = (),
    omitted_count: int | None = None,
    truncated: bool | None = None,
) -> None:
    if execution is None:
        return
    _emit(
        execution,
        Phase.COMPLETED,
        outcome=outcome,
        refs=refs,
        ranks_scores=ranks_scores,
        omitted_count=omitted_count,
        truncated=truncated,
    )


def fail(execution: Execution | None, error: BaseException) -> None:
    """Record the error category for a service exception; the caller re-raises it."""
    interrupted = isinstance(error, KeyboardInterrupt | SystemExit)
    complete(execution, Outcome.INTERRUPTED if interrupted else Outcome.ERROR)


def _milli(score: Any) -> int:
    value = float(score)
    return int(round(value * 1000)) if math.isfinite(value) else 0


def finish_search(execution: Execution | None, response: Any) -> None:
    """Completed event for a search response (outcome ok/empty, bounded refs/ranks)."""
    if execution is None:
        return
    try:
        cards = response.results[:_MAX_REFS]
        refs = tuple(str(card.ref) for card in cards)
        ranks = tuple((rank, _milli(card.score)) for rank, card in enumerate(cards, 1))
        empty = not response.results and response.total_matches == 0
        complete(
            execution,
            Outcome.EMPTY if empty else Outcome.OK,
            refs=refs,
            ranks_scores=ranks,
            omitted_count=int(response.omitted_count),
            truncated=bool(response.excerpts_truncated),
        )
    except Exception as error:
        _failed(execution.profile, error, root=execution.root)
        complete(execution, Outcome.UNKNOWN)  # unreadable response: never imply success


def finish_show(execution: Execution | None, ref: str, detail: Any) -> None:
    """Completed event for a show: EMPTY when the ref does not resolve."""
    if execution is None:
        return
    try:
        if detail is None:
            complete(execution, Outcome.EMPTY)
            return
        truncated = bool(detail.prompt_truncated or detail.response_truncated)
        complete(execution, Outcome.OK, refs=(ref,), truncated=truncated)
    except Exception as error:
        _failed(execution.profile, error, root=execution.root)
        complete(execution, Outcome.UNKNOWN)


def rendered(
    execution: Execution | None,
    *,
    truncated: bool | None = None,
    omitted_count: int | None = None,
    rendered_bytes: Quantity | None = None,
    bytes_unavailable: str = "serialization_not_observed",
) -> None:
    """Rendered event describing the shape the calling layer actually produced."""
    if execution is None:
        return
    try:
        quantity = rendered_bytes or Quantity.unavailable("bytes", bytes_unavailable)
        _emit(
            execution,
            Phase.RENDERED,
            truncated=truncated,
            omitted_count=omitted_count,
            rendered_bytes=quantity,
        )
    except Exception as error:
        _failed(execution.profile, error, root=execution.root)
