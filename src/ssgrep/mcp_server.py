"""MCP stdio server for ssgrep.

Exposes exactly three tools — ``search_sessions``, ``show_session``,
``index_status`` — as thin adapters over ``ssgrep.api``. The fastmcp SDK is
imported lazily when the server is constructed; importing this module pulls in
no SDK dependencies. model2vec is loaded by ``ssgrep.embed`` on first use,
not at import.

stdout is the MCP transport: nothing in this module may write to stdout.
All diagnostics go to stderr.

On startup, the server drains any pending work items from the queue (D13).
This is best-effort and does not block server startup or prevent it from
serving if the drain fails.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ssgrep import api
from ssgrep.types import EmptyQueryError, IndexNotFoundError, IndexNotReadyError

#: Explicit project scope, set by `ssgrep mcp --project-dir <path>` via
#: set_project_dir() before the server starts. None (the default) preserves
#: the original behavior: scope resolves from the server process's working
#: directory at tool invocation time. Exists because MCP client configs
#: (claude_desktop_config.json, .mcp.json, etc.) universally support
#: `command`/`args` but not always `cwd` -- a field-deployment report
#: documented needing a shell wrapper that cd'd before exec, for exactly
#: this gap. No tool parameter can widen this scope: it is fixed for the
#: server's lifetime before the first tool call.
_PROJECT_DIR = None

_INSTRUCTIONS = (
    "Search and browse AI coding session transcripts. Use search_sessions to "
    "find relevant past work, show_session for full context on one episode, "
    "index_status to check index health."
)

_mcp = None


def set_project_dir(project_dir: Path | None) -> None:
    """Fix the server's project scope explicitly (CLI --project-dir).

    Must be called before the first tool invocation; passing None restores
    the default cwd-resolution behavior. The MCP tools themselves expose no
    parameter that can change scope after startup.
    """
    global _PROJECT_DIR
    _PROJECT_DIR = project_dir


def _get_project_dir() -> Path:
    """Resolve the project scope for a tool invocation.

    An explicit scope set via set_project_dir() (the CLI's --project-dir
    flag) wins; otherwise the server process's current working directory is
    resolved at call time -- dynamically rather than at module import time,
    so tests that chdir after import see the correct directory.
    """
    if _PROJECT_DIR is not None:
        return _PROJECT_DIR
    return Path.cwd()


def _drain_startup_workqueue() -> None:
    """Drain pending work items on MCP startup (D13 reconciliation).

    This is best-effort: failures do not prevent the server from serving.
    Empty or unreadable queues are silent. The drain is performed by calling
    indexer.index(), which internally drains the entire queue via its own
    _drain_workqueue() before proceeding with discovery-based indexing.

    A single indexer.index() call processes all pending and abandoned work
    items, so no explicit per-item loop is needed here.

    If the transcript root (~/.claude/projects) does not exist, indexer.index()
    raises IndexNotFoundError. This is not a fatal error for the MCP server —
    search operations will provide actionable error messages. Log it and continue.
    """
    # Let index() do the draining via its internal _drain_workqueue().
    # Drain failures don't prevent server startup.
    try:
        from ssgrep import indexer

        indexer.index(_get_project_dir(), quiet=True)
    except IndexNotFoundError as error:
        # Missing transcript root is not a fatal server error; search operations
        # will provide actionable messages. Log and continue.
        print(f"MCP startup: {error}", file=sys.stderr)
    except Exception:
        # Other drain failures are logged to stderr by indexer; don't raise.
        pass


def _plain(value: Any) -> Any:
    """Reduce frozen contract dataclasses to JSON-serializable plain data."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _fail(tool: str, error: Exception) -> dict[str, Any]:
    """Log an unexpected failure to stderr and return it as structured data."""
    print(f"{tool} failed: {error}", file=sys.stderr)
    return {"error": f"{tool} failed: {error}"}


def search_sessions(query: str, limit: int = 10) -> dict[str, Any]:
    """Search session transcripts. Returns ranked cards (title, date, score,
    files, excerpt, ref) plus total_matches and truncation flags."""
    try:
        response = api.search(_get_project_dir(), query, limit=limit)
    except EmptyQueryError:
        return {"error": "Query must not be empty."}
    except (IndexNotFoundError, IndexNotReadyError) as error:
        # API messages are already actionable (e.g. "Run `ssgrep index` first.").
        return {"error": str(error)}
    except Exception as error:
        return _fail("search_sessions", error)

    payload: dict[str, Any] = {
        "results": [_plain(card) for card in response.results],
        "total_matches": response.total_matches,
        "omitted_count": response.omitted_count,
        "excerpts_truncated": response.excerpts_truncated,
        "clamped": response.clamped,
        "index_empty": response.index_empty,
        "stale": response.stale,
        "stale_count": response.stale_count,
    }
    if response.index_empty:
        # A bare `index_empty: true` is the same undiagnosable observable as
        # "Indexed 0 sessions" on the terminal, landing on the caller LEAST
        # able to investigate and MOST likely to suggest `ssgrep index
        # --rebuild` -- the one command that destroys a healthy index whose
        # only problem is scope. The census that stops a human reaching for
        # it has to reach this caller too. Best-effort: a failed diagnostic
        # must never turn a successful (if empty) search into an error.
        try:
            from ssgrep import indexer_support, scope_report

            # Census the EFFECTIVE (persisted) scope: this is the literal
            # auto-executing-agent channel -- a wrong-baseline remedy here
            # is exactly the hazard the census-scope fixes exist to close.
            payload["zero_discovery"] = scope_report.build_scope_report(
                indexer_support.effective_scope(_get_project_dir())
            ).as_payload()
        except Exception:  # pragma: no cover - defensive
            pass
    return payload


def show_session(ref: str, max_chars: int = 20000) -> dict[str, Any]:
    """Full prompt and response for one episode; ref comes from
    search_sessions. Content beyond max_chars is truncated."""
    try:
        detail = api.show(_get_project_dir(), ref)
    except (IndexNotFoundError, IndexNotReadyError) as error:
        return {"error": str(error)}
    except Exception as error:
        return _fail("show_session", error)

    if detail is None:
        # api.show collapses "no index" and "ref not found" into None;
        # disambiguate so a missing index never reads as "never solved before".
        try:
            if not api.status(_get_project_dir()).index_exists:
                return {"error": "No index found. Run `ssgrep index` first."}
        except Exception:
            # api.status failed (corrupt index, bad schema); treat as missing.
            return {"error": "No index found. Run `ssgrep index` first."}
        return {"error": f"Episode not found: {ref}. Run search_sessions to find valid refs."}

    result = _plain(detail)
    budget = max(max_chars, 0)
    prompt = result["prompt_text"]
    response = result["response_text"]
    truncated = False
    if len(prompt) > budget:
        result["prompt_text"] = prompt[:budget]
        result["response_text"] = ""
        truncated = True
    elif len(prompt) + len(response) > budget:
        result["response_text"] = response[: budget - len(prompt)]
        truncated = True
    # OR-fold in the detail-level truncation flags: an episode whose stored
    # text was already capped by detail.py is truncated even when this
    # function's own max_chars budget was generous.
    result["truncated"] = (
        truncated
        or result.get("prompt_truncated", False)
        or result.get("response_truncated", False)
    )
    return result


def index_status() -> dict[str, Any]:
    """Index health: existence, session/episode/chunk counts, last-indexed
    time, model, schema version, staleness."""
    try:
        stats = api.status(_get_project_dir())
    except Exception as error:
        return _fail("index_status", error)

    result = _plain(stats)
    if not stats.index_exists:
        result["message"] = "No index found. Run `ssgrep index` to build one."
    return result


def _ssgrep_version() -> str:
    """ssgrep's installed version, or "unknown" if it cannot be determined.

    Never raises: an un-introspectable install must still get a server.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("ssgrep")
    except PackageNotFoundError:  # pragma: no cover - only for non-installed trees
        return "unknown"


def get_mcp_server():
    """Lazily initialize and return the MCP server instance.

    The fastmcp SDK is imported only when this function is called, not when
    this module is imported. This defers SDK initialization cost until the
    server is actually constructed.

    On first initialization, drains any pending work items from the queue (D13).
    The drain calls indexer.index() synchronously; startup blocks until the
    drain completes. Drain failures (missing transcript root, corrupt index) do
    not prevent the server from starting; search operations provide actionable
    error messages to the user instead.
    """
    global _mcp
    if _mcp is None:
        import fastmcp
        from fastmcp import FastMCP

        # Silence FastMCP's startup banner and its update check BEFORE the
        # server is constructed or run. Left alone, log_server_banner() calls
        # check_for_newer_version(), which makes an outbound HTTPS request to
        # https://pypi.org/pypi/fastmcp/json on every cold start (12-hour
        # cache) and writes a version cache under the user's config dir.
        #
        # That breaks the product's flagship promise. The README states
        # "after that, no network access is needed or used" and then claims
        # "These assertions are verified by tests in every CI run" -- while
        # tests/test_privacy.py never started the MCP server, so CI stayed
        # green over a false guarantee. `ssgrep mcp` is a documented,
        # first-class integration, so an offline-by-choice buyer got
        # unexplained egress to PyPI several times a day, plus an advert
        # telling them to `pip install --upgrade fastmcp` -- which pyproject
        # pins, so following it breaks their paid install.
        #
        # Both knobs, not just the banner: show_banner=False suppresses the
        # call site, and check_for_updates="off" disables the check itself so
        # any other path into it stays offline too.
        fastmcp.settings.show_server_banner = False
        fastmcp.settings.check_for_updates = "off"

        # ssgrep's own version, not fastmcp's. serverInfo.version was
        # reporting the SDK's, so every MCP-client bug report carried the
        # wrong one.
        _mcp = FastMCP("ssgrep", instructions=_INSTRUCTIONS, version=_ssgrep_version())

        # Register tools with the server
        _mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})(search_sessions)
        _mcp.tool(annotations={"readOnlyHint": True})(show_session)
        _mcp.tool(annotations={"readOnlyHint": True})(index_status)

        # Drain startup work queue (D13). This blocks until the drain completes
        # (typically warm: ~2s, cold: full-index cost). Best-effort: failures do
        # not prevent the server from serving.
        _drain_startup_workqueue()

    return _mcp
