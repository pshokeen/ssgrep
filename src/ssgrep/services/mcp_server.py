"""MCP stdio server for ssgrep.

Exposes exactly three tools — ``search_sessions``, ``show_session``,
``index_status`` — as thin adapters over ``ssgrep.api``. The fastmcp SDK is
imported lazily when the server is constructed; importing this module pulls in
no SDK dependencies. The embedding and re-ranker models are loaded lazily on
first use (index or search), not at import.

stdout is the MCP transport: nothing in this module may write to stdout.
All diagnostics go to stderr.

Startup performs one best-effort global reconciliation.

The reconciliation runs in the background so startup doesn't block MCP client
handshake timeout windows.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ssgrep.services import api
from ssgrep.utilities.types import (
    EmptyQueryError,
    IndexNotFoundError,
    IndexNotReadyError,
    InvalidPredicateError,
)

_INSTRUCTIONS = (
    "Search and browse AI coding session transcripts. Use search_sessions to "
    "find relevant past work, show_session for bounded context on one episode, "
    "index_status to check index health."
)

_mcp = None


def _start_reconcile_background() -> None:
    """Reconcile changed transcripts without blocking process startup."""

    threading.Thread(
        target=_reconcile_on_startup,
        daemon=True,
        name="ssgrep-mcp-reconcile",
    ).start()


def _reconcile_on_startup() -> None:
    """Refresh changed transcripts; failures do not prevent serving."""
    try:
        api.index()
    except Exception as error:
        print(f"MCP startup reconciliation failed: {error}", file=sys.stderr)


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


def search_sessions(query: str, limit: int = 10, where: str | None = None) -> dict[str, Any]:
    """Search all indexed projects with optional Lance metadata ``where``.

    Returns ranked cards (project, source, title, date, score, files, excerpt,
    ref) plus total_matches and truncation flags.
    """
    try:
        response = api.search(query, limit=limit, where=where)
    except EmptyQueryError:
        return {"error": "Query must not be empty."}
    except InvalidPredicateError as error:
        return {"error": str(error)}
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
    }
    return payload


def show_session(ref: str, max_chars: int = 20000) -> dict[str, Any]:
    """Bounded prompt and response for a ref returned by search_sessions."""
    try:
        detail = api.show(ref)
    except (IndexNotFoundError, IndexNotReadyError) as error:
        return {"error": str(error)}
    except Exception as error:
        return _fail("show_session", error)

    if detail is None:
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
    """Return global index counts, size, model, schema, and archive status."""
    try:
        stats = api.status()
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

    On first initialization, run one best-effort global reconciliation so
    search sees transcript changes made since the previous index run.
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

        # Reconcile global sources in the background.
        _start_reconcile_background()

    return _mcp
