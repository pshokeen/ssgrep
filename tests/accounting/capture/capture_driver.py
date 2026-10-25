"""Subprocess entry point: the real usecli CLI with only the retrieval leaf stubbed.

Run as ``python capture_driver.py <ssgrep args...>``. ``SSGREP_DRIVER`` selects the
leaf behavior; everything above it (api capture, command handlers, usecli
rendering, JSON ``os._exit`` paths, exit codes) is the production code.

* ``ok``       -- one result card (default)
* ``many``     -- many large cards (output larger than any pipe buffer)
* ``empty``    -- zero matches
* ``missing``  -- ``IndexNotFoundError`` (JSON ``os._exit`` path, exit 4)
* ``crash``    -- ``RuntimeError`` (unexpected service failure)
* ``slow``     -- sleeps (interruption after ``started``)

``SSGREP_DRIVER_JOURNAL=fail`` additionally makes every journal append raise,
so a failing service and a failing journal can be combined.
"""

from __future__ import annotations

import os
import resource
import sys
import time
from datetime import UTC, datetime

import usecli

import ssgrep.search as search_module
import ssgrep.search.detail as detail_module
from ssgrep.accounting.capture import journal
from ssgrep.utilities.types import (
    EpisodeDetail,
    IndexNotFoundError,
    ResultCard,
    SearchResponse,
)

BEHAVIOR = os.environ.get("SSGREP_DRIVER", "ok")


def _card(index: int, size: int = 60) -> ResultCard:
    return ResultCard(
        ref=f"claude:session-{index}:episode:{index}",
        title=f"Result {index}",
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        score=1.0 - index / 1000,
        excerpt="payload " * size,
        files_touched=("src/a.py",),
        is_subagent=False,
    )


def fake_search(query: str, **_kwargs: object) -> SearchResponse:
    if BEHAVIOR == "missing":
        raise IndexNotFoundError("No global index found. Run `ssgrep index` first.")
    if BEHAVIOR == "crash":
        raise RuntimeError("service exploded")
    if BEHAVIOR == "slow":
        print("SLOW-STARTED", file=sys.stderr, flush=True)
        time.sleep(60)
    if BEHAVIOR == "empty":
        return SearchResponse(results=[], total_matches=0)
    count = 300 if BEHAVIOR == "many" else 1
    cards = [_card(i, 400 if BEHAVIOR == "many" else 60) for i in range(count)]
    return SearchResponse(results=cards, total_matches=count)


def fake_show(ref: str) -> EpisodeDetail | None:
    if BEHAVIOR == "missing":
        raise IndexNotFoundError("No global index found. Run `ssgrep index` first.")
    if ref == "unknown":
        return None
    return EpisodeDetail(
        episode_id=ref,
        session_id="session-1",
        title="A title",
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        git_branch="main",
        cwd="/work/project",
        prompt_text="prompt " * 20,
        response_text="response " * 20,
        files_touched=("src/a.py",),
        tool_names=("bash",),
        is_subagent=False,
        response_truncated=True,
    )


def _failing_append(*_args: object, **_kwargs: object) -> bool:
    raise OSError(28, "No space left on device")


vars(search_module)["search"] = fake_search
vars(detail_module)["show"] = fake_show
if os.environ.get("SSGREP_DRIVER_JOURNAL") == "fail":
    vars(journal)["append"] = _failing_append

if os.environ.get("SSGREP_DRIVER_RENDER_METADATA") == "fail":
    from pathlib import Path
    from unittest import mock

    from ssgrep.accounting.capture import capture

    _actual_rendered = capture.rendered

    def _render_quantity_failure(unit, _reason):
        marker = Path(os.environ["SSGREP_DATA_DIR"]) / "render-metadata-fault"
        marker.write_text(unit)
        raise ValueError("injected render metadata construction failure")

    def _render_with_metadata_fault(
        execution: capture.Execution | None,
        *,
        truncated: bool | None = None,
        omitted_count: int | None = None,
        rendered_bytes: capture.Quantity | None = None,
        bytes_unavailable: str = "serialization_not_observed",
    ) -> None:
        with mock.patch.object(
            capture.Quantity, "unavailable", side_effect=_render_quantity_failure
        ):
            _actual_rendered(
                execution,
                truncated=truncated,
                omitted_count=omitted_count,
                rendered_bytes=rendered_bytes,
                bytes_unavailable=bytes_unavailable,
            )

    _render_fault_patch = mock.patch.object(capture, "rendered", _render_with_metadata_fault)
    _render_fault_patch.start()

# Apply real storage pressure in the freshly exec'd child, not a Python
# preexec_fn forked from pytest's multithreaded/native-library process.
if limit := os.environ.get("SSGREP_DRIVER_FSIZE_LIMIT"):
    resource.setrlimit(resource.RLIMIT_FSIZE, (int(limit), int(limit)))

if os.environ.get("SSGREP_DRIVER_TRANSPORT") == "mcp":
    # Real FastMCP server and in-memory client: the tool functions and capture wiring are
    # production code. Only startup reconciliation (an index build) is disabled.
    import asyncio
    import json

    from fastmcp import Client

    from ssgrep.services import mcp_server

    vars(mcp_server)["_start_reconcile_background"] = lambda: None
    tool, argument = sys.argv[1], sys.argv[2]
    arguments = {"query": argument} if tool == "search_sessions" else {"ref": argument}

    async def _call() -> object:
        async with Client(mcp_server.get_mcp_server()) as client:
            return (await client.call_tool(tool, arguments)).structured_content

    sys.stdout.write(json.dumps(asyncio.run(_call()), sort_keys=True) + "\n")
    raise SystemExit(0)

sys.argv = ["ssgrep", *sys.argv[1:]]
usecli.main()
