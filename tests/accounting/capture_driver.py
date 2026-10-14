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
import sys
import time
from datetime import UTC, datetime

import usecli

import ssgrep.search as search_module
import ssgrep.search.detail as detail_module
from ssgrep.accounting import journal
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

sys.argv = ["ssgrep", *sys.argv[1:]]
usecli.main()
