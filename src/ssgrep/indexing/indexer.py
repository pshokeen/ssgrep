"""Global multi-runtime transcript ingestion.

Public entry point for the CLI and MCP service: serializes index runs across
local processes with the advisory ``index.lock`` and delegates the actual
incremental reconciliation to the CocoIndex pipeline
(:mod:`ssgrep.pipeline.app`). Model identity constants are re-exported here
for callers that report bindings.
"""

from __future__ import annotations

import fcntl
from pathlib import Path

from ssgrep.indexing.embed import DIMENSION, MODEL_ID, MODEL_REVISION
from ssgrep.pipeline import run as _pipeline_run
from ssgrep.store import LanceStore, ensure_data_dir
from ssgrep.utilities.types import IndexStats


def index(
    *,
    rebuild: bool = False,
    no_subagents: bool = False,
    allow_shrink: bool = False,
    scope: str | None = None,
    quiet: bool = False,
    live: bool = False,
    full_reprocess: bool = False,
) -> IndexStats:
    """Serialize global, cross-table reconciliation across local processes."""
    repository = LanceStore()
    ensure_data_dir()
    with (repository.root / "index.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _pipeline_run(
            rebuild=rebuild,
            no_subagents=no_subagents,
            allow_shrink=allow_shrink,
            scope=str(Path(scope).expanduser().absolute()) if scope else None,
            quiet=quiet,
            live=live,
            full_reprocess=full_reprocess,
        )


__all__ = ["DIMENSION", "MODEL_ID", "MODEL_REVISION", "index"]
