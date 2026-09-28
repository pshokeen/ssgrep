"""Drive CocoIndex updates and reject incomplete child execution."""

from __future__ import annotations

import cocoindex as coco
from usecli import ProgressBar

#: CocoIndex processor whose executions correspond one-to-one with transcript
#: sources (one ``process_source`` component per source key, aggregated under
#: the mounted function's name). Progress reports track these executions, so
#: the bar advances per source as parsing and embedding complete.
_PROCESS_SOURCE_COMPONENT = "process_source"


def _source_progress(stats: coco.UpdateStats, total: int) -> int:
    """How many ``process_source`` executions have finished, clamped to total.

    ``stats.by_component`` groups per-processor counters under the mounted
    function name; children of a LiveMap mount are aggregated in that one
    bucket. Falls back to the engine-wide finished count (accepting that a few
    scaffold components may finish first) rather than failing.
    """
    for name, group in stats.by_component.items():
        if name.split(".")[-1] == _PROCESS_SOURCE_COMPONENT:
            return min(max(group.num_finished, 0), total)
    return min(max(stats.total.num_finished, 0), total)


async def _drive_update(
    app: coco.App,
    *,
    total: int,
    full_reprocess: bool,
    quiet: bool,
) -> None:
    """Run one engine update, rendering a per-source progress bar when interactive.

    ``app.update_blocking`` suppresses all output, so the CLI would sit silent
    while hundreds of sources are embedded. This drives the async update handle
    instead: Rich's ``ProgressBar`` (stderr, self-disabling under ``--quiet`` /
    JSON mode / non-TTY) advances once per source as CocoIndex's per-component
    stats report finished executions. Errors surface through ``watch()``.
    """
    handle = app.update(full_reprocess=full_reprocess)
    if total <= 0:
        await handle.result()
    else:
        # Drain the stream before reporting child failures: workers must no longer
        # be writing when the caller handles an unsuccessful reconciliation.
        with ProgressBar(total=total, description="Indexing transcripts", quiet=quiet) as progress:
            async for snapshot in handle.watch():
                if snapshot.stats is not None:
                    progress.update(completed=_source_progress(snapshot.stats, total))
    # A successful root result does not imply every mounted child succeeded.
    # Progress snapshots may omit the final statistics, so read the handle itself.
    stats = handle.stats()
    if stats is None:
        raise RuntimeError("Index update failed: final statistics unavailable")
    errors = stats.total.num_errors
    if errors:
        raise RuntimeError(f"Index update failed: {errors} component errors")
