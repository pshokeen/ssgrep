"""The traced CocoIndex processing components.

One component per source key (mounted via ``mount_each`` over the registry
LiveMap). Each component is memoized on its frozen descriptor, so an
unchanged source is normally skipped wholesale; a changed source re-parses,
re-segments, re-embeds, and re-declares its rows, and the engine reconciles
the targets. Deleted sources keep their registry descriptor, so they usually
memo-hit and are skipped, which is what preserves tombstoned rows -- but a
change to the pipeline code itself invalidates every memo, including theirs.
When that happens, a deleted source's transcript file is gone, so
``process_source`` cannot re-parse it; instead it redeclares the rows already
recorded for that source from ``ARCHIVED_ROWS`` (see ``pipeline/archive.py``),
so the engine's reconciliation never sees an empty declaration and never
deletes the archived rows.
"""

from __future__ import annotations

from pathlib import Path

import cocoindex as coco
from cocoindex.connectors import lancedb
from cocoindex.resources.live_map import LiveMap

from ssgrep.pipeline import rows as rows_mod
from ssgrep.pipeline.diagnostics import current as diagnostics
from ssgrep.pipeline.episodes import build_episodes, enrich_session
from ssgrep.pipeline.sources import SourceDescriptor, to_transcript_source
from ssgrep.pipeline.state import ARCHIVED_ROWS, EMBEDDER
from ssgrep.sessions import adapters as transcript_adapters


@coco.fn
async def produce_entries(
    lm: LiveMap[str, SourceDescriptor], entries: dict[str, SourceDescriptor]
) -> None:
    """Declare the full source map: registry (frozen) plus fresh discovery."""
    for key, descriptor in entries.items():
        lm.declare_entry(key, descriptor)


@coco.fn(memo=True)
async def process_source(
    descriptor: SourceDescriptor,
    chunk_table: lancedb.TableTarget[rows_mod.ChunkRow],
    episode_table: lancedb.TableTarget[rows_mod.EpisodeRow],
    session_table: lancedb.TableTarget[rows_mod.SessionRow],
) -> None:
    """Parse one transcript source and declare its session/episode/chunk rows.

    Runs only when ``descriptor`` (or the pipeline code) changed: unchanged
    sources are skipped entirely and keep their existing rows.
    """
    source = to_transcript_source(descriptor)
    try:
        parsed = transcript_adapters.read_source(source)
    except FileNotFoundError as error:
        # Only the adapter's missing transcript qualifies, not other read failures.
        if error.filename is None or Path(error.filename) != source.session.path:
            raise
        snapshot = coco.use_context(ARCHIVED_ROWS)
        if source.key not in snapshot:
            raise ValueError("Source disappeared after archive snapshot; retry indexing") from error
        sessions, episodes, chunks = snapshot[source.key]
        for target, items in (
            (session_table, sessions),
            (episode_table, episodes),
            (chunk_table, chunks),
        ):
            for row in items:
                target.declare_row(row=row)
        diagnostics.record(archived=1)
        return
    diagnostics.record(
        malformed=parsed.malformed_records,
        skipped=parsed.skipped_records,
    )
    enriched = enrich_session(source.session, list(parsed.records))
    episodes = build_episodes(list(parsed.records), enriched)

    session_table.declare_row(
        row=rows_mod.build_session_row(
            session_id=enriched.session_id,
            path=source.key,
            runtime=enriched.runtime,
        )
    )
    for episode in episodes:
        episode_table.declare_row(row=rows_mod.build_episode_row(episode, enriched))

    payloads = [
        (episode, part, search_text)
        for episode in episodes
        for part, search_text in rows_mod.chunk_payloads(episode, enriched)
    ]
    if not payloads:
        return
    # The embedder's encode_many_async coalesces every concurrently-running
    # source's chunks into one large model batch (deduplicating byte-identical
    # texts) instead of one tiny ColBERT.encode call per source, and is awaited
    # so the event loop stays free for other sources while the device is busy.
    provider = coco.use_context(EMBEDDER)
    texts = [search_text for (_episode, _part, search_text) in payloads]
    vectors = await provider.encode_many_async(texts, is_query=False)
    for (episode, part, search_text), vector in zip(payloads, vectors, strict=True):
        chunk_table.declare_row(
            row=rows_mod.build_chunk_row(part, search_text, vector, episode, enriched)
        )


__all__ = ["process_source", "produce_entries"]
