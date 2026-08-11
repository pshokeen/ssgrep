"""Text chunking for episode content."""

from __future__ import annotations

import hashlib

from ssgrep.types import Chunk, ContentType, Episode

# Sliding-window size and overlap, in characters. Tuned together (they are
# not independent: recall breaks when overlap falls below ~25% of target)
# with eval/harness.py against the frozen corpus snapshot, 2026-08-07 --
# see eval/README.md, "Update 2026-08-07", for the measurement account.
#
# Mechanism, not numerology: a multi-hop query needs two facts that were
# stated together in one episode to land in the SAME chunk to be findable
# (episode roll-up is MAX-based, so facts split across separately-scored
# chunks each look like weak partial evidence), and a paraphrase query
# needs enough surrounding prose context in the chunk for mean-pooled and
# per-token embeddings to capture the topic rather than a fragment. Both
# push toward larger windows with proportionally larger overlap. The
# original 1200/200 predates all five support legs and the MaxSim rerank
# leg; re-sweeping after those shipped found the surface had moved.
#
# Sweep summary (48-query labelled set, frozen snapshot): every tested
# config in the 1600-1800 target range with overlap >= ~25% of target
# holds recall@10 at the baseline 87.5% with the identical miss set, and
# lifts overall MRR to 0.645-0.700 (shipped 1200/300 measured 0.641 --
# multi-hop and paraphrase MRR rise in nearly every cell, paraphrase up to
# ~0.46 from 0.26). Below that overlap ratio (1600/300, 1750/350,
# 1800/400) individual queries fall out of the top 10; above target 1800
# (1900/475, 2000/500) recall breaks regardless of overlap. 1750/450 is a
# plateau-interior pick: all six nearest tested neighbors (1750/400,
# 1750/500, 1750/550, 1700/450, 1800/450, 1750/440) hold recall with MRR
# 0.66-0.70. Larger chunks also mean FEWER chunks (6,544 -> 5,148 on the
# eval corpus, -21%), so index size and brute-force cosine cost both drop.
CHUNK_TARGET_SIZE = 1750
CHUNK_OVERLAP = 450

# Hex digits of a sha256 kept in the chunk id. 16 hex chars = 64 bits, far
# more than enough to avoid an accidental collision among the handful of
# chunks one episode's one content type ever produces.
_HASH_LEN = 16


def _chunk_id(episode_id: str, content_type: ContentType, start: int, text: str) -> str:
    """Content-derived chunk id: readable episode/content-type prefix, plus
    a hash standing in for the old positional index.

    The hash covers `start` (the window's offset into the episode's text,
    always unique within one chunk_text() call -- see the loop below) as
    well as the chunk's own text, not text alone. Chunk rows are not just a
    search-retrieval convenience: they are what both retrieval legs (FTS5
    and cosine) match against. detail.show() does NOT reconstruct an
    episode's text from them -- it reads the episodes table's own
    prompt_text/response_text columns directly (see detail.py), which are
    the canonical, un-chunked copies stored once per episode at index time.
    Two chunks at different positions that happen to hold byte-identical
    text -- e.g. consecutive windows over a long run of a single repeated
    character, or any other highly repetitive input -- are therefore NOT
    interchangeable: collapsing them under a text-only hash would make
    INSERT OR REPLACE silently discard one, leaving that position
    unsearchable even though the episode's canonical text (unaffected by
    chunk ids) still has it. Folding `start` in keeps every positionally
    distinct chunk's id distinct while still being content-derived: unlike
    the old bare index, the id also changes if the text at that position
    changes, rather than silently pointing an unchanged id at different
    content.

    Stable across runs: a pure function of its inputs, so re-chunking the
    same episode text -- e.g. on retry after a crash mid-index, per
    indexer.py's crash-recovery notes -- reproduces the same ids and lets
    INSERT OR REPLACE make the re-insertion idempotent, rather than
    appending a second row under a new position-derived id.
    """
    digest = hashlib.sha256(f"{start}:{text}".encode()).hexdigest()[:_HASH_LEN]
    return f"{episode_id}:{content_type.value}:{digest}"


def chunk_text(
    text: str, episode_id: str, session_id: str, content_type: ContentType
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_TARGET_SIZE, len(text))

        if end < len(text):
            last_newline = text.rfind("\n\n", start, end)
            if last_newline > start + CHUNK_TARGET_SIZE // 2:
                end = last_newline + 2

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(episode_id, content_type, start, chunk_text_str),
                    episode_id=episode_id,
                    session_id=session_id,
                    text=chunk_text_str,
                    content_type=content_type,
                )
            )

        start = end - CHUNK_OVERLAP if end < len(text) else end

    return chunks


def chunk_episode(episode: Episode) -> list[Chunk]:
    chunks = []
    if episode.prompt_text:
        chunks.extend(
            chunk_text(
                episode.prompt_text, episode.episode_id, episode.session_id, ContentType.PROMPT
            )
        )
    if episode.response_text:
        chunks.extend(
            chunk_text(
                episode.response_text, episode.episode_id, episode.session_id, ContentType.RESPONSE
            )
        )
    return chunks
