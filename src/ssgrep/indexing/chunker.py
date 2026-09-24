"""Chunk transcript episodes with Chonkie."""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache

from chonkie import OverlapRefinery, RecursiveChunker

from ssgrep.indexing.embed import MODEL_ID, MODEL_REVISION
from ssgrep.utilities.types import Chunk, ContentType, Episode

logger = logging.getLogger(__name__)

# The default model's context window is 299 model tokens (tokenizer.
# model_max_length).  The chunker guarantees a chunk plus its prefix overlap
# stays within the window (OverlapRefinery prepends up to the resolved overlap
# tokens).  The pipeline's ``search_text`` adds a title prefix on top of the
# chunk, and ``rows._contextual_search_text`` trims that title so the full
# embedded sequence still fits the model window.  CHUNK_TOKEN_BUDGET is sized
# so that budget + overlap stays under 299 with headroom -- at the shipped
# defaults 235 + 25 = 260 <= 299 -- keeping the chunk content itself
# embeddable.  Recursive splitting keeps paragraphs and sentences intact.
CHUNK_TOKEN_BUDGET = 235
#: Shipped default from the T6 overlap sweep (harness output is not kept in the repo):
#: 25 kept every measured quality proxy at or above the overlap-50 run while
#: cutting stored token vectors 7.4%; 12 was rejected (recall -0.07, MRR -0.05).
CHUNK_TOKEN_OVERLAP = 25
#: Operator overrides via SSGREP_CHUNK_OVERLAP are clamped to
#: [0, CHUNK_TOKEN_OVERLAP_MAX]; the ceiling caps prefix duplication at half
#: the chunk budget (values between 50 and 117 trade window headroom for
#: duplication and remain legal operator choices).
CHUNK_TOKEN_OVERLAP_MAX = CHUNK_TOKEN_BUDGET // 2
_HASH_LEN = 16

_chunker = None
_overlap = None

#: Transformers emits this warning when a tokenizer is asked to encode a
#: sequence longer than the model's window.  Chonkie deliberately tokenizes
#: the FULL episode text to find chunk boundaries, which legitimately exceeds
#: the 299-token window, but that full text is never fed to the model — only
#: the small chunks are embedded.  The warning is therefore expected noise, so
#: it is filtered to avoid alarming users at the start of an index run.
_OVERLONG_TOKEN_WARNING = (
    "Token indices sequence length is longer than the specified maximum sequence length"
)


class _FullTextTokenizeFilter(logging.Filter):
    """Drop transformers' over-long-sequence warning emitted during chunking."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _OVERLONG_TOKEN_WARNING not in record.getMessage()


def _suppress_overlong_warning() -> None:
    """Idempotently silence the benign full-text tokenization warning."""
    logger = logging.getLogger("transformers.tokenization_utils_base")
    if not any(isinstance(f, _FullTextTokenizeFilter) for f in logger.filters):
        logger.addFilter(_FullTextTokenizeFilter())


_suppress_overlong_warning()


@lru_cache(maxsize=1)
def _tokenizer():
    """Return a cached tokenizer for the embedding model, without loading it."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION or None)


def _resolved_overlap() -> int:
    """Resolve SSGREP_CHUNK_OVERLAP, clamped to [0, CHUNK_TOKEN_OVERLAP_MAX]."""
    raw = os.environ.get("SSGREP_CHUNK_OVERLAP", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return CHUNK_TOKEN_OVERLAP
    clamped = max(0, min(CHUNK_TOKEN_OVERLAP_MAX, value))
    if clamped != value:
        logger.warning(
            "SSGREP_CHUNK_OVERLAP=%s out of range [0, %d]; clamped to %d",
            raw,
            CHUNK_TOKEN_OVERLAP_MAX,
            clamped,
        )
    return clamped


def _get_chunker() -> RecursiveChunker:
    global _chunker
    if _chunker is None:
        _chunker = RecursiveChunker(
            tokenizer=_tokenizer(),
            chunk_size=CHUNK_TOKEN_BUDGET,
        )
    return _chunker


def _get_overlap() -> OverlapRefinery:
    global _overlap
    if _overlap is None:
        _overlap = OverlapRefinery(
            tokenizer=_tokenizer(),
            context_size=_resolved_overlap(),
            mode="token",
            method="prefix",
            merge=True,
            inplace=False,
        )
    return _overlap


def _chunk_id(episode_id: str, content_type: ContentType, start: int, text: str) -> str:
    """Return a stable, content-derived identifier for a Chonkie chunk."""
    digest = hashlib.sha256(f"{start}:{text}".encode()).hexdigest()[:_HASH_LEN]
    return f"{episode_id}:{content_type.value}:{digest}"


def chunk_text(text: str, episode_id: str, content_type: ContentType) -> list[Chunk]:
    """Split one prompt or response on structure, then add retrieval context."""
    if not text or not text.strip():
        return []

    chunks: list[Chunk] = []
    pieces = _get_overlap()(_get_chunker().chunk(text))
    for piece in pieces:
        value = piece.text.strip()
        if not value:
            continue
        start = piece.start_index
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(episode_id, content_type, start, value),
                text=value,
                content_type=content_type,
            )
        )
    return chunks


def chunk_episode(episode: Episode) -> list[Chunk]:
    """Chunk prompt and response independently so content filters stay exact."""
    chunks: list[Chunk] = []
    chunks.extend(
        chunk_text(
            episode.prompt_text,
            episode.episode_id,
            ContentType.PROMPT,
        )
    )
    chunks.extend(
        chunk_text(
            episode.response_text,
            episode.episode_id,
            ContentType.RESPONSE,
        )
    )
    return chunks
