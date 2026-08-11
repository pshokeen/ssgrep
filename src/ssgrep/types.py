"""Frozen contract types for ssgrep.

Every module in the system codes against these types. No CLI or framework
types may appear here. This file gates 10 parallel tasks in wave 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict


# Exception types for degenerate query and index states
class SearchException(Exception):
    """Base exception for search operation failures."""

    def __init__(self, message: str, condition: str | None = None, command: str | None = None):
        """Initialize with message and optional machine-readable condition code."""
        super().__init__(message)
        self.condition = condition
        self.command = command


class EmptyQueryError(SearchException):
    """Raised when search receives an empty or whitespace-only query."""

    def __init__(self, message: str, condition: str = "empty_query", command: str | None = None):
        super().__init__(message, condition, command)


class IndexNotFoundError(SearchException):
    """Raised when no index exists for the project."""

    def __init__(
        self, message: str, condition: str = "missing_index", command: str = "ssgrep index"
    ):
        super().__init__(message, condition, command)


class IndexNotReadyError(SearchException):
    """Raised when index exists but is not yet ready for queries."""

    def __init__(
        self,
        message: str,
        condition: str = "corrupt_index",
        command: str | None = "ssgrep index --rebuild",
    ):
        super().__init__(message, condition, command)


class RebuildWouldShrinkError(SearchException):
    """Raised when a rebuild would replace a populated index with a much
    smaller one, and the caller has not explicitly opted in.

    Carries the old and new (session, episode, chunk) counts so the message
    can name exactly what would have been lost. Nothing has been committed
    when this is raised: the live index is still intact and still current.

    ``command`` defaults to None, and must never be defaulted to
    ``--allow-shrink``. Every other SearchException uses that field to carry a
    recovery command, so a `--json` consumer, a CI step, or a `jq -r .command
    | sh` wrapper treats it as "the thing to run next". On this one condition
    the only command that would clear the refusal is the one that discards the
    buyer's indexed history -- so defaulting to it handed every automated
    caller a loaded gun, on the single code path that had just detected
    impending data loss and stopped it. The refusal is a decision for a human;
    an honest machine-readable answer is "no command will fix this for you".
    cli/commands/index.py fills this in with a corrective, NON-destructive
    command when the census finds one.
    """

    def __init__(
        self,
        message: str,
        old_counts: tuple[int, int, int],
        new_counts: tuple[int, int, int],
        condition: str = "rebuild_would_shrink",
        command: str | None = None,
    ):
        super().__init__(message, condition, command)
        self.old_counts = old_counts
        self.new_counts = new_counts


@dataclass(frozen=True)
class ErrorResponse:
    """Structured error response for JSON mode.

    Returned from commands when an error occurs in JSON mode.
    The usecli envelope wraps this as the single JSON document.
    Preserves condition code, human message, and recovery command.
    """

    ok: bool = False
    condition: str | None = None
    message: str = ""
    command: str | None = None


class ContentType(Enum):
    """Content type for chunks — prompt or response only.

    There is no 'thinking' type: all 2135 thinking blocks in the corpus are
    empty on disk, holding only an opaque replay signature.
    """

    PROMPT = "prompt"
    RESPONSE = "response"


class Record(TypedDict, total=False):
    """A parsed record from a transcript JSONL line.

    Records are dictionaries parsed from JSONL transcript files. Each record
    represents one line of the transcript and carries a 'type' field that
    determines its semantic meaning. Records are the primary interchange
    format between records.py (line-by-line reader and classifier),
    episodes.py (episode segmentation), signal.py (signal/noise classification),
    metadata.py (metadata harvesting), and chunker.py.

    Required fields:
    - type: str — the record type (e.g. 'user', 'assistant', 'system', etc.)

    Optional fields (depending on type):
    - subtype: str — for system records (e.g. 'away_summary', 'compact_boundary')
    - [other fields]: Any — various content fields depending on record type

    Records.py yields these dicts as-is from JSONL parsing, without additional
    wrapping. The dicts represent the full parsed JSON object plus any parsed
    content blocks. Consuming modules (episodes, signal, metadata, chunker)
    read these dicts and may extract or transform their fields, but the
    interchange type remains a dict.
    """

    type: str  # Required field
    subtype: str  # Optional, used by system records
    # All other fields are optional and type-specific
    # Additional fields will be present depending on record type


@dataclass(frozen=True)
class SessionFile:
    """A discovered transcript file.

    Subagent identity derives from parent session id plus agent file identity,
    since a subagent's internal sessionId is its parent's. For subagent files,
    session_id MUST hold the derived unique document identity (computed from
    parent_session_id + agent_hash), not the raw internal sessionId; the raw
    parent sessionId is stored in parent_session_id. For main sessions,
    session_id is the transcript's internal sessionId and parent_session_id
    is None.
    """

    path: Path
    session_id: str
    is_main: bool
    size: int
    mtime: float
    parent_session_id: str | None = None
    agent_hash: str | None = None
    agent_type: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    agent_model: str | None = None


@dataclass(frozen=True)
class Episode:
    """One user prompt plus every assistant message it triggered.

    Episodes are the retrieval unit — a question like "how was this solved"
    needs the problem and its resolution together.
    """

    episode_id: str
    session_id: str
    prompt_text: str
    response_text: str
    title: str
    timestamp: datetime | None = None
    git_branch: str | None = None
    cwd: str | None = None
    files_touched: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    is_subagent: bool = False
    agent_type: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    parent_session_id: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A chunk of indexable prose, linked to its parent episode.

    Chunk ids are content-derived and therefore stable across runs.
    Target size ~1200 chars with ~200 char overlap.

    Note: byte_offset is reserved for future use; do not interpret it as a
    transcript file offset (which are disavowed by design to avoid stale
    pointers after file pruning). Current consumers should ignore this field.
    """

    chunk_id: str
    episode_id: str
    session_id: str
    text: str
    content_type: ContentType
    byte_offset: int = 0
    vec_row: int | None = None


@dataclass(frozen=True)
class FileCursor:
    """Per-file cursor for incremental append-offset re-indexing.

    Stores (path, size, mtime, byte_offset, first_line_hash) for resuming
    from where the previous index run left off.
    """

    path: Path
    size: int
    mtime: float
    byte_offset: int
    first_line_hash: str


@dataclass(frozen=True)
class SearchFilters:
    """Composable filters for search results.

    Filters compose: applying multiple filters narrows results to the
    intersection of each filter's individual criteria.
    """

    date_from: datetime | None = None
    date_to: datetime | None = None
    file_path: str | None = None
    content_type: ContentType | None = None
    branch: str | None = None


@dataclass(frozen=True)
class ResultCard:
    """Compact search result card.

    Never returns raw transcript content beyond the bounded excerpt.
    Full transcript drill-down requires a separate show() call.
    """

    ref: str
    title: str
    timestamp: datetime | None
    score: float
    excerpt: str
    files_touched: tuple[str, ...]
    is_subagent: bool
    agent_name: str | None = None
    agent_description: str | None = None
    parent_session_id: str | None = None
    content_type: ContentType | None = None
    source_absent: bool = False  # True if the source transcript has disappeared
    #: `score` divided by the ranking's theoretical ceiling (a result ranked
    #: first in every fusion leg with a maximal roll-up tail), clamped to
    #: [0, 1]. Unlike `score` -- a weighted-RRF sum whose magnitude is an
    #: artifact of the fusion constants -- this is stable across corpora and
    #: config retunes, so thresholds and cross-index comparisons can use it.
    #: The ceiling is derived from the live constants at import
    #: (search.SCORE_CEILING), never hardcoded, so a retune cannot silently
    #: skew it. 0.0 when the producing path predates the field.
    score_normalized: float = 0.0


@dataclass(frozen=True)
class SearchResponse:
    """Response envelope for search operations.

    Encapsulates results, omitted count for token-budget trimming, and
    index state to distinguish degenerate outcomes (no index, no matches,
    empty index, empty query) from successful searches. Includes total match
    count and truncation signals required by the session-search spec and
    mcp-server spec to allow callers to distinguish complete results from
    truncated ones and to know whether the requested limit was clamped to the
    hard maximum.

    The index_exists and index_empty fields work together:
    - index_exists=False: no index for this project (IndexNotFoundError raised)
    - index_exists=True, index_empty=True: index exists but contains no chunks
    - index_exists=True, index_empty=False: index exists and has chunks
      (either query matched nothing or results are in the list)
    """

    results: list[ResultCard]
    omitted_count: int = 0
    index_exists: bool = True
    index_empty: bool = False
    total_matches: int = 0
    excerpts_truncated: bool = False
    clamped: bool = False
    stale: bool = False
    stale_count: int = 0


@dataclass(frozen=True)
class EpisodeDetail:
    """Full episode context for the show command.

    This is the only command that returns untruncated transcript content.
    Carries the same stale/stale_count staleness signal as SearchResponse
    and IndexStats, since content drilled into via show() can be just as
    misleadingly out of date as a search result.
    """

    episode_id: str
    session_id: str
    title: str
    timestamp: datetime | None
    git_branch: str | None
    cwd: str | None
    prompt_text: str
    response_text: str
    files_touched: tuple[str, ...]
    tool_names: tuple[str, ...]
    is_subagent: bool
    agent_type: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    parent_session_id: str | None = None
    stale: bool = False
    stale_count: int = 0
    prompt_truncated: bool = False
    response_truncated: bool = False


@dataclass(frozen=True)
class IndexStats:
    """Index observability data for the status command.

    Reports corpus counts, freshness, model binding, and parsing degradation.
    The index_exists field distinguishes a missing or uninitialized index from
    one that exists but contains no data.
    """

    session_count: int
    episode_count: int
    chunk_count: int
    index_size_bytes: int
    last_index_time: datetime | None
    model_id: str
    vector_dimension: int
    skipped_records: int
    malformed_records: int
    schema_version: int
    tombstoned_source_count: int = 0
    tombstoned_chunk_count: int = 0
    index_exists: bool = True
    stale: bool = False
    stale_count: int = 0
    cwd_cache_degraded: bool = False
    cwd_cache_fallback_scans: int = 0
    #: Work-queue hints the drain's fail-closed scope filter dropped this
    #: run (completed without indexing: the transcript recorded no cwd at or
    #: beneath the index's effective scope). Zero in normal operation; a
    #: non-zero value means something enqueued out-of-scope hints -- an old
    #: binary, a foreign hook, or a crafted item -- and the filter did its
    #: job. Surfaced per the 2026-08-07 blind-review nit: the drop was
    #: counted internally but observable nowhere.
    queue_items_out_of_scope: int = 0
    #: Sessions from the auto-discovered corpus only (scope-matched transcripts).
    #: Excludes index-owned notes and SSGREP_TRANSCRIPT_DIRS external roots, which
    #: are always included regardless of scope. Used to gate scope-diagnostic
    #: heuristics that only make sense when the corpus count is small -- external
    #: content must not trigger "suspiciously small discovery" warnings.
    corpus_session_count: int = 0
