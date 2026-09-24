"""Frozen contract types for ssgrep.

Every module in the system codes against these types. No CLI or framework
types may appear here. This file gates 10 parallel tasks in wave 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


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


class InvalidPredicateError(SearchException):
    """Raised when a metadata predicate cannot be parsed or evaluated."""

    def __init__(self, message: str, condition: str = "invalid_predicate"):
        super().__init__(message, condition, None)


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
    cli/commands/index_command.py fills this in with a corrective, NON-destructive
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


class ContentType(Enum):
    """Searchable prompt or response content."""

    PROMPT = "prompt"
    RESPONSE = "response"


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
    parent_session_id: str | None = None
    agent_type: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    agent_model: str | None = None
    project_paths: tuple[str, ...] = ()
    source_project: str | None = None
    claude_version: str | None = None
    entrypoint: str | None = None
    permission_mode: str | None = None
    user_type: str | None = None
    runtime: str = "claude"


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
    agent_model: str | None = None
    project: str | None = None
    source_path: str | None = None
    source_project: str | None = None
    claude_version: str | None = None
    entrypoint: str | None = None
    permission_mode: str | None = None
    user_type: str | None = None
    runtime: str = "claude"


@dataclass(frozen=True)
class Chunk:
    """A stable, structure-aware piece of one episode."""

    chunk_id: str
    text: str
    content_type: ContentType


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
    project: str | None = None
    source_path: str | None = None
    session_id: str | None = None
    is_subagent: bool | None = None
    agent_type: str | None = None
    agent_model: str | None = None
    tool_name: str | None = None
    runtime: str | None = None


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
    project: str | None = None
    source_path: str | None = None
    source_project: str | None = None
    git_branch: str | None = None
    agent_model: str | None = None
    source_absent: bool = False
    runtime: str = "claude"


@dataclass(frozen=True)
class SearchResponse:
    """Bounded results plus match, omission, and clamping metadata."""

    results: list[ResultCard]
    omitted_count: int = 0
    index_empty: bool = False
    total_matches: int = 0
    excerpts_truncated: bool = False
    clamped: bool = False


@dataclass(frozen=True)
class EpisodeDetail:
    """Bounded episode context for the show command."""

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
    project: str | None = None
    source_path: str | None = None
    source_project: str | None = None
    agent_model: str | None = None
    claude_version: str | None = None
    entrypoint: str | None = None
    permission_mode: str | None = None
    user_type: str | None = None
    prompt_truncated: bool = False
    response_truncated: bool = False
    runtime: str = "claude"


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
    last_optimize_time: datetime | None = None
    index_exists: bool = True
    data_dir: str | None = None
    runtime_counts: tuple[tuple[str, int], ...] = ()
