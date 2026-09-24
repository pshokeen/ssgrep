"""Session discovery for Claude Code transcripts."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

from ssgrep.store import CWD_CACHE_TABLE, LanceStore
from ssgrep.utilities import paths
from ssgrep.utilities.types import SessionFile

# In-process cache for the cwd index: cache_key -> (file_path -> set of cwds).
# Lives for the lifetime of one process (a single CLI invocation never sees
# the corpus change mid-command), so once built it is never invalidated
# here. A LanceDB table provides cross-process persistence.
_cwd_index_cache: dict[str, dict[str, set[str]]] = {}
_cwd_index_lock = threading.Lock()


def _extract_cwds_from_line(line: str) -> set[str]:
    """Extract cwd values from a single JSON line.

    Returns a set of cwds found in this line (zero or one element).
    Skips blank lines and malformed JSON; continues on parse errors.
    """
    cwds: set[str] = set()
    line = line.strip()
    if not line:
        return cwds
    try:
        record = json.loads(line)
        cwd = record.get("cwd") if isinstance(record, dict) else None
        # Newlines cannot round-trip through the newline-delimited cache value.
        if isinstance(cwd, str) and "\n" not in cwd:
            cwds.add(cwd)
    except json.JSONDecodeError:
        pass
    return cwds


def _should_exclude_path(jsonl_file: Path, claude_dir: Path) -> bool:
    """Check if a path should be excluded from discovery.

    Position-aware exclusions, each checked at the depth it is actually
    written at -- confirmed against the live ~/.claude/projects corpus
    (2026-07-28), not assumed to share one depth:
    - <projectDir>/memory/ and its contents (parts[1], a direct child of the
      *project* directory) -- one memory store per project, shared across
      all its sessions. Every observed instance sits here; none is nested
      under a session.
    - <projectDir>/<sessionId>/tool-results/ and its contents (parts[2], a
      direct child of the *session* directory -- one level deeper than
      memory/). Every observed instance sits here, with zero exceptions;
      none is at the project level the way memory/ is.
    - <projectDir>/<sessionId>/workflows/ and its contents (parts[2]), but
      NOT <projectDir>/<sessionId>/subagents/workflows/ -- that is real
      subagent transcript data, one level deeper still (parts[3], under
      "subagents"), and must stay included.

    Returns True if the file should be excluded.
    """
    try:
        relative = jsonl_file.relative_to(claude_dir)
    except ValueError:
        return False

    parts = relative.parts

    # "memory" is a direct child of the project directory: <projectDir>/memory/...
    if len(parts) >= 2 and parts[1] == "memory":
        return True

    # "tool-results" and "workflows" are direct children of the *session*
    # directory, one level deeper than "memory": <projectDir>/<sessionId>/....
    # parts[1] here is the sessionId itself, never "tool-results" or
    # "workflows" literally -- checking them at parts[1] (memory's depth)
    # would silently never match, which was the bug.
    if len(parts) >= 3 and parts[2] in ("tool-results", "workflows"):
        return True

    # subagents/ (including subagents/workflows/, at parts[3]) is included.
    return False


def iter_transcript_files(claude_dir: Path) -> Iterator[Path]:
    """Yield transcript files considered by discovery and cwd projection."""
    for jsonl_file in claude_dir.rglob("*.jsonl"):
        # Position-aware sidecar exclusion (not "any component").
        if _should_exclude_path(jsonl_file, claude_dir):
            continue
        if jsonl_file.suffix == ".json":
            continue
        yield jsonl_file


def _scan_cwds(path: Path) -> set[str]:
    cwds: set[str] = set()
    with path.open(errors="replace") as stream:
        for line in stream:
            cwds.update(_extract_cwds_from_line(line))
    return cwds


def _build_cwd_index(claude_dir: Path) -> dict[str, set[str]]:
    """Load the global Lance projection and rescan only changed sources."""
    cached: dict[str, dict] = {}
    repository = LanceStore()
    if repository.exists():
        cached = {
            str(row["path"]): row for row in repository.rows(CWD_CACHE_TABLE, limit=1_000_000)
        }
    index: dict[str, set[str]] = {}
    for jsonl_file in iter_transcript_files(claude_dir):
        try:
            disk_stat = jsonl_file.stat()
        except OSError:
            continue
        existing = cached.get(str(jsonl_file))
        if (
            existing
            and int(existing["size"]) == disk_stat.st_size
            and float(existing["mtime"]) == disk_stat.st_mtime
        ):
            value = str(existing.get("cwds") or "")
            index[str(jsonl_file)] = set(value.split("\n")) if value else set()
            continue
        try:
            index[str(jsonl_file)] = _scan_cwds(jsonl_file)
        except OSError:
            continue
    return index


def cwd_index_for(claude_dir: Path) -> dict[str, set[str]]:
    """The process-wide cwd index for claude_dir, building it at most once.

    Shared by discovery calls so each process scans the corpus at most once.
    """
    with _cwd_index_lock:
        cache_key = str(claude_dir)
        if cache_key not in _cwd_index_cache:
            _cwd_index_cache[cache_key] = _build_cwd_index(claude_dir)
        return _cwd_index_cache[cache_key]


def _scope_matches_cwd(cwd: str, scope: str) -> bool:
    """Check if a cwd matches the requested scope.

    A cwd matches if it is exactly the scope or is a subdirectory of the scope.

    Both sides go through the single canonicalization in ssgrep.paths --
    never one side only. The cwd is a *historical* string read out of a
    transcript, so it is canonicalized lexically and never resolve()d: the
    directory it names may not exist any more, and resolve() would then
    invent a path anchored to the current working directory. The scope
    arrives already resolve()d (paths.resolve_live), and canonical() is
    idempotent, so canonicalizing it again here is free and makes this
    function correct for any caller, not just the CLI.
    """
    return paths.is_at_or_beneath(cwd, scope)


def _file_matches_scope(jsonl_file: Path, scope: str, cwd_index: dict[str, set[str]]) -> bool:
    """Check if a file's cwds match the requested scope."""
    file_path_str = str(jsonl_file)

    if file_path_str not in cwd_index:
        # File not in index, need to scan it (fallback).
        # This is a cache miss; the file is scanned inline rather than using
        # the pre-built index. Increments the fallback counter under lock.
        try:
            cwds = _scan_cwds(jsonl_file)
        except OSError:
            return False
        with _cwd_index_lock:
            cwd_index[file_path_str] = cwds

        if not cwds:
            return False

    # Check if any cwd in this file matches the scope
    for cwd in cwd_index[file_path_str]:
        if _scope_matches_cwd(cwd, scope):
            return True

    return False


def classify_session(path: Path, claude_dir: Path) -> tuple[bool, str | None, str | None]:
    """Classify main vs subagent by path position.

    Returns (is_main, parent_session_id, agent_hash).
    Classification is based on path depth and structure, not filename:
    - Main session: <projectDir>/<sessionId>.jsonl (depth 2 under claude_dir)
    - Subagent/workflow: <projectDir>/<sessionId>/subagents/**/*.jsonl

    For subagents, parent_session_id is the session directory name (parts[1]).
    For agent_hash, derive from the relative path starting from "subagents/" to
    ensure uniqueness even when the same filename appears at different paths.
    """
    try:
        relative = path.relative_to(claude_dir)
    except ValueError:
        # File is outside claude_dir, treat as main
        return True, None, None

    parts = relative.parts

    # Main session: exactly 2 parts (projectDir, sessionId.jsonl)
    if len(parts) == 2:
        return True, None, None

    # Subagent: at least 4 parts and must contain "subagents"
    if len(parts) >= 4 and "subagents" in parts:
        # parts[0] = projectDir (encoded)
        # parts[1] = sessionId (the UUID)
        # parts[2+] = subagent path (subagents/...)
        parent_session_id = parts[1]

        # Derive agent_hash from path relative to session directory.
        # This ensures unique identities even when the same filename appears
        # at different paths (e.g., agent-X at both subagents/ and subagents/workflows/wf_Y/).
        filename = path.stem
        subagents_idx = parts.index("subagents")
        # Build path from "subagents" onwards, using filename stem for the last component
        remaining_parts = list(parts[subagents_idx:])
        # Replace the last part (filename with extension) with the stem
        remaining_parts[-1] = filename
        agent_hash = "-".join(remaining_parts)

        return False, parent_session_id, agent_hash

    # Not main and not under subagents, treat as main (default)
    return True, None, None


def discover_sessions(
    scope: str | None = None,
    no_subagents: bool = False,
) -> list[SessionFile]:
    """Discover Claude Code transcripts across every recorded project.

    Passing ``scope`` narrows ingestion by recorded cwd; the default discovers
    the entire Claude transcript corpus.
    """
    claude_dir = paths.resolve_claude_dir() / "projects"
    if not claude_dir.exists():
        return []

    cwd_index = cwd_index_for(claude_dir)
    sessions: list[SessionFile] = []
    for jsonl_file in iter_transcript_files(claude_dir):
        if scope and not _file_matches_scope(jsonl_file, scope, cwd_index):
            continue

        is_main, parent_id, agent_hash = classify_session(jsonl_file, claude_dir)
        if no_subagents and not is_main:
            continue
        session_id = jsonl_file.stem
        if not is_main and agent_hash and parent_id:
            session_id = f"{parent_id}:agent:{agent_hash}"
        try:
            source_project = jsonl_file.relative_to(claude_dir).parts[0]
        except (ValueError, IndexError):
            source_project = jsonl_file.parent.name

        sessions.append(
            SessionFile(
                path=jsonl_file,
                session_id=session_id,
                is_main=is_main,
                parent_session_id=parent_id,
                project_paths=tuple(sorted(cwd_index.get(str(jsonl_file), set()))),
                source_project=source_project,
            )
        )
    return sessions
