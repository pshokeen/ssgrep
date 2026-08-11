"""Session discovery for Claude Code transcripts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ssgrep import paths, store
from ssgrep.types import SessionFile

# In-process cache for the cwd index: cache_key -> (file_path -> set of cwds).
# Lives for the lifetime of one process (a single CLI invocation never sees
# the corpus change mid-command), so once built it is never invalidated
# here. Cross-process persistence -- the part that actually matters, since
# every CLI invocation is a fresh process -- is _build_cwd_index's job via
# the cwd_projection cache database (see _cwd_cache_db_path).
_cwd_index_cache: dict[str, dict[str, set[str]]] = {}
_cwd_index_lock = threading.Lock()

# Monotonic statistics for fallback scans: incremented when a file is not in
# the cache and must be scanned inline. Tracks degradation visibility; never reset
# during a process lifetime. Used in tests to verify cache is being used.
_cwd_index_stats = {"fallback_scans": 0}


def encode_path(path: Path) -> str:
    """Forward-only encoding: / and . become -"""
    return str(path).replace("/", "-").replace(".", "-")


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
        # A cwd containing a newline is dropped rather than carried. It cannot
        # survive this module's own storage anyway: the persisted cwd cache
        # round-trips the set through "\n".join(...) / .split("\n"), so such a
        # value is silently shredded into several bogus entries on the second
        # run. Carrying it forward was also an injection vector -- scope_report
        # interpolates recorded cwds into the remedy message, and a multi-line
        # value forged a complete second "Point ssgrep there:" block above the
        # genuine one, with an arbitrary command in it. Dropping at the source
        # is the honest resolution: the value was never usable, and the
        # histogram no longer has to be trusted to neutralize it.
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
    """Every transcript file under claude_dir that discovery considers at all.

    The single enumeration used by _build_cwd_index, discover_sessions, and
    scope_report's census. That sharing is the point, not tidiness: the
    diagnostic reports "N transcripts exist under this root, R of them were
    rejected by scope", and those numbers are only true if the census walks
    exactly the set discovery walks. A second, independently written rglob
    with its own copy of the sidecar exclusions would drift from this one
    the first time an exclusion rule changed, and the diagnostic would then
    confidently state a total the indexer never saw.
    """
    for jsonl_file in claude_dir.rglob("*.jsonl"):
        # Position-aware sidecar exclusion (not "any component").
        if _should_exclude_path(jsonl_file, claude_dir):
            continue
        if jsonl_file.suffix == ".json":
            continue
        yield jsonl_file


@dataclass(frozen=True)
class _CwdCacheEntry:
    """One file's persisted cwd-projection evidence (D11).

    Mirrors FileCursor's (size, mtime, byte_offset, first_line_hash) cursor
    shape. byte_offset is how far the incremental line-reader got, which can
    trail size by a few bytes when the last line on disk isn't newline-
    terminated yet (an in-progress write, picked up whole next time --
    mirrors indexer._read_from_offset). cwds is the accumulated evidence:
    every distinct cwd value seen among the file's records up to byte_offset.
    """

    size: int
    mtime: float
    byte_offset: int
    first_line_hash: str
    cwds: frozenset[str]


def _cwd_cache_db_path(claude_dir: Path) -> Path:
    """Where the persistent cwd-projection cache lives for this claude_dir.

    A sibling of claude_dir, never inside it: claude_dir.rglob("*.jsonl")
    (both here and in discover_sessions' own scan) must never see it. It is
    not nested under any one project's .ssgrep/ either, since the
    projection it caches spans the whole corpus (D11), not one project.
    """
    return claude_dir.parent / "ssgrep-cache" / store.CWD_CACHE_DB_NAME


def _first_line_hash(path: Path) -> str:
    """sha256 of the raw first line: the Rewrite/Truncation Guard's content
    fingerprint. Same approach as indexer._first_line_hash, but a separate
    implementation on purpose -- indexer.py already imports this module, so
    the reverse import would cycle, and the two caches serve different
    purposes and need not be coupled.
    """
    with open(path, "rb") as f:
        first_line = f.readline()
    return hashlib.sha256(first_line).hexdigest()


def _read_cwds_from_offset(path: Path, start_offset: int) -> tuple[set[str], int]:
    """Extract cwd values from every complete line at or after start_offset.

    Returns (cwds_found, end_offset). A final line with no trailing newline
    yet is left unconsumed -- end_offset stops before it -- so a file still
    being written is picked up whole on a later scan rather than parsed
    half-written, mirroring indexer._read_from_offset exactly.
    """
    cwds: set[str] = set()
    pos = start_offset
    with open(path, "rb") as f:
        f.seek(start_offset)
        for raw_line in f:
            if not raw_line.endswith(b"\n"):
                break
            pos += len(raw_line)
            line = raw_line.decode("utf-8", errors="replace")
            cwds.update(_extract_cwds_from_line(line))
    return cwds, pos


def _rescan_cwd_file(
    path: Path, disk_size: int, disk_mtime: float, cached: _CwdCacheEntry | None
) -> _CwdCacheEntry:
    """Refresh one file's cwd evidence, extending the cached evidence rather
    than rebuilding from scratch whenever it is safe to do so.

    The Rewrite/Truncation Guard (the same rule indexer._needs_full_reparse
    applies to session_files): resuming from the cached byte_offset and
    UNIONING in newly found cwds is only safe if the file has grown-or-held
    (disk_size >= cached.size) AND its first line still hashes the same.
    Otherwise the file was truncated and rewritten out from under the old
    evidence, which is discarded rather than merged, and the file is
    rescanned whole from byte 0.
    """
    current_hash = _first_line_hash(path)

    can_extend = (
        cached is not None and disk_size >= cached.size and current_hash == cached.first_line_hash
    )

    if can_extend:
        assert cached is not None  # narrows for mypy; can_extend already checked this
        new_cwds, end_offset = _read_cwds_from_offset(path, cached.byte_offset)
        merged_cwds = set(cached.cwds) | new_cwds
    else:
        merged_cwds, end_offset = _read_cwds_from_offset(path, 0)

    return _CwdCacheEntry(
        size=disk_size,
        mtime=disk_mtime,
        byte_offset=end_offset,
        first_line_hash=current_hash,
        cwds=frozenset(merged_cwds),
    )


def _load_persisted_cwd_cache(
    claude_dir: Path,
) -> tuple[sqlite3.Connection | None, dict[str, _CwdCacheEntry]]:
    """Best-effort open of the persistent cache.

    Any failure (missing home directory, unwritable disk, corrupt db)
    degrades to "no cache" rather than failing discovery -- the cache is a
    speed optimization, never a correctness dependency. A cold or absent
    cache falls back to exactly the full scan discovery has always done.
    """
    try:
        conn = store.init_cwd_cache(_cwd_cache_db_path(claude_dir))
        rows = store.load_cwd_cache_rows(conn)
    except (sqlite3.Error, OSError):
        return None, {}

    cached = {
        path: _CwdCacheEntry(
            size=size,
            mtime=mtime,
            byte_offset=byte_offset,
            first_line_hash=first_line_hash,
            cwds=frozenset(cwds_blob.split("\n")) if cwds_blob else frozenset(),
        )
        for path, size, mtime, byte_offset, first_line_hash, cwds_blob in rows
    }
    return conn, cached


def _build_cwd_index(claude_dir: Path) -> dict[str, set[str]]:
    """Build the complete cwd index, reusing persisted per-file evidence.

    Returns a dict mapping file path strings to sets of cwd values found in
    that file -- same contract as always. What changed is how each file's
    evidence is obtained: a file whose size AND mtime both match its cached
    row is trusted with zero file I/O; a changed file is extended or fully
    rescanned (see _rescan_cwd_file); a file with no cached row is scanned
    in full, exactly as before. D11 requires walking every file under
    claude_dir regardless of which project's scope triggered the call, so
    the persisted cache is likewise global (see _cwd_cache_db_path): one
    process's first search warms every later search, of any project.
    """
    conn, cached = _load_persisted_cwd_cache(claude_dir)

    index: dict[str, set[str]] = {}
    changed_rows: list[tuple[str, int, float, int, str, str]] = []
    seen_paths: set[str] = set()

    for jsonl_file in iter_transcript_files(claude_dir):
        path_str = str(jsonl_file)
        try:
            disk_stat = jsonl_file.stat()
        except OSError:
            # Vanished between enumeration and stat: normal skip. Omitted
            # from seen_paths, so any stale cache row for it is pruned
            # below rather than trusted.
            continue

        seen_paths.add(path_str)
        disk_size, disk_mtime = disk_stat.st_size, disk_stat.st_mtime
        cached_entry = cached.get(path_str)

        if (
            cached_entry is not None
            and cached_entry.size == disk_size
            and cached_entry.mtime == disk_mtime
        ):
            # Unchanged since the cache was written: trust it, no file I/O.
            # Store the cached cwds even if empty -- an empty file should not
            # trigger a fallback rescan on every call.
            index[path_str] = set(cached_entry.cwds)
            continue

        try:
            new_entry = _rescan_cwd_file(jsonl_file, disk_size, disk_mtime, cached_entry)
        except OSError:
            # Vanished or became unreadable between stat and open: normal
            # skip. (File may have been deleted or become inaccessible.)
            continue

        # Store the cwds even if empty -- an empty file should not trigger
        # a fallback rescan on every call. The persistence layer (changed_rows)
        # runs unconditionally, so empty-cwd files are still persisted.
        index[path_str] = set(new_entry.cwds)
        changed_rows.append(
            (
                path_str,
                new_entry.size,
                new_entry.mtime,
                new_entry.byte_offset,
                new_entry.first_line_hash,
                "\n".join(sorted(new_entry.cwds)),
            )
        )

    if conn is not None:
        try:
            store.save_cwd_cache_rows(conn, changed_rows)
            stale_paths = [p for p in cached if p not in seen_paths]
            store.delete_cwd_cache_rows(conn, stale_paths)
        except sqlite3.Error:
            # Persistence is best-effort; the in-memory index built above is
            # already correct and is what this call returns regardless.
            pass
        finally:
            conn.close()

    return index


def cwd_index_for(claude_dir: Path) -> dict[str, set[str]]:
    """The process-wide cwd index for claude_dir, building it at most once.

    Shared by discover_sessions and scope_report so a zero-discovery
    diagnostic explains the *same* evidence the scan actually used, and does
    not pay to rebuild it: on the index command's zero path this cache is
    already warm from the discover_sessions call that returned nothing.
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
        cwds: set[str] = set()
        try:
            with open(jsonl_file, errors="replace") as f:
                for line in f:
                    cwds.update(_extract_cwds_from_line(line))
        except OSError:
            return False

        # Increment fallback counter under the same lock that guards the cache.
        with _cwd_index_lock:
            _cwd_index_stats["fallback_scans"] += 1
            # Memoize the result even if empty, so this file is not rescanned
            # on every call. An empty set means no cwds were found, which is
            # distinct from "not yet scanned" (not in the dict).
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
    project_dir: Path,
    scope: str | None = None,
    no_subagents: bool = False,
) -> list[SessionFile]:
    """Discover transcript files under Claude Code's projects/ directory.

    When scope is provided, discovers only files whose records contain a cwd
    that is at or beneath the given scope path. When scope is None, defaults
    to filtering by project_dir. This is a full streaming scan of transcript
    records and is cached.
    """
    claude_dir = paths.resolve_claude_dir() / "projects"
    if not claude_dir.exists():
        return []

    # If scope not explicitly provided, default to the project_dir
    if scope is None:
        scope = str(project_dir)

    sessions = []

    # Build or use cached cwd index if scope filtering is needed
    cwd_index: dict[str, set[str]] = {}
    if scope:
        cwd_index = cwd_index_for(claude_dir)

    for jsonl_file in iter_transcript_files(claude_dir):
        if scope:
            if not _file_matches_scope(jsonl_file, scope, cwd_index):
                continue

        is_main, parent_id, agent_hash = classify_session(jsonl_file, claude_dir)

        if no_subagents and not is_main:
            continue

        try:
            stat = jsonl_file.stat()
        except FileNotFoundError:
            continue

        session_id = jsonl_file.stem
        if not is_main and agent_hash and parent_id:
            session_id = f"{parent_id}:agent:{agent_hash}"

        sessions.append(
            SessionFile(
                path=jsonl_file,
                session_id=session_id,
                is_main=is_main,
                size=stat.st_size,
                mtime=stat.st_mtime,
                parent_session_id=parent_id,
                agent_hash=agent_hash,
            )
        )

    return sessions
