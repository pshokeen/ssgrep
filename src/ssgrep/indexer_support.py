"""Support routines for indexer.index() runs.

Extracted from indexer.py to keep that module under the repo's file-size
gate: these two helpers are self-contained side quests of an index run
(draining SessionEnd hook hints, and keeping .ssgrep/ out of the working
tree), not part of the core discovery/parse/embed/commit pipeline.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ssgrep import discovery, paths, store, vectors
from ssgrep.workqueue import WorkQueue

if TYPE_CHECKING:
    from ssgrep.indexer import _RunStats


def drain_workqueue(
    conn: sqlite3.Connection,
    index_dir: Path,
    project_dir: Path,
    vec_store: vectors.VectorStore,
    stats: _RunStats,
    quiet: bool,
    scope: str | None = None,
) -> None:
    """Drain pending work items from the queue, as per D13 reconciliation.

    This processes hints enqueued by SessionEnd hooks. Items are not source of
    truth — they are just hints. A corrupt or unreadable queue is a no-op with
    a stderr note; an empty queue is silent. All claimed items are re-tried to
    handle crashes mid-processing.

    `scope` is the index's effective scope (indexer.index passes the same
    value it discovers with). Queue items whose transcript records no cwd at
    or beneath it are DROPPED (completed without indexing), with a counted
    stat: the enqueue side already filters by the persisted scope, but the
    drain must not trust the queue -- items enqueued by an older binary, a
    concurrently-running old process, or a hook pointed here from another
    project would otherwise smuggle out-of-scope sessions into a scoped
    index with no rebuild and no consent (a unanimous blind-review blocker:
    both reviewers reproduced session_count 3->4 on a scoped index from the
    plain MCP-startup call shape). Note shards under this index's own
    notes/ dir are always in scope (index-owned; see notes.py). scope=None
    preserves the legacy unfiltered drain for callers that predate scoping.

    Args:
        conn: Open database connection for writing indexed data.
        index_dir: Path to .ssgrep/ directory where workqueue.db lives.
        project_dir: Project directory for discovery if needed.
        vec_store: Open vector store for embeddings.
        stats: RunStats to accumulate counts.
        quiet: Suppress status messages.
        scope: Effective index scope for item filtering (None = no filter).
    """
    queue = WorkQueue(index_dir)
    try:
        queue.open()
    except Exception:
        # Queue is optional; corrupt or unreadable queue is a no-op.
        print("ssgrep: workqueue unreadable, skipping drain", file=sys.stderr)
        return

    try:
        # Collect all items to drain: unclaimed + abandoned (claimed but incomplete).
        # Abandoned items are re-tried; they were left in a claimed state by a
        # process that crashed before marking them complete.
        to_process = queue.pending() + queue.abandoned()
        if not to_process:
            return  # empty queue is completely normal and silent

        if not quiet:
            print(f"ssgrep: draining {len(to_process)} queued work item(s)", file=sys.stderr)

        for item in to_process:
            try:
                # Claim the item (or re-claim if it was abandoned).
                # If it's already claimed, this is a no-op (pending() returns only
                # unclaimed items, and abandoned() already have claimed_at set).
                if item.claimed_at is None:
                    claimed = queue.claim()
                    if claimed is None:
                        continue
                    item = claimed

                # Index the session file. If the file is gone or unreadable,
                # _index_file silently skips it; that's normal behavior.
                # If indexing succeeds, mark the item complete.
                # CRITICAL: Do work BEFORE calling complete(). This claim-then-delete
                # ordering ensures that if this process crashes mid-work, the item
                # remains in claimed state and can be retried on recovery. If a
                # mutation test swaps this order, the item is lost on any crash and
                # test_crashed_drain_leaves_item_replayable will fail (it verifies
                # that abandoned items can be re-tried). Tested by mutation.
                try:
                    session_path = Path(item.transcript_path)
                    # Classify the session (main vs subagent) based on path structure.
                    # Queue items come from discover_sessions, which includes both
                    # main and subagent sessions, so we must classify them properly.
                    # Same claude_dir discover_sessions uses: one resolver, so
                    # CLAUDE_CONFIG_DIR cannot split classification from discovery.
                    try:
                        is_main, parent_session_id, agent_hash = discovery.classify_session(
                            session_path, paths.resolve_claude_dir() / "projects"
                        )
                    except Exception:
                        # Fallback: treat as main session if classification fails
                        is_main, parent_session_id, agent_hash = True, None, None

                    if scope is not None and not _item_in_scope(session_path, index_dir, scope):
                        # Out-of-scope hint: drop it (complete without
                        # indexing) so it neither pollutes the scoped index
                        # nor wedges the queue by being retried forever.
                        stats.queue_items_out_of_scope += 1
                        queue.complete(item.item_id)
                        continue

                    session = discovery.SessionFile(
                        path=session_path,
                        session_id=item.session_id,
                        is_main=is_main,
                        parent_session_id=parent_session_id,
                        agent_hash=agent_hash,
                        size=0,  # will be re-stat'd in _index_file
                        mtime=0.0,  # will be re-stat'd in _index_file
                    )
                    # Imported lazily: indexer imports this module at load time, and
                    # the file-indexing core stays in indexer.py.
                    from ssgrep.indexer import _index_file

                    _index_file(conn, vec_store, session, stats=stats)
                    queue.complete(item.item_id)
                except Exception:
                    # Indexing failure: leave the item in claimed state
                    # so it can be re-tried on the next run.
                    pass
            except Exception:
                # Claim or complete failed: move to next item.
                # The exception handler above will have left any mid-process
                # items in a claimed state for recovery.
                pass
    finally:
        queue.close()


def _item_in_scope(session_path: Path, index_dir: Path, scope: str) -> bool:
    """Whether a queued transcript belongs to this index's scope.

    Index-owned note shards (under this index's notes/ dir) are always in
    scope. Everything else must record at least one cwd at or beneath the
    scope, resolved through the same cwd index discovery itself uses -- one
    matcher, so the drain can never disagree with discovery about what
    belongs. Unreadable/unknown files fail CLOSED (not in scope): a hint the
    index cannot attribute must not be indexed into a scoped index.
    """
    try:
        if session_path.is_relative_to(index_dir / "notes"):
            return True
    except (ValueError, OSError):
        pass
    try:
        claude_dir = paths.resolve_claude_dir() / "projects"
        cwd_index = discovery.cwd_index_for(claude_dir)
        cwds = cwd_index.get(str(session_path), set())
        return any(discovery._scope_matches_cwd(cwd, scope) for cwd in cwds)
    except Exception:
        return False


def add_to_gitignore(project_dir: Path, entry: str = ".ssgrep/") -> None:
    """Add an entry to .gitignore if not already present.

    This is idempotent: calling it multiple times with the same entry
    will not duplicate the line.

    Does nothing when project_dir does not exist. It used to `mkdir -p` it
    first, which meant the census's own moved-project remedy
    (``ssgrep index --project-dir /the/old/path``) RESURRECTED the directory
    the buyer had deleted or moved away from and dropped a .gitignore into
    it. Creating source-tree files in a directory the user removed is not
    this function's business: .gitignore exists to keep .ssgrep/ out of a
    working tree, and a path with no working tree has nothing to protect.
    The index directory itself is still created where it is needed, by
    store.init_db() and vectors.open_vectors(), so the remedy keeps working.

    Args:
        project_dir: Path to the project directory.
        entry: The entry to add to .gitignore (e.g., ".ssgrep/").
    """
    if not project_dir.is_dir():
        return

    gitignore_path = project_dir / ".gitignore"

    # Read existing content if .gitignore exists
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        lines = content.splitlines(keepends=False)
    else:
        lines = []

    # Check if entry already exists (exact match)
    if entry in lines or entry.rstrip("/") in lines:
        # Already present, nothing to do
        return

    # Append the entry with a newline
    if lines and not content.endswith("\n"):
        lines.append("")  # Add blank line separator if needed

    lines.append(entry)

    # Write back to .gitignore
    gitignore_path.write_text("\n".join(lines) + "\n")


def read_persisted_scope(db_path: Path) -> str | None:
    """The scope an existing index was built with (meta key), or None.

    THE single implementation -- indexer.index()'s effective-scope default
    and search.staleness_summary()'s discovery scope both import this, so
    the two consumers can never drift apart (a blind review found them as
    verbatim duplicates; a divergence here is exactly the
    scope-reversion-class bug it flagged). Best-effort and never raises: a
    missing index, a pre-scope index (no meta key), or an unreadable db all
    mean "no persisted scope", which callers treat as the legacy default
    (project_dir).
    """
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            return store.get_meta(conn, "scope")
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def effective_scope_from(explicit: str | None, persisted: str | None, project_dir: Path) -> str:
    """THE precedence rule, in one place: explicit --scope, else the
    persisted scope, else project_dir. Every consumer must route through
    this (or effective_scope() below) -- two independently-maintained
    expressions of this rule were exactly the bug class two blind-review
    rounds found live (a call site deriving scope differently than the
    canonical path)."""
    if explicit is not None:
        return explicit
    if persisted is not None:
        return persisted
    return str(project_dir)


def effective_scope(project_dir: Path, explicit: str | None = None) -> str:
    """Convenience form: reads the persisted scope from the project's
    default index location. For call sites that don't already hold the
    persisted value (CLI diagnostics, MCP tools, search's zero-discovery
    paths)."""
    try:
        persisted = read_persisted_scope(
            store.GenerationalStore(project_dir / ".ssgrep").get_index_path()
        )
    except Exception:
        persisted = None
    return effective_scope_from(explicit, persisted, project_dir)
