"""Prune command for ssgrep — permanently delete tombstoned content.

Tombstoning (design decision) retains chunks from vanished sources as
searchable; ``prune`` is the ONLY operation permitted to delete indexed
content — ``store.delete_session_chunks()`` is called from here and from
nowhere else. Sessions whose transcripts still exist on disk are never
touched: the deletion filter is ``source_status = 'absent'`` only.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from usecli.cli.core.base_command import BaseCommand
from usecli.cli.core.runtime import is_json_mode

from ssgrep import store
from ssgrep.cli import exit_codes
from ssgrep.cli.commands import resolve_project_dir
from ssgrep.types import IndexNotFoundError

#: Index directory name inside the project (mirrors indexer.py's convention).
INDEX_DIRNAME = ".ssgrep"

#: Seconds per day for --older-than conversion.
_DAY_SECONDS = 86_400


def _tombstoned_sessions(conn: sqlite3.Connection) -> list[dict]:
    """List tombstoned sessions with their last-seen-on-disk timestamp.

    The age proxy is the source file's recorded mtime from session_files
    (the last moment the file was seen on disk before it vanished). Falls
    back to the global last_index_time when no cursor row exists; a session
    with neither is reported with last_seen=None and is excluded whenever
    --older-than is in effect (never guess ages for a destructive operation).
    """
    rows = conn.execute(
        """
        SELECT s.session_id, s.path,
               (SELECT COUNT(*) FROM chunks c
                 WHERE c.session_id = s.session_id AND c.source_status = 'absent')
                   AS chunk_count,
               (SELECT COUNT(*) FROM episodes e
                 WHERE e.session_id = s.session_id AND e.source_status = 'absent')
                   AS episode_count
        FROM sessions s
        WHERE s.source_status = 'absent'
        ORDER BY s.session_id
        """
    ).fetchall()

    last_index_time_str = store.get_meta(conn, "last_index_time")
    fallback_ts: float | None = None
    if last_index_time_str:
        try:
            fallback_ts = datetime.fromisoformat(last_index_time_str).timestamp()
        except ValueError:
            fallback_ts = None

    sessions = []
    for session_id, path, chunk_count, episode_count in rows:
        cursor = conn.execute("SELECT mtime FROM session_files WHERE path = ?", (path,)).fetchone()
        last_seen = cursor[0] if cursor else fallback_ts
        sessions.append(
            {
                "session_id": session_id,
                "path": path,
                "chunk_count": chunk_count,
                "episode_count": episode_count,
                "last_seen": last_seen,
            }
        )
    return sessions


def _filter_by_age(sessions: list[dict], older_than: int) -> list[dict]:
    """Keep only sessions not seen on disk for at least ``older_than`` days."""
    if older_than <= 0:
        return sessions
    cutoff = datetime.now(UTC).timestamp() - older_than * _DAY_SECONDS
    return [s for s in sessions if s["last_seen"] is not None and s["last_seen"] < cutoff]


class PruneCommand(BaseCommand):
    """Delete tombstoned (vanished) content from the index."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "prune"

    def description(self) -> str:
        """Command description."""
        return "Delete tombstoned (vanished) content from the index"

    def handle(
        self,
        older_than: int = 0,
        dry_run: bool = False,
        yes: bool = False,
        project_dir: str = ".",
    ) -> object:
        """Hard-delete chunks/episodes/sessions whose sources vanished."""
        project = resolve_project_dir(project_dir)
        gen_store = store.GenerationalStore(project / INDEX_DIRNAME)
        db_path = gen_store.get_index_path()

        if not db_path.exists():
            msg = f"No index found for {project}. Run `ssgrep index` to build the index."
            error = IndexNotFoundError(msg)
            if is_json_mode():
                # Write error JSON directly and exit, bypassing usecli's exception wrapping.
                # Use os._exit() to avoid all Python exception handling and usecli's
                # stdout redirection. Use sys.__stdout__ to write to the real stdout.
                error_doc = {
                    "ok": False,
                    "condition": error.condition,
                    "message": str(error),
                    "command": error.command,
                }
                sys.__stdout__.write(json.dumps(error_doc) + "\n")  # type: ignore
                sys.__stdout__.flush()  # type: ignore
                os._exit(exit_codes.MISSING_INDEX)
            else:
                print(error, file=sys.stderr)
                raise SystemExit(exit_codes.MISSING_INDEX) from error

        conn = sqlite3.connect(str(db_path))
        try:
            sessions = _filter_by_age(_tombstoned_sessions(conn), older_than)

            if dry_run:
                document = {
                    "ok": True,
                    "dry_run": True,
                    "older_than": older_than,
                    "session_count": len(sessions),
                    "chunk_count": sum(s["chunk_count"] for s in sessions),
                    "episode_count": sum(s["episode_count"] for s in sessions),
                    "sessions": [
                        {k: v for k, v in s.items() if k != "last_seen"} for s in sessions
                    ],
                }
                if is_json_mode():
                    return document
                if not sessions:
                    print("Nothing to prune: no tombstoned content matches.", file=sys.stderr)
                    return None
                print(
                    f"Would prune {document['session_count']} tombstoned sessions "
                    f"({document['chunk_count']} chunks, "
                    f"{document['episode_count']} episodes):",
                    file=sys.stderr,
                )
                for s in sessions:
                    print(
                        f"  {s['session_id']}  {s['chunk_count']} chunks  {s['path']}",
                        file=sys.stderr,
                    )
                print("Re-run without --dry-run to delete.", file=sys.stderr)
                return None

            if not sessions:
                if is_json_mode():
                    return {
                        "ok": True,
                        "dry_run": False,
                        "older_than": older_than,
                        "session_count": 0,
                        "chunk_count": 0,
                        "episode_count": 0,
                        "sessions": [],
                    }
                print("Nothing to prune: no tombstoned content matches.", file=sys.stderr)
                return None

            # Bug 1 + Bug 2: Handle non-interactive stdin and JSON mode confirmation
            if not yes:
                if is_json_mode():
                    # Never prompt in JSON mode — stdout must stay machine-clean.
                    # This is a usage error (missing required flag), not an internal failure.
                    # Bug 2: Write structured JSON error document
                    total_chunks = sum(s["chunk_count"] for s in sessions)
                    error_doc = {
                        "ok": False,
                        "condition": "confirmation_required",
                        "message": (
                            f"Refusing to prune {len(sessions)} tombstoned sessions "
                            f"({total_chunks} chunks) without explicit confirmation. "
                            "Pass --yes to confirm."
                        ),
                        "command": "ssgrep prune --yes",
                    }
                    sys.__stdout__.write(json.dumps(error_doc) + "\n")  # type: ignore
                    sys.__stdout__.flush()  # type: ignore
                    os._exit(exit_codes.USAGE_ERROR)
                # Bug 1: Check if stdin is a terminal before prompting
                if not sys.stdin.isatty():
                    print(
                        "Error: prune requires interactive confirmation (a TTY for stdin). "
                        "Pass --yes to confirm deletion without a prompt, or run in an "
                        "interactive terminal.",
                        file=sys.stderr,
                    )
                    raise SystemExit(exit_codes.USAGE_ERROR)
                total_chunks = sum(s["chunk_count"] for s in sessions)
                print(
                    f"About to permanently delete {len(sessions)} tombstoned sessions "
                    f"({total_chunks} chunks) from {db_path}.",
                    file=sys.stderr,
                    flush=True,
                )
                # Write prompt to stderr with no trailing newline, read answer from stdin
                sys.stderr.write("Type 'yes' to confirm: ")
                sys.stderr.flush()
                try:
                    answer = input()
                except (EOFError, KeyboardInterrupt):
                    print("Aborted; nothing deleted.", file=sys.stderr)
                    raise SystemExit(exit_codes.USAGE_ERROR) from None
                if answer.strip().lower() != "yes":
                    print("Aborted; nothing deleted.", file=sys.stderr)
                    raise SystemExit(exit_codes.USAGE_ERROR)

        finally:
            # The pre-lock connection served display and confirmation only;
            # the deletion phase below re-opens from paths resolved under the
            # generation lock.
            with contextlib.suppress(Exception):
                conn.close()

        deleted = self._delete_under_lock(project, older_than)

        document = {
            "ok": True,
            "dry_run": False,
            "older_than": older_than,
            "session_count": len(deleted),
            "chunk_count": sum(s["chunk_count"] for s in deleted),
            "episode_count": sum(s["episode_count"] for s in deleted),
            "sessions": deleted,
        }
        if is_json_mode():
            return document

        print(
            f"Pruned {document['session_count']} tombstoned sessions "
            f"({document['chunk_count']} chunks, "
            f"{document['episode_count']} episodes).",
            file=sys.stderr,
        )
        return None

    def _delete_under_lock(self, project: Path, older_than: int) -> list[dict]:
        """Run the destructive phase entirely under the generation lock.

        The confirmation prompt in ``handle()`` can wait on a human
        indefinitely, during which a concurrent rebuild may commit a new live
        generation (the classic TOCTOU window: prune.py used to lock a
        generation number cached before the prompt). Closing that window:

        1. Re-read the manifest (fresh ``GenerationalStore``) and acquire
           ``hold_generation()`` on the freshly-read generation.
        2. Re-read the manifest under the lock; if the live generation moved
           between read and lock, release and retry.
        3. Resolve paths, open the connection, and re-derive the tombstoned
           session set under the lock — the pre-prompt set is display-only.
        4. Delete and run ``cleanup_orphaned_vectors`` with the explicitly
           passed generation, never a stale cached ``current_generation``.
        """
        index_dir = project / INDEX_DIRNAME
        for _attempt in range(5):
            gen_store = store.GenerationalStore(index_dir)
            generation = gen_store.current_generation
            with gen_store.hold_generation(generation):
                # Revalidate under the lock: retry if a rebuild landed between
                # the manifest read above and the flock acquisition.
                if store.GenerationalStore(index_dir).current_generation != generation:
                    continue
                db_path = gen_store.get_index_path(generation)
                if not db_path.exists():
                    continue
                conn = sqlite3.connect(str(db_path))
                try:
                    sessions = _filter_by_age(_tombstoned_sessions(conn), older_than)
                    deleted = []
                    for s in sessions:
                        store.delete_session_chunks(conn, s["session_id"])
                        deleted.append({k: v for k, v in s.items() if k != "last_seen"})
                    conn.commit()

                    # cleanup_orphaned_vectors closes the connection internally
                    # and stages/commits the compacted generation on top of the
                    # generation we hold — passed explicitly.
                    store.cleanup_orphaned_vectors(conn, gen_store, generation=generation)
                finally:
                    with contextlib.suppress(Exception):
                        conn.close()  # Idempotent if already closed by cleanup
                return deleted

        # The live generation moved on every attempt (pathological rebuild
        # churn). Nothing has been deleted; report rather than guess.
        msg = (
            "The live index generation kept changing while prune was acquiring "
            "its lock (concurrent rebuilds). Nothing was deleted; re-run prune."
        )
        if is_json_mode():
            error_doc = {
                "ok": False,
                "condition": "generation_contention",
                "message": msg,
                "command": "ssgrep prune --yes",
            }
            sys.__stdout__.write(json.dumps(error_doc) + "\n")  # type: ignore
            sys.__stdout__.flush()  # type: ignore
            os._exit(exit_codes.INTERNAL_FAILURE)
        print(f"Error: {msg}", file=sys.stderr)
        raise SystemExit(exit_codes.INTERNAL_FAILURE)
