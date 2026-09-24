"""Permanently delete tombstoned rows from the global LanceDB store."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from datetime import datetime
from typing import Annotated, TypedDict

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.cli import exit_codes
from ssgrep.store import (
    CHUNKS_TABLE,
    CURSORS_TABLE,
    EPISODES_TABLE,
    SESSIONS_TABLE,
    LanceStore,
    quote,
)
from ssgrep.utilities.types import IndexNotFoundError

_DAY_SECONDS = 86_400


class TombstonedSession(TypedDict):
    session_id: str
    path: str
    chunk_count: int
    episode_count: int
    absent_since: float | None


def _tombstoned_sessions(repository: LanceStore) -> list[TombstonedSession]:
    result: list[TombstonedSession] = []
    for row in repository.rows(SESSIONS_TABLE, where="source_status = 'absent'", limit=100_000):
        session_id = str(row["session_id"])
        path = str(row["path"])
        absent_since = row.get("absent_since")
        if isinstance(absent_since, datetime):
            absent_timestamp: float | None = absent_since.timestamp()
        elif isinstance(absent_since, str):
            absent_timestamp = datetime.fromisoformat(absent_since).timestamp()
        else:
            absent_timestamp = None
        result.append(
            {
                "session_id": session_id,
                "path": path,
                "chunk_count": repository.count(CHUNKS_TABLE, f"session_id = {quote(session_id)}"),
                "episode_count": repository.count(
                    EPISODES_TABLE, f"session_id = {quote(session_id)}"
                ),
                "absent_since": absent_timestamp,
            }
        )
    return result


def _filter_by_age(sessions: list[TombstonedSession], older_than: int) -> list[TombstonedSession]:
    if older_than <= 0:
        return sessions
    cutoff = time.time() - older_than * _DAY_SECONDS
    return [
        item
        for item in sessions
        if item["absent_since"] is not None and item["absent_since"] < cutoff
    ]


class PruneCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "prune"

    def description(self) -> str:
        return "Permanently delete globally indexed content whose source vanished"

    def handle(
        self,
        older_than: Annotated[
            int,
            Option(
                "--older-than",
                min=0,
                help="Only prune sources absent for this many days",
            ),
        ] = 0,
        dry_run: Annotated[bool, Option("--dry-run", help="Preview without deleting")] = False,
        yes: Annotated[bool, Option("--yes", help="Skip interactive confirmation")] = False,
    ) -> dict[str, object] | None:
        repository = LanceStore()
        if not repository.exists():
            error = IndexNotFoundError("No global index found. Run `ssgrep index` first.")
            if is_json_mode():
                stdout = sys.__stdout__
                assert stdout is not None
                stdout.write(
                    json.dumps(
                        {
                            "ok": False,
                            "condition": error.condition,
                            "message": str(error),
                            "command": error.command,
                        }
                    )
                    + "\n"
                )
                stdout.flush()
                os._exit(exit_codes.MISSING_INDEX)
            print(error, file=sys.stderr)
            raise SystemExit(exit_codes.MISSING_INDEX) from error

        sessions = _filter_by_age(_tombstoned_sessions(repository), older_than)
        document: dict[str, object] = {
            "ok": True,
            "dry_run": dry_run,
            "older_than": older_than,
            "session_count": len(sessions),
            "chunk_count": sum(item["chunk_count"] for item in sessions),
            "episode_count": sum(item["episode_count"] for item in sessions),
            "sessions": [
                {k: v for k, v in item.items() if k != "absent_since"} for item in sessions
            ],
        }
        if dry_run or not sessions:
            if is_json_mode():
                return document
            if not sessions:
                print("Nothing to prune: no tombstoned content matches.", file=sys.stderr)
            else:
                print(
                    f"Would prune {len(sessions)} sessions ({document['chunk_count']} chunks).",
                    file=sys.stderr,
                )
            return None
        if not yes:
            if is_json_mode() or not sys.stdin.isatty():
                print("Prune requires --yes when non-interactive.", file=sys.stderr)
                raise SystemExit(exit_codes.USAGE_ERROR)
            answer = input(f"Permanently delete {len(sessions)} tombstoned sessions? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                print("Cancelled.", file=sys.stderr)
                return None
        with (repository.root / "index.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Revalidate after confirmation and after any concurrent index run.
            sessions = _filter_by_age(_tombstoned_sessions(repository), older_than)
            document.update(
                session_count=len(sessions),
                chunk_count=sum(item["chunk_count"] for item in sessions),
                episode_count=sum(item["episode_count"] for item in sessions),
                sessions=[
                    {key: value for key, value in item.items() if key != "absent_since"}
                    for item in sessions
                ],
            )
            for item in sessions:
                session_id = str(item["session_id"])
                predicate = f"session_id = {quote(session_id)}"
                repository.delete(CHUNKS_TABLE, predicate)
                repository.delete(EPISODES_TABLE, predicate)
                repository.delete(SESSIONS_TABLE, predicate)
                repository.delete(CURSORS_TABLE, f"path = {quote(str(item['path']))}")
        if is_json_mode():
            document["dry_run"] = False
            return document
        print(f"Pruned {len(sessions)} tombstoned sessions.", file=sys.stderr)
        return None
