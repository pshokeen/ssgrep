"""OpenCode SQLite: a read-only, query-only, transactional snapshot.

* ``file:...?mode=ro`` (never ``immutable``, which would ignore live WAL content),
  ``PRAGMA query_only=ON`` and an explicit ``BEGIN`` so every read in one cycle sees
  one committed snapshot, WAL included. The source file/WAL are never written.
* A SQLite authorizer allows reads of ``session``/``message``/``part`` only, so the
  redundant ``event`` history (and credential tables) can never be consumed or
  leaked, and any write attempt is denied.
* Each JSON value is capped at 8 MiB *inside SQL*: an oversized value is replaced
  by NULL before it reaches Python and is reported as a coverage gap.
* Incomplete (pending/running) tool parts and unfinished assistant messages are
  held as pending ids and accepted once, after they complete.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES, RawEvidence
from ssgrep.accounting.readers import roots
from ssgrep.accounting.readers.base import Gap, ReadResult, Source, SourceState
from ssgrep.accounting.readers.opencode_cursor import watermarks_lost
from ssgrep.accounting.readers.values import (
    as_int,
    loads,
    make_evidence,
    nano_usd,
    obj,
    pick_ints,
    text_of,
    to_utc,
)

TABLES = frozenset({"session", "message", "part"})
SCHEMA_TABLES = frozenset({"sqlite_master", "sqlite_schema", "sqlite_temp_master"})
TOKEN_NAMES = {
    "total": "total_tokens",
    "input": "input_tokens",
    "output": "output_tokens",
    "reasoning": "reasoning_tokens",
}
IGNORED_PARTS = frozenset(
    {
        "text",
        "reasoning",
        "step-start",
        "step-finish",
        "snapshot",
        "patch",
        "file",
        "agent",
        "subtask",
        "compaction",
        "retry",
    }
)
# The owning session's directory, joined in SQL so it never depends on paging order.
DIRECTORY = "(SELECT s.directory FROM session AS s WHERE s.id = {table}.session_id) AS cwd, "
PENDING_STATES = frozenset({"pending", "running"})
PENDING_CAP = 500
DEFAULT_BUDGET = 1000
REQUIRED = {
    "session": {"id", "parent_id", "directory", "project_id", "time_updated"},
    "message": {"id", "session_id", "time_updated", "data"},
    "part": {"id", "message_id", "session_id", "time_updated", "data"},
}


class OpenCodeReader:
    runtime: ClassVar[str] = "opencode"

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self.env = env
        self.tables_read: set[str] = set()

    # ---- discovery ---------------------------------------------------------
    def source_for(self, path: Path) -> Source:
        return Source(self.runtime, path)

    def discover(self) -> list[Source]:
        path = roots.opencode_db(self.env)
        return [self.source_for(path)] if path.is_file() and not path.is_symlink() else []

    # ---- connection --------------------------------------------------------
    def _authorize(self, action: int, arg1: str | None, arg2: str | None, *_: Any) -> int:
        if action == sqlite3.SQLITE_READ:
            table = (arg1 or "").lower()
            if table in TABLES:
                self.tables_read.add(table)
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_OK if table in SCHEMA_TABLES else sqlite3.SQLITE_DENY
        allowed = (
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_TRANSACTION,
            sqlite3.SQLITE_RECURSIVE,
        )
        if action in allowed:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in {"query_only", "table_info"}:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    @contextmanager
    def snapshot(self, database: Path) -> Iterator[sqlite3.Connection]:
        """One consistent committed snapshot; read-only, query-only, WAL-aware."""
        uri = f"{database.absolute().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0, isolation_level=None)
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("BEGIN")
            connection.execute("SELECT count(*) FROM sqlite_master").fetchone()  # pins snapshot
            connection.set_authorizer(self._authorize)
            yield connection
        finally:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()

    # ---- reading -------------------------------------------------------------
    @staticmethod
    def _identity(path: Path) -> dict[str, Any]:
        info = path.stat()
        birth = getattr(info, "st_birthtime_ns", None)
        return {"dev": info.st_dev, "inode": info.st_ino, "birth": birth}

    def read(
        self, source: Source, state: SourceState | None = None, budget: int | None = None
    ) -> ReadResult:
        path = source.path
        result = ReadResult()
        try:
            identity = self._identity(path)
        except OSError:
            return self._unavailable(source, result, state)
        state = state or SourceState()
        previous = state.extra.get("identity")
        if previous and previous != identity:
            result.reset = "rotated"
            state = SourceState()
        extra = deepcopy(state.extra)
        extra["identity"] = identity
        try:
            with self.snapshot(path) as connection:
                self._scan(connection, source, extra, budget or DEFAULT_BUDGET, result)
        except sqlite3.DatabaseError:
            return self._unavailable(source, result, state)
        result.state = SourceState(extra=extra, dev=identity["dev"], inode=identity["inode"])
        return result

    @staticmethod
    def _unavailable(source: Source, result: ReadResult, state: SourceState | None) -> ReadResult:
        result.gaps.append(Gap(source.source_id, "database_unavailable", 0, "database"))
        result.state = state or SourceState()
        return result

    def _scan(
        self,
        connection: sqlite3.Connection,
        source: Source,
        extra: dict[str, Any],
        budget: int,
        result: ReadResult,
    ) -> None:
        for table, columns in REQUIRED.items():
            found = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            if not columns <= found:
                result.gaps.append(Gap(source.source_id, "unsupported_schema", 0, table))
                return
        marks = extra.setdefault("wm", {})
        if watermarks_lost(connection, marks):
            result.reset = "rotated"
            extra["wm"], extra["pending"] = {}, {}
            marks = extra["wm"]
        pending = extra.setdefault("pending", {})
        for table in ("session", "message", "part"):
            rows = self._fetch(connection, table, marks, pending.get(table, []), budget)
            if len(rows) >= budget:
                result.more = True
            held: list[str] = []
            for row in rows:
                mark = [row["time_updated"], row["id"]]
                marks[table] = max(marks.get(table, mark), mark)
                self._row(source, table, row, held, result)
            pending[table] = held[:PENDING_CAP]
            if len(held) > PENDING_CAP:
                result.gaps.append(Gap(source.source_id, "pending_overflow", len(held), table))
            result.pending_records += len(held)

    @staticmethod
    def _fetch(
        connection: sqlite3.Connection,
        table: str,
        marks: dict[str, Any],
        pending: list[str],
        budget: int,
    ) -> list[sqlite3.Row]:
        mark = marks.get(table)
        holders = ",".join("?" for _ in pending)
        extras = {
            "session": "parent_id, directory, project_id, directory AS cwd, NULL AS data, 0 AS n",
            "message": "session_id, NULL AS message_id, "
            + DIRECTORY.format(table="message")
            + "CASE WHEN length(CAST(data AS BLOB)) > ? THEN NULL ELSE data END AS data, "
            "length(CAST(data AS BLOB)) AS n",
            "part": "session_id, message_id, "
            + DIRECTORY.format(table="part")
            + "CASE WHEN length(CAST(data AS BLOB)) > ? THEN NULL ELSE data END AS data, "
            "length(CAST(data AS BLOB)) AS n",
        }[table]
        capped = table != "session"
        clause = "(time_updated > ? OR (time_updated = ? AND id > ?))" if mark else "1"
        if pending:
            clause += f" OR id IN ({holders})"
        # table and column fragments above are constants; every value is bound.
        query = (  # noqa: S608
            f"SELECT id, time_updated, {extras} FROM {table} WHERE {clause} "
            "ORDER BY time_updated, id LIMIT ?"
        )
        args: list[Any] = [MAX_RAW_RECORD_BYTES] if capped else []
        args += ([mark[0], mark[0], mark[1]] if mark else []) + [*pending, budget]
        connection.row_factory = sqlite3.Row
        return connection.execute(query, args).fetchall()

    # ---- decoding ------------------------------------------------------------
    def _row(
        self, source: Source, table: str, row: sqlite3.Row, held: list[str], result: ReadResult
    ) -> None:
        locator = f"{table}:{row['id']}"
        if table == "session":
            result.evidence.append(self._session(source, row, locator))
            return
        if row["data"] is None:
            result.gaps.append(Gap(source.source_id, "oversized_record", int(row["n"]), locator))
            return
        try:
            data = loads(row["data"])
        except ValueError:
            result.gaps.append(Gap(source.source_id, "malformed_json", int(row["n"]), locator))
            return
        if not isinstance(data, dict):
            result.unrecognized["<non-object>"] = result.unrecognized.get("<non-object>", 0) + 1
            return
        if table == "message":
            self._message(source, row, data, locator, held, result)
        else:
            self._part(source, row, data, locator, held, result)

    def _evidence(
        self,
        source: Source,
        session: str,
        record_id: str,
        kind: str,
        locator: str,
        ts: Any,
        payload: dict[str, Any],
        branch: str = "main",
    ) -> RawEvidence:
        return make_evidence(
            runtime=self.runtime,
            source_id=source.source_id,
            branch_id=branch,
            session_id=session,
            record_id=record_id,
            locator=locator,
            kind=kind,
            ts=to_utc(ts),
            payload=payload,
        )

    def _session(self, source: Source, row: sqlite3.Row, locator: str) -> RawEvidence:
        parent = row["parent_id"]
        body = {
            "id": row["id"],
            "parent_id": parent,
            "directory": row["directory"],
            "project_id": row["project_id"],
        }
        branch = "child" if parent else "main"
        return self._evidence(
            source, row["id"], f"session:{row['id']}", "session_header", locator, None, body, branch
        )

    def _message(
        self,
        source: Source,
        row: sqlite3.Row,
        data: dict,
        locator: str,
        held: list[str],
        result: ReadResult,
    ) -> None:
        if data.get("role") != "assistant":
            return
        stamps = obj(data.get("time"))
        if not (stamps.get("completed") or data.get("finish") or data.get("error")):
            held.append(row["id"])
            return
        tokens = obj(data.get("tokens"))
        counts = pick_ints(tokens, TOKEN_NAMES)
        counts.update(pick_ints(tokens.get("cache"), {"write": "cache_write_tokens"}))
        counts.update(pick_ints(tokens.get("cache"), {"read": "cache_read_tokens"}))
        body = {
            "message_id": row["id"],
            "model": data.get("modelID"),
            "provider": data.get("providerID"),
            "tokens": counts or None,
            "cost_nano_usd": nano_usd(data.get("cost")),
            "completed_ms": as_int(stamps.get("completed")),
        }
        result.evidence.append(
            self._evidence(
                source,
                row["session_id"],
                f"usage:{row['id']}",
                "usage",
                locator,
                stamps.get("completed") or stamps.get("created"),
                body,
            )
        )

    def _part(
        self,
        source: Source,
        row: sqlite3.Row,
        data: dict,
        locator: str,
        held: list[str],
        result: ReadResult,
    ) -> None:
        kind = data.get("type")
        if kind != "tool":
            if not isinstance(kind, str) or kind not in IGNORED_PARTS:
                name = str(kind)[:64]
                result.unrecognized[name] = result.unrecognized.get(name, 0) + 1
            return
        state = obj(data.get("state"))
        status = state.get("status")
        call_id = data.get("callID")
        if not isinstance(call_id, str) or not isinstance(status, str):
            result.unrecognized["tool"] = result.unrecognized.get("tool", 0) + 1
            return
        if status in PENDING_STATES:
            held.append(row["id"])
            return
        stamps = obj(state.get("time"))
        start, end = as_int(stamps.get("start")), as_int(stamps.get("end"))
        meta = obj(state.get("metadata"))
        call = {
            "call_id": call_id,
            "name": data.get("tool"),
            "arguments": state.get("input"),
            "status": status,
            "start_ms": start,
            "end_ms": end,
            "duration_ms": end - start if start is not None and end is not None else None,
            "message_id": row["message_id"],
            "cwd": row["cwd"],
        }
        output = text_of(state.get("output")) or text_of(state.get("error"))
        failed = status == "error"
        reply = {
            "call_id": call_id,
            "output": output,
            "output_bytes": len(output.encode()),
            "is_error": failed,
            "exit_code": as_int(meta.get("exit")),
        }
        session = row["session_id"]
        result.evidence.append(
            self._evidence(source, session, f"call:{call_id}", "tool_call", locator, start, call)
        )
        result.evidence.append(
            self._evidence(source, session, f"result:{call_id}", "tool_result", locator, end, reply)
        )
