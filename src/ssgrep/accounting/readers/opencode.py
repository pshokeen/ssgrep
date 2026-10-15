"""OpenCode SQLite: a read-only, query-only, transactional snapshot.

* Ordinary read-only WAL connections can still mutate sidecars. A guarded,
  nonmutating snapshot uses SQLite's private heap WAL index for active WALs and
  immutable only when no WAL frames exist; any observed source race discards the read.
* A SQLite authorizer allows reads of ``session``/``message``/``part`` only, so the
  redundant ``event`` history (and credential tables) can never be consumed or
  leaked, and any write attempt is denied.
* Each JSON value is capped at 8 MiB *inside SQL*: an oversized value is replaced
  by NULL before it reaches Python and is reported as a coverage gap.
* Incomplete (pending/running) tool parts and unfinished assistant messages are
  held as pending ids and accepted once, after they complete.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES, RawEvidence
from ssgrep.accounting.readers import roots
from ssgrep.accounting.readers.base import Gap, ReadResult, Source, SourceState
from ssgrep.accounting.readers.opencode_cursor import abandon, finish_generation, progress, scan
from ssgrep.accounting.readers.opencode_rows import read_row
from ssgrep.accounting.readers.opencode_snapshot import (
    SnapshotChanged,
    SnapshotTracker,
    SnapshotUnavailable,
    authorize,
    snapshot,
)
from ssgrep.accounting.readers.values import (
    as_int,
    make_evidence,
    obj,
    text_of,
    to_utc,
)

if TYPE_CHECKING:
    import apsw

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
        return authorize(self.tables_read, action, arg1, arg2)

    @contextmanager
    def snapshot(
        self, database: Path, *, descriptor=None, tracker=None
    ) -> Iterator[apsw.Connection]:
        """An unchanged committed view, with no native main/WAL/SHM writes."""
        with snapshot(
            database, self._authorize, descriptor=descriptor, tracker=tracker
        ) as connection:
            yield connection

    # ---- reading -------------------------------------------------------------
    @staticmethod
    def _identity(path: Path) -> dict[str, Any]:
        info = path.stat()
        birth = getattr(info, "st_birthtime_ns", None)
        return {"dev": info.st_dev, "inode": info.st_ino, "birth": birth}

    def read(
        self,
        source: Source,
        state: SourceState | None = None,
        budget: int | None = None,
        *,
        byte_budget: int | None = None,
        maintenance: bool = False,
    ) -> ReadResult:
        path = source.path
        prior_state = state
        result = ReadResult()
        try:
            identity = self._identity(path)
        except OSError:
            return self._unavailable(source, result, state)
        try:
            import apsw
        except ImportError:
            return self._unavailable(source, result, state, "database_snapshot_unavailable")
        state = state or SourceState()
        previous = state.extra.get("identity")
        if previous and previous != identity:
            result.reset = "rotated"
            prior = deepcopy(state.extra)
            for table, ids in prior.get("pending", {}).items():
                for record_id in ids:
                    abandon(
                        prior, table, record_id, "source_rotated", source, result, self._evidence
                    )
            state = SourceState(
                extra={
                    "abandoned": prior.get("abandoned", {}),
                    "abandoned_count": progress(prior)["abandoned_records"],
                }
            )
        extra = deepcopy(state.extra)
        extra["identity"] = identity
        tracker = SnapshotTracker()
        try:
            with self.snapshot(path, tracker=tracker) as connection:
                self._scan(
                    connection,
                    source,
                    extra,
                    budget or DEFAULT_BUDGET,
                    result,
                    byte_budget,
                    maintenance,
                    tracker.before,
                )
        except SnapshotChanged:
            return self._unavailable(source, result, prior_state, "database_changed", "changed")
        except SnapshotUnavailable:
            return self._unavailable(source, result, prior_state, "database_snapshot_unavailable")
        except apsw.Error:
            return self._unavailable(source, result, prior_state)
        finish_generation(extra, result, tracker.before, tracker.after)
        result.state = SourceState(extra=extra, dev=identity["dev"], inode=identity["inode"])
        return result

    @staticmethod
    def _unavailable(
        source: Source,
        result: ReadResult,
        state: SourceState | None,
        reason: str = "database_unavailable",
        status: str = "unavailable",
    ) -> ReadResult:
        result.evidence.clear()
        result.reset = None
        result.gaps.append(Gap(source.source_id, reason, 0, "database"))
        result.state = deepcopy(state) if state is not None else SourceState()
        progress(result.state.extra).update(
            status=status,
            pending_records=None,
            pending_count_complete=False,
            snapshot_checked=False,
        )
        return result

    def _scan(
        self,
        connection: apsw.Connection,
        source: Source,
        extra: dict[str, Any],
        budget: int,
        result: ReadResult,
        byte_budget: int | None = None,
        maintenance: bool = False,
        generation: dict | None = None,
    ) -> None:
        for table, columns in REQUIRED.items():
            found = {row["name"] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            if not columns <= found:
                result.gaps.append(Gap(source.source_id, "unsupported_schema", 0, table))
                progress(extra).update(
                    status="unsupported",
                    pending_records=None,
                    pending_count_complete=False,
                    snapshot_checked=False,
                )
                return
        scan(
            connection,
            source,
            extra,
            budget,
            byte_budget,
            result,
            self._fetch,
            self._row,
            self._evidence,
            maintenance,
            generation,
        )

    @staticmethod
    def _fetch(
        connection: apsw.Connection,
        table: str,
        marks: dict[str, Any],
        pending: list[str],
        budget: int,
        *,
        recheck: bool = False,
    ) -> apsw.Cursor:
        mark = marks.get(table)
        holders = ",".join("?" for _ in pending)
        extras = {
            "session": "parent_id, directory, project_id, directory AS cwd, NULL AS data, "
            "length(CAST(id AS BLOB)) + length(CAST(directory AS BLOB)) + "
            "length(CAST(project_id AS BLOB)) + "
            "coalesce(length(CAST(parent_id AS BLOB)), 0) AS n",
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
        if recheck:
            clause = f"id IN ({holders})"
        # table and column fragments above are constants; every value is bound.
        query = (  # noqa: S608
            f"SELECT id, time_updated, {extras} FROM {table} WHERE id IN "
            f"(SELECT id FROM {table} WHERE {clause} ORDER BY time_updated, id LIMIT ?) "
            "ORDER BY time_updated, id"
        )
        args: list[Any] = [MAX_RAW_RECORD_BYTES] if capped else []
        args += [*pending] if recheck else [mark[0], mark[0], mark[1]] if mark else []
        args.append(budget)
        return connection.execute(query, args)

    # ---- decoding ------------------------------------------------------------
    def _row(
        self,
        source: Source,
        table: str,
        row: Mapping[str, Any],
        held: list[str],
        result: ReadResult,
    ) -> None:
        read_row(self, source, table, row, held, result)

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

    def _part(
        self,
        source: Source,
        row: Mapping[str, Any],
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
                result.gaps.append(
                    Gap(source.source_id, "unsupported_part_type", int(row["n"]), locator)
                )
            return
        state = obj(data.get("state"))
        status = state.get("status")
        call_id = data.get("callID")
        if not isinstance(call_id, str) or not isinstance(status, str):
            result.unrecognized["tool"] = result.unrecognized.get("tool", 0) + 1
            result.gaps.append(Gap(source.source_id, "unsupported_tool", int(row["n"]), locator))
            return
        if status in PENDING_STATES:
            held.append(row["id"])
            result.evidence.append(
                self._evidence(
                    source,
                    row["session_id"],
                    f"pending-tool:{row['id']}",
                    "pending_tool",
                    locator,
                    None,
                    {
                        "id": row["id"],
                        "message_id": row["message_id"],
                        "session_id": row["session_id"],
                        "call_id": call_id,
                        "status": "unfinished",
                    },
                )
            )
            return
        if status not in {"completed", "error"}:
            result.gaps.append(
                Gap(source.source_id, "unsupported_tool_status", int(row["n"]), locator)
            )
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
