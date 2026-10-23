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

from ssgrep.accounting.ledger.contracts import MAX_RAW_RECORD_BYTES, RawEvidence
from ssgrep.accounting.readers import classify as roots
from ssgrep.accounting.readers.base import (
    Gap,
    ReadResult,
    Source,
    SourceState,
    as_int,
    loads,
    make_evidence,
    nano_usd,
    obj,
    pick_ints,
    text_of,
    to_utc,
)
from ssgrep.accounting.readers.opencode_storage import (
    DEFAULT_BUDGET,
    DIRECTORY,
    IGNORED_PARTS,
    PENDING_STATES,
    REQUIRED,
    SnapshotChanged,
    SnapshotTracker,
    SnapshotUnavailable,
    authorize,
    snapshot,
)

if TYPE_CHECKING:
    import apsw


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


TABLES = ("session", "message", "part")
CURSOR_VERSION = 3
DETAIL_CAP = 32


def watermarks_lost(connection: apsw.Connection, marks: dict[str, Any]) -> bool:
    """Missing/regressed anchors prove lost continuity; ordinary WAL updates do not."""
    for table in TABLES:
        if table not in marks:
            continue
        time, record_id = marks[table]
        row = connection.execute(
            f"SELECT time_updated FROM {table} WHERE id = ?",
            (record_id,),  # noqa: S608
        ).fetchone()
        if row is None or row["time_updated"] < time:
            return True
    return False


def progress(extra: dict[str, Any]) -> dict[str, Any]:
    return extra.setdefault(
        "opencode",
        {
            "cursor_version": CURSOR_VERSION,
            "status": "unvisited",
            "rows_visited": {},
            "row_count_basis": "decoded_visits_including_maintenance_and_rechecks",
            "bytes_read": 0,
            "bytes_examined": 0,
            "pending_records": None,
            "known_pending_lower_bound": 0,
            "pending_count_complete": False,
            "abandoned_records": int(extra.get("abandoned_count", len(extra.get("abandoned", {})))),
            "unsupported_records": 0,
            "next_table": 0,
            "next_lane": 0,
            "as_of": "last_committed_chunk",
            "maintenance_consistency": "per_chunk_committed_snapshots",
        },
    )


def sample(mapping, key, value):
    if key in mapping or len(mapping) < DETAIL_CAP:
        mapping[key] = value


def abandon(extra, table, record_id, reason, source, result, emit):
    locator = f"{table}:{record_id}"
    metadata = extra.setdefault("pending_provenance", {}).pop(locator, {"id": record_id})
    body = {**metadata, "reason": reason}
    sample(extra.setdefault("abandoned", {}), locator, body)
    progress(extra)["abandoned_records"] += 1
    result.gaps.append(Gap(source.source_id, "abandoned_pending", 0, locator))
    result.evidence.append(
        emit(
            source,
            str(metadata.get("session_id", "unknown")),
            f"abandoned:{locator}",
            "abandoned_pending",
            locator,
            None,
            body,
        )
    )


def prepare(connection, extra, source, result, emit):
    old_version = extra.get("opencode", {}).get("cursor_version")
    migrated = bool(extra.get("wm")) and old_version != CURSOR_VERSION
    lost = watermarks_lost(connection, extra.get("wm", {}))
    if migrated or lost:
        if lost:
            for table, ids in extra.get("pending", {}).items():
                for record_id in ids[:DETAIL_CAP]:
                    abandon(extra, table, record_id, "cursor_continuity_lost", source, result, emit)
            extra["pending"] = {}
        extra["wm"], extra["census"] = {}, {}
        extra.pop("census_epoch", None)
        stats = progress(extra)
        stats.update(cursor_version=CURSOR_VERSION, next_lane=0, next_table=0)
        result.reset = "cursor_migration" if migrated else "rotated"
    progress(extra)
    extra.setdefault("wm", {})
    extra.setdefault("pending", {})


def has_fresh(connection, table, mark):
    clause = "time_updated > ? OR (time_updated = ? AND id > ?)" if mark else "1"
    args = (mark[0], mark[0], mark[1]) if mark else ()
    return (
        connection.execute(
            f"SELECT 1 FROM {table} WHERE {clause} LIMIT 1",
            args,  # noqa: S608
        ).fetchone()
        is not None
    )


def id_page(connection, table, anchor, budget):
    clause, args = ("id > ?", [anchor]) if anchor is not None else ("1", [])
    return connection.execute(
        f"SELECT id FROM {table} WHERE {clause} ORDER BY id LIMIT ?",  # noqa: S608
        [*args, budget],
    )


def scan(
    connection,
    source,
    extra,
    budget,
    byte_budget,
    result,
    fetch,
    decode,
    emit,
    maintenance,
    generation,
):
    """Bounded fresh, sampled-recheck and ID-discovery lanes, one quantum each.

    No JSON predicate or COUNT scans an unbounded table. Native ID-keyset
    maintenance revisits records (including backdated inserts and unsampled tool
    completions); the canonical ledger dedupes terminal observations. Checkpoint
    detail samples are bounded, while pending provenance is emitted separately
    into existing evidence segments. Pending totals are unknown, not sample size.
    Only missing sampled rows or lost continuity abandon observed pending tools;
    unsampled provenance remains pending/unavailable if the native source is lost.
    """
    prepare(connection, extra, source, result, emit)
    stats = progress(extra)
    marks, pending = extra["wm"], extra["pending"]
    census = extra.setdefault("census", {})
    if census and all(census.get(table, {}).get("done", False) for table in TABLES):
        # Keep last completed sweep's lifecycle facts while the next is in flight.
        stats["last_sweep_pending"] = any(census[table].get("pending", False) for table in TABLES)
        census.clear()
    epoch = extra.get("census_epoch")
    if not census or epoch is None:
        epoch = {"generation": generation, "dirty": bool(census)}
        extra["census_epoch"] = epoch
    if epoch["generation"] != generation:
        epoch["dirty"] = True  # Finish the forward sweep, then resweep; never restart each chunk.
    old_pending = {table: set(ids) for table, ids in pending.items()}
    start = int(stats.get("next_lane", 0)) if byte_budget is not None else 0
    charged = 0
    for delta in range(9):
        lane = (start + delta) % 9
        table, kind = TABLES[lane // 3], lane % 3
        ids = pending.setdefault(table, [])
        selected, sweep = [], None
        if kind == 0:
            rows = fetch(connection, table, marks, [], budget)
        elif kind == 1:
            selected = [
                record_id for record_id in ids if record_id in old_pending.get(table, set())
            ][: min(budget, DETAIL_CAP)]
            rows = (
                fetch(connection, table, marks, selected, budget, recheck=True) if selected else []
            )
        elif maintenance:
            sweep = census.setdefault(table, {})
            rows = (
                id_page(connection, table, sweep.get("id"), budget) if not sweep.get("done") else []
            )
        else:
            continue
        handled, held = set(), []
        for row in rows:
            record_id = row["id"]
            if sweep is not None:
                sweep["id"] = record_id
                stats["metadata_rows_examined"] = stats.get("metadata_rows_examined", 0) + 1
                row = next(iter(fetch(connection, table, marks, [record_id], 1, recheck=True)))
            handled.add(record_id)
            if kind == 0:
                mark = [row["time_updated"], record_id]
                marks[table] = max(marks.get(table, mark), mark)
            visits = stats["rows_visited"]
            visits[table] = visits.get(table, 0) + 1
            size = int(row["n"])
            decoded = size if table == "session" or row["data"] is not None else 0
            result.bytes_read += decoded
            stats["bytes_read"] += decoded
            stats["bytes_examined"] += size
            lane_bytes = (
                "fresh_bytes_read",
                "pending_recheck_bytes_read",
                "maintenance_bytes_read",
            )[kind]
            stats[lane_bytes] = stats.get(lane_bytes, 0) + decoded
            charged += max(size, 1)
            before = len(result.gaps)
            decode(source, table, row, held, result)
            locator = f"{table}:{record_id}"
            provenance = extra.setdefault("pending_provenance", {})
            metadata = {"id": record_id, "time_updated": row["time_updated"]}
            if table != "session":
                metadata.update(session_id=row["session_id"], message_id=row["message_id"])
            if record_id in held:
                if sweep is not None:
                    sweep["pending"] = True
                else:
                    stats["last_sweep_pending"] = True
                if table == "part":
                    data = loads(row["data"])
                    metadata.update(call_id=data.get("callID"), status="unfinished")
                sample(provenance, locator, metadata)
            else:
                provenance.pop(locator, None)
            problems = extra.setdefault("unsupported", {})
            problems.pop(locator, None)
            details = extra.setdefault("unsupported_provenance", {})
            details.pop(locator, None)
            for gap in result.gaps[before:]:
                sample(problems, locator, gap.reason)
                if gap.reason == "unsupported_tool_status":
                    data = loads(row["data"])
                    metadata.update(
                        call_id=data.get("callID"), status=obj(data.get("state")).get("status")
                    )
                sample(details, locator, metadata)
                stats["unsupported_records"] += 1
            stats["next_lane"] = (lane + 1) % 9
            stats["next_table"] = stats["next_lane"] // 3
            if byte_budget is not None and charged >= byte_budget:
                break
        if kind == 1:
            attempted = set(selected) if byte_budget is None or charged < byte_budget else handled
            for record_id in attempted - handled:
                abandon(extra, table, record_id, "pending_row_missing", source, result, emit)
            handled |= attempted
        pending[table] = [record_id for record_id in ids if record_id not in handled]
        for record_id in held:
            if record_id not in pending[table]:
                if sum(len(values) for values in pending.values()) < DETAIL_CAP:
                    pending[table].append(record_id)
                else:
                    stats["pending_unsampled"] = True
        if sweep is not None:
            sweep["done"] = next(iter(id_page(connection, table, sweep.get("id"), 1)), None) is None
        if byte_budget is not None and charged >= byte_budget:
            break
    stats["discovery_pending"] = maintenance and (
        epoch["dirty"] or any(not census.get(table, {}).get("done", False) for table in TABLES)
    )
    if maintenance and not stats["discovery_pending"]:
        stats["last_sweep_pending"] = any(census[table].get("pending", False) for table in TABLES)
        if not stats["last_sweep_pending"]:
            stats["pending_unsampled"] = False
    result.more = stats["discovery_pending"] or any(
        has_fresh(connection, table, marks.get(table)) for table in TABLES
    )
    result.pending_records = sum(len(ids) for ids in pending.values())
    stats["known_pending_lower_bound"] = result.pending_records
    if not result.pending_records and not stats.get("pending_unsampled"):
        stats["last_sweep_pending"] = False
    possibly_pending = bool(result.pending_records or stats.get("last_sweep_pending"))
    pending_unknown = possibly_pending or result.more
    stats["pending_records"] = None if pending_unknown else 0
    stats["pending_count_complete"] = not pending_unknown
    stats["snapshot_checked"] = True
    stats["status"] = (
        "lag"
        if result.more
        else "pending"
        if possibly_pending
        else "unsupported"
        if stats["unsupported_records"]
        else "abandoned"
        if stats["abandoned_records"]
        else "complete"
    )


def finish_generation(extra, result, before, after):
    """Only the accepted descriptor generation may become an idle-cache proof."""
    stats = progress(extra)
    stats["source_generation"] = after
    if before != after:
        extra.setdefault("census_epoch", {"generation": before})["dirty"] = True
        stats.update(
            discovery_pending=True, status="lag", pending_records=None, pending_count_complete=False
        )
        result.more = True


TOKEN_NAMES = {
    "total": "total_tokens",
    "input": "input_tokens",
    "output": "output_tokens",
    "reasoning": "reasoning_tokens",
}


def read_row(reader, source, table, row, held, result):
    """Reject one unrepresentable row, never its following rows or earlier evidence.

    Input bytes and fresh watermarks are owned by the cursor before this call.
    Roll back partial emissions/held IDs if any later normalization stage exceeds
    Python's recursion budget; no half-call can survive as accepted evidence.
    """
    evidence_start, held_start, gap_start = len(result.evidence), len(held), len(result.gaps)
    try:
        decode_row(reader, source, table, row, held, result)
    except RecursionError:
        del result.evidence[evidence_start:]
        del held[held_start:]
        del result.gaps[gap_start:]
        result.gaps.append(
            Gap(source.source_id, "normalization_recursion", int(row["n"]), f"{table}:{row['id']}")
        )


def decode_row(reader, source, table, row, held, result):
    locator = f"{table}:{row['id']}"
    if table == "session":
        result.evidence.append(session_evidence(reader, source, row, locator))
        return
    if row["data"] is None:
        result.gaps.append(Gap(source.source_id, "oversized_record", int(row["n"]), locator))
        return
    try:
        data = loads(row["data"])
    except (ValueError, RecursionError):
        result.gaps.append(Gap(source.source_id, "malformed_json", int(row["n"]), locator))
        return
    if not isinstance(data, dict):
        result.unrecognized["<non-object>"] = result.unrecognized.get("<non-object>", 0) + 1
        result.gaps.append(Gap(source.source_id, "unsupported_record", int(row["n"]), locator))
        return
    if table == "message":
        read_message(reader, source, row, data, locator, held, result)
    else:
        reader._part(source, row, data, locator, held, result)


def session_evidence(
    reader: OpenCodeReader, source: Source, row: Mapping[str, Any], locator: str
) -> RawEvidence:
    parent = row["parent_id"]
    body = {
        "id": row["id"],
        "parent_id": parent,
        "directory": row["directory"],
        "project_id": row["project_id"],
    }
    branch = "child" if parent else "main"
    return reader._evidence(
        source, row["id"], f"session:{row['id']}", "session_header", locator, None, body, branch
    )


def read_message(
    reader: OpenCodeReader,
    source: Source,
    row: Mapping[str, Any],
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
        result.evidence.append(
            reader._evidence(
                source,
                row["session_id"],
                f"pending-message:{row['id']}",
                "pending_message",
                locator,
                stamps.get("created"),
                {
                    "id": row["id"],
                    "message_id": row["id"],
                    "session_id": row["session_id"],
                    "status": "pending",
                },
            )
        )
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
        reader._evidence(
            source,
            row["session_id"],
            f"usage:{row['id']}",
            "usage",
            locator,
            stamps.get("completed") or stamps.get("created"),
            body,
        )
    )
