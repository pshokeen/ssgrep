"""Bounded snapshot-local OpenCode keysets; native SQLite remains row authority."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ssgrep.accounting.readers.base import Gap
from ssgrep.accounting.readers.values import loads, obj

if TYPE_CHECKING:
    import apsw

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
