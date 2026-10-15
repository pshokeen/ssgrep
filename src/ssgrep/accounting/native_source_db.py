"""OpenCode historical source rows from one scoped, bounded read-only snapshot."""

from __future__ import annotations

import hashlib
import json
import os

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES
from ssgrep.accounting.native_source import (
    MAX_SOURCE_RECORDS,
    SOURCE_KINDS,
    NativeSourceStatus,
    SourceReadError,
)
from ssgrep.accounting.readers.base import ReadResult
from ssgrep.accounting.readers.opencode import REQUIRED
from ssgrep.accounting.readers.opencode_snapshot import (
    SnapshotChanged,
    SnapshotUnavailable,
    named_row,
)
from ssgrep.accounting.readers.values import loads

USER_ROLE_SQL = (
    "(SELECT CASE WHEN length(CAST(m.data AS BLOB)) > 8388608 THEN 'source_read_bound' "
    "ELSE json_extract(m.data, '$.role') END FROM message AS m "
    "WHERE m.id = part.message_id AND m.session_id = part.session_id) AS message_role"
)


def session_row(connection, session_id, budget):
    """One native owner row, bounded before strings enter Python."""
    fields = ", ".join(
        f"CASE WHEN length(CAST({name} AS BLOB)) > 16384 THEN NULL ELSE {name} END AS {name}"
        for name in ("parent_id", "directory", "project_id")
    )
    size = " + ".join(
        f"coalesce(length(CAST({name} AS BLOB)), 0)"
        for name in ("id", "parent_id", "directory", "project_id")
    )
    row = connection.execute(
        f"SELECT id, {fields}, time_updated, NULL AS data, ({size}) AS n "
        "FROM session WHERE id = ? LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None:
        raise SourceReadError(NativeSourceStatus.MISSING)
    if row["n"] > min(16384, budget):
        raise SourceReadError(NativeSourceStatus.BOUNDED)
    return row


def user_part(reader, source, row, result):
    """Native user text is prior knowledge, not a discovered tool witness."""
    if row["message_role"] == "source_read_bound":
        raise SourceReadError(NativeSourceStatus.BOUNDED)
    if row["message_role"] != "user":
        return False
    try:
        data = loads(row["data"])
    except (ValueError, TypeError, RecursionError):
        raise SourceReadError(NativeSourceStatus.INCOMPLETE) from None
    if not isinstance(data, dict) or data.get("type") != "text":
        return False
    text = data.get("text")
    if isinstance(text, str) and text:
        result.evidence.append(
            reader._evidence(
                source,
                row["session_id"],
                f"turn:{row['id']}",
                "user_turn",
                f"part:{row['id']}",
                row["time_updated"],
                {"text": text, "cwd": row["cwd"]},
            )
        )
    return True


def source_rows(connection, table, reference, cwd, limit, budget):
    """Whole selected session once, or only a previously verified native row set."""
    if len(reference.row_locators) > MAX_SOURCE_RECORDS:
        raise SourceReadError(NativeSourceStatus.BOUNDED)
    ids = sorted(
        {
            locator.split(":", 1)[1]
            for locator in reference.row_locators
            if locator.startswith(table + ":")
        }
    )
    if any(len(row_id) > 512 for row_id in ids):
        raise SourceReadError(NativeSourceStatus.BOUNDED)
    if reference.row_locators and not ids:
        return
    batches = (
        [ids[start : start + 500] for start in range(0, len(ids), 500)]
        if reference.row_locators
        else [()]
    )
    message_id = (
        "CASE WHEN length(message_id) > 512 THEN NULL ELSE message_id END AS message_id"
        if table == "part"
        else "NULL AS message_id"
    )
    role = USER_ROLE_SQL if table == "part" else "NULL AS message_role"
    for batch in batches:
        clause = " AND id IN (" + ",".join("?" for _ in batch) + ")" if batch else ""
        query = (
            "SELECT CASE WHEN length(id) > 512 THEN NULL ELSE id END AS id, "
            f"session_id, {message_id}, {role}, time_updated, ? AS cwd, "
            "CASE WHEN length(CAST(data AS BLOB)) > ? THEN NULL ELSE data END AS data, "
            f"length(CAST(data AS BLOB)) AS n FROM {table} WHERE session_id = ?{clause} "
            "ORDER BY time_updated, id LIMIT ?"
        )
        yield from connection.execute(
            query, (cwd, min(MAX_RAW_RECORD_BYTES, budget), reference.session_id, *batch, limit)
        )


def read_session(reader, reference, descriptor, budget, max_records):
    """Use native IDs and the existing decoder; never scan unrelated sessions.

    SQL bounds data before materialization; a progress handler bounds query work.
    The database digest covers selected logical rows (including timing), not the
    main file alone, so committed WAL changes remain observable.
    """
    try:
        import apsw
    except ImportError:
        raise SourceReadError(NativeSourceStatus.UNREADABLE) from None
    source = reader.source_for(reference.path)
    info = os.fstat(descriptor)
    result = ReadResult()
    count = rows_read = 0
    row_limit = min(MAX_SOURCE_RECORDS, max_records)
    steps = 0

    def progress():
        nonlocal steps
        steps += 1
        return int(steps > 1000)

    try:
        with reader.snapshot(reference.path, descriptor=descriptor) as connection:
            connection.set_progress_handler(progress, 1000)
            connection.set_row_trace(named_row)
            for table, columns in REQUIRED.items():
                found = {row["name"] for row in connection.execute(f'PRAGMA table_info("{table}")')}
                if not columns <= found:
                    raise SourceReadError(NativeSourceStatus.UNSUPPORTED)
            session = session_row(connection, reference.session_id, budget)
            count += session["n"]
            reader._row(source, "session", session, [], result)
            for table in ("message", "part"):
                cursor = source_rows(
                    connection,
                    table,
                    reference,
                    session["directory"],
                    row_limit + 1,
                    max(0, budget - count),
                )
                held = []
                for row in cursor:
                    rows_read += 1
                    if rows_read > row_limit or row["data"] is None or row["id"] is None:
                        raise SourceReadError(NativeSourceStatus.BOUNDED)
                    size = int(row["n"] or 0)
                    if count + size > budget:
                        raise SourceReadError(NativeSourceStatus.BOUNDED)
                    count += size
                    if table != "part" or not user_part(reader, source, row, result):
                        reader._row(source, table, row, held, result)
                    if len(result.evidence) > row_limit:
                        raise SourceReadError(NativeSourceStatus.BOUNDED)
                if held or result.gaps:
                    raise SourceReadError(NativeSourceStatus.INCOMPLETE)
            connection.set_progress_handler(None, 0)
    except SourceReadError as error:
        error.bytes_read = count
        raise
    except SnapshotChanged:
        raise SourceReadError(NativeSourceStatus.CHANGED, count) from None
    except SnapshotUnavailable:
        raise SourceReadError(NativeSourceStatus.UNREADABLE, count) from None
    except apsw.Error as error:
        status = (
            NativeSourceStatus.BOUNDED
            if isinstance(error, apsw.InterruptError)
            else NativeSourceStatus.UNREADABLE
        )
        raise SourceReadError(status, count) from None
    # SQLite has no descriptor-open API: reject replacement between the securely
    # checked descriptor and the path used for the native WAL-aware connection.
    try:
        after = os.stat(reference.path, follow_symlinks=False)
    except FileNotFoundError:
        raise SourceReadError(NativeSourceStatus.MISSING, count) from None
    except OSError:
        raise SourceReadError(NativeSourceStatus.UNREADABLE, count) from None
    if (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino):
        raise SourceReadError(NativeSourceStatus.UNAPPROVED, count)
    evidence = [item for item in result.evidence if item.kind in SOURCE_KINDS]
    # Pin native measurements, not session.time_updated or uncredited usage rows:
    # those can legitimately change when new messages append to the same session.
    manifest = sorted((item.native_identity, item.digest, item.ts_utc) for item in evidence)
    digest = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    return evidence, digest, count
