"""Read existing index locator metadata only; never normalized transcript text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ssgrep.accounting.native_source import NativeSourceReference, NativeSourceStatus

COLUMNS = ["episode_id", "session_id", "runtime", "source_path", "cwd", "source_status"]
MAX_LOCATOR_BYTES = 16 * 1024


def index_reference(ref: str, repository: Any = None):
    """One bounded episode lookup using today's existing v6 index fields.

    No prior native digests exist in that schema: they are observed by the native
    adapter and may subsequently be persisted as minimal source references.
    """
    if not isinstance(ref, str) or len(ref.encode()) > 512 or ":ep:" not in ref:
        return None, NativeSourceStatus.UNRESOLVED
    owned = repository is None
    try:
        if owned:
            from ssgrep.store import LanceStore

            repository = LanceStore()
        if not repository.exists():
            return None, NativeSourceStatus.UNRESOLVED
        if repository.get_meta("index_state") != "ready":
            return None, NativeSourceStatus.UNRESOLVED
        generation = repository.get_meta("last_index_time")
        literal = "'" + ref.replace("'", "''") + "'"
        rows = list(
            repository.rows("episodes", where=f"episode_id = {literal}", columns=COLUMNS, limit=2)
        )
        if len(rows) != 1:
            return None, NativeSourceStatus.UNRESOLVED
        row = rows[0]
        if sum(len(str(row.get(column, "")).encode()) for column in COLUMNS) > MAX_LOCATOR_BYTES:
            return None, NativeSourceStatus.BOUNDED
        if (
            repository.get_meta("index_state") != "ready"
            or repository.get_meta("last_index_time") != generation
        ):
            return None, NativeSourceStatus.CHANGED
        status = row.get("source_status", "available")
        if status in {"expired", "purged"}:
            return None, NativeSourceStatus.EXPIRED
        if status in {"absent", "missing"}:
            return None, NativeSourceStatus.MISSING
        if status != "available":
            return None, NativeSourceStatus.UNAPPROVED
        runtime, session, path = row.get("runtime"), row.get("session_id"), row.get("source_path")
        if not (
            isinstance(runtime, str)
            and runtime
            and isinstance(session, str)
            and session
            and isinstance(path, str)
            and path
        ):
            return None, NativeSourceStatus.IDENTITY_MISMATCH
        # Native index adapters namespace non-Claude session IDs once.
        if runtime != "claude" and session.startswith(runtime + ":"):
            session = session[len(runtime) + 1 :]
        cwd = row.get("cwd")
        indexed_session = row["session_id"] if runtime == "claude" else None
        return NativeSourceReference(
            ref,
            runtime,
            session,
            Path(path),
            cwd=cwd if isinstance(cwd, str) else None,
            indexed_session_id=indexed_session,
        ), NativeSourceStatus.OK
    except (OSError, ValueError, KeyError, RuntimeError, ImportError):
        return None, NativeSourceStatus.UNRESOLVED
    finally:
        if owned and repository is not None:
            repository.close()
