"""Author native JSONL note episodes beneath the global data root."""

from __future__ import annotations

import fcntl
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ssgrep.store.paths import data_dir, ensure_data_dir
from ssgrep.utilities.types import SessionFile

NOTES_DIRNAME = "notes"


def notes_dir() -> Path:
    return data_dir() / NOTES_DIRNAME


def write_note(
    project_dir: Path,
    title: str,
    body: str,
) -> Path:
    """Append one note (a native user/assistant record pair) and return its file.

    The user turn carries the title -- phrase it as the question you will
    later search for; that is the measured-best shape. The assistant turn
    carries the body. Records carry this project's cwd so scope matching
    treats notes like any transcript recorded here. Files are append-only
    monthly shards (notes-YYYYMM.jsonl), using the native transcript shape.
    """
    if not title.strip():
        raise ValueError("note title must not be empty")
    if not body.strip():
        raise ValueError("note body must not be empty")

    ensure_data_dir()
    target_dir = notes_dir()
    target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    now = datetime.now(UTC)
    shard = target_dir / f"notes-{now:%Y%m}.jsonl"
    session_id = f"note-{now:%Y%m}"
    note_uuid = uuid.uuid4().hex[:12]
    timestamp = now.isoformat().replace("+00:00", "Z")
    cwd = str(project_dir)

    user_record = {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": title},
        "uuid": f"note-{note_uuid}-u",
        "timestamp": timestamp,
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "",
    }
    assistant_record = {
        "parentUuid": f"note-{note_uuid}-u",
        "isSidechain": False,
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": body}]},
        "uuid": f"note-{note_uuid}-a",
        "timestamp": timestamp,
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "",
    }
    # Keep each prompt/response pair contiguous across concurrent writers.
    payload = json.dumps(user_record) + "\n" + json.dumps(assistant_record) + "\n"
    with open(shard, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(payload)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return shard


def discover_notes() -> list[SessionFile]:
    """SessionFile entries for every note shard in this index, best-effort.

    Notes are ssgrep-owned and always in scope (they were written FOR this
    index); they bypass cwd matching entirely, so a scope override
    (`--scope /old/path`) never hides this index's own notes. Missing or
    unreadable dirs yield [] -- notes are an adjunct, never a reason to fail
    a global index run.
    """
    root = notes_dir()
    if not root.is_dir():
        return []
    out: list[SessionFile] = []
    for shard in sorted(root.glob("notes-*.jsonl")):
        out.append(
            SessionFile(
                path=shard,
                session_id=shard.stem,
                is_main=True,
            )
        )
    return out
