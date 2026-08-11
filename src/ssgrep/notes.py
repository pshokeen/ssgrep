"""Durable, searchable notes: authored content written into the index's own root.

Field-validated product direction: a deployment hand-synthesized "lesson"
episodes -- user/assistant pairs phrased as the questions they'd later search
for -- and got rank-1 retrieval on every probe (scores above the corpus's
measured max). Their only obstacle was reverse-engineering the record schema.
This module is that path, supported: `ssgrep note` writes a NATIVE transcript
record pair, so notes ride the exact records -> episodes -> chunker -> embed
pipeline real transcripts do. No parallel code path exists to fork.

Trust boundary (the design's "guard inversion"): this is ssgrep's first
intentional write path to indexed content, and it writes ONLY under
`<index_dir>/notes/` -- never under the transcript root. The buyer's Claude
Code history at ~/.claude is read-only to ssgrep, unconditionally; the test
suite's session guards enforce it, and test_notes.py mutation-guards the
write target directly. Notes are also deliberately CLI-only: no MCP write
tool, because a write capability reachable by a prompt-injected agent is a
new trust class (it could pollute the index its user trusts).
"""

from __future__ import annotations

import fcntl
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ssgrep.types import SessionFile

NOTES_DIRNAME = "notes"


def notes_dir(index_dir: Path) -> Path:
    """The notes root inside an index directory. Never under ~/.claude."""
    return index_dir / NOTES_DIRNAME


def write_note(
    project_dir: Path,
    title: str,
    body: str,
    *,
    index_dir: Path | None = None,
) -> Path:
    """Append one note (a native user/assistant record pair) and return its file.

    The user turn carries the title -- phrase it as the question you will
    later search for; that is the measured-best shape. The assistant turn
    carries the body. Records carry this project's cwd so scope matching
    treats notes like any transcript recorded here. Files are append-only
    monthly shards (notes-YYYYMM.jsonl), so staleness cursors and bounded
    tail repair work on them unchanged.
    """
    if not title.strip():
        raise ValueError("note title must not be empty")
    if not body.strip():
        raise ValueError("note body must not be empty")

    target_dir = notes_dir(index_dir or (project_dir / ".ssgrep"))
    target_dir.mkdir(parents=True, exist_ok=True)

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
    # ONE buffered payload, ONE write, under an exclusive flock: two
    # concurrent `ssgrep note` invocations used to interleave their two
    # separate f.write() calls at the line level ([A-user, B-user, A-asst,
    # B-asst]), and because episode segmentation is record-ORDER based, that
    # silently lost one note's answer and contaminated the other's -- a
    # blind-review blocker reproduced end-to-end through the real
    # segmentation code. The lock is advisory-POSIX (macOS/Linux, the only
    # supported platforms) and held only for the single append.
    payload = json.dumps(user_record) + "\n" + json.dumps(assistant_record) + "\n"
    with open(shard, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(payload)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return shard


def discover_notes(index_dir: Path) -> list[SessionFile]:
    """SessionFile entries for every note shard in this index, best-effort.

    Notes are ssgrep-owned and always in scope (they were written FOR this
    index); they bypass cwd matching entirely, so a scope override
    (`--scope /old/path`) never hides this index's own notes. Missing or
    unreadable dirs yield [] -- notes are an adjunct, never a reason to fail
    an index run or a staleness check.
    """
    root = notes_dir(index_dir)
    if not root.is_dir():
        return []
    out: list[SessionFile] = []
    for shard in sorted(root.glob("notes-*.jsonl")):
        try:
            stat = shard.stat()
        except OSError:
            continue
        out.append(
            SessionFile(
                path=shard,
                session_id=shard.stem,
                is_main=True,
                size=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    return out
