"""Save knowledge and its optional original investment, without changing index schemas."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from ssgrep.utilities.types import EpisodeDetail


def load_work_file(path: str) -> dict[str, Any]:
    """Read one bounded JSON object, never block on a FIFO or follow a link."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("work metadata must be a regular JSON file")
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("work metadata exceeds 64 KiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("work metadata must be a JSON object")
    return value


def work_for(detail: EpisodeDetail) -> dict[str, Any] | None:
    """Read optional metadata; never infer history or write during retrieval."""
    from ssgrep.store.paths import data_dir

    try:
        (data_dir() / "work-metadata").lstat()
    except OSError:
        return None
    from ssgrep.sessions.work import read_work, subject_digest

    return read_work(detail.episode_id, subject_digest(detail))


def _index_saved_note() -> None:
    """A fresh process owns the index engine; MCP may already hold its startup engine."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c", "from ssgrep.services.api import index; index(quiet=True)"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=300,
    )


def save_note(
    title: str | None = None,
    body: str | None = None,
    project_dir: str | None = None,
    work: dict[str, Any] | None = None,
    ref: str | None = None,
    infer_work: bool = False,
) -> dict[str, Any]:
    """Create a native note or enrich an existing ref; original investment is not savings.

    Explicit metadata is caller-supplied, not an attestation of provider billing.
    Historical inference is opt-in. Updating a shared work_id replaces its current
    investment revision for every linked note, rather than adding the cost again.
    """
    from ssgrep.search import detail as episode_detail
    from ssgrep.sessions import notes
    from ssgrep.sessions.work import attach_work, subject_digest, validate_work

    if ref is not None:
        if title is not None or body is not None or project_dir is not None:
            raise ValueError("enrichment accepts ref and work, not note content or project_dir")
        if (work is None) == (not infer_work):
            raise ValueError("enrichment needs exactly one of work or infer_work")
        original = episode_detail.show(ref)
        if original is None:
            return {"ok": False, "condition": "unknown_ref", "message": "Episode not found."}
        if infer_work:
            from ssgrep.sessions.work_evidence import infer_work as recover

            work = recover(original)
            if work is None:
                return {
                    "ok": False,
                    "condition": "work_evidence_unavailable",
                    "message": "No trustworthy work metadata could be recovered; nothing changed.",
                    "ref": ref,
                }
        assert work is not None
        envelope = attach_work(ref, subject_digest(original), validate_work(work))
        return {"ok": True, "ref": ref, "enriched": True, "work": envelope}
    if infer_work:
        raise ValueError("infer_work requires an existing ref")
    if title is None or not title.strip() or body is None or not body.strip():
        raise ValueError("title and body are required when saving a new note")
    canonical = validate_work(work) if work is not None else None
    project = Path(project_dir or Path.cwd()).expanduser().absolute()
    from ssgrep.accounting.capture import capture
    from ssgrep.accounting.ledger.contracts import Operation, Outcome

    execution = capture.begin(Operation.NOTE, text=title, cwd=str(project))
    try:
        shard, saved_ref = notes.append_note(project, title, body)
    except BaseException as error:
        capture.fail(execution, error)
        raise
    capture.complete(execution, Outcome.OK)
    result: dict[str, Any] = {
        "ok": True,
        "saved": True,
        "indexed": False,
        "ref": saved_ref,
        "note_file": str(shard),
    }
    if canonical is not None:
        from ssgrep.sessions.signal import strip_markup

        # This is the unchanged native note text exposed by the existing detail
        # reader, including its display bounds. Persist metadata before indexing
        # so an index failure does not discard the supplied investment.
        saved = EpisodeDetail(
            episode_id=saved_ref,
            session_id=shard.stem,
            title=title,
            timestamp=None,
            git_branch=None,
            cwd=str(project),
            prompt_text=episode_detail._apply_bound(
                strip_markup(title) or title, episode_detail.MAX_PROMPT_CHARS
            ),
            response_text=episode_detail._apply_bound(
                strip_markup(body) or body, episode_detail.MAX_RESPONSE_CHARS
            ),
            files_touched=(),
            tool_names=(),
            is_subagent=False,
            runtime="ssgrep",
        )
        try:
            result["work"] = attach_work(saved_ref, subject_digest(saved), canonical)
        except (OSError, ValueError):
            return {
                **result,
                "ok": False,
                "condition": "work_metadata_failed",
                "message": "Note saved; metadata failed. Index the note, then enrich its ref.",
            }
    try:
        _index_saved_note()
    except Exception as error:
        return {
            **result,
            "ok": False,
            "condition": "index_failed",
            "index_error": {
                "type": type(error).__name__,
                "exit_code": getattr(error, "returncode", None),
            },
            "command": "ssgrep index",
            "message": (
                f"Note saved; indexing failed ({type(error).__name__}). "
                "Run ssgrep index for diagnostics, not another save_note."
            ),
        }
    return {**result, "indexed": True}


def work_text(envelope: dict[str, Any]) -> str:
    """A compact local detail; never present investment as avoided work."""
    work = envelope["work"]
    providers = work.get("contributions") or [work]
    models = ", ".join(
        f"{item.get('provider', 'unknown provider')} / {item.get('model', 'unknown model')}"
        for item in providers
    )
    duration = work.get("elapsed_seconds")
    elapsed = f"{duration} s elapsed" if duration is not None else "elapsed time unknown"
    return (
        f"Original investigation: {models}; {elapsed}; {work.get('basis', 'unknown')} "
        f"(work {envelope['work_id']}, revision {envelope['revision']}). "
        "This is original investment, not measured savings."
    )
