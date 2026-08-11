"""Note command for ssgrep: write a durable, searchable note into the index."""

from __future__ import annotations

import json
import os
import sys

from usecli.cli.core.base_command import BaseCommand
from usecli.cli.core.runtime import is_json_mode, is_quiet

from ssgrep import api, notes
from ssgrep.cli import exit_codes
from ssgrep.cli.commands import resolve_project_dir, to_jsonable


class NoteCommand(BaseCommand):
    """Write a note into the project's index."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "note"

    def description(self) -> str:
        """Command description."""
        return "Write a durable, searchable note into the project's index"

    def handle(
        self,
        title: str = "",
        body: str = "",
        project_dir: str = ".",
    ) -> object:
        """Write the note as a native transcript record pair, then reindex.

        --title should be phrased as the question you will later search for
        (the measured-best shape -- see notes.py). --body is the content;
        pass `-` to read it from stdin. The note lands under
        `<project>/.ssgrep/notes/` -- never under ~/.claude -- and is
        immediately searchable after the quiet incremental reindex this
        command runs.
        """
        if not title.strip():
            print(
                "ssgrep note: --title is required (and becomes the searchable question).",
                file=sys.stderr,
            )
            raise SystemExit(exit_codes.USAGE_ERROR)
        body_text = sys.stdin.read() if body == "-" else body
        if not body_text.strip():
            print("ssgrep note: --body is required (use '-' to read from stdin).", file=sys.stderr)
            raise SystemExit(exit_codes.USAGE_ERROR)

        project = resolve_project_dir(project_dir)
        shard = notes.write_note(project, title, body_text)
        try:
            stats = api.index(project, quiet=True)
        except Exception as error:
            # Mirror IndexCommand's convention: never a raw traceback, and
            # the --json contract (one machine-readable document) holds on
            # every path. The note itself is SAFE on disk -- say so, always:
            # the failure is the reindex, not the write (blind-review
            # finding: a missing transcript root crashed this command with
            # a bare traceback and zero JSON output).
            condition = getattr(error, "condition", "index_failed")
            command = getattr(error, "command", None)
            if is_json_mode():
                error_doc = {
                    "ok": False,
                    "condition": condition,
                    "message": f"Note written to {shard}, but reindexing failed: {error}",
                    "note_file": str(shard),
                    "command": command,
                }
                sys.__stdout__.write(json.dumps(error_doc) + "\n")  # type: ignore[union-attr]
                sys.__stdout__.flush()  # type: ignore[union-attr]
                os._exit(exit_codes.INTERNAL_FAILURE)
            print(f"Note written to {shard}", file=sys.stderr)
            print(f"ssgrep note: reindexing failed: {error}", file=sys.stderr)
            print(
                "The note is saved and will be indexed by the next successful "
                "`ssgrep index` run.",
                file=sys.stderr,
            )
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from error

        if is_json_mode():
            payload = {"ok": True, "note_file": str(shard), "index": to_jsonable(stats)}
            return payload
        if not is_quiet():
            print(f"Note written to {shard}")
            print(
                f"Indexed {stats.session_count} sessions, "
                f"{stats.episode_count} episodes, {stats.chunk_count} chunks."
            )
        return None
