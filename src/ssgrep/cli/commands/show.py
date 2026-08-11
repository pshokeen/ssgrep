"""Show command for ssgrep."""

from __future__ import annotations

import json
import os
import sys

from usecli.cli.core.base_command import BaseCommand
from usecli.cli.core.runtime import is_json_mode

from ssgrep import api
from ssgrep.cli import exit_codes
from ssgrep.cli.commands import resolve_project_dir, to_jsonable
from ssgrep.render import render_episode_detail
from ssgrep.types import EpisodeDetail, IndexNotFoundError, IndexNotReadyError


class ShowCommand(BaseCommand):
    """Show full context for one episode."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "show"

    def description(self) -> str:
        """Command description."""
        return "Show full, untruncated context for a single episode"

    def handle(self, ref: str, project_dir: str = ".") -> object:
        """Print the full prompt and response for the episode identified by ref."""
        project = resolve_project_dir(project_dir)

        try:
            detail: EpisodeDetail | None = api.show(project, ref)
        except (IndexNotFoundError, IndexNotReadyError) as error:
            if is_json_mode():
                # Write error JSON directly and exit, bypassing usecli's exception wrapping.
                # Use os._exit() to avoid all Python exception handling and usecli's
                # stdout redirection. Use sys.__stdout__ to write to the real stdout.
                error_doc = {
                    "ok": False,
                    "condition": error.condition,
                    "message": str(error),
                    "command": error.command,
                }
                sys.__stdout__.write(json.dumps(error_doc) + "\n")  # type: ignore
                sys.__stdout__.flush()  # type: ignore
                os._exit(exit_codes.MISSING_INDEX)
            else:
                print(error, file=sys.stderr)
                raise SystemExit(exit_codes.MISSING_INDEX) from error

        if detail is None:
            message = (
                f"Episode not found: {ref}. " "Run `ssgrep search <query>` to find valid refs."
            )
            if is_json_mode():
                # Write error JSON directly and exit with the same code as plain mode.
                # Unknown ref is NO_MATCHING_DATA (exit 3), not MISSING_INDEX (exit 4).
                error_doc = {
                    "ok": False,
                    "condition": "unknown_ref",
                    "message": message,
                    "command": "ssgrep search <query>",
                }
                sys.__stdout__.write(json.dumps(error_doc) + "\n")  # type: ignore
                sys.__stdout__.flush()  # type: ignore
                os._exit(exit_codes.NO_MATCHING_DATA)
            else:
                print(message, file=sys.stderr)
                raise SystemExit(exit_codes.NO_MATCHING_DATA)

        if is_json_mode():
            return to_jsonable(detail)

        print(render_episode_detail(detail))
        return None
