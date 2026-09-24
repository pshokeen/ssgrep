"""Show global index state."""

from __future__ import annotations

import json
import os
import sys

from usecli import BaseCommand
from usecli.cli.core.runtime import is_json_mode

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import to_jsonable
from ssgrep.cli.render import render_index_stats_table
from ssgrep.services import api
from ssgrep.utilities.types import IndexNotReadyError


class StatusCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "status"

    def description(self) -> str:
        return "Show global index counts, size, model, and archive status"

    def handle(self) -> dict[str, object] | None:
        try:
            stats = api.status()
        except IndexNotReadyError as error:
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
            print(error, file=sys.stderr)
            raise SystemExit(exit_codes.MISSING_INDEX) from error
        except Exception as error:
            if is_json_mode():
                raise
            print(f"Status failed: {error}", file=sys.stderr)
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from error

        if is_json_mode():
            return to_jsonable(stats)
        if not stats.index_exists:
            print("No global index found. Run `ssgrep index` to build it.")
            return None

        render_index_stats_table(stats)
        return None
