"""Build or update the global transcript index."""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated, NoReturn

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode, is_quiet

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import to_jsonable
from ssgrep.cli.render import render_index_stats_table
from ssgrep.services import api
from ssgrep.utilities.types import IndexNotReadyError, RebuildWouldShrinkError, SearchException

#: Conventional exit status for a process ended by SIGINT (128 + 2).
INTERRUPTED = 130


def _fail(error: SearchException, code: int, *, immediate: bool = False) -> NoReturn:
    if is_json_mode():
        stdout = sys.__stdout__
        assert stdout is not None
        stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "condition": error.condition,
                    "message": str(error),
                    "command": error.command,
                }
            )
            + "\n"
        )
        stdout.flush()
        os._exit(code)
    print(error, file=sys.stderr)
    if immediate:
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)


class IndexCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "index"

    def description(self) -> str:
        return "Build or update the global all-project session index"

    def handle(
        self,
        rebuild: Annotated[bool, Option("--rebuild", help="Recreate all Lance tables")] = False,
        no_subagents: Annotated[
            bool, Option("--no-subagents", help="Exclude subagent transcripts")
        ] = False,
        quiet: Annotated[bool, Option("--quiet", help="Suppress progress output")] = False,
        allow_shrink: Annotated[
            bool,
            Option("--allow-shrink", help="Confirm a potentially shrinking rebuild"),
        ] = False,
        scope: Annotated[
            str | None,
            Option("--scope", help="Only ingest transcripts recorded beneath this path"),
        ] = None,
        live: Annotated[
            bool,
            Option(
                "--live",
                help="Keep polling for changes in the foreground until interrupted "
                "(no daemon is installed)",
            ),
        ] = False,
        full_reprocess: Annotated[
            bool,
            Option("--full-reprocess", help="Re-run every source, ignoring memoized state"),
        ] = False,
    ) -> dict[str, object] | None:
        try:
            stats = api.index(
                rebuild=rebuild,
                no_subagents=no_subagents,
                allow_shrink=allow_shrink,
                scope=scope,
                quiet=quiet,
                live=live,
                full_reprocess=full_reprocess,
            )
        except RebuildWouldShrinkError as error:
            _fail(error, exit_codes.USAGE_ERROR)
        except IndexNotReadyError as error:
            _fail(error, exit_codes.MISSING_INDEX)
        except KeyboardInterrupt:
            # The engine driver has already stopped what it could, so the
            # index is left in a resumable "building" state. The engine's own
            # worker tasks cannot be stopped from here and would print
            # "Task was destroyed but it is pending!" while the interpreter
            # tears down, hence the immediate exit.
            _fail(
                SearchException(
                    "Index build interrupted; run `ssgrep index` to resume it.",
                    condition="interrupted",
                    command="ssgrep index",
                ),
                INTERRUPTED,
                immediate=True,
            )
        except Exception as error:
            if is_json_mode():
                raise
            print(f"Index operation failed: {error}", file=sys.stderr)
            if not rebuild and getattr(error, "command", None) is None:
                print("Try `ssgrep index --rebuild` to force a full rebuild.", file=sys.stderr)
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from error

        if is_json_mode():
            return to_jsonable(stats)
        if not (quiet or is_quiet()):
            render_index_stats_table(stats)
        return None
