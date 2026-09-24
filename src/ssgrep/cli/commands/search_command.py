"""Search command for ssgrep."""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated

from usecli import Argument, BaseCommand, Console, Option, Spinner, console
from usecli.cli.core.runtime import is_json_mode

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import to_jsonable
from ssgrep.search.render import render_search_response
from ssgrep.services import api
from ssgrep.utilities.types import (
    EmptyQueryError,
    IndexNotFoundError,
    IndexNotReadyError,
    InvalidPredicateError,
    SearchResponse,
)


class SearchCommand(BaseCommand):
    """Search the global session index, optionally using a Lance ``where`` predicate."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "search"

    def description(self) -> str:
        """Command description."""
        return "Search transcripts across all projects"

    def handle(
        self,
        query: Annotated[str, Argument(help="Text to find across indexed sessions")],
        limit: Annotated[int, Option("--limit", min=0, help="Maximum result cards")] = 10,
        token_budget: Annotated[
            int,
            Option("--token-budget", min=0, help="Approximate response token budget"),
        ] = 1500,
        where: Annotated[
            str | None,
            Option("--where", help="Lance SQL metadata predicate applied before ranking"),
        ] = None,
    ) -> dict[str, object] | None:
        """Search the global index and print ranked result cards."""
        try:
            if not is_json_mode():
                console.line()
            # The Spinner auto-suppresses in JSON mode (and when not a TTY),
            # so it only renders during interactive, human-readable searches.
            with Spinner("Searching transcripts"):
                response: SearchResponse = api.search(
                    query,
                    limit=limit,
                    token_budget=token_budget,
                    where=where,
                )
        except (EmptyQueryError, InvalidPredicateError) as error:
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
                os._exit(exit_codes.USAGE_ERROR)
            else:
                print(error, file=sys.stderr)
                raise SystemExit(exit_codes.USAGE_ERROR) from error
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

        # Check if query executed but found no results
        if not response.results and response.total_matches == 0:
            # Distinguish between empty scope (no sessions) and populated index (no matches)
            if response.index_empty:
                if is_json_mode():
                    document = {
                        "ok": False,
                        "condition": "index_empty",
                        "message": "The global index contains no searchable chunks.",
                        "command": "ssgrep index",
                    }
                    sys.__stdout__.write(json.dumps(document) + "\n")  # type: ignore
                    sys.__stdout__.flush()  # type: ignore
                    os._exit(exit_codes.NO_MATCHING_DATA)
                print("The global index contains no searchable chunks.", file=sys.stderr)
            else:
                print("No results found for this query.", file=sys.stderr)
            raise SystemExit(exit_codes.NO_MATCHING_DATA)

        if is_json_mode():
            return to_jsonable(response)

        Console().print(render_search_response(response, query=query))
        return None
