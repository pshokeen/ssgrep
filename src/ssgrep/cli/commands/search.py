"""Search command for ssgrep."""

from __future__ import annotations

import json
import os
import sys

from usecli.cli.core.base_command import BaseCommand
from usecli.cli.core.runtime import is_json_mode

from ssgrep import api, indexer_support, scope_report
from ssgrep.cli import exit_codes
from ssgrep.cli.commands import resolve_project_dir, to_jsonable
from ssgrep.render import render_search_response
from ssgrep.types import (
    EmptyQueryError,
    IndexNotFoundError,
    IndexNotReadyError,
    SearchResponse,
)


class SearchCommand(BaseCommand):
    """Search the local session index."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "search"

    def description(self) -> str:
        """Command description."""
        return "Search session transcripts for a query"

    def handle(
        self,
        query: str,
        limit: int = 10,
        token_budget: int = 1500,
        project_dir: str = ".",
    ) -> object:
        """Search the current project's index and print ranked result cards."""
        try:
            project = resolve_project_dir(project_dir)
            response: SearchResponse = api.search(
                project,
                query,
                limit=limit,
                token_budget=token_budget,
            )
        except EmptyQueryError as error:
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
        if not response.results:
            # Distinguish between empty scope (no sessions) and populated index (no matches)
            if response.index_empty:
                # An index that exists but holds nothing is the downstream
                # face of a zero discovery: the same scope mismatch that
                # printed "Indexed 0 sessions" shows up here as "nothing to
                # search". Explain it here too -- search is where a user
                # notices, and where they decide whether to reach for the
                # destructive --rebuild.
                if is_json_mode():
                    # A structured caller must get the explanation as data on
                    # stdout, not prose on stderr: the same zero-discovery
                    # census `index --json` already emits, plus the condition
                    # and remedy. Same direct-write/os._exit pattern as the
                    # error handlers above, for the same reason.
                    document = {
                        "ok": False,
                        "condition": "index_empty",
                        "message": "No sessions recorded in this project to search.",
                        "command": "ssgrep index",
                        # Census the EFFECTIVE (persisted) scope, not raw
                        # project_dir -- a scoped index's empty result must
                        # name the scope actually responsible (blind-review
                        # finding: the diagnostic + remedy were computed
                        # against the wrong baseline on 3 sites).
                        "zero_discovery": scope_report.build_scope_report(
                            indexer_support.effective_scope(project)
                        ).as_payload(),
                    }
                    sys.__stdout__.write(json.dumps(document) + "\n")  # type: ignore
                    sys.__stdout__.flush()  # type: ignore
                    os._exit(exit_codes.NO_MATCHING_DATA)
                print("No sessions recorded in this project to search.", file=sys.stderr)
                print(
                    scope_report.describe_zero_discovery(indexer_support.effective_scope(project)),
                    file=sys.stderr,
                )
            else:
                print("No results found for this query.", file=sys.stderr)
            raise SystemExit(exit_codes.NO_MATCHING_DATA)

        if is_json_mode():
            return to_jsonable(response)

        print(render_search_response(response))
        return None
