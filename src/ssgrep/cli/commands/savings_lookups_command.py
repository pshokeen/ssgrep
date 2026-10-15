"""ssgrep savings lookups: bounded read-only verdicts including no-receipt calls."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode


class SavingsLookupsCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings lookups"

    def description(self) -> str:
        return "Page lookup execution, exposure, reuse and valuation evidence"

    def handle(
        self,
        since: Annotated[str | None, Option("--since", help="UTC interval start")] = None,
        until: Annotated[str | None, Option("--until", help="UTC interval end, exclusive")] = None,
        project: Annotated[
            str | None, Option("--project", help="Canonical project selection")
        ] = None,
        runtime: Annotated[str | None, Option("--runtime", help="Native runtime selection")] = None,
        session: Annotated[str | None, Option("--session", help="Native session selection")] = None,
        task: Annotated[str | None, Option("--task", help="Task or root selection")] = None,
        include_children: Annotated[
            bool, Option("--include-children", help="Include selected session descendants")
        ] = False,
        snapshot: Annotated[
            str | None, Option("--snapshot", help="Exact published snapshot identity")
        ] = None,
        generation: Annotated[
            int | None, Option("--generation", help="Exact published generation")
        ] = None,
        limit: Annotated[int, Option("--limit", help="Rows per page (1-100)")] = 20,
        cursor: Annotated[
            str | None, Option("--cursor", help="Continue the same snapshot and selection")
        ] = None,
        profile: Annotated[str, Option("--profile", help="Accounting profile")] = "personal-local",
    ) -> dict[str, object] | None:
        from ssgrep.accounting.policy_rules import PolicyError
        from ssgrep.accounting.scope import build_scope
        from ssgrep.accounting.service_lookups import read_lookups
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            scope = build_scope(
                since=since,
                until=until,
                project=project,
                runtime=runtime,
                session=session,
                task=task,
                include_children=include_children,
            )
            view = read_lookups(
                profile,
                scope,
                snapshot_id=snapshot,
                generation=generation,
                limit=limit,
                cursor=cursor,
            )
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.savings_json import lookups_document

            return lookups_document(view)
        from ssgrep.cli.savings_lookup_render import render_lookups
        from ssgrep.cli.savings_render import emit

        emit(render_lookups(view, terminal_width()))
        return None
