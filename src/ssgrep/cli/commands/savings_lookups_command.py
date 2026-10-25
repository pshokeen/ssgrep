"""ssgrep savings lookups: bounded read-only verdicts including no-receipt calls."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE


class SavingsLookupsCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings lookups"

    def description(self) -> str:
        return "Everyday: review each lookup, its cost, and its reuse verdict"

    def handle(
        self,
        from_: Annotated[
            str | None, Option("--from", "-f", help="Start of the UTC interval")
        ] = None,
        to: Annotated[
            str | None, Option("--to", "-t", help="Exclusive end of the UTC interval")
        ] = None,
        project: Annotated[
            str | None, Option("--project", "-p", help="Canonical project selection")
        ] = None,
        runtime: Annotated[
            str | None, Option("--runtime", "-r", help="Native runtime selection")
        ] = None,
        session: Annotated[
            str | None, Option("--session", "-s", help="Native session selection")
        ] = None,
        task: Annotated[str | None, Option("--task", "-k", help="Task or root selection")] = None,
        include_children: Annotated[
            bool, Option("--children", "-c", help="Include selected session descendants")
        ] = False,
        cursor: Annotated[
            str | None, Option("--page", "-n", help="Continue from this lookup page")
        ] = None,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.lookups import read_lookups
        from ssgrep.accounting.views.types import build_scope
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            scope = build_scope(
                since=from_,
                until=to,
                project=project,
                runtime=runtime,
                session=session,
                task=task,
                include_children=include_children,
            )
            view = read_lookups(profile, scope, limit=20, cursor=cursor)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.savings_json import lookups_document

            return lookups_document(view)
        from ssgrep.cli.savings_lookup_render import render_lookups
        from ssgrep.cli.savings_render import emit

        emit(render_lookups(view, terminal_width()))
        return None
