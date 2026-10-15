"""``ssgrep savings diagnostics``: complete bounded, snapshot-bound warning pages."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode


class SavingsDiagnosticsCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings diagnostics"

    def description(self) -> str:
        return "Page snapshot diagnostics; private warning text is hidden by default"

    def handle(
        self,
        snapshot: Annotated[
            str | None, Option("--snapshot", help="Exact snapshot identity")
        ] = None,
        generation: Annotated[
            int | None, Option("--generation", help="Exact snapshot generation")
        ] = None,
        since: Annotated[str | None, Option("--since", help="Interval start (UTC)")] = None,
        until: Annotated[
            str | None, Option("--until", help="Interval end, exclusive (UTC)")
        ] = None,
        project: Annotated[str | None, Option("--project", help="Project selection")] = None,
        runtime: Annotated[str | None, Option("--runtime", help="Runtime selection")] = None,
        session: Annotated[str | None, Option("--session", help="Session selection")] = None,
        task: Annotated[str | None, Option("--task", help="Task selection")] = None,
        by: Annotated[
            str | None, Option("--by", help="Preserve summary grouping selection")
        ] = None,
        include_children: Annotated[
            bool, Option("--include-children", help="Include selected session's children")
        ] = False,
        code: Annotated[str | None, Option("--code", help="One stable diagnostic code")] = None,
        limit: Annotated[int, Option("--limit", help="Records per page (1-100)")] = 20,
        cursor: Annotated[
            str | None, Option("--cursor", help="Continue a snapshot-bound page")
        ] = None,
        include_text: Annotated[
            bool, Option("--include-text", help="Explicitly display private warning text locally")
        ] = False,
        profile: Annotated[str, Option("--profile", help="Accounting profile")] = "personal-local",
    ) -> dict[str, object] | None:
        from ssgrep.accounting.policy_rules import PolicyError
        from ssgrep.accounting.scope import build_scope
        from ssgrep.accounting.service_diagnostics import diagnostics_document, read_diagnostics
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            scope = build_scope(
                since=since,
                until=until,
                project=project,
                runtime=runtime,
                session=session,
                task=task,
                by=by,
                include_children=include_children,
            )
            view = read_diagnostics(
                profile,
                scope,
                snapshot_id=snapshot,
                generation=generation,
                limit=limit,
                cursor=cursor,
                code=code,
                include_text=include_text,
            )
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            return diagnostics_document(view)
        from ssgrep.cli.savings_render import emit
        from ssgrep.cli.savings_state_render import render_diagnostics

        emit(render_diagnostics(view, terminal_width()))
        return None
