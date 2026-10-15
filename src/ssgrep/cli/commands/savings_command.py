"""``ssgrep savings``: read the latest snapshot; never collects, indexes, or sends."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.base_command import ClickContext, NestedCommandRegistry
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.contracts import DEFAULT_LIMIT, PERSONAL_PROFILE


class SavingsCommand(BaseCommand):
    """The ``savings`` group's own summary.

    usecli turns ``savings explain``/``savings history`` into a command group whose
    callback merely lists subcommands. ``register`` keeps that group but installs this
    command as its callback, so ``ssgrep savings [options]`` runs the summary while
    ``ssgrep savings explain ...`` still dispatches to the subcommand.
    """

    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def register(self) -> None:
        if not self.visible():
            return
        group = NestedCommandRegistry().get_or_create_group(self.app, "savings")
        group.info.help = self.description()
        info = group.registered_callback
        assert info is not None and callable(info.callback)
        self._listing_callback = info.callback
        info.callback = self._group_callback

    def _group_callback(
        self,
        ctx: ClickContext,
        help_flag: Annotated[
            bool, Option("--help", "-h", is_eager=True, help="Show help for this group")
        ] = False,
        interactive: Annotated[
            bool, Option("--interactive", "-i", help="Run in interactive mode.")
        ] = False,
        json_output: Annotated[
            bool, Option("--json", help="Emit one machine-readable JSON document.")
        ] = False,
        since: Annotated[
            str | None, Option("--since", help="Interval start: RFC3339 or YYYY-MM-DD (UTC)")
        ] = None,
        until: Annotated[
            str | None, Option("--until", help="Interval end, exclusive: RFC3339 or YYYY-MM-DD")
        ] = None,
        project: Annotated[str | None, Option("--project", help="Only this project path")] = None,
        runtime: Annotated[str | None, Option("--runtime", help="Only this runtime")] = None,
        session: Annotated[str | None, Option("--session", help="Only this session id")] = None,
        task: Annotated[str | None, Option("--task", help="Only this task id")] = None,
        by: Annotated[
            str | None, Option("--by", help="Group by task, runtime, project, or day")
        ] = None,
        limit: Annotated[int, Option("--limit", help="Rows per page (1-100)")] = DEFAULT_LIMIT,
        cursor: Annotated[str | None, Option("--cursor", help="Continue a previous page")] = None,
        include_children: Annotated[
            bool, Option("--include-children", help="Include descendants of --session")
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        if ctx.invoked_subcommand is not None:
            return None
        if help_flag or interactive:
            return self._listing_callback(ctx, help_flag, interactive, json_output)
        return self.handle(
            since,
            until,
            project,
            runtime,
            session,
            task,
            by,
            limit,
            cursor,
            include_children,
            profile,
        )

    def signature(self) -> str:
        return "savings"

    def description(self) -> str:
        return "Show measured lookup usage and modeled benefit from the latest snapshot"

    def handle(
        self,
        since: Annotated[
            str | None, Option("--since", help="Interval start: RFC3339 or YYYY-MM-DD (UTC)")
        ] = None,
        until: Annotated[
            str | None, Option("--until", help="Interval end, exclusive: RFC3339 or YYYY-MM-DD")
        ] = None,
        project: Annotated[str | None, Option("--project", help="Only this project path")] = None,
        runtime: Annotated[str | None, Option("--runtime", help="Only this runtime")] = None,
        session: Annotated[str | None, Option("--session", help="Only this session id")] = None,
        task: Annotated[str | None, Option("--task", help="Only this task id")] = None,
        by: Annotated[
            str | None, Option("--by", help="Group by task, runtime, project, or day")
        ] = None,
        limit: Annotated[int, Option("--limit", help="Rows per page (1-100)")] = DEFAULT_LIMIT,
        cursor: Annotated[str | None, Option("--cursor", help="Continue a previous page")] = None,
        include_children: Annotated[
            bool, Option("--include-children", help="Include descendants of --session")
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        """Summarize the selected profile; validation failures exit 2 before any read."""
        from ssgrep.accounting.policy_rules import PolicyError
        from ssgrep.accounting.scope import build_scope, check_limit, check_profile
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            check_profile(profile)
            check_limit(limit)
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
            from ssgrep.accounting import service

            view = service.read_summary(profile, scope, limit=limit, cursor=cursor)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.savings_json import summary_document

            return summary_document(view)
        from ssgrep.cli import savings_render

        savings_render.emit(savings_render.render_summary(view, terminal_width()))
        return None
