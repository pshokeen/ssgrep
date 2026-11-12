"""``ssgrep savings``: read the latest snapshot; never collects, indexes, or sends."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.base_command import ClickContext, NestedCommandRegistry
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import DEFAULT_LIMIT, PERSONAL_PROFILE


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
        no_refresh: Annotated[
            bool, Option("--no-refresh", "-N", help="Use the existing snapshot without collecting")
        ] = False,
        from_: Annotated[
            str | None,
            Option("--from", "-f", help="Include activity from this UTC date or instant"),
        ] = None,
        to: Annotated[
            str | None,
            Option("--to", "-t", help="End the interval before this UTC date or instant"),
        ] = None,
        project: Annotated[
            str | None, Option("--project", "-p", help="Limit results to this project path")
        ] = None,
        runtime: Annotated[
            str | None, Option("--runtime", "-r", help="Limit results to this agent runtime")
        ] = None,
        session: Annotated[
            str | None, Option("--session", "-s", help="Limit results to this conversation")
        ] = None,
        group_by: Annotated[
            str | None,
            Option("--group-by", "-g", help="Group totals by task, runtime, project, or day"),
        ] = None,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
        for_agent: Annotated[
            bool,
            Option(
                "--for-agent",
                help="One-screen honest verdict mix and reuse rate for agent self-correction",
            ),
        ] = False,
    ) -> dict[str, object] | None:
        if help_flag:
            self._show_savings_help()
            ctx.exit()
        if interactive:
            return self._listing_callback(ctx, help_flag, interactive, json_output)
        if ctx.invoked_subcommand is not None:
            return None
        return self.handle(
            from_, to, project, runtime, session, group_by, profile, no_refresh, for_agent
        )

    @staticmethod
    def _show_savings_help() -> None:
        print("Usage: ssgrep savings [COMMAND] [OPTIONS]")
        print()
        print("Start here")
        print("  ssgrep savings         How much ssgrep has saved you; refreshes stale numbers")
        print("  ssgrep savings setup   Turn on local-only measurement")
        print(
            "  ssgrep savings status  Check measurement, freshness, missing data; "
            "--sources counts runtimes"
        )
        print()
        print("Everyday")
        print("  ssgrep savings collect          Refresh saved numbers")
        print("  ssgrep savings explain RECEIPT  Why one lookup earned or missed credit")
        print("  ssgrep savings lookups          Each lookup's cost and reuse verdict")
        print("  ssgrep savings history          Closed periods and revisions")
        print()
        print("Save or share")
        print("  ssgrep savings export --output FILE    Save a private JSON report")
        print("  ssgrep savings export [DESTINATION]    Send all approved organization data")
        print("  ssgrep savings serve                   Run the foreground metrics server")
        print()
        print("Maintenance")
        print("  ssgrep savings pause")
        print("  ssgrep savings resume")
        print("  ssgrep savings purge BEFORE_DATE")
        print()
        print("Options")
        print("  --from, -f       Include activity from this UTC date or instant")
        print("  --to, -t         End the interval before this UTC date or instant")
        print("  --project, -p    Limit results to this project path")
        print("  --runtime, -r    Limit results to this agent runtime")
        print("  --session, -s    Limit results to this conversation")
        print("  --group-by, -g   Group totals by task, runtime, project, or day")
        print("  --profile        Select the accounting profile")
        print("  --no-refresh, -N Use the existing snapshot without collecting")
        print("  --for-agent       Honest verdict mix and reuse rate for agent self-correction")
        print("  --interactive, -i  Open the interactive picker")
        print("  --help, -h       Show this message")

    def signature(self) -> str:
        return "savings"

    def description(self) -> str:
        return "Show measured lookup usage and modeled benefit from the latest snapshot"

    def handle(
        self,
        from_: str | None = None,
        to: str | None = None,
        project: str | None = None,
        runtime: str | None = None,
        session: str | None = None,
        group_by: str | None = None,
        profile: str = PERSONAL_PROFILE,
        no_refresh: bool = False,
        for_agent: Annotated[
            bool,
            Option(
                "--for-agent",
                help="One-screen honest verdict mix and reuse rate for agent self-correction",
            ),
        ] = False,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import build_scope, check_profile
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            check_profile(profile)
            scope = build_scope(
                since=from_,
                until=to,
                project=project,
                runtime=runtime,
                session=session,
                by=group_by,
            )
            from ssgrep.accounting.views import service

            view = service.read_summary(profile, scope, limit=DEFAULT_LIMIT)
            if (
                not no_refresh
                and view.collection_enabled
                and not view.paused
                and (view.stale or view.state.value == "no_snapshot")
            ):
                from ssgrep.accounting.collect.collector import (
                    CollectError,
                    CollectOptions,
                    collect_once,
                )
                from ssgrep.cli.savings_text import safe

                try:
                    collect_once(profile, CollectOptions())
                except CollectError as error:
                    view = replace(
                        view,
                        warnings=(*view.warnings, f"refresh_failed: {safe(error.condition)}"),
                    )
                else:
                    view = service.read_summary(profile, scope, limit=DEFAULT_LIMIT)
        except PolicyError as error:
            fail(error)
        if for_agent:
            from ssgrep.accounting.views.reuse_rate import agent_summary

            document = agent_summary(profile, scope)
            if is_json_mode():
                return document
            from ssgrep.cli.savings_render import emit
            from ssgrep.cli.savings_text import Doc

            doc = Doc(terminal_width())
            doc.line("ssgrep savings --for-agent", style="bold")
            doc.pair("Lookups in scope", str(document["total_lookups"]), indent=2)
            doc.pair("Operative reuses", str(document["operative_reuses"]), indent=2)
            rate = document["operative_reuse_rate"]
            doc.pair("Reuse rate", f"{rate:.0%}" if rate is not None else "unknown", indent=2)
            mix: object = document["verdict_mix"]
            doc.pair(
                "Verdict mix",
                ", ".join(f"{k}: {v}" for k, v in sorted(mix.items()))  # type: ignore[attr-defined]
                if isinstance(mix, dict)
                else "unknown",
                indent=2,
            )
            doc.line(
                "Search BEFORE solving: operative reuse earns credit; browsing does not.",
                indent=2,
            )
            emit(doc.text())
            return None
        if is_json_mode():
            from ssgrep.cli.savings_json import summary_document

            return summary_document(view)
        from ssgrep.cli import savings_render

        savings_render.emit(savings_render.render_summary(view, terminal_width()))
        return None
