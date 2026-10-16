"""``ssgrep savings history``: closed periods and their revisions (read-only)."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import DEFAULT_LIMIT, PERSONAL_PROFILE


class SavingsHistoryCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings history"

    def description(self) -> str:
        return "Everyday: review closed periods and revisions without double counting"

    def handle(
        self,
        from_: Annotated[
            str | None, Option("--from", "-f", help="Start of the UTC interval")
        ] = None,
        to: Annotated[
            str | None, Option("--to", "-t", help="Exclusive end of the UTC interval")
        ] = None,
        limit: Annotated[
            int, Option("--limit", "-n", help="Number of closed periods to show (1-100)")
        ] = DEFAULT_LIMIT,
        cursor: Annotated[
            str | None, Option("--page", "-p", help="Continue from this history page")
        ] = None,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import check_limit, check_profile
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            check_profile(profile)
            check_limit(limit)
            from ssgrep.accounting.views import service

            view = service.read_history(profile, from_, to, limit, cursor)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.savings_json import history_document

            return history_document(view)
        from ssgrep.cli import savings_render

        savings_render.emit(savings_render.render_history(view, terminal_width()))
        return None
