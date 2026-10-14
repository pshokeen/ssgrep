"""``ssgrep savings history``: closed periods and their revisions (read-only)."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting import service
from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import DEFAULT_LIMIT, check_limit, check_profile
from ssgrep.cli import savings_render
from ssgrep.cli.accounting_cli import fail, terminal_width
from ssgrep.cli.savings_json import history_document


class SavingsHistoryCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "savings history"

    def description(self) -> str:
        return "List closed periods and revisions without double counting replacements"

    def handle(
        self,
        since: Annotated[
            str | None, Option("--since", help="Interval start: RFC3339 or YYYY-MM-DD (UTC)")
        ] = None,
        until: Annotated[
            str | None, Option("--until", help="Interval end, exclusive: RFC3339 or YYYY-MM-DD")
        ] = None,
        limit: Annotated[int, Option("--limit", help="Rows per page (1-100)")] = DEFAULT_LIMIT,
        cursor: Annotated[str | None, Option("--cursor", help="Continue a previous page")] = None,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        try:
            check_profile(profile)
            check_limit(limit)
            view = service.read_history(profile, since, until, limit, cursor)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            return history_document(view)
        savings_render.emit(savings_render.render_history(view, terminal_width()))
        return None
