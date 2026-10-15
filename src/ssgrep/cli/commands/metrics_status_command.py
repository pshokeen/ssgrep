"""``ssgrep metrics status``: operational state; works without a search index."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.contracts import PERSONAL_PROFILE


class MetricsStatusCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "metrics status"

    def description(self) -> str:
        return "Show capture, snapshot freshness, and per-destination delivery state"

    def handle(
        self,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.policy_rules import PolicyError
        from ssgrep.accounting.scope import check_profile
        from ssgrep.accounting.service_status import read_status
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            check_profile(profile)
            view = read_status(profile)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.metrics_json import status_document

            return status_document(view)
        from ssgrep.cli.metrics_render import render_status
        from ssgrep.cli.savings_render import emit

        emit(render_status(view, terminal_width()))
        return None
