"""``ssgrep savings pause``: stop capture, collection, and sending (non-destructive)."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE


class MetricsPauseCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings pause"

    def description(self) -> str:
        return (
            "Maintenance: stop capture, scheduled collection, and sending without deleting anything"
        )

    def handle(
        self,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy import policy
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import check_profile, command_for
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            check_profile(profile)
            result = policy.pause(profile)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.metrics_json import lifecycle_document

            return lifecycle_document(
                "paused", result, profile, "savings resume", "Resume when you want capture back."
            )
        from ssgrep.cli.metrics_render import render_lifecycle
        from ssgrep.cli.savings_render import emit

        emit(
            render_lifecycle(
                "Measurement paused.",
                (
                    f"Profile {profile} paused since {result.paused_at}. Capture, scheduled "
                    "collection and sending are stopped; history stays readable.",
                    "The paused interval is not imported on resume unless you backfill it.",
                ),
                command_for(profile, "savings resume"),
                terminal_width(),
            )
        )
        return None
