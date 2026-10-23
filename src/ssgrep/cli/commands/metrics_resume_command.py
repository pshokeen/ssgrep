"""``ssgrep savings resume``: lift a pause; never re-enables exports or imports the gap."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE


class MetricsResumeCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings resume"

    def description(self) -> str:
        return "Maintenance: resume a paused profile; export signals stay disabled unless enabled"

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
            result = policy.resume(profile)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.metrics_json import lifecycle_document

            return lifecycle_document(
                "resumed", result, profile, "savings status", "Review capture and delivery state."
            )
        from ssgrep.cli.metrics_render import render_lifecycle
        from ssgrep.cli.savings_render import emit

        emit(
            render_lifecycle(
                "Measurement resumed.",
                (
                    f"Profile {profile} is no longer paused. Export signals that were disabled "
                    "stay disabled; the paused interval was not imported.",
                ),
                command_for(profile, "savings status"),
                terminal_width(),
            )
        )
        return None
