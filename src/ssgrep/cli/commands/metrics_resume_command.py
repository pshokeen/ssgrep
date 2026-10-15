"""``ssgrep metrics resume``: lift a pause; never re-enables exports or imports the gap."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.contracts import PERSONAL_PROFILE


class MetricsResumeCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "metrics resume"

    def description(self) -> str:
        return "Resume a paused profile; disabled export signals stay disabled"

    def handle(
        self,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting import policy
        from ssgrep.accounting.policy_rules import PolicyError
        from ssgrep.accounting.scope import check_profile
        from ssgrep.accounting.service_types import command_for
        from ssgrep.cli.accounting_cli import fail, terminal_width

        try:
            check_profile(profile)
            result = policy.resume(profile)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.metrics_json import lifecycle_document

            return lifecycle_document(
                "resumed", result, profile, "metrics status", "Review capture and delivery state."
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
                command_for(profile, "metrics status"),
                terminal_width(),
            )
        )
        return None
