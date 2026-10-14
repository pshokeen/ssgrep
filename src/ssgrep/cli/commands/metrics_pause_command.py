"""``ssgrep metrics pause``: stop capture, collection, and sending (non-destructive)."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting import policy
from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import check_profile
from ssgrep.accounting.service_types import command_for
from ssgrep.cli.accounting_cli import fail, terminal_width
from ssgrep.cli.metrics_json import lifecycle_document
from ssgrep.cli.metrics_render import render_lifecycle
from ssgrep.cli.savings_render import emit


class MetricsPauseCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "metrics pause"

    def description(self) -> str:
        return "Pause capture, scheduled collection, and sending without deleting anything"

    def handle(
        self,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        try:
            check_profile(profile)
            result = policy.pause(profile)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            return lifecycle_document(
                "paused", result, profile, "metrics resume", "Resume when you want capture back."
            )
        emit(
            render_lifecycle(
                "Measurement paused.",
                (
                    f"Profile {profile} paused since {result.paused_at}. Capture, scheduled "
                    "collection and sending are stopped; history stays readable.",
                    "The paused interval is not imported on resume unless you backfill it.",
                ),
                command_for(profile, "metrics resume"),
                terminal_width(),
            )
        )
        return None
