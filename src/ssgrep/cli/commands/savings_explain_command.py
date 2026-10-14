"""``ssgrep savings explain``: show how one receipt was valued (local, read-only)."""

from __future__ import annotations

from typing import Annotated

from usecli import Argument, BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting import service
from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import check_profile
from ssgrep.accounting.service_types import command_for
from ssgrep.cli import exit_codes, savings_render
from ssgrep.cli.accounting_cli import exit_with, fail, terminal_width
from ssgrep.cli.savings_json import receipt_document


class SavingsExplainCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "savings explain"

    def description(self) -> str:
        return "Explain one savings receipt: sources, overlap removal, costs, and exclusions"

    def handle(
        self,
        receipt_id: Annotated[str, Argument(help="Receipt id shown by `ssgrep savings`")],
        include_content: Annotated[
            bool, Option("--include-content", help="Show private local content (hidden by default)")
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        try:
            check_profile(profile)
            view = service.read_receipt(profile, receipt_id, include_content)
        except PolicyError as error:
            fail(error)
        if view is None:
            exit_with(
                "receipt_not_found",
                "No receipt with that id exists for this profile.",
                command_for(profile, "savings"),
                exit_codes.NO_MATCHING_DATA,
            )
        if is_json_mode():
            return receipt_document(view, profile)
        savings_render.emit(
            savings_render.render_receipt(view, profile, terminal_width(), include_content)
        )
        return None
