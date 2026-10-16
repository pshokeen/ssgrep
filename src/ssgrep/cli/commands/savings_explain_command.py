"""``ssgrep savings explain``: show how one receipt was valued (local, read-only)."""

from __future__ import annotations

from typing import Annotated

from usecli import Argument, BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE


class SavingsExplainCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings explain"

    def description(self) -> str:
        return "Everyday: explain why one lookup earned or did not earn credit"

    def handle(
        self,
        receipt_id: Annotated[str, Argument(help="Receipt id shown by `ssgrep savings`")],
        include_content: Annotated[
            bool,
            Option(
                "--include-content", "-c", help="Show private local content (hidden by default)"
            ),
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import check_profile, command_for
        from ssgrep.cli import exit_codes
        from ssgrep.cli.accounting_cli import exit_with, fail, terminal_width

        try:
            check_profile(profile)
            from ssgrep.accounting.views import service

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
            from ssgrep.cli.savings_json import receipt_document

            return receipt_document(view, profile)
        from ssgrep.cli import savings_render

        savings_render.emit(
            savings_render.render_receipt(view, profile, terminal_width(), include_content)
        )
        return None
