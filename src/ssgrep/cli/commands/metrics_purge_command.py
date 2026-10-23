"""``ssgrep savings purge``: delete owned accounting data before a day (scoped, confirmed).

Every argument is validated and confirmation is demanded BEFORE any side effect. ``--dry-run``
creates and modifies nothing. Source transcripts, the search index, notes, policy and other
profiles are never touched; remote erasure is never claimed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated

from usecli import Argument, BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE

if TYPE_CHECKING:
    from ssgrep.accounting.privacy import purge
    from ssgrep.accounting.privacy.purge import PurgeError


def _refuse(error: PurgeError) -> None:
    from ssgrep.cli.accounting_cli import exit_with

    exit_with(error.condition, str(error), error.command, error.exit_code)


class MetricsPurgeCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings purge"

    def description(self) -> str:
        return "Maintenance: remove old accounting data; transcripts and index are untouched"

    def handle(
        self,
        before: Annotated[str, Argument(..., help="Delete data before this day or UTC instant")],
        workspace: Annotated[
            str | None,
            Option("--workspace", "-w", help="Only conversations inside this workspace root path"),
        ] = None,
        dry_run: Annotated[
            bool, Option("--dry-run", "-n", help="Preview which data would be deleted")
        ] = False,
        yes: Annotated[
            bool, Option("--yes", "-y", help="Confirm permanent deletion without prompting")
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy, valid_profile_id
        from ssgrep.accounting.privacy import policy, purge
        from ssgrep.accounting.privacy.purge import USAGE, PurgeError, parse_selection
        from ssgrep.accounting.views.types import command_for
        from ssgrep.cli import exit_codes
        from ssgrep.cli.accounting_cli import exit_with, stdin_is_tty, terminal_width

        try:
            selection = parse_selection(
                profile=profile,
                before=before,
                workspace=workspace,
                policy=policy.load_profile_policy(profile)
                if valid_profile_id(profile)
                else ProfilePolicy(),
            )
        except PurgeError as error:
            _refuse(error)
        interactive = stdin_is_tty() and not is_json_mode()
        if not dry_run and not yes and not interactive:
            exit_with(
                "confirmation_required",
                "Purge deletes data for good: re-run with --yes (non-interactive) or --dry-run "
                "to preview. Nothing was changed.",
                command_for(profile, f"{USAGE.replace('ssgrep ', '', 1)} --yes"),
                exit_codes.USAGE_ERROR,
            )

        try:
            if dry_run or not yes:
                preview = purge.dry_run(selection)
                if not dry_run:
                    from ssgrep.cli.purge_render import render_report
                    from ssgrep.cli.savings_render import emit

                    emit(render_report(preview, terminal_width()))
                    print(file=sys.stdout)
                    try:
                        answer = input("Delete this data permanently? [y/N] ")
                    except EOFError:
                        answer = ""
                    if answer.strip().lower() not in {"y", "yes"}:
                        print("Cancelled. Nothing was changed.", file=sys.stderr)
                        return None
                else:
                    return self._show(preview, profile)
            done = purge.execute(selection)
        except PurgeError as error:
            _refuse(error)
        return self._show(done, profile)

    @staticmethod
    def _show(report: purge.PurgeReport, profile: str) -> dict[str, object] | None:
        if is_json_mode():
            from ssgrep.cli.purge_render import report_document

            return report_document(report, profile)
        from ssgrep.cli.accounting_cli import terminal_width
        from ssgrep.cli.purge_render import render_report
        from ssgrep.cli.savings_render import emit

        emit(render_report(report, terminal_width()))
        return None
