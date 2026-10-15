"""``ssgrep metrics purge``: delete owned accounting data before a date (scoped, confirmed).

Every argument is validated and confirmation is demanded BEFORE any side effect. ``--dry-run``
creates and modifies nothing. Source transcripts, the search index, notes, policy and other
profiles are never touched; remote erasure is never claimed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.contracts import PERSONAL_PROFILE

if TYPE_CHECKING:
    from ssgrep.accounting import purge
    from ssgrep.accounting.purge_model import PurgeError


def _refuse(error: PurgeError) -> None:
    from ssgrep.cli.accounting_cli import exit_with

    exit_with(error.condition, str(error), error.command, error.exit_code)


class MetricsPurgeCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "metrics purge"

    def description(self) -> str:
        return "Delete accounting data dated before a day (transcripts and index untouched)"

    def handle(
        self,
        before: Annotated[
            str | None,
            Option("--before", help="Delete data dated before this day or RFC3339 instant (UTC)"),
        ] = None,
        workspace: Annotated[
            str | None,
            Option("--workspace", help="Only conversations inside this workspace root path"),
        ] = None,
        dry_run: Annotated[
            bool, Option("--dry-run", help="Describe what would be deleted; change nothing")
        ] = False,
        yes: Annotated[bool, Option("--yes", help="Confirm deletion without prompting")] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting import policy
        from ssgrep.accounting.ledger_contracts import ProfilePolicy, valid_profile_id
        from ssgrep.accounting.purge_model import PurgeError
        from ssgrep.accounting.purge_select import USAGE, parse_selection
        from ssgrep.accounting.service_types import command_for
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
        from ssgrep.accounting import purge

        try:
            if dry_run or not yes:
                preview = purge.dry_run(selection)
                if not dry_run:
                    from ssgrep.cli.purge_render import render_report
                    from ssgrep.cli.savings_render import emit

                    emit(render_report(preview, terminal_width()))
                    print(file=sys.stdout)
                    answer = input("Delete this data permanently? [y/N] ")
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
