"""``ssgrep savings collect``: one bounded collection pass (live, backfill or recompute).

Everything is validated before any side effect; a paused or disabled profile
fails before the worker lease, progress file, checkpoint or snapshot is touched.
The reducer stack is imported only once validation has passed.
"""

from __future__ import annotations

from typing import Annotated

from usecli import Argument, BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE

_COLLECT_HELP = "ssgrep savings collect --help"


def _validate(
    profile: str,
    from_: str | None,
    to: str | None,
    backfill: bool,
    recompute: bool,
) -> tuple[str | None, str | None]:
    """Exit 2 on any inconsistent argument combination; returns normalized bounds."""
    from ssgrep.accounting.privacy.policy import PolicyError
    from ssgrep.accounting.views.types import check_profile, interval
    from ssgrep.cli.accounting_cli import fail, usage_error

    if backfill and not (from_ and to):
        usage_error(
            "backfill_requires_interval",
            "--backfill needs both FROM and TO positional dates. Nothing was changed.",
            _COLLECT_HELP,
        )
    if (from_ or to) and not (backfill or recompute):
        usage_error(
            "interval_requires_mode",
            "FROM and TO only apply with --backfill or --recompute. Nothing was changed.",
            _COLLECT_HELP,
        )
    if backfill and recompute:
        usage_error(
            "conflicting_modes",
            "--backfill and --recompute cannot be combined. Nothing was changed.",
            _COLLECT_HELP,
        )
    try:
        start, end = interval(from_, to)
        check_profile(profile)
    except PolicyError as error:
        fail(error)
    return start, end


class MetricsCollectCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings collect"

    def description(self) -> str:
        return "Everyday: refresh saved numbers now; backfill or recompute when needed"

    def handle(
        self,
        from_: str | None = Argument(None, help="Backfill start date (UTC); use with --backfill"),
        to: str | None = Argument(None, help="Backfill end date, exclusive (UTC)"),
        backfill: Annotated[
            bool, Option("--backfill", "-b", help="Import the positional FROM/TO interval")
        ] = False,
        recompute: Annotated[
            bool, Option("--recompute", "-r", help="Recompute from retained evidence only")
        ] = False,
        new_epoch: Annotated[
            bool, Option("--new-epoch", "-n", help="Restart live counters after a declared restore")
        ] = False,
        repin_frequency: Annotated[
            bool, Option("--repin-frequency", "-R", help="Create a replacement rarity revision")
        ] = False,
        journal_legacy_writers_stopped: Annotated[
            bool,
            Option(
                "--journal-legacy-writers-stopped",
                "-j",
                help=(
                    "Confirm every pre-v2 capture process is stopped before expiring legacy "
                    "journals"
                ),
            ),
        ] = False,
        reset_frequency_authority: Annotated[
            bool,
            Option(
                "--reset-frequency-authority",
                "-F",
                help="Reapprove narrowed rarity scope after purge",
            ),
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
        budget_seconds: Annotated[
            int | None,
            Option("--budget-seconds", "-B", min=1, help="Cooperative per-pass budget (seconds)"),
        ] = None,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy import policy
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import command_for
        from ssgrep.cli import exit_codes
        from ssgrep.cli.accounting_cli import exit_with, fail, terminal_width, usage_error
        from ssgrep.cli.savings_text import safe

        start, end = _validate(profile, from_, to, backfill, recompute)
        if reset_frequency_authority and not repin_frequency:
            usage_error(
                "reset_frequency_requires_repin",
                "--reset-frequency-authority requires --repin-frequency; nothing was changed.",
                _COLLECT_HELP,
            )
        try:
            policy.require_not_paused(profile, "collection")
            enabled = policy.load_profile_policy(profile).collection_enabled
        except PolicyError as error:
            fail(error)
        if not enabled:
            exit_with(
                "collection_disabled",
                "Local measurement is not enabled for this profile; nothing was collected.",
                command_for(profile, "savings setup"),
                exit_codes.INTERNAL_FAILURE,
            )
        from ssgrep.accounting.collect.collector import CollectOptions, CycleBudget
        from ssgrep.cli.collect_render import collect_document

        options = CollectOptions(
            since=start,
            until=end,
            backfill=backfill,
            recompute=recompute,
            new_epoch=new_epoch,
            repin_frequency=repin_frequency,
            journal_legacy_writers_stopped=journal_legacy_writers_stopped,
            reset_frequency_authority=reset_frequency_authority,
            budget=CycleBudget() if budget_seconds is None else CycleBudget(seconds=budget_seconds),
        )
        from ssgrep.accounting.collect.collector import CollectError, collect_once

        try:
            result = collect_once(profile, options)
        except CollectError as error:
            exit_with(error.condition, safe(str(error)), error.command, exit_codes.INTERNAL_FAILURE)
        document, data, actions = collect_document(profile, options, result)
        if is_json_mode():
            return document
        from ssgrep.cli.collect_render import render_collect, render_running
        from ssgrep.cli.savings_render import emit

        warnings = tuple(safe(w) for w in result.warnings)
        if result.state == "already_running":
            emit(render_running(data, actions, terminal_width()))
        else:
            emit(render_collect(data, warnings, actions, terminal_width()))
        return None
