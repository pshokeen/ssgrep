"""``ssgrep metrics collect``: one bounded collection pass (live, backfill or recompute).

Everything is validated before any side effect; a paused or disabled profile
fails before the worker lease, progress file, checkpoint or snapshot is touched.
The reducer stack is imported only once validation has passed.
"""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting import policy
from ssgrep.accounting.collector_types import CollectOptions
from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import check_profile, interval
from ssgrep.accounting.service_types import command_for
from ssgrep.cli import exit_codes
from ssgrep.cli.accounting_cli import exit_with, fail, terminal_width, usage_error
from ssgrep.cli.collect_json import collect_document
from ssgrep.cli.collect_render import render_collect, render_running
from ssgrep.cli.savings_render import emit
from ssgrep.cli.savings_text import safe

_COLLECT_HELP = "ssgrep metrics collect --help"


def _validate(
    profile: str,
    since: str | None,
    until: str | None,
    backfill: bool,
    recompute: bool,
) -> tuple[str | None, str | None]:
    """Exit 2 on any inconsistent argument combination; returns normalized bounds."""
    if backfill and not (since and until):
        usage_error(
            "backfill_requires_interval",
            "--backfill needs both --since and --until: history is only imported for an "
            "explicit interval. Nothing was changed.",
            "ssgrep metrics collect --backfill --since <start> --until <end>",
        )
    if (since or until) and not (backfill or recompute):
        usage_error(
            "interval_requires_mode",
            "--since/--until only apply together with --backfill or --recompute; a normal "
            "collection never imports a chosen interval. Nothing was changed.",
            _COLLECT_HELP,
        )
    if backfill and recompute:
        usage_error(
            "conflicting_modes",
            "--backfill and --recompute cannot be combined. Nothing was changed.",
            _COLLECT_HELP,
        )
    try:
        start, end = interval(since, until)
        check_profile(profile)
    except PolicyError as error:
        fail(error)
    return start, end


class MetricsCollectCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "metrics collect"

    def description(self) -> str:
        return "Run one bounded collection pass; backfill, recompute or start a new counter epoch"

    def handle(
        self,
        since: Annotated[
            str | None,
            Option("--since", help="Interval start (RFC3339 or YYYY-MM-DD, UTC); with a mode"),
        ] = None,
        until: Annotated[
            str | None,
            Option("--until", help="Interval end, exclusive (RFC3339 or YYYY-MM-DD, UTC)"),
        ] = None,
        backfill: Annotated[
            bool,
            Option("--backfill", help="Import historical evidence for --since/--until (separate)"),
        ] = False,
        recompute: Annotated[
            bool,
            Option("--recompute", help="Recompute from retained frozen evidence only"),
        ] = False,
        new_epoch: Annotated[
            bool,
            Option("--new-epoch", help="Declare a reset or restored backup: restart live counters"),
        ] = False,
        repin_frequency: Annotated[
            bool,
            Option(
                "--repin-frequency",
                help="Pin a new candidate-frequency view: a NEW methodology namespace",
            ),
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        start, end = _validate(profile, since, until, backfill, recompute)
        try:
            policy.require_not_paused(profile, "collection")
            enabled = policy.load_profile_policy(profile).collection_enabled
        except PolicyError as error:
            fail(error)
        if not enabled:
            exit_with(
                "collection_disabled",
                "Local measurement is not enabled for this profile; nothing was collected.",
                command_for(profile, "metrics setup"),
                exit_codes.INTERNAL_FAILURE,
            )
        options = CollectOptions(
            since=start,
            until=end,
            backfill=backfill,
            recompute=recompute,
            new_epoch=new_epoch,
            repin_frequency=repin_frequency,
        )
        from ssgrep.accounting.collector import collect_once
        from ssgrep.accounting.collector_types import CollectError

        try:
            result = collect_once(profile, options)
        except CollectError as error:
            exit_with(error.condition, safe(str(error)), error.command, exit_codes.INTERNAL_FAILURE)
        document, data, actions = collect_document(profile, options, result)
        if is_json_mode():
            return document
        warnings = tuple(safe(w) for w in result.warnings)
        if result.state == "already_running":
            emit(render_running(data, actions, terminal_width()))
        else:
            emit(render_collect(data, warnings, actions, terminal_width()))
        return None
