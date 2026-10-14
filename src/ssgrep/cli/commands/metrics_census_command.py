"""``ssgrep metrics census``: read-only runtime and repository coverage census."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from rich.text import Text
from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.census import render_census_text, run_census
from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import check_profile
from ssgrep.cli.accounting_cli import envelope, fail
from ssgrep.cli.savings_render import emit


class MetricsCensusCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "metrics census"

    def description(self) -> str:
        return "Count runtime sources, repositories, and recognized invocations (aggregate only)"

    def handle(
        self,
        root: Annotated[
            str | None, Option("--root", help="Directory to scan for repositories (default: cwd)")
        ] = None,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        try:
            check_profile(profile)
        except PolicyError as error:
            fail(error)
        report = run_census(Path(root).expanduser() if root else None)
        if is_json_mode():
            data = report.to_dict()
            state = (
                "partial" if data["coverage"]["runtimes_covered"] < len(data["runtimes"]) else "ok"
            )
            return envelope(
                state,
                data,
                warnings=("An empty or partial census is not evidence of no historical usage.",),
                provenance={"read_only": True, "content_included": False},
            )
        emit(Text(render_census_text(report)))
        return None
