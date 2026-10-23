"""``ssgrep savings status``: operational state; works without a search index."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE


def _source_total(counts: Iterable[int | None]) -> int | None:
    total = 0
    for count in counts:
        if count is None:
            return None
        total += count
    return total


class MetricsStatusCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings status"

    def description(self) -> str:
        return (
            "Start here: see if measurement is on, how fresh the numbers are, and what is missing"
        )

    def handle(
        self,
        details: Annotated[
            bool, Option("--details", "-d", help="Page the snapshot's diagnostic warnings")
        ] = False,
        page: Annotated[
            str | None, Option("--page", "-n", help="Continue the diagnostic page")
        ] = None,
        include_text: Annotated[
            bool, Option("--include-text", "-t", help="Show private warning text locally")
        ] = False,
        sources: Annotated[
            bool, Option("--sources", "-s", help="Scan runtime sources for aggregate counts")
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.diagnostics import diagnostics_document, read_diagnostics
        from ssgrep.accounting.views.service import read_status
        from ssgrep.accounting.views.types import Scope, check_profile
        from ssgrep.cli.accounting_cli import fail, terminal_width, usage_error

        if (page is not None or include_text) and not details:
            usage_error(
                "details_required",
                "--page and --include-text apply only with --details.",
                "ssgrep savings status --details",
            )

        try:
            check_profile(profile)
            view = read_status(profile)
            diagnostics = (
                read_diagnostics(
                    profile,
                    Scope(),
                    snapshot_id=view.snapshot_id,
                    generation=view.generation,
                    cursor=page,
                    include_text=include_text,
                )
                if details
                else None
            )
        except PolicyError as error:
            fail(error)
        source_census: dict[str, object] = {
            "basis": "latest_collected_source_watermarks",
            "sources": _source_total(row.sources for row in view.lag.by_runtime),
            "by_runtime": [
                {
                    "runtime": row.runtime,
                    "sources": row.sources,
                    "total_bytes": row.total_bytes,
                    "lag_bytes": row.lag_bytes,
                }
                for row in view.lag.by_runtime
            ],
        }
        if sources:
            from ssgrep.accounting.readers.classify import run_census

            census = run_census()
            source_census["inventory"] = census.to_dict()
        if is_json_mode():
            from ssgrep.cli.metrics_json import status_document

            document = status_document(view)
            data = document["data"]
            assert isinstance(data, dict)
            data["source_census"] = source_census
            if diagnostics is not None:
                data["diagnostics"] = diagnostics_document(diagnostics)
            return document
        from ssgrep.cli.metrics_render import render_inventory, render_status
        from ssgrep.cli.savings_render import emit
        from ssgrep.cli.savings_state_render import add_diagnostics
        from ssgrep.cli.savings_text import Doc

        def extra(doc: Doc) -> None:
            if sources:
                render_inventory(doc, census.to_dict())
            if diagnostics is not None:
                add_diagnostics(doc, diagnostics)

        emit(render_status(view, terminal_width(), source_census=source_census, extra=extra))
        return None
