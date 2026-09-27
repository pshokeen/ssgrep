"""Install coding-agent integration and initialize the global index."""

from __future__ import annotations

import sys

from usecli import BaseCommand, console
from usecli.cli.core.runtime import is_json_mode, is_quiet

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import to_jsonable
from ssgrep.cli.commands.index_command import _fail
from ssgrep.cli.commands.mcp_command import _display_registrations
from ssgrep.cli.render import render_index_stats_table
from ssgrep.services import api, integrations, mcp_install
from ssgrep.utilities.types import IndexNotReadyError


def _display_skills(skill_results: tuple[tuple[str, str], ...]) -> None:
    """Print styled skill-installation results."""
    for name, status in skill_results:
        styled = _skill_style(status)
        console.print(f"  [cyan]{name:<12}[/cyan] {styled}")


def _skill_style(status: str) -> str:
    if status == "already_installed":
        return "[gray]already installed[/gray]"
    if status == "installed":
        return "[green]installed[/green]"
    if status == "user_modified":
        return "[yellow]kept your edits[/yellow]"
    if status.startswith("error:"):
        return f"[red]{status}[/red]"
    return status


class InitCommand(BaseCommand):
    """First-run setup shared by every supported local runtime."""

    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "init"

    def description(self) -> str:
        return (
            "Install agent skills, register MCP clients, and build the global multi-runtime index"
        )

    def handle(self) -> dict[str, object] | None:
        try:
            mcp_install._launcher_mode()
        except ValueError as error:
            print(f"Usage error: {error}", file=sys.stderr)
            raise SystemExit(exit_codes.USAGE_ERROR) from error
        skill_results = integrations.install_skills()
        mcp_results = mcp_install.install_mcp_registrations()
        try:
            stats = api.index()
        except IndexNotReadyError as error:
            _fail(error, exit_codes.MISSING_INDEX)
        except Exception as error:
            if is_json_mode():
                raise
            print(f"Initialization failed: {error}", file=sys.stderr)
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from error

        if is_json_mode():
            return {
                "ok": True,
                "skills": {name: status for name, status in skill_results},
                "mcp": {
                    name: {"status": status, "config": str(config) if config else None}
                    for name, status, config in mcp_results
                },
                "sources": {name: count for name, count in stats.runtime_counts},
                "index": to_jsonable(stats),
            }
        if not is_quiet():
            if skill_results:
                console.print("[bold]Agent skills[/bold]")
                _display_skills(skill_results)
            if mcp_results:
                if skill_results:
                    console.print()
                console.print("[bold]MCP clients[/bold]")
                _display_registrations(mcp_results)
            console.print()
            render_index_stats_table(stats)
        return None
