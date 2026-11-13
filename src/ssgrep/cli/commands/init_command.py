"""Install coding-agent integration and initialize the global index."""

from __future__ import annotations

import sys
from typing import Annotated

from usecli import BaseCommand, Option, console
from usecli.cli.core.runtime import is_json_mode, is_quiet

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import to_jsonable
from ssgrep.cli.commands.index_command import _fail
from ssgrep.cli.commands.mcp_command import (
    _display_registrations,
    _exit_failed,
    _registrations_json,
)
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

    def handle(
        self,
        apply_rules: Annotated[
            bool,
            Option(
                "--apply-rules",
                help="Also install the search-first block into this repo's CLAUDE.md/AGENTS.md",
            ),
        ] = False,
        remove_rules: Annotated[
            bool,
            Option("--remove-rules", help="Strip the managed rules block from this repo"),
        ] = False,
        install_hooks: Annotated[
            bool,
            Option(
                "--install-hooks",
                help="Install the pre-edit search hook into harnesses with a hook bus",
            ),
        ] = False,
        remove_hooks: Annotated[
            bool, Option("--remove-hooks", help="Remove the pre-edit search hook")
        ] = False,
    ) -> dict[str, object] | None:
        from pathlib import Path

        if (apply_rules and remove_rules) or (install_hooks and remove_hooks):
            print(
                "Choose --apply-rules or --remove-rules, and --install-hooks or "
                "--remove-hooks; not both of any pair.",
                file=sys.stderr,
            )
            raise SystemExit(exit_codes.USAGE_ERROR)
        rule_status: dict[str, str] = {}
        if apply_rules or remove_rules:
            target = integrations.project_rule_instruction_file(Path.cwd())
            rule_status[target.name] = (
                integrations.remove_project_rule(target)
                if remove_rules
                else integrations.apply_project_rule(target)
            )
        hook_results: tuple[tuple[str, str], ...] = ()
        if install_hooks or remove_hooks:
            hook_results = (
                integrations.install_hooks() if install_hooks else integrations.remove_hooks()
            )
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

        payload: dict[str, object] = {
            "skills": {name: status for name, status in skill_results},
            "mcp": _registrations_json(mcp_results),
            "sources": {name: count for name, count in stats.runtime_counts},
            "index": to_jsonable(stats),
        }
        if rule_status:
            payload["project_rules"] = rule_status
        if hook_results:
            payload["hooks"] = {name: status for name, status in hook_results}
        if not (is_json_mode() or is_quiet()):
            if rule_status:
                console.print("[bold]Project rules[/bold]")
                for filename, status in rule_status.items():
                    console.print(f"  [cyan]{filename:<12}[/cyan] {_skill_style(status)}")
            if hook_results:
                console.print("[bold]Hooks[/bold]")
                _display_skills(hook_results)
            if skill_results:
                if hook_results or rule_status:
                    console.print()
                console.print("[bold]Agent skills[/bold]")
                _display_skills(skill_results)
            if mcp_results:
                if skill_results or hook_results or rule_status:
                    console.print()
                console.print("[bold]MCP clients[/bold]")
                _display_registrations(mcp_results)
            console.print()
            render_index_stats_table(stats)
        # Every client has been reported above; any that failed still fails the run.
        failed = (
            tuple(f"skill {name}" for name, status in skill_results if status.startswith("error:"))
            + tuple(f"hook {name}" for name, status in hook_results if status.startswith("error:"))
            + tuple(f"mcp {name}" for name in mcp_install.failed_clients(mcp_results))
        )
        if failed:
            _exit_failed(failed, payload)
        return payload if is_json_mode() else None
