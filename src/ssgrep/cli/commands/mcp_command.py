"""MCP server command: start the stdio server or install client registrations."""

from __future__ import annotations

import sys
from typing import Annotated

from usecli import Argument, BaseCommand, console
from usecli.cli.core.runtime import is_json_mode, is_quiet

from ssgrep.cli import exit_codes
from ssgrep.services import mcp_install


def _display_registrations(results: tuple[mcp_install.InstallResult, ...]) -> None:
    """Print styled per-client registration results with config locations."""
    for name, status, config in results:
        styled = _status_style(status)
        location = f" ({config})" if config else ""
        console.print(f"  [cyan]{name:<10}[/cyan] {styled}{location}")


def _status_style(status: str) -> str:
    if status.startswith("error:"):
        return f"[red]{status}[/red]"
    if status.startswith("skipped:"):
        return f"[gray]{status}[/gray]"
    if status == "already_installed":
        return "[gray]already installed[/gray]"
    if status == "updated":
        return "[yellow]updated[/yellow]"
    if status in ("installed",):
        return "[green]installed[/green]"
    return status


class McpCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "mcp"

    def description(self) -> str:
        return "Start the MCP stdio server (or install client registrations)"

    def handle(
        self,
        action: Annotated[
            str | None,
            Argument(
                help="`install` writes the ssgrep registration into every supported MCP client"
            ),
        ] = None,
        clients: Annotated[
            list[str] | None,
            Argument(help="Optional subset: claude, cursor, zed, codex, opencode"),
        ] = None,
    ) -> dict[str, object] | None:
        if action is not None and action != "install":
            message = f"unknown mcp action: {action!r} (only `ssgrep mcp install` is available)"
            if is_json_mode():
                sys.stderr.write(f"{message}\n")
            else:
                print(message, file=sys.stderr)
            raise SystemExit(exit_codes.USAGE_ERROR)
        if action == "install":
            return self._install(clients)
        self._serve()

    def _install(self, clients: list[str] | None) -> dict[str, object] | None:
        """Register ssgrep with each supported MCP client, independently."""
        try:
            results = mcp_install.install_mcp_registrations(tuple(clients or ()))
        except ValueError as error:
            print(f"Usage error: {error}", file=sys.stderr)
            raise SystemExit(exit_codes.USAGE_ERROR) from error
        if is_json_mode():
            return {
                "ok": True,
                "clients": {
                    name: {"status": status, "config": str(config) if config else None}
                    for name, status, config in results
                },
            }
        if not is_quiet():
            _display_registrations(results)
        return None

    def _serve(self) -> None:
        # usecli loads this file as top-level ``mcp_command``. Defensively
        # evict any stray ``mcp`` shadow (one that is not the real package)
        # before FastMCP imports the actual ``mcp`` package.
        if sys.modules.get("mcp") is not None and not hasattr(sys.modules["mcp"], "types"):
            del sys.modules["mcp"]
        from ssgrep.services.mcp_server import get_mcp_server

        get_mcp_server().run()
